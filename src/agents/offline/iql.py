"""
Implicit Q-Learning (IQL) for GeMS datasets with Dual-Stream E2E GRU
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2110.06169.pdf

Enhancements:
- Dual-Stream End-to-End GRU Architecture
- SwanLab logging support
- TrajectoryReplayBuffer for episode-based sampling

Key Features:
- Expectile regression for V-function
- Advantage Weighted Regression (AWR) for policy
- No explicit Q-target backup
"""
import copy
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from config.offline import paths
from config.offline.config import IQLConfig, auto_generate_paths, auto_generate_swanlab_config
from common.offline.buffer import TrajectoryReplayBuffer
from common.offline.utils import set_seed, soft_update
from common.offline.networks import (
    TanhGaussianActor,
    DeterministicActor,
    FixedGaussianActor,
    Critic,
    ValueFunction,
    LOG_STD_MIN,
    LOG_STD_MAX,
)
from common.offline.eval_env import OfflineEvalEnv
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.item_embeddings import ItemEmbeddings

# SwanLab Logger
try:
    from common.offline.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available")


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """
    Expectile regression loss (核心IQL loss)
    L(u) = |tau - I(u < 0)| * u^2

    Args:
        diff: Difference tensor (V(s) - Q(s,a))
        expectile: Expectile parameter (default 0.7)

    Returns:
        Expectile loss
    """
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)


def compute_svd_rank(states: torch.Tensor, eps: float = 1e-8) -> tuple:
    """
    Compute SVD-based effective rank and condition number for representation collapse detection.

    Args:
        states: State tensor [batch_size, state_dim]
        eps: Small constant for numerical stability

    Returns:
        effective_rank: Effective rank (normalized by state_dim)
        condition_number: Ratio of max to min singular value
    """
    if states.dim() != 2:
        states = states.view(-1, states.size(-1))

    # Center the data
    states_centered = states - states.mean(dim=0, keepdim=True)

    # SVD decomposition
    try:
        U, S, V = torch.svd(states_centered)

        # Effective rank: (sum of singular values)^2 / sum of squared singular values
        sum_s = S.sum()
        sum_s2 = (S ** 2).sum()
        effective_rank = (sum_s ** 2) / (sum_s2 + eps)

        # Condition number: max / min singular value
        condition_number = S[0] / (S[-1] + eps)

        return effective_rank.item(), condition_number.item()
    except:
        return 0.0, 0.0


def compute_gradient_conflict(grad1: torch.Tensor, grad2: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two gradient tensors.

    Args:
        grad1: First gradient tensor
        grad2: Second gradient tensor
        eps: Small constant for numerical stability

    Returns:
        cos_sim: Cosine similarity (-1 to 1)
                 -1.0 = completely opposite
                  0.0 = orthogonal
                 +1.0 = completely aligned
    """
    # Flatten gradients
    g1_flat = grad1.view(-1)
    g2_flat = grad2.view(-1)

    # Compute cosine similarity
    dot_product = torch.dot(g1_flat, g2_flat)
    norm1 = torch.norm(g1_flat)
    norm2 = torch.norm(g2_flat)

    cos_sim = dot_product / (norm1 * norm2 + eps)

    return cos_sim.item()


class IQLAgent:
    """Implicit Q-Learning Agent with Dual-Stream E2E GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config: IQLConfig,
        ranker_params: Dict,
        ranker=None,  # 🔥 Solution B: Accept ranker for real-time inference
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0
        self.total_it = 0

        # 0. Ranker for real-time action inference (Solution B)
        self.ranker = ranker

        # Extract action normalization parameters from ranker_params
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        
        # 🔥 NEW: Extract dataset latent space global range (for probe)
        self.dataset_center = ranker_params.get('dataset_center', self.action_center).to(self.device)
        self.action_range = ranker_params.get('action_range', self.action_scale * 2).to(self.device)

        # Extract GeMS-trained embeddings from ranker_params
        self.item_embeddings = ranker_params['item_embeddings']
        logging.info(f"✅ Using GeMS-trained embeddings: {self.item_embeddings.num_items} items, "
                    f"{self.item_embeddings.embedd_dim} dims")

        # Calculate explicit input_dim for GRU
        input_dim = config.rec_size * (config.item_embedd_dim + 1)

        # Initialize Dual-Stream GRU
        logging.info("Initializing Dual-Stream GRU belief encoder...")
        self.belief = GRUBelief(
            item_embeddings=self.item_embeddings,
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=self.device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=["actor", "critic"],  # DUAL-STREAM
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim
        )

        # Double-freeze embeddings (after GRUBelief's deepcopy)
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()
        logging.info("✅ Item embeddings frozen (double-checked)")

        # Actor - support multiple architectures
        if config.actor_type == "gaussian":
            self.actor = TanhGaussianActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
            ).to(self.device)
            logging.info(f"✅ Using TanhGaussianActor (learnable variance)")
        elif config.actor_type == "deterministic":
            self.actor = DeterministicActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
            ).to(self.device)
            logging.info(f"✅ Using DeterministicActor (no variance)")
        elif config.actor_type == "fixed_gaussian":
            self.actor = FixedGaussianActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
                fixed_std=config.fixed_std,
            ).to(self.device)
            logging.info(f"✅ Using FixedGaussianActor (fixed_std={config.fixed_std})")
        else:
            raise ValueError(f"Unknown actor_type: {config.actor_type}")


        # Critics
        self.critic_1 = Critic(config.belief_hidden_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_2 = Critic(config.belief_hidden_dim, action_dim, config.hidden_dim).to(self.device)

        # Value function
        self.value = ValueFunction(config.belief_hidden_dim, config.hidden_dim, config.n_hidden).to(self.device)

        # Target critics
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Three separate optimizers (IQL-specific)
        # Value optimizer: includes Value Network + Critic GRU
        self.value_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["critic"].parameters()},
            {'params': self.value.parameters()}
        ], lr=config.value_lr)

        # Critic optimizer: only Critic networks (no GRU)
        self.critic_optimizer = torch.optim.Adam([
            {'params': self.critic_1.parameters()},
            {'params': self.critic_2.parameters()}
        ], lr=config.critic_lr)

        # Actor optimizer: includes Actor + Actor GRU
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.actor_lr)

        logging.info("IQLAgent initialized with Dual-Stream E2E GRU")

    def train(self, batch) -> Dict[str, float]:
        """
        Train one step with IQL loss (Three-step training)

        Step 1: Value Network Update (uses s_critic)
        Step 2: Critic Update (uses s_critic and ns_critic)
        Step 3: Actor Update (uses s_actor for policy, s_critic for advantage)
        """
        self.total_it += 1

        # Dual-Stream GRU forward
        states, next_states = self.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic = states["critic"]
        ns_critic = next_states["critic"]

        # Real-time action inference for label actions (A/B via CLI)
        flat_slates = torch.cat(batch.obs["slate"], dim=0)
        flat_clicks = torch.cat(batch.obs["clicks"], dim=0)

        if self.config.label_click_mode == "real":
            label_clicks = flat_clicks.float()
        else:
            label_clicks = torch.zeros_like(flat_slates, dtype=torch.float32)

        with torch.no_grad():
            true_actions, _ = self.ranker.run_inference(flat_slates, label_clicks)
            true_actions = (true_actions - self.action_center) / self.action_scale

            # 🔥 SAFETY CLAMP: Architecture Alignment
            # Reason: GeMS outputs unbounded space (-∞, +∞), but Actor expects bounded space (-1, 1)
            # This clamp prevents NaN in atanh() and log(1 - x²) computations
            # Using 0.99 instead of 0.999 to leave a larger safety margin
            true_actions = torch.clamp(true_actions, min=-0.99, max=0.99)

        # ========================================================================
        # 🔥 CONVICTION METRICS: Quantify Out-of-Bounds Behavior
        # ========================================================================
        with torch.no_grad():
            # 1. Out-of-Bounds (OOB) Counter
            # Count how many normalized actions exceed [-1, 1] range
            oob_mask = (true_actions.abs() >= 1.0)
            oob_count = oob_mask.sum().item()
            oob_rate = oob_count / true_actions.numel()

            # 2. The "Atanh Test" - Direct Evidence of NaN Source
            # Compute both protected (current) and raw atanh to prove the issue
            action_ratio = true_actions / self.max_action

            # Protected version (what we currently use with clamping)
            action_ratio_protected = torch.clamp(action_ratio, -0.999, 0.999)
            atanh_protected = torch.atanh(action_ratio_protected)
            atanh_protected_has_nan = torch.isnan(atanh_protected).any().item()

            # Raw version (what would happen without clamping)
            # This will produce NaN if |action_ratio| > 1
            # 🔥 FIX: Compute domain violation rate BEFORE clamping for proper diagnosis
            atanh_domain_violation_rate = (action_ratio.abs() > 1.0).float().mean().item()
            atanh_raw = torch.atanh(action_ratio)  # True raw computation without clamp
            atanh_raw_has_nan = torch.isnan(atanh_raw).any().item()
            atanh_raw_has_inf = torch.isinf(atanh_raw).any().item()

            # 3. Extreme Values - Maximum Deviation
            true_action_abs_max = true_actions.abs().max().item()
            action_ratio_abs_max = action_ratio.abs().max().item()

        rewards = torch.cat(batch.rewards, dim=0) if batch.rewards else None
        dones = torch.cat(batch.dones, dim=0) if batch.dones else None

        # ========================================================================
        # Step 1: Value Network Update (Expectile Regression)
        # ========================================================================
        with torch.no_grad():
            # Compute target Q-values
            target_q1, target_q2 = self.critic_1_target.both(s_critic, true_actions)
            target_q = torch.min(target_q1, target_q2)

        # Current V-value (keep gradient flow to GRU)
        current_v = self.value(s_critic)

        # 🔥 Numerical Stability: Clamp V values to prevent explosion
        current_v = torch.clamp(current_v, min=-100.0, max=100.0)

        # Expectile loss
        value_loss = expectile_loss(target_q - current_v, self.config.tau).mean()

        # Optimize value network (retain graph for later use of s_critic)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["critic"].parameters()) + list(self.value.parameters()),
            10.0
        )
        self.value_optimizer.step()

        # ========================================================================
        # Step 2: Critic Update (Standard Bellman Backup)
        # ========================================================================
        with torch.no_grad():
            # Next state value
            next_v = self.value(ns_critic)
            # 🔥 Numerical Stability: Clamp next V values
            next_v = torch.clamp(next_v, min=-100.0, max=100.0)

            if rewards is not None and dones is not None:
                target_q = rewards + (1 - dones) * self.config.gamma * next_v
            else:
                target_q = next_v * self.config.gamma

            # 🔥 Numerical Stability: Clamp target Q values to prevent explosion
            target_q = torch.clamp(target_q, min=-100.0, max=100.0)

        # Current Q-values (detach s_critic to avoid gradient conflict)
        # Reason: Value optimizer already updated GRU in Step 1
        current_q1, current_q2 = self.critic_1.both(s_critic.detach(), true_actions)

        # 🔥 Numerical Stability: Clamp current Q values
        current_q1 = torch.clamp(current_q1, min=-100.0, max=100.0)
        current_q2 = torch.clamp(current_q2, min=-100.0, max=100.0)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            10.0
        )
        self.critic_optimizer.step()

        # ========================================================================
        # Step 3: Actor Update (Advantage Weighted Regression)
        # ========================================================================

        # 🔍 FORENSIC MONITOR 0A: Input Health (GRU state)
        with torch.no_grad():
            s_actor_has_nan = torch.isnan(s_actor).any().item()
            s_actor_has_inf = torch.isinf(s_actor).any().item()
            s_actor_min = s_actor.min().item()
            s_actor_max = s_actor.max().item()
            s_actor_mean = s_actor.mean().item()

        # 🔍 FORENSIC MONITOR 0B: Target Health (GeMS output)
        with torch.no_grad():
            true_actions_has_nan = torch.isnan(true_actions).any().item()
            true_actions_has_inf = torch.isinf(true_actions).any().item()
            true_actions_min = true_actions.min().item()
            true_actions_max = true_actions.max().item()
            true_actions_mean = true_actions.mean().item()

        with torch.no_grad():
            # Compute advantage using s_critic
            v = self.value(s_critic.detach())
            q1, q2 = self.critic_1.both(s_critic.detach(), true_actions)
            q = torch.min(q1, q2)
            advantage = q - v

            # 🔍 FORENSIC MONITOR 1: Advantage Extremes (before clipping)
            advantage_max_raw = advantage.max().item()
            advantage_min_raw = advantage.min().item()

            # Compute weights (clamp before exp to prevent overflow)
            advantage_scaled = advantage * self.config.beta

            # 🔍 FORENSIC MONITOR 2: Weight Explosion (before clipping)
            weight_before_clip_max = advantage_scaled.max().item()
            weight_before_clip_min = advantage_scaled.min().item()

            advantage_clipped = torch.clamp(advantage_scaled, min=-5.0, max=5.0)
            exp_adv = torch.exp(advantage_clipped)

            # 🔍 FORENSIC MONITOR 3: Weight Explosion (after exp)
            weight_max = exp_adv.max().item()
            weight_mean = exp_adv.mean().item()

        # FIX P2: align true_actions to the valid atanh domain [-0.999, 0.999]
        # The previous clamp to [-3.0, 3.0] was dead code (true_actions already
        # clamped to [-0.99, 0.99] above) and misaligned with networks.py safe_action
        true_actions_clamped = torch.clamp(true_actions, min=-0.999, max=0.999)

        # 🔍 FORENSIC MONITOR 4: Policy Internal Diagnostics
        # Branch by actor_type to avoid AttributeError
        with torch.no_grad():
            if self.config.actor_type == "deterministic":
                # Deterministic actor: only has mu, no log_std
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)

                # Actor mu statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                # ⚠️ Dummy values for log_std (deterministic has no variance)
                actor_log_std_raw_min = 0.0
                actor_log_std_raw_mean = 0.0
                actor_log_std_raw_max = 0.0
                actor_log_std_min = 0.0
                actor_log_std_mean = 0.0
                actor_log_std_max = 0.0
                actor_log_std_floor_hit_rate = 0.0
                actor_log_std_ceiling_hit_rate = 0.0

                # Jacobian term (still relevant for action clamping)
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared
                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # Mu distribution analysis
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()
                mu_iqr = mu_p75 - mu_p10
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

                # ⚠️ Dummy values for per-dimension log_std analysis
                max_per_dim_floor_hit = 0.0
                mean_per_dim_floor_hit = 0.0
                num_dims_with_high_floor_hit = 0

            elif self.config.actor_type == "fixed_gaussian":
                # Fixed Gaussian: mu is learnable, log_std is fixed buffer
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)
                actor_log_std = self.actor.log_std  # Fixed buffer, not a function call

                # Actor mu statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                # Log_std statistics (fixed, so raw = clamped)
                actor_log_std_raw_min = actor_log_std.min().item()
                actor_log_std_raw_mean = actor_log_std.mean().item()
                actor_log_std_raw_max = actor_log_std.max().item()
                actor_log_std_min = actor_log_std.min().item()
                actor_log_std_mean = actor_log_std.mean().item()
                actor_log_std_max = actor_log_std.max().item()
                # Floor/ceiling hit rate = 0 (log_std is fixed, never hits bounds)
                actor_log_std_floor_hit_rate = 0.0
                actor_log_std_ceiling_hit_rate = 0.0

                # Jacobian term
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared
                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # Mu distribution analysis
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()
                mu_iqr = mu_p75 - mu_p10
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

                # ⚠️ Dummy values for per-dimension log_std analysis (fixed variance)
                max_per_dim_floor_hit = 0.0
                mean_per_dim_floor_hit = 0.0
                num_dims_with_high_floor_hit = 0

            else:
                # Gaussian actor: learnable mu and log_std
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)
                actor_log_std_raw = self.actor.log_std(hidden)
                actor_log_std = torch.clamp(actor_log_std_raw, min=LOG_STD_MIN, max=LOG_STD_MAX)

                # Actor output statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                actor_log_std_raw_min = actor_log_std_raw.min().item()
                actor_log_std_raw_mean = actor_log_std_raw.mean().item()
                actor_log_std_raw_max = actor_log_std_raw.max().item()

                actor_log_std_min = actor_log_std.min().item()
                actor_log_std_mean = actor_log_std.mean().item()
                actor_log_std_max = actor_log_std.max().item()
                actor_log_std_floor_hit_rate = (actor_log_std_raw <= LOG_STD_MIN).float().mean().item()
                actor_log_std_ceiling_hit_rate = (actor_log_std_raw >= LOG_STD_MAX).float().mean().item()

                # 🔍 FORENSIC MONITOR 4B: Gaussian Intermediate Terms
                # Compute the problematic term: 1 - (action / max_action)^2
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared

                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # 🔍 FORENSIC MONITOR 4C: Distribution Analysis (Gemini suggestion)
                # 1. Tanh Saturation - ratio of mu values pushed to boundaries
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()

                # 2. Mu Distribution Percentiles (Q10, Q25, Q50, Q75, Q90)
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()

                # 3. Per-dimension log_std analysis (32 dimensions)
                # Compute floor hit rate per dimension
                per_dim_floor_hit = (actor_log_std_raw <= LOG_STD_MIN).float().mean(dim=0)  # [32]
                per_dim_log_std_mean = actor_log_std.mean(dim=0)  # [32]

                # Summary stats for per-dimension analysis
                max_per_dim_floor_hit = per_dim_floor_hit.max().item()
                mean_per_dim_floor_hit = per_dim_floor_hit.mean().item()
                num_dims_with_high_floor_hit = (per_dim_floor_hit > 0.5).sum().item()  # How many dims have >50% floor hit

                # Mu interquartile range (IQR) - measure of distribution spread
                mu_iqr = mu_p75 - mu_p10

                # 🔥 NEW: Policy distance to origin (data mean) - for BC gravity evidence
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

        # 🔥 Actor Loss - branch by actor_type
        if self.config.actor_type == "deterministic":
            # Deterministic actor: use weighted MSE instead of log_prob
            actor_mu = self.actor.get_mu(s_actor)
            actor_action = torch.tanh(actor_mu) * self.max_action
            # Weighted MSE: exp_adv weights each sample
            mse_per_sample = F.mse_loss(actor_action, true_actions_clamped, reduction='none').mean(dim=-1, keepdim=True)
            awr_loss = (exp_adv * mse_per_sample).mean()

            # BC loss (same as before)
            bc_loss = F.mse_loss(actor_action, true_actions_clamped)

            # No log_prob for deterministic actor
            policy_entropy = 0.0
            log_prob_min_raw = 0.0
            log_prob_max_raw = 0.0
        else:
            # Gaussian actors (gaussian or fixed_gaussian): use log_prob
            log_prob = self.actor.log_prob(s_actor, true_actions_clamped)

            # 🔍 FORENSIC MONITOR 5: Log Probability Stability (before clipping)
            log_prob_min_raw = log_prob.min().item()
            log_prob_max_raw = log_prob.max().item()

            # NOTE: clamp is now inside networks.py log_prob() at -100 instead of -20
            # No additional clamp here - log_prob already has gradient-safe range

            # 🔥 NEW METRIC: Policy Entropy (监控策略是否坍缩)
            policy_entropy = -log_prob.mean().item()

            # AWR loss
            awr_loss = -(exp_adv * log_prob).mean()

            # FIX P0: BC Loss - provides stable gradient independent of Advantage
            # When AWR weights are near-uniform (advantage≈0), bc_loss keeps Actor
            # tracking the data distribution and prevents gradient death
            actor_mu, _ = self.actor(s_actor, deterministic=True)
            bc_loss = F.mse_loss(actor_mu, true_actions_clamped)

        lambda_bc = getattr(self.config, 'lambda_bc', 0.5)
        actor_loss = awr_loss + lambda_bc * bc_loss

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["actor"].parameters()) + list(self.actor.parameters()),
            1.0  # FIX Phase 3: reduced from 10.0 — smaller steps prevent overshooting optimal policy
        )
        self.actor_optimizer.step()

        # Update target networks (use iql_tau for soft update, not expectile tau)
        soft_update(self.critic_1_target, self.critic_1, self.config.iql_tau)
        soft_update(self.critic_2_target, self.critic_2, self.config.iql_tau)

        # ========================================================================
        # 🔥 Enhanced Monitoring Metrics
        # ========================================================================
        # 🔥 OOD Distance Probe: Actor 输出与数据集真实动作的距离
        with torch.no_grad():
            # 确定性动作的 OOD 距离
            pred_actions_det, _ = self.actor(s_actor, deterministic=True)
            ood_distances_det = torch.norm(pred_actions_det - true_actions_clamped, dim=-1)
            ood_distance_mean_det = ood_distances_det.mean().item()
            ood_distance_max_det = ood_distances_det.max().item()

            # 采样动作的 OOD 距离（仅 Gaussian Actor）
            if self.config.actor_type in ["gaussian", "fixed_gaussian"]:
                pred_actions_samp, _ = self.actor(s_actor, deterministic=False)
                ood_distances_samp = torch.norm(pred_actions_samp - true_actions_clamped, dim=-1)
                ood_distance_mean_samp = ood_distances_samp.mean().item()
                ood_distance_max_samp = ood_distances_samp.max().item()
            else:
                ood_distance_mean_samp = 0.0
                ood_distance_max_samp = 0.0
            
            # 🔥 NEW: 潜空间全局定位探针 (到数据集中心/边界的距离)
            # 1. 到数据集中心的距离
            dist_to_center = torch.norm(pred_actions_det - self.dataset_center.to(pred_actions_det.device), dim=-1)
            z_to_dataset_center_mean = dist_to_center.mean().item()
            
            # 2. 到数据集边界的距离 (相对值，0=在中心，1=在边界)
            pred_normalized = (pred_actions_det - self.dataset_center.to(pred_actions_det.device)) / (self.action_range.to(pred_actions_det.device) / 2 + 1e-6)
            dist_to_boundary = torch.abs(pred_normalized).max(dim=-1).values
            z_to_boundary_mean = dist_to_boundary.mean().item()

        # 🔥 采样时序监控：batch 平均 reward
        if batch.rewards is not None:
            batch_reward_mean = torch.cat(batch.rewards, dim=0).mean().item()
        else:
            batch_reward_mean = 0.0

        metrics = {
            # Loss metrics
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "awr_loss": awr_loss.item(),
            "bc_loss": bc_loss.item(),

            # V-value statistics (enhanced)
            "v_value_mean": current_v.mean().item(),
            "v_value_min": current_v.min().item(),
            "v_value_max": current_v.max().item(),
            "v_value_std": current_v.std().item(),

            # Q-value statistics (enhanced)
            "q_value_mean": current_q1.mean().item(),
            "q_value_min": current_q1.min().item(),
            "q_value_max": current_q1.max().item(),
            "q_value_std": current_q1.std().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_min": target_q.min().item(),
            "target_q_max": target_q.max().item(),

            # TD error
            "td_error": (current_q1 - target_q).abs().mean().item(),

            # Advantage statistics (basic)
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item(),

            # 🔍 FORENSIC LEVEL 0: Input & Target Health
            "s_actor_has_nan": s_actor_has_nan,
            "s_actor_has_inf": s_actor_has_inf,
            "s_actor_min": s_actor_min,
            "s_actor_max": s_actor_max,
            "s_actor_mean": s_actor_mean,
            "true_actions_has_nan": true_actions_has_nan,
            "true_actions_has_inf": true_actions_has_inf,
            "true_actions_min": true_actions_min,
            "true_actions_max": true_actions_max,
            "true_actions_mean": true_actions_mean,

            # 🔍 FORENSIC LEVEL 1: Advantage Extremes
            "advantage_max": advantage_max_raw,
            "advantage_min": advantage_min_raw,

            # 🔍 FORENSIC: Weight Explosion Monitor
            "weight_before_clip_max": weight_before_clip_max,
            "weight_before_clip_min": weight_before_clip_min,
            "weight_max": weight_max,
            "weight_mean": weight_mean,

            # 🔥 NEW METRICS: AWR Weight Distribution & Policy Entropy
            "awr_weight_std": exp_adv.std().item(),
            "policy_entropy": policy_entropy,

            # 🔍 FORENSIC: Policy Internal Diagnostics
            "actor_mu_min": actor_mu_min,
            "actor_mu_max": actor_mu_max,
            "actor_mu_mean": actor_mu_mean,
            # Distribution percentiles
            "mu_p10": mu_p10,
            "mu_p25": mu_p25,
            "mu_p50": mu_p50,
            "mu_p75": mu_p75,
            "mu_p90": mu_p90,
            "mu_iqr": mu_iqr,
            # Tanh saturation
            "tanh_saturation_ratio": tanh_saturation_ratio,
            # Per-dimension analysis
            "max_per_dim_floor_hit": max_per_dim_floor_hit,
            "mean_per_dim_floor_hit": mean_per_dim_floor_hit,
            "num_dims_with_high_floor_hit": num_dims_with_high_floor_hit,
            # Log std stats
            "actor_log_std_raw_min": actor_log_std_raw_min,
            "actor_log_std_raw_mean": actor_log_std_raw_mean,
            "actor_log_std_raw_max": actor_log_std_raw_max,
            "actor_log_std_min": actor_log_std_min,
            "actor_log_std_mean": actor_log_std_mean,
            "actor_log_std_max": actor_log_std_max,
            "actor_log_std_floor_hit_rate": actor_log_std_floor_hit_rate,
            "actor_log_std_ceiling_hit_rate": actor_log_std_ceiling_hit_rate,

            # 🔍 FORENSIC: Jacobian Term (1 - (action/max_action)^2)
            "jacobian_term_min": jacobian_term_min,
            "jacobian_term_max": jacobian_term_max,
            "jacobian_term_has_negative": jacobian_term_has_negative,
            "jacobian_term_has_nan": jacobian_term_has_nan,

            # 🔍 FORENSIC: Log Probability Stability
            "log_prob_min_raw": log_prob_min_raw,
            "log_prob_max_raw": log_prob_max_raw,

            # 🔥 CONVICTION METRICS: Quantitative Proof of OOB Problem
            "oob_count": oob_count,
            "oob_rate": oob_rate,
            "atanh_protected_has_nan": atanh_protected_has_nan,
            "atanh_raw_has_nan": atanh_raw_has_nan,
            "atanh_raw_has_inf": atanh_raw_has_inf,
            "atanh_domain_violation_rate": atanh_domain_violation_rate,
            "true_action_abs_max": true_action_abs_max,
            "action_ratio_abs_max": action_ratio_abs_max,

            # Gradient norms
            "value_grad_norm": value_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),

            # 🔥 NEW: Policy distance to origin (BC gravity evidence)
            "policy_distance_to_origin": policy_distance_to_origin,

            # 🔥 OOD Distance Probe: Actor 输出与数据集真实动作的距离
            "ood_distance_mean_det": ood_distance_mean_det,
            "ood_distance_max_det": ood_distance_max_det,
            "ood_distance_mean_samp": ood_distance_mean_samp,
            "ood_distance_max_samp": ood_distance_max_samp,

            # 🔥 NEW: 潜空间全局定位探针
            "z_to_dataset_center_mean": z_to_dataset_center_mean,
            "z_to_boundary_mean": z_to_boundary_mean,

            # 🔥 采样时序监控探针
            "batch_reward_mean": batch_reward_mean,
        }

        return metrics

    def compute_representation_diagnostics(self, batch) -> Dict[str, float]:
        """
        Compute representation diagnostics for monitoring training health.

        Returns:
            Dictionary with SVD rank and representation consistency metrics
        """
        # Forward pass to get states
        states, _ = self.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic = states["critic"]

        # Compute SVD rank for both streams
        actor_rank, actor_condition = compute_svd_rank(s_actor)
        critic_rank, critic_condition = compute_svd_rank(s_critic)

        # Compute representation consistency (cosine similarity between actor and critic states)
        s_actor_flat = s_actor.view(-1)
        s_critic_flat = s_critic.view(-1)

        dot_product = torch.dot(s_actor_flat, s_critic_flat)
        norm_actor = torch.norm(s_actor_flat)
        norm_critic = torch.norm(s_critic_flat)

        representation_consistency = (dot_product / (norm_actor * norm_critic + 1e-8)).item()

        return {
            "actor_svd_rank": actor_rank,
            "actor_condition_number": actor_condition,
            "critic_svd_rank": critic_rank,
            "critic_condition_number": critic_condition,
            "representation_consistency": representation_consistency,
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        Select action using Actor GRU and decode to slate.

        Returns:
            slate: numpy array of shape [rec_size] containing item IDs
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # Use Actor GRU only
        belief_state = self.belief.forward(obs_tensor, done=False)["actor"]

        # Actor prediction
        raw_action, _ = self.actor(belief_state, deterministic=deterministic, need_log_prob=False)

        # Denormalize
        latent_action = raw_action * self.action_scale + self.action_center

        # 🔥 NEW: 使用 GeMS ranker 解码 latent action 为 slate
        if self.ranker is None:
            raise RuntimeError(
                "IQLAgent.act() requires a ranker for slate decoding. "
                "Please provide ranker during initialization."
            )

        # 🔧 FIX: 确保设备一致性 - 使用 ranker 的设备而非 agent 的设备
        # 原因：ranker 可能在不同设备上（CPU/CUDA/CUDA:0/CUDA:1）
        ranker_device = next(self.ranker.parameters()).device
        latent_action = latent_action.to(ranker_device)

        # 添加 batch 维度 (ranker 期望 [batch_size, latent_dim])
        latent_action_batched = latent_action.unsqueeze(0)  # [1, latent_dim]

        # 解码为 slate
        slate_tensor = self.ranker.rank(latent_action_batched)  # [1, rec_size]

        # 移除 batch 维度并转换为 numpy
        slate_output = slate_tensor.squeeze(0).cpu().numpy()  # [rec_size]

        return slate_output

    def reset_hidden(self):
        """Reset dual-stream GRU hidden states"""
        dummy_obs = {
            "slate": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
            "clicks": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device)
        }
        self.belief.forward(dummy_obs, done=True)

    def save(self, filepath: str):
        """Save model with embeddings metadata for standalone loading"""
        torch.save({
            'belief_state_dict': self.belief.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'total_it': self.total_it,
            'embeddings_meta': {
                'num_items': self.item_embeddings.num_items,
                'embedd_dim': self.item_embeddings.embedd_dim,
            },
            'action_dim': self.action_dim,
            'config': self.config,
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.belief.load_state_dict(checkpoint['belief_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, embedding_path: str, device: torch.device):
        """
        Load IQLAgent from checkpoint without requiring GeMS.

        Args:
            checkpoint_path: Path to saved agent checkpoint
            embedding_path: Path to item embeddings (.pt file)
            device: Device to load model on

        Returns:
            Loaded IQLAgent instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract metadata
        config = checkpoint['config']
        action_dim = checkpoint['action_dim']
        embeddings_meta = checkpoint['embeddings_meta']

        # Load embeddings
        embedding_weights = torch.load(embedding_path, map_location=device)
        item_embeddings = ItemEmbeddings(
            num_items=embeddings_meta['num_items'],
            item_embedd_dim=embeddings_meta['embedd_dim'],
            device=device,
            weights=embedding_weights
        )

        # Freeze embeddings
        for param in item_embeddings.parameters():
            param.requires_grad = False

        # Construct ranker_params
        ranker_params = {
            'item_embeddings': item_embeddings,
            'action_center': checkpoint['action_center'],
            'action_scale': checkpoint['action_scale'],
            'num_items': embeddings_meta['num_items'],
            'item_embedd_dim': embeddings_meta['embedd_dim'],
        }

        # Create agent
        agent = cls(action_dim=action_dim, config=config, ranker_params=ranker_params)

        # Load state dicts
        agent.belief.load_state_dict(checkpoint['belief_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        agent.value.load_state_dict(checkpoint['value_state_dict'])
        agent.total_it = checkpoint['total_it']

        logging.info(f"IQLAgent loaded from {checkpoint_path}")
        return agent


def train_iql(config: IQLConfig):
    """Train IQL with Dual-Stream E2E GRU"""
    timestamp = datetime.now().strftime("%Y%m%d")
    config = auto_generate_paths(config, timestamp)
    config = auto_generate_swanlab_config(config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    ranker_type = getattr(config, 'ranker_type', 'unknown')
    log_filename = f"{config.env_name}_{config.dataset_quality}_{ranker_type}_expectile{config.tau}_seed{config.seed}_{config.run_id}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ],
        force=True
    )

    set_seed(config.seed)
    logging.info(f"Global seed set to {config.seed}")

    # Initialize SwanLab
    swan_logger = None
    if config.use_swanlab and SWANLAB_AVAILABLE:
        try:
            swan_logger = SwanlabLogger(
                project=config.swan_project,
                experiment_name=config.run_name,
                workspace=config.swan_workspace,
                config=config.__dict__,
                mode=config.swan_mode,
                logdir=config.swan_logdir,
            )
            logging.info(f"SwanLab initialized")
        except Exception as e:
            logging.warning(f"SwanLab init failed: {e}")

    logging.info("=" * 80)
    logging.info("=== IQL Training Configuration ===")
    logging.info("=" * 80)
    logging.info(f"Environment: {config.env_name}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Expectile: {config.tau}")
    logging.info(f"Beta: {config.beta}")
    logging.info(f"Log file: {log_filepath}")
    logging.info("=" * 80)

    # Load dataset
    logging.info(f"\nLoading dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)

    logging.info(f"Dataset statistics:")
    logging.info(f"  Slates shape: {dataset['slates'].shape}")
    logging.info(f"  Clicks shape: {dataset['clicks'].shape}")
    logging.info(f"  Next slates shape: {dataset['next_slates'].shape}")
    logging.info(f"  Next clicks shape: {dataset['next_clicks'].shape}")
    logging.info(f"  Total transitions: {len(dataset['slates'])}")

    # Note: Buffer creation moved after action relabeling

    # 🔥 Load ranker dynamically using RankerFactory
    from common.offline.ranker_factory import RankerFactory

    ranker, action_dim, agent_embeddings = RankerFactory.create(
        ranker_type=config.ranker_type,
        config=config,
        device=config.device
    )

    # ========================================================================
    # 🔥 Solution B: Real-time Action Inference (No Pre-computed Actions)
    # ========================================================================
    logging.info("")
    logging.info("✅ Using real-time action inference (on-the-fly from slates/clicks)")
    logging.info("")

    # Get action dimension from ranker (unified interface for all rankers)
    action_dim, _ = ranker.get_action_dim()

    # Create buffer
    replay_buffer = TrajectoryReplayBuffer(device=config.device)
    # Load data (V4 format - no pre-computed actions)
    dataset_dict = {
        'episode_ids': dataset['episode_ids'],
        'slates': dataset['slates'],
        'clicks': dataset['clicks'],
        'next_slates': dataset['next_slates'],  # ✅ Required by TrajectoryReplayBuffer
        'next_clicks': dataset['next_clicks'],  # ✅ Required by TrajectoryReplayBuffer
    }
    if 'rewards' in dataset:
        if config.normalize_rewards:
            dataset_dict['rewards'] = dataset['rewards'] / config.reward_scale
            logging.info(f"⚡ Applied reward scaling: rewards / {config.reward_scale}")
        else:
            dataset_dict['rewards'] = dataset['rewards']
    if 'terminals' in dataset:
        dataset_dict['terminals'] = dataset['terminals']

    replay_buffer.load_d4rl_dataset(dataset_dict)
    logging.info(f"✅ Buffer loaded successfully")

    # ========================================================================
    # 🔥 Compute Action Normalization (TD3+BC approach with fake_clicks=0)
    # ========================================================================
    logging.info("")
    logging.info("Computing action normalization parameters...")
    logging.info(f"Label click mode: {config.label_click_mode}")

    sample_size = min(10000, len(dataset['slates']))
    sample_indices = np.random.choice(len(dataset['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(dataset['slates'][sample_indices], device=config.device, dtype=torch.long)

    if config.label_click_mode == "real":
        sample_clicks = torch.tensor(dataset['clicks'][sample_indices], device=config.device, dtype=torch.float32)
    else:
        sample_clicks = torch.zeros_like(sample_slates, dtype=torch.float32)

    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, sample_clicks)

    action_min = sample_actions.min(dim=0)[0]
    action_max = sample_actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6
    
    # 🔥 NEW: 计算数据集潜空间全局范围 (用于探针)
    dataset_center = (action_max + action_min) / 2  # 潜空间中心点 [latent_dim]
    action_range = action_max - action_min  # 潜空间范围 [latent_dim]

    logging.info(f"✅ Action normalization computed from {sample_size} samples")
    logging.info(f"  center mean: {action_center.mean().item():.6f}")
    logging.info(f"  scale mean: {action_scale.mean().item():.6f}")
    logging.info(f"  dataset_center mean: {dataset_center.mean().item():.6f}")
    logging.info(f"  action_range mean: {action_range.mean().item():.6f}")
    logging.info("")

    # 🔥 agent_embeddings already created by RankerFactory
    # No need to create again - it's returned from factory

    # Freeze embeddings
    for param in agent_embeddings.parameters():
        param.requires_grad = False

    # Construct ranker_params
    ranker_params = {
        'item_embeddings': agent_embeddings,
        'action_center': action_center,
        'action_scale': action_scale,
        'dataset_center': dataset_center,  # 🔥 NEW: for global probe
        'action_range': action_range,  # 🔥 NEW: for global probe
        'num_items': ranker.num_items,
        'item_embedd_dim': config.item_embedd_dim,
    }

    # Initialize agent
    agent = IQLAgent(
        action_dim=action_dim,
        config=config,
        ranker_params=ranker_params,
        ranker=ranker,  # 🔥 Solution B: Pass ranker for real-time inference
    )

    # Initialize evaluation environment
    logging.info(f"\n{'='*80}")
    logging.info(f"Initializing evaluation environment")
    logging.info(f"{'='*80}")

    try:
        eval_env = OfflineEvalEnv(
            env_name=config.env_name,
            dataset_quality=config.dataset_quality,
            device=config.device,
            seed=config.seed,
            verbose=False,
            dataset_path=config.dataset_path  # 🔥 NEW: for frequency probe
        )
        logging.info(f"✅ Evaluation environment initialized for {config.env_name}")
    except Exception as e:
        logging.warning(f"⚠️  Failed to initialize evaluation environment: {e}")
        eval_env = None

    # Training loop
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting IQL training")
    logging.info(f"{'='*80}\n")

    # 🔥 NEW: Best checkpoint tracking
    best_eval_reward = float('-inf')

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        metrics = agent.train(batch)

        # Logging
        if (t + 1) % config.log_freq == 0:
            # 🔥 计算诊断指标（合并到主日志）
            diag_metrics = agent.compute_representation_diagnostics(batch)

            # 🔥 构建分组的 SwanLab 指标字典（13个类别）
            swanlab_metrics = {
                # 1️⃣ Loss指标（3个）
                "1_Loss/actor_loss": metrics['actor_loss'],
                "1_Loss/critic_loss": metrics['critic_loss'],
                "1_Loss/value_loss": metrics['value_loss'],

                # 2️⃣ V-value统计（4个）
                "2_V-value/mean": metrics['v_value_mean'],
                "2_V-value/min": metrics['v_value_min'],
                "2_V-value/max": metrics['v_value_max'],
                "2_V-value/std": metrics['v_value_std'],

                # 3️⃣ Q-value统计（7个）
                "3_Q-value/mean": metrics['q_value_mean'],
                "3_Q-value/min": metrics['q_value_min'],
                "3_Q-value/max": metrics['q_value_max'],
                "3_Q-value/std": metrics['q_value_std'],
                "3_Q-value/target_mean": metrics['target_q_mean'],
                "3_Q-value/target_min": metrics['target_q_min'],
                "3_Q-value/target_max": metrics['target_q_max'],

                # 4️⃣ TD Error（1个）
                "4_TD-Error/td_error": metrics['td_error'],

                # 5️⃣ Advantage统计（4个）
                "5_Advantage/mean": metrics['advantage_mean'],
                "5_Advantage/std": metrics['advantage_std'],
                "5_Advantage/max": metrics['advantage_max'],
                "5_Advantage/min": metrics['advantage_min'],

                # 6️⃣ 输入健康检查（8个）
                "6_Input-Health/s_actor_has_nan": metrics['s_actor_has_nan'],
                "6_Input-Health/s_actor_has_inf": metrics['s_actor_has_inf'],
                "6_Input-Health/s_actor_min": metrics['s_actor_min'],
                "6_Input-Health/s_actor_max": metrics['s_actor_max'],
                "6_Input-Health/true_actions_has_nan": metrics['true_actions_has_nan'],
                "6_Input-Health/true_actions_has_inf": metrics['true_actions_has_inf'],
                "6_Input-Health/true_actions_min": metrics['true_actions_min'],
                "6_Input-Health/true_actions_max": metrics['true_actions_max'],

                # 7️⃣ AWR权重（4个）
                "7_AWR-Weight/before_clip_max": metrics['weight_before_clip_max'],
                "7_AWR-Weight/before_clip_min": metrics['weight_before_clip_min'],
                "7_AWR-Weight/max": metrics['weight_max'],
                "7_AWR-Weight/mean": metrics['weight_mean'],

                # 8️⃣ 策略统计（9个 - added distance_to_origin）
                "8_Policy/awr_weight_std": metrics['awr_weight_std'],
                "8_Policy/entropy": metrics['policy_entropy'],
                "8_Policy/mu_min": metrics['actor_mu_min'],
                "8_Policy/mu_max": metrics['actor_mu_max'],
                "8_Policy/mu_mean": metrics['actor_mu_mean'],
                "8_Policy/distance_to_origin": metrics['policy_distance_to_origin'],  # 🔥 NEW: BC gravity evidence
                "8_Policy/z_to_dataset_center_mean": metrics['z_to_dataset_center_mean'],
                "8_Policy/z_to_boundary_mean": metrics['z_to_boundary_mean'],
                "8_Policy/tanh_saturation_ratio": metrics['tanh_saturation_ratio'],
                "8_Policy/mu_iqr": metrics['mu_iqr'],
                "8_Policy/max_per_dim_floor_hit": metrics['max_per_dim_floor_hit'],
                "8_Policy/mean_per_dim_floor_hit": metrics['mean_per_dim_floor_hit'],
                "8_Policy/num_dims_with_high_floor_hit": metrics['num_dims_with_high_floor_hit'],
                "8_Policy/raw_log_std_min": metrics['actor_log_std_raw_min'],
                "8_Policy/raw_log_std_mean": metrics['actor_log_std_raw_mean'],
                "8_Policy/raw_log_std_max": metrics['actor_log_std_raw_max'],
                "8_Policy/log_std_min": metrics['actor_log_std_min'],
                "8_Policy/log_std_mean": metrics['actor_log_std_mean'],
                "8_Policy/log_std_max": metrics['actor_log_std_max'],
                "8_Policy/log_std_floor_hit_rate": metrics['actor_log_std_floor_hit_rate'],
                "8_Policy/log_std_ceiling_hit_rate": metrics['actor_log_std_ceiling_hit_rate'],
                "8_Policy/ood_distance_mean_det": metrics['ood_distance_mean_det'],
                "8_Policy/ood_distance_max_det": metrics['ood_distance_max_det'],
                "8_Policy/ood_distance_mean_samp": metrics['ood_distance_mean_samp'],
                "8_Policy/ood_distance_max_samp": metrics['ood_distance_max_samp'],

                # 9️⃣ Jacobian项（4个）
                "9_Jacobian/min": metrics['jacobian_term_min'],
                "9_Jacobian/max": metrics['jacobian_term_max'],
                "9_Jacobian/has_negative": metrics['jacobian_term_has_negative'],
                "9_Jacobian/has_nan": metrics['jacobian_term_has_nan'],

                # 🔟 Log概率（2个）
                "10_LogProb/min_raw": metrics['log_prob_min_raw'],
                "10_LogProb/max_raw": metrics['log_prob_max_raw'],

                # 1️⃣1️⃣ OOB监控（6个）
                "11_OOB/count": metrics['oob_count'],
                "11_OOB/rate": metrics['oob_rate'],
                "11_OOB/atanh_protected_has_nan": metrics['atanh_protected_has_nan'],
                "11_OOB/atanh_raw_has_nan": metrics['atanh_raw_has_nan'],
                "11_OOB/atanh_raw_has_inf": metrics['atanh_raw_has_inf'],
                "11_OOB/atanh_domain_violation_rate": metrics['atanh_domain_violation_rate'],
                "11_OOB/true_action_abs_max": metrics['true_action_abs_max'],
                "11_OOB/action_ratio_abs_max": metrics['action_ratio_abs_max'],

                # 1️⃣2️⃣ 梯度范数（3个）
                "12_Gradient/actor_norm": metrics['actor_grad_norm'],
                "12_Gradient/critic_norm": metrics['critic_grad_norm'],
                "12_Gradient/value_norm": metrics['value_grad_norm'],

                # 1️⃣3️⃣ 表示诊断（5个）
                "13_Representation/actor_svd_rank": diag_metrics['actor_svd_rank'],
                "13_Representation/critic_svd_rank": diag_metrics['critic_svd_rank'],
                "13_Representation/actor_condition_number": diag_metrics['actor_condition_number'],
                "13_Representation/critic_condition_number": diag_metrics['critic_condition_number'],
                "13_Representation/consistency": diag_metrics['representation_consistency'],

                # 1️⃣4️⃣ 采样时序监控（1个）- 🔥 数据顺序效应探针
                "Batch_Data/reward_mean": metrics['batch_reward_mean'],
            }

            # 完整分类日志记录
            progress_pct = (t + 1) / config.max_timesteps * 100
            logging.info(f"[Training] Step {t+1}/{config.max_timesteps} ({progress_pct:.1f}%)")

            # 1. Loss指标
            logging.info(f"  [1] Loss: actor={metrics['actor_loss']:.6f}, critic={metrics['critic_loss']:.6f}, value={metrics['value_loss']:.6f}")

            # 2. V-value统计
            logging.info(f"  [2] V-value: mean={metrics['v_value_mean']:.6f}, min={metrics['v_value_min']:.6f}, max={metrics['v_value_max']:.6f}, std={metrics['v_value_std']:.6f}")

            # 3. Q-value统计
            logging.info(f"  [3] Q-value: mean={metrics['q_value_mean']:.6f}, min={metrics['q_value_min']:.6f}, max={metrics['q_value_max']:.6f}, std={metrics['q_value_std']:.6f}, target_mean={metrics['target_q_mean']:.6f}, target_min={metrics['target_q_min']:.6f}, target_max={metrics['target_q_max']:.6f}")

            # 4. TD Error
            logging.info(f"  [4] TD-Error: td_error={metrics['td_error']:.6f}")

            # 5. Advantage统计
            logging.info(f"  [5] Advantage: mean={metrics['advantage_mean']:.6f}, std={metrics['advantage_std']:.6f}, max={metrics['advantage_max']:.6f}, min={metrics['advantage_min']:.6f}")

            # 6. 输入健康检查
            logging.info(f"  [6] Input-Health: s_actor(nan={metrics['s_actor_has_nan']:.1f}, inf={metrics['s_actor_has_inf']:.1f}, min={metrics['s_actor_min']:.6f}, max={metrics['s_actor_max']:.6f}), true_actions(nan={metrics['true_actions_has_nan']:.1f}, inf={metrics['true_actions_has_inf']:.1f}, min={metrics['true_actions_min']:.6f}, max={metrics['true_actions_max']:.6f})")

            # 7. AWR权重
            logging.info(f"  [7] AWR-Weight: before_clip(max={metrics['weight_before_clip_max']:.6f}, min={metrics['weight_before_clip_min']:.6f}), max={metrics['weight_max']:.6f}, mean={metrics['weight_mean']:.6f}")

            # 8. 策略统计（新增 z_center, z_boundary, tanh_sat, mu_iqr, per_dim_floor）
            logging.info(
                f"  [8] Policy: awr_weight_std={metrics['awr_weight_std']:.6f}, entropy={metrics['policy_entropy']:.6f}, "
                f"mu(min={metrics['actor_mu_min']:.6f}, max={metrics['actor_mu_max']:.6f}, mean={metrics['actor_mu_mean']:.6f}), "
                f"z_center={metrics['z_to_dataset_center_mean']:.4f}, z_boundary={metrics['z_to_boundary_mean']:.4f}, "
                f"tanh_sat={metrics['tanh_saturation_ratio']:.4f}, mu_iqr={metrics['mu_iqr']:.4f}, "
                f"per_dim_floor(max={metrics['max_per_dim_floor_hit']:.4f}, mean={metrics['mean_per_dim_floor_hit']:.4f}, dims>0.5={int(metrics['num_dims_with_high_floor_hit'])}), "
                f"log_std_raw(min={metrics['actor_log_std_raw_min']:.6f}, mean={metrics['actor_log_std_raw_mean']:.6f}, max={metrics['actor_log_std_raw_max']:.6f}), "
                f"log_std_clamped(min={metrics['actor_log_std_min']:.6f}, mean={metrics['actor_log_std_mean']:.6f}, max={metrics['actor_log_std_max']:.6f}, "
                f"floor_hit={metrics['actor_log_std_floor_hit_rate']:.4f}, ceil_hit={metrics['actor_log_std_ceiling_hit_rate']:.4f}), "
                f"ood_det(mean={metrics['ood_distance_mean_det']:.4f}, max={metrics['ood_distance_max_det']:.4f}), "
                f"ood_samp(mean={metrics['ood_distance_mean_samp']:.4f}, max={metrics['ood_distance_max_samp']:.4f})"
            )

            # 9. Jacobian项
            logging.info(f"  [9] Jacobian: min={metrics['jacobian_term_min']:.6f}, max={metrics['jacobian_term_max']:.6f}, has_negative={metrics['jacobian_term_has_negative']:.1f}, has_nan={metrics['jacobian_term_has_nan']:.1f}")

            # 10. Log概率
            logging.info(f"  [10] LogProb: min_raw={metrics['log_prob_min_raw']:.6f}, max_raw={metrics['log_prob_max_raw']:.6f}")

            # 11. OOB监控
            logging.info(f"  [11] OOB: count={metrics['oob_count']:.1f}, rate={metrics['oob_rate']:.6f}, atanh_protected_nan={metrics['atanh_protected_has_nan']:.1f}, atanh_raw_nan={metrics['atanh_raw_has_nan']:.1f}, atanh_raw_inf={metrics['atanh_raw_has_inf']:.1f}, domain_violation_rate={metrics['atanh_domain_violation_rate']:.6f}, true_action_abs_max={metrics['true_action_abs_max']:.6f}, action_ratio_abs_max={metrics['action_ratio_abs_max']:.6f}")

            # 12. 梯度范数
            logging.info(f"  [12] Gradient: actor_norm={metrics['actor_grad_norm']:.6f}, critic_norm={metrics['critic_grad_norm']:.6f}, value_norm={metrics['value_grad_norm']:.6f}")

            # 13. 表示诊断
            logging.info(f"  [13] Representation: actor_svd_rank={diag_metrics['actor_svd_rank']:.1f}, critic_svd_rank={diag_metrics['critic_svd_rank']:.1f}, actor_condition={diag_metrics['actor_condition_number']:.6f}, critic_condition={diag_metrics['critic_condition_number']:.6f}, consistency={diag_metrics['representation_consistency']:.6f}")
            logging.info("")  # 空行分隔

            if swan_logger:
                swan_logger.log_metrics(swanlab_metrics, step=t+1)

        # Evaluation
        if eval_env is not None and (t + 1) % config.eval_freq == 0:
            eval_metrics = eval_env.evaluate_policy(
                agent=agent,
                num_episodes=config.eval_episodes,
                deterministic=True
            )

            # 简洁的 Evaluation 日志
            logging.info(f"[Evaluation] Step {t+1}/{config.max_timesteps}")
            logging.info(
                f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} "
                f"(min={eval_metrics.get('min_reward', 0):.2f}, max={eval_metrics.get('max_reward', 0):.2f})"
            )
            logging.info(
                f"  Median Reward: {eval_metrics.get('median_reward', 0.0):.2f} | "
                f"IQM: {eval_metrics.get('iqm_reward', eval_metrics['mean_reward']):.2f}"
            )
            logging.info(f"  Episode Length: {eval_metrics['mean_episode_length']:.1f}")
            logging.info("")  # 空行分隔

            if swan_logger:
                swan_logger.log_metrics({
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/std_reward': eval_metrics['std_reward'],
                    'eval/median_reward': eval_metrics.get('median_reward', eval_metrics['mean_reward']),
                    'eval/iqm_reward': eval_metrics.get('iqm_reward', eval_metrics['mean_reward']),
                    'eval/mean_episode_length': eval_metrics['mean_episode_length'],
                    # 🔥 Add early termination rate for Boredom evidence
                    'eval/early_termination_rate': eval_metrics.get('early_termination_rate', 0.0),
                }, step=t+1)

            # 🔥 NEW: Save best checkpoint
            best_metric_name = getattr(config, "best_checkpoint_metric", "iqm")
            if best_metric_name == "iqm":
                current_reward = eval_metrics.get('iqm_reward', eval_metrics['mean_reward'])
            elif best_metric_name == "median":
                current_reward = eval_metrics.get('median_reward', eval_metrics['mean_reward'])
            else:
                current_reward = eval_metrics['mean_reward']
            if current_reward > best_eval_reward:
                best_eval_reward = current_reward
                best_checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f"iql_{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_seed{config.seed}_{config.run_id}_best.pt"
                )
                agent.save(best_checkpoint_path)
                logging.info(
                    f"🏆 New best model saved! {best_metric_name}={best_eval_reward:.2f} at step {t+1}"
                )
                logging.info(f"   Checkpoint: {best_checkpoint_path}")

                # 🔥 NEW: 自动触发Ranker蝴蝶效应测试（Perturbation Sensitivity Test）
                if config.enable_perturbation_test:
                    import json
                    logging.info("")
                    logging.info("=" * 80)
                    logging.info("🔬 Triggering Perturbation Sensitivity Test (Ranker Butterfly Effect)")
                    logging.info("=" * 80)
                    logging.info(f"Baseline IQM: {best_eval_reward:.2f}")
                    logging.info(f"Testing noise levels: {config.perturbation_sigmas}")
                    logging.info("")

                    perturbation_results = {}
                    for sigma in config.perturbation_sigmas:
                        logging.info(f"📊 Testing σ={sigma}...")

                        # 运行加噪评估
                        perturbed_metrics = eval_env.evaluate_policy(
                            agent=agent,
                            num_episodes=config.perturbation_episodes,
                            deterministic=True,
                            noise_sigma=sigma,
                            return_hamming_stats=True
                        )

                        # 计算对比指标
                        baseline_iqm = best_eval_reward
                        perturbed_iqm = perturbed_metrics['iqm_reward']
                        reward_drop_pct = (baseline_iqm - perturbed_iqm) / baseline_iqm * 100 if baseline_iqm > 0 else 0.0
                        hamming_dist = perturbed_metrics.get('hamming_distance_mean', 0.0)
                        hamming_std = perturbed_metrics.get('hamming_distance_std', 0.0)
                        early_term_rate = perturbed_metrics.get('early_termination_rate', 0.0)

                        perturbation_results[f"sigma_{sigma}"] = {
                            'iqm_reward': float(perturbed_iqm),
                            'reward_drop_pct': float(reward_drop_pct),
                            'hamming_distance_mean': float(hamming_dist),
                            'hamming_distance_std': float(hamming_std),
                            'early_termination_rate': float(early_term_rate),
                        }

                        logging.info(f"  IQM: {baseline_iqm:.2f} → {perturbed_iqm:.2f} (↓{reward_drop_pct:.1f}%)")
                        logging.info(f"  Hamming Distance: {hamming_dist:.3f} ± {hamming_std:.3f} ({hamming_dist*10:.1f} items changed)")
                        logging.info(f"  Early Termination Rate: {early_term_rate:.1%}")
                        logging.info("")

                    # 保存结果到JSON（专门的motivation_test目录）
                    # 🔥 NEW: 保存到专门的动机实验目录
                    motivation_json_dir = "/data/liyuefeng/offline-slate-rl/motivation_test/json/perturbation_butterfly_effect"
                    os.makedirs(motivation_json_dir, exist_ok=True)

                    json_filename = f"perturbation_tau{config.tau}_beta{config.beta}_seed{config.seed}_{config.run_id}_step{t+1}.json"
                    perturbation_log_path = os.path.join(motivation_json_dir, json_filename)

                    perturbation_data = {
                        'step': t+1,
                        'baseline_iqm': float(baseline_iqm),
                        'config': {
                            'tau': config.tau,
                            'beta': config.beta,
                            'seed': config.seed,
                            'ranker_type': config.ranker_type,
                            'lambda_bc': config.lambda_bc,
                            'experiment_tag': config.experiment_tag,
                        },
                        'results': perturbation_results
                    }

                    with open(perturbation_log_path, 'w') as f:
                        json.dump(perturbation_data, f, indent=2)

                    logging.info(f"✅ Perturbation test completed! Results saved to:")
                    logging.info(f"   {perturbation_log_path}")
                    logging.info("=" * 80)
                    logging.info("")

        # Save checkpoint
        if (t + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"iql_{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Final save
    final_path = os.path.join(
        config.checkpoint_dir,
        f"iql_{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_final.pt"
    )
    agent.save(final_path)

    # Final evaluation
    if eval_env is not None:
        logging.info(f"\n{'='*80}")
        logging.info(f"Final Evaluation")
        logging.info(f"{'='*80}")

        final_eval_metrics = eval_env.evaluate_policy(
            agent=agent,
            num_episodes=config.final_eval_episodes,
            deterministic=True
        )

        logging.info(f"Final Results:")
        logging.info(f"  Mean Reward: {final_eval_metrics['mean_reward']:.2f} ± {final_eval_metrics['std_reward']:.2f}")
        logging.info(
            f"  Median Reward: {final_eval_metrics.get('median_reward', 0.0):.2f} | "
            f"IQM: {final_eval_metrics.get('iqm_reward', final_eval_metrics['mean_reward']):.2f}"
        )
        logging.info(f"  Mean Episode Length: {final_eval_metrics['mean_episode_length']:.2f}")

        if swan_logger:
            swan_logger.log_metrics({
                'final_eval/mean_reward': final_eval_metrics['mean_reward'],
                'final_eval/std_reward': final_eval_metrics['std_reward'],
                'final_eval/median_reward': final_eval_metrics.get('median_reward', final_eval_metrics['mean_reward']),
                'final_eval/iqm_reward': final_eval_metrics.get('iqm_reward', final_eval_metrics['mean_reward']),
                'final_eval/mean_episode_length': final_eval_metrics['mean_episode_length'],
                # 🔥 Add early termination rate for Boredom evidence
                'final_eval/early_termination_rate': final_eval_metrics.get('early_termination_rate', 0.0),
            }, step=config.max_timesteps)

    logging.info(f"\n{'='*80}")
    logging.info(f"IQL training completed!")
    logging.info(f"{'='*80}")

    if swan_logger:
        swan_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IQL (Implicit Q-Learning) on offline datasets")

    # Experiment configuration
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment",
                        help="实验名称")
    parser.add_argument("--experiment_tag", type=str, default="",
                        help="实验标签 (用于 swanlab 实验名称，如 'exp05_beta3')")
    parser.add_argument("--env_name", type=str, default="diffuse_mix",
                        help="环境名称")
    parser.add_argument("--dataset_quality", type=str, default="expert",
                        help="数据集质量 (旧benchmark: random/medium/expert, 新benchmark: v2_b3/v2_b5)")
    parser.add_argument("--seed", type=int, default=58407201,
                        help="随机种子")
    parser.add_argument("--run_id", type=str, default="",
                        help="唯一运行标识符 (格式: MMDD_HHMM, 如果为空则自动生成)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="",
                        help="数据集路径 (如果为空则自动生成)")

    # Training configuration
    parser.add_argument("--max_timesteps", type=int, default=int(1e6),
                        help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="批次大小")
    parser.add_argument("--eval_freq", type=int, default=500,
                        help="评估频率 (训练步数)")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="每次评估回合数")
    parser.add_argument("--final_eval_episodes", type=int, default=100,
                        help="最终评估回合数")
    parser.add_argument("--best_checkpoint_metric", type=str, default="iqm",
                        choices=["mean", "median", "iqm"],
                        help="保存best checkpoint使用的评估指标")
    parser.add_argument("--save_freq", type=int, default=int(5e4),
                        help="保存频率 (训练步数)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="日志记录频率")

    # IQL-specific hyperparameters
    parser.add_argument("--expectile", type=float, default=0.7,
                        help="Expectile parameter for value function")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Temperature parameter for AWR")
    parser.add_argument("--value_lr", type=float, default=3e-4,
                        help="Value network learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Critic learning rate")
    parser.add_argument("--actor_lr", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="Reward scaling factor (reward / reward_scale)")
    parser.add_argument("--actor_type", type=str, default="gaussian",
                        choices=["gaussian", "deterministic", "fixed_gaussian"],
                        help="Actor architecture type: gaussian (learnable std), deterministic (no std), fixed_gaussian (fixed std)")
    parser.add_argument("--fixed_std", type=float, default=0.1,
                        help="Fixed standard deviation for fixed_gaussian actor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate")

    # Network configuration
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--n_hidden", type=int, default=2,
                        help="隐藏层数量")

    # 🔥 Ranker & Agent configuration (解耦架构)
    parser.add_argument("--ranker", type=str, default="gems",
                        choices=["gems", "topk", "kheadargmax", "wolpertinger", "wolpertinger_slate", "greedy"],
                        help="Ranker类型: gems, topk, kheadargmax, wolpertinger, wolpertinger_slate, greedy")
    parser.add_argument("--agent", type=str, default="iql",
                        choices=["iql", "td3bc", "cql"],
                        help="Agent类型 (为未来扩展准备)")

    # 🔥 Wolpertinger Ranker 参数
    parser.add_argument("--wolpertinger_k", type=int, default=50,
                        help="Wolpertinger kNN 候选数量")
    parser.add_argument("--wolpertinger_hidden_dims", type=int, nargs='+', default=[256, 128],
                        help="Wolpertinger Actor 隐层维度")

    # 🔥 GreedySlateRanker 参数
    parser.add_argument("--greedy_s_no_click", type=float, default=-1.0,
                        help="GreedySlateRanker 无点击基准分数")

    # BC Loss & Entropy 正则化
    parser.add_argument("--lambda_bc", type=float, default=0.5,
                        help="BC Loss 权重，0=纯 AWR，1=AWR+BC 等权")
    parser.add_argument("--entropy_alpha", type=float, default=0.01,
                        help="熵正则化系数（暂未启用，预留接口）")
    parser.add_argument("--label_click_mode", type=str, default="fake_zero",
                        choices=["fake_zero", "real"],
                        help="动作标签推断click模式: fake_zero(全0点击) 或 real(真实点击)")

    # 🔥 Perturbation Sensitivity Test (Ranker Butterfly Effect Experiment)
    parser.add_argument("--enable_perturbation_test", action="store_true",
                        help="是否在达到Best IQM时自动触发Ranker加噪敏感性测试")
    parser.add_argument("--perturbation_episodes", type=int, default=50,
                        help="加噪测试的评估轮数")

    # SwanLab configuration
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")
    # 🔥 Add SwanLab project configuration arguments
    parser.add_argument("--swan_project", type=str, default="Offline_Slate_RL_202603",
                        help="SwanLab项目名称")
    parser.add_argument("--swan_workspace", type=str, default="Cliff",
                        help="SwanLab工作空间")
    parser.add_argument("--swan_mode", type=str, default="cloud",
                        choices=["cloud", "local", "offline"],
                        help="SwanLab运行模式")
    parser.add_argument("--swan_logdir", type=str, default="experiments/swanlog",
                        help="SwanLab本地日志目录")

    args = parser.parse_args()

    config = IQLConfig(
        experiment_name=args.experiment_name,
        experiment_tag=args.experiment_tag,
        env_name=args.env_name,
        dataset_quality=args.dataset_quality,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        dataset_path=args.dataset_path,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        tau=args.expectile,  # Expectile parameter (CLI: --expectile, Config: tau)
        beta=args.beta,
        value_lr=args.value_lr,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        gamma=args.gamma,
        reward_scale=args.reward_scale,  # 🔥 NEW: Reward scaling factor
        iql_tau=args.tau,  # Soft update tau (CLI: --tau, Config: iql_tau)
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        use_swanlab=args.use_swanlab,
        # 🔥 SwanLab configuration from CLI
        swan_project=args.swan_project,
        swan_workspace=args.swan_workspace,
        swan_mode=args.swan_mode,
        swan_logdir=args.swan_logdir,
        ranker_type=args.ranker,  # 🔥 NEW: Ranker selection
        agent_type=args.agent,    # 🔥 NEW: Agent selection
        wolpertinger_k=args.wolpertinger_k,  # 🔥 NEW: Wolpertinger kNN k
        wolpertinger_hidden_dims=args.wolpertinger_hidden_dims,  # 🔥 NEW: Wolpertinger Actor hidden dims
        greedy_s_no_click=args.greedy_s_no_click,  # 🔥 NEW: Greedy s_no_click
        lambda_bc=args.lambda_bc,
        entropy_alpha=args.entropy_alpha,
        label_click_mode=args.label_click_mode,
        # 🔥 NEW: Perturbation Test
        enable_perturbation_test=args.enable_perturbation_test,
        perturbation_episodes=args.perturbation_episodes,
        # 🔥 NEW: Actor Architecture Ablation
        actor_type=args.actor_type,
        fixed_std=args.fixed_std,
    )
    config.best_checkpoint_metric = args.best_checkpoint_metric

    train_iql(config)
