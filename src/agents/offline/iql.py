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

from config import paths
from config.offline_config import IQLConfig, auto_generate_paths, auto_generate_swanlab_config
from common.offline.buffer import TrajectoryReplayBuffer
from common.offline.utils import set_seed, soft_update
from common.offline.networks import TanhGaussianActor, Critic, ValueFunction
from common.offline.eval_env import OfflineEvalEnv
from common.offline.checkpoint_utils import resolve_gems_checkpoint
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.item_embeddings import ItemEmbeddings
from rankers.gems.rankers import GeMS

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

        # Actor
        self.actor = TanhGaussianActor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=self.max_action,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
        ).to(self.device)

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

        # Real-time action inference with fake_clicks=0 (zero-padding for consistency)
        flat_slates = torch.cat(batch.obs["slate"], dim=0)

        # 🔥 Use fake_clicks=0 to eliminate click noise (consistent supervision)
        fake_clicks = torch.zeros_like(flat_slates, dtype=torch.float32)

        with torch.no_grad():
            true_actions, _ = self.ranker.run_inference(flat_slates, fake_clicks)
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
            atanh_raw = torch.atanh(action_ratio.clamp(-1.0, 1.0))  # clamp to valid domain
            atanh_raw_has_nan = torch.isnan(atanh_raw).any().item()

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
            target_q1 = self.critic_1_target.q1(s_critic, true_actions)
            target_q2 = self.critic_2_target.q1(s_critic, true_actions)
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
        current_q1 = self.critic_1.q1(s_critic.detach(), true_actions)
        current_q2 = self.critic_2.q1(s_critic.detach(), true_actions)

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
            q1 = self.critic_1.q1(s_critic.detach(), true_actions)
            q2 = self.critic_2.q1(s_critic.detach(), true_actions)
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

        # Actor log probability (uses s_actor, keep gradient flow to GRU)
        # 🔧 NUMERICAL STABILITY FIX: Clamp true_actions to prevent out-of-distribution values
        # that cause NaN in log_prob computation (before calling log_prob)
        true_actions_clamped = torch.clamp(true_actions, min=-3.0, max=3.0)

        # 🔍 FORENSIC MONITOR 4: Policy Internal Diagnostics
        # Get actor distribution parameters (mu, log_std)
        with torch.no_grad():
            hidden = self.actor.trunk(s_actor)
            actor_mu = self.actor.mu(hidden)
            actor_log_std = self.actor.log_std(hidden)

            # Actor output statistics
            actor_mu_min = actor_mu.min().item()
            actor_mu_max = actor_mu.max().item()
            actor_mu_mean = actor_mu.mean().item()

            actor_log_std_min = actor_log_std.min().item()
            actor_log_std_mean = actor_log_std.mean().item()
            actor_log_std_max = actor_log_std.max().item()

            # 🔍 FORENSIC MONITOR 4B: Gaussian Intermediate Terms
            # Compute the problematic term: 1 - (action / max_action)^2
            action_ratio = true_actions_clamped / self.max_action
            action_ratio_squared = action_ratio.pow(2)
            jacobian_term = 1 - action_ratio_squared

            jacobian_term_min = jacobian_term.min().item()
            jacobian_term_max = jacobian_term.max().item()
            jacobian_term_has_negative = (jacobian_term <= 0).any().item()
            jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

        log_prob = self.actor.log_prob(s_actor, true_actions_clamped)

        # 🔍 FORENSIC MONITOR 5: Log Probability Stability (before clipping)
        log_prob_min_raw = log_prob.min().item()
        log_prob_max_raw = log_prob.max().item()

        # 🔧 NUMERICAL STABILITY FIX: Clamp log_prob to prevent -inf underflow and NaN
        log_prob = torch.clamp(log_prob, min=-20.0, max=0.0)

        # 🔥 NEW METRIC: Policy Entropy (监控策略是否坍缩)
        policy_entropy = -log_prob.mean().item()

        # Actor loss (AWR)
        actor_loss = -(exp_adv * log_prob).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["actor"].parameters()) + list(self.actor.parameters()),
            10.0
        )
        self.actor_optimizer.step()

        # Update target networks (use iql_tau for soft update, not expectile tau)
        soft_update(self.critic_1_target, self.critic_1, self.config.iql_tau)
        soft_update(self.critic_2_target, self.critic_2, self.config.iql_tau)

        # ========================================================================
        # 🔥 Enhanced Monitoring Metrics
        # ========================================================================
        metrics = {
            # Loss metrics
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),

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
            "actor_log_std_min": actor_log_std_min,
            "actor_log_std_mean": actor_log_std_mean,
            "actor_log_std_max": actor_log_std_max,

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
            "true_action_abs_max": true_action_abs_max,
            "action_ratio_abs_max": action_ratio_abs_max,

            # Gradient norms
            "value_grad_norm": value_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
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

    log_filename = f"{config.env_name}_{config.dataset_quality}_expectile{config.tau}_seed{config.seed}_{config.run_id}.log"
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

    # Load GeMS and extract embeddings
    logging.info(f"\n{'='*80}")
    logging.info("Loading GeMS checkpoint and extracting embeddings")
    logging.info(f"{'='*80}")

    # 使用统一配置模块解析checkpoint路径和参数
    gems_path, lambda_click = resolve_gems_checkpoint(
        env_name=config.env_name,
        dataset_quality=config.dataset_quality
    )

    logging.info(f"GeMS checkpoint: {gems_path}")
    logging.info(f"Item embeddings: {config.item_embedds_path}")

    # Load temporary embeddings for GeMS initialization
    temp_embeddings = ItemEmbeddings.from_pretrained(config.item_embedds_path, config.device)

    # Load GeMS checkpoint
    ranker = GeMS.load_from_checkpoint(
        gems_path,
        map_location=config.device,
        item_embeddings=temp_embeddings,
        item_embedd_dim=config.item_embedd_dim,
        device=config.device,
        rec_size=config.rec_size,
        latent_dim=32,
        lambda_click=lambda_click,
        lambda_KL=1.0,
        lambda_prior=1.0,
        ranker_lr=3e-3,
        fixed_embedds="scratch",
        ranker_sample=False,
        hidden_layers_infer=[512, 256],
        hidden_layers_decoder=[256, 512]
    )
    ranker.freeze()
    logging.info("✅ GeMS checkpoint loaded")

    # 显式强制设备同步 (对标eval_env.py的做法)
    ranker = ranker.to(config.device)
    logging.info(f"✅ GeMS moved to {config.device}")

    # Extract GeMS-trained embeddings
    gems_embedding_weights = ranker.item_embeddings.weight.data.clone()
    logging.info(f"✅ Extracted embeddings: shape={gems_embedding_weights.shape}")

    # ========================================================================
    # 🔥 Solution B: Real-time Action Inference (No Pre-computed Actions)
    # ========================================================================
    logging.info("")
    logging.info("✅ Using real-time action inference (on-the-fly from slates/clicks)")
    logging.info("")

    # Get action dimension from ranker
    action_dim = ranker.latent_dim

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
            dataset_dict['rewards'] = dataset['rewards'] / 100.0
            logging.info("⚡ Applied reward scaling: rewards / 100.0")
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

    sample_size = min(10000, len(dataset['slates']))
    sample_indices = np.random.choice(len(dataset['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(dataset['slates'][sample_indices], device=config.device, dtype=torch.long)

    # 🔥 Use fake_clicks=0 for consistent normalization (same as training)
    fake_clicks = torch.zeros_like(sample_slates, dtype=torch.float32)

    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, fake_clicks)

    action_min = sample_actions.min(dim=0)[0]
    action_max = sample_actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6

    logging.info(f"✅ Action normalization computed from {sample_size} samples")
    logging.info(f"  center mean: {action_center.mean().item():.6f}")
    logging.info(f"  scale mean: {action_scale.mean().item():.6f}")
    logging.info("")

    # Create agent's ItemEmbeddings with GeMS weights
    agent_embeddings = ItemEmbeddings(
        num_items=ranker.num_items,
        item_embedd_dim=config.item_embedd_dim,
        device=config.device,
        weights=gems_embedding_weights
    )

    # Freeze embeddings
    for param in agent_embeddings.parameters():
        param.requires_grad = False

    # Construct ranker_params
    ranker_params = {
        'item_embeddings': agent_embeddings,
        'action_center': action_center,
        'action_scale': action_scale,
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
            verbose=False
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
            # 构建统一的 SwanLab 指标字典（带命名空间前缀）
            swanlab_metrics = {
                # Loss metrics
                "train/actor_loss": metrics['actor_loss'],
                "train/critic_loss": metrics['critic_loss'],
                "train/value_loss": metrics['value_loss'],

                # V-value statistics (enhanced)
                "train/v_value_mean": metrics['v_value_mean'],
                "train/v_value_min": metrics['v_value_min'],
                "train/v_value_max": metrics['v_value_max'],
                "train/v_value_std": metrics['v_value_std'],

                # Q-value statistics (enhanced)
                "train/q_value_mean": metrics['q_value_mean'],
                "train/q_value_min": metrics['q_value_min'],
                "train/q_value_max": metrics['q_value_max'],
                "train/q_value_std": metrics['q_value_std'],
                "train/target_q_mean": metrics['target_q_mean'],
                "train/target_q_min": metrics['target_q_min'],
                "train/target_q_max": metrics['target_q_max'],

                # TD error
                "train/td_error": metrics['td_error'],

                # Advantage statistics
                "train/advantage_mean": metrics['advantage_mean'],
                "train/advantage_std": metrics['advantage_std'],

                # 🔍 FORENSIC LEVEL 0: Input & Target Health
                "train/s_actor_has_nan": metrics['s_actor_has_nan'],
                "train/s_actor_has_inf": metrics['s_actor_has_inf'],
                "train/s_actor_min": metrics['s_actor_min'],
                "train/s_actor_max": metrics['s_actor_max'],
                "train/true_actions_has_nan": metrics['true_actions_has_nan'],
                "train/true_actions_has_inf": metrics['true_actions_has_inf'],
                "train/true_actions_min": metrics['true_actions_min'],
                "train/true_actions_max": metrics['true_actions_max'],

                # 🔍 FORENSIC LEVEL 1: Advantage Extremes
                "train/advantage_max": metrics['advantage_max'],
                "train/advantage_min": metrics['advantage_min'],

                # 🔍 FORENSIC: Weight Explosion Monitor
                "train/weight_before_clip_max": metrics['weight_before_clip_max'],
                "train/weight_before_clip_min": metrics['weight_before_clip_min'],
                "train/weight_max": metrics['weight_max'],
                "train/weight_mean": metrics['weight_mean'],

                # 🔥 NEW METRICS: AWR Weight Distribution & Policy Entropy
                "train/awr_weight_std": metrics['awr_weight_std'],
                "train/policy_entropy": metrics['policy_entropy'],

                # 🔍 FORENSIC: Policy Internal Diagnostics
                "train/actor_mu_min": metrics['actor_mu_min'],
                "train/actor_mu_max": metrics['actor_mu_max'],
                "train/actor_mu_mean": metrics['actor_mu_mean'],
                "train/actor_log_std_min": metrics['actor_log_std_min'],
                "train/actor_log_std_mean": metrics['actor_log_std_mean'],
                "train/actor_log_std_max": metrics['actor_log_std_max'],

                # 🔍 FORENSIC: Jacobian Term (1 - (action/max_action)^2)
                "train/jacobian_term_min": metrics['jacobian_term_min'],
                "train/jacobian_term_max": metrics['jacobian_term_max'],
                "train/jacobian_term_has_negative": metrics['jacobian_term_has_negative'],
                "train/jacobian_term_has_nan": metrics['jacobian_term_has_nan'],

                # 🔍 FORENSIC: Log Probability Stability
                "train/log_prob_min_raw": metrics['log_prob_min_raw'],
                "train/log_prob_max_raw": metrics['log_prob_max_raw'],

                # 🔥 CONVICTION METRICS: Quantitative Proof of OOB Problem
                "train/oob_count": metrics['oob_count'],
                "train/oob_rate": metrics['oob_rate'],
                "train/atanh_protected_has_nan": metrics['atanh_protected_has_nan'],
                "train/atanh_raw_has_nan": metrics['atanh_raw_has_nan'],
                "train/true_action_abs_max": metrics['true_action_abs_max'],
                "train/action_ratio_abs_max": metrics['action_ratio_abs_max'],

                # Gradient norms
                "train/actor_grad_norm": metrics['actor_grad_norm'],
                "train/critic_grad_norm": metrics['critic_grad_norm'],
                "train/value_grad_norm": metrics['value_grad_norm'],
            }

            # 全量本地日志记录（与 SwanLab 完全一致）
            log_parts = [f"Step {t+1}/{config.max_timesteps}:"]
            for key, value in swanlab_metrics.items():
                short_key = key.replace("train/", "")
                log_parts.append(f"{short_key}={value:.6f}")
            logging.info(", ".join(log_parts))

            if swan_logger:
                swan_logger.log_metrics(swanlab_metrics, step=t+1)

        # 🔥 Representation Diagnostics (every 100 steps)
        if (t + 1) % 100 == 0:
            diag_metrics = agent.compute_representation_diagnostics(batch)

            diag_log_parts = [f"[Diagnostics @ Step {t+1}]"]
            diag_log_parts.append(f"actor_rank={diag_metrics['actor_svd_rank']:.2f}/{config.belief_hidden_dim}")
            diag_log_parts.append(f"critic_rank={diag_metrics['critic_svd_rank']:.2f}/{config.belief_hidden_dim}")
            diag_log_parts.append(f"actor_cond={diag_metrics['actor_condition_number']:.2f}")
            diag_log_parts.append(f"critic_cond={diag_metrics['critic_condition_number']:.2f}")
            diag_log_parts.append(f"repr_consistency={diag_metrics['representation_consistency']:.4f}")
            logging.info(", ".join(diag_log_parts))

            if swan_logger:
                swan_logger.log_metrics({
                    "diagnostics/actor_svd_rank": diag_metrics['actor_svd_rank'],
                    "diagnostics/critic_svd_rank": diag_metrics['critic_svd_rank'],
                    "diagnostics/actor_condition_number": diag_metrics['actor_condition_number'],
                    "diagnostics/critic_condition_number": diag_metrics['critic_condition_number'],
                    "diagnostics/representation_consistency": diag_metrics['representation_consistency'],
                }, step=t+1)

        # Evaluation
        if eval_env is not None and (t + 1) % config.eval_freq == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"Evaluating at step {t+1}")
            logging.info(f"{'='*80}")

            eval_metrics = eval_env.evaluate_policy(
                agent=agent,
                num_episodes=10,
                deterministic=True
            )

            log_msg = (f"Evaluation: mean_reward={eval_metrics['mean_reward']:.2f} ± "
                      f"{eval_metrics['std_reward']:.2f}")
            logging.info(log_msg)

            if swan_logger:
                swan_logger.log_metrics({
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/std_reward': eval_metrics['std_reward'],
                    'eval/mean_episode_length': eval_metrics['mean_episode_length'],
                }, step=t+1)

            # 🔥 NEW: Save best checkpoint
            current_reward = eval_metrics['mean_reward']
            if current_reward > best_eval_reward:
                best_eval_reward = current_reward
                best_checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f"iql_{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_seed{config.seed}_{config.run_id}_best.pt"
                )
                agent.save(best_checkpoint_path)
                logging.info(f"🏆 New best model saved! Reward: {best_eval_reward:.2f} at step {t+1}")
                logging.info(f"   Checkpoint: {best_checkpoint_path}")

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
            num_episodes=100,
            deterministic=True
        )

        logging.info(f"Final Results:")
        logging.info(f"  Mean Reward: {final_eval_metrics['mean_reward']:.2f} ± {final_eval_metrics['std_reward']:.2f}")
        logging.info(f"  Mean Episode Length: {final_eval_metrics['mean_episode_length']:.2f}")

        if swan_logger:
            swan_logger.log_metrics({
                'final_eval/mean_reward': final_eval_metrics['mean_reward'],
                'final_eval/std_reward': final_eval_metrics['std_reward'],
                'final_eval/mean_episode_length': final_eval_metrics['mean_episode_length'],
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
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate")

    # Network configuration
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--n_hidden", type=int, default=2,
                        help="隐藏层数量")

    # SwanLab configuration
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")

    args = parser.parse_args()

    config = IQLConfig(
        experiment_name=args.experiment_name,
        env_name=args.env_name,
        dataset_quality=args.dataset_quality,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        dataset_path=args.dataset_path,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        tau=args.expectile,  # Expectile parameter (CLI: --expectile, Config: tau)
        beta=args.beta,
        value_lr=args.value_lr,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        gamma=args.gamma,
        iql_tau=args.tau,  # Soft update tau (CLI: --tau, Config: iql_tau)
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        use_swanlab=args.use_swanlab,
    )

    train_iql(config)

