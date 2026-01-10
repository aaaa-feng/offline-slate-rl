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


class IQLAgent:
    """Implicit Q-Learning Agent with Dual-Stream E2E GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config: IQLConfig,
        ranker_params: Dict,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0
        self.total_it = 0

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

        # Concatenate data
        true_actions = torch.cat(batch.actions, dim=0)
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

        # Expectile loss
        value_loss = expectile_loss(target_q - current_v, self.config.tau).mean()

        # Optimize value network (retain graph for later use of s_critic)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), float('inf'))
        self.value_optimizer.step()

        # ========================================================================
        # Step 2: Critic Update (Standard Bellman Backup)
        # ========================================================================
        with torch.no_grad():
            # Next state value
            next_v = self.value(ns_critic)

            if rewards is not None and dones is not None:
                target_q = rewards + (1 - dones) * self.config.gamma * next_v
            else:
                target_q = next_v * self.config.gamma

        # Current Q-values (detach s_critic to avoid gradient conflict)
        # Reason: Value optimizer already updated GRU in Step 1
        current_q1 = self.critic_1.q1(s_critic.detach(), true_actions)
        current_q2 = self.critic_2.q1(s_critic.detach(), true_actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            float('inf')
        )
        self.critic_optimizer.step()

        # ========================================================================
        # Step 3: Actor Update (Advantage Weighted Regression)
        # ========================================================================
        with torch.no_grad():
            # Compute advantage using s_critic
            v = self.value(s_critic.detach())
            q1 = self.critic_1.q1(s_critic.detach(), true_actions)
            q2 = self.critic_2.q1(s_critic.detach(), true_actions)
            q = torch.min(q1, q2)
            advantage = q - v

            # Compute weights
            exp_adv = torch.exp(advantage * self.config.beta)
            exp_adv = torch.clamp(exp_adv, max=100.0)

        # Actor log probability (uses s_actor, keep gradient flow to GRU)
        log_prob = self.actor.log_prob(s_actor, true_actions)

        # Actor loss (AWR)
        actor_loss = -(exp_adv * log_prob).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        self.actor_optimizer.step()

        # Update target networks (use iql_tau for soft update, not expectile tau)
        soft_update(self.critic_1_target, self.critic_1, self.config.iql_tau)
        soft_update(self.critic_2_target, self.critic_2, self.config.iql_tau)

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "v_value": current_v.mean().item(),
            "q_value": current_q1.mean().item(),
            "q_value_std": current_q1.std().item(),
            "target_q": target_q.mean().item(),
            "advantage": advantage.mean().item(),
            "advantage_std": advantage.std().item(),
            "value_grad_norm": value_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """Select action using Actor GRU"""
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
        action = raw_action * self.action_scale + self.action_center
        return action.cpu().numpy().flatten()

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
    action_dim = dataset['actions'].shape[1]

    logging.info(f"Dataset statistics:")
    logging.info(f"  Slates shape: {dataset['slates'].shape}")
    logging.info(f"  Actions shape: {dataset['actions'].shape}")

    # Note: Buffer creation moved after action relabeling

    # Load GeMS and extract embeddings
    logging.info(f"\n{'='*80}")
    logging.info("Loading GeMS checkpoint and extracting embeddings")
    logging.info(f"{'='*80}")

    gems_checkpoint_name = f"GeMS_{config.env_name}_{config.dataset_quality}_latent32_beta1.0_click0.5_seed58407201"
    gems_path = f"/data/liyuefeng/offline-slate-rl/checkpoints/gems/offline/{gems_checkpoint_name}.ckpt"

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
        lambda_click=0.5,
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
    # 内存重打标 (In-Memory Action Relabeling) - Zero Trust Strategy
    # ========================================================================
    logging.info("")
    logging.info("=" * 80)
    logging.info("⚠️  IN-MEMORY ACTION RELABELING")
    logging.info("=" * 80)
    logging.info("Strategy: Zero Trust - Regenerate all actions using current GeMS")
    logging.info("Reason:   Ensure absolute consistency between training and inference")
    logging.info("")

    # 1. Extract raw discrete data
    raw_slates = torch.tensor(dataset['slates'], device=config.device, dtype=torch.long)
    raw_clicks = torch.tensor(dataset['clicks'], device=config.device, dtype=torch.float)
    total_samples = len(raw_slates)

    logging.info(f"Total samples: {total_samples:,}")
    logging.info("Starting action relabeling...")

    # 2. Batch inference to regenerate actions
    batch_size = 1000
    new_actions_list = []

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_slates = raw_slates[i:i+batch_size]
            batch_clicks = raw_clicks[i:i+batch_size]

            # Key: Use current GeMS Encoder to infer latent actions
            mu, _ = ranker.run_inference(batch_slates, batch_clicks)
            new_actions_list.append(mu.cpu().numpy())

            # Progress logging
            if (i + batch_size) % 100000 == 0 or (i + batch_size) >= total_samples:
                processed = min(i + batch_size, total_samples)
                logging.info(f"  Progress: {processed:,}/{total_samples:,} ({processed/total_samples*100:.1f}%)")

    new_actions = np.concatenate(new_actions_list, axis=0)
    logging.info(f"✅ Relabeling complete: {len(new_actions):,} actions generated")

    # 3. Action statistics validation (Primary quality indicator)
    logging.info("")
    logging.info("Action Statistics (Primary Quality Indicator):")
    logging.info(f"  Shape: {new_actions.shape}")
    logging.info(f"  Mean:  {new_actions.mean():.6f} (expect ≈ 0)")
    logging.info(f"  Std:   {new_actions.std():.6f}  (expect ≈ 1)")
    logging.info(f"  Min:   {new_actions.min():.6f}")
    logging.info(f"  Max:   {new_actions.max():.6f}")

    # 4. GeMS reconstruction quality test (Informational only, no blocking)
    logging.info("")
    logging.info("GeMS Reconstruction Quality Test (Informational Only):")
    test_size = min(100, len(raw_slates))
    test_slates = raw_slates[:test_size]
    test_clicks = raw_clicks[:test_size]
    with torch.no_grad():
        test_actions, _ = ranker.run_inference(test_slates, test_clicks)
        # Loop decoding (ranker.rank does not support batch input)
        matches_list = []
        for i in range(test_size):
            reconstructed = ranker.rank(test_actions[i])
            match = (test_slates[i] == reconstructed).float().mean().item()
            matches_list.append(match)
        matches = np.mean(matches_list)
    logging.info(f"  Exact match accuracy: {matches:.4f}")
    logging.info("  Note: Low accuracy is normal for slate ranking tasks")

    # 5. Overwrite old actions
    logging.info("")
    logging.info("✅ Action relabeling complete. Overwriting dataset actions.")
    logging.info("=" * 80)
    logging.info("")

    # 6. Create buffer with relabeled actions
    replay_buffer = TrajectoryReplayBuffer(device=config.device)
    dataset_dict = {
        'episode_ids': dataset['episode_ids'],
        'slates': dataset['slates'],
        'clicks': dataset['clicks'],
        'actions': new_actions,  # Use relabeled actions!
    }
    if 'rewards' in dataset:
        if config.normalize_rewards:
            # Apply scaling / 100.0 as standard practice
            dataset_dict['rewards'] = dataset['rewards'] / 100.0
            logging.info("⚡ Applied reward scaling: rewards / 100.0")
        else:
            dataset_dict['rewards'] = dataset['rewards']
    if 'terminals' in dataset:
        dataset_dict['terminals'] = dataset['terminals']

    replay_buffer.load_d4rl_dataset(dataset_dict)
    action_center, action_scale = replay_buffer.get_action_normalization_params()
    logging.info("✅ Buffer loaded with relabeled actions")

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
    logging.info("✅ Agent embeddings frozen")

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

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            # 构建统一的 SwanLab 指标字典（带命名空间前缀）
            swanlab_metrics = {
                # 公共指标
                "train/actor_loss": metrics['actor_loss'],
                "train/actor_grad_norm": metrics['actor_grad_norm'],
                # Critic/Q 指标
                "train/critic_loss": metrics['critic_loss'],
                "train/critic_grad_norm": metrics['critic_grad_norm'],
                "train/q_value_mean": metrics['q_value'],
                "train/q_value_std": metrics['q_value_std'],
                "train/target_q_mean": metrics['target_q'],
                # IQL 特有指标
                "train/value_loss": metrics['value_loss'],
                "train/v_value_mean": metrics['v_value'],
                "train/advantage_mean": metrics['advantage'],
                "train/advantage_std": metrics['advantage_std'],
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
                        choices=["random", "medium", "expert"],
                        help="数据集质量")
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
    parser.add_argument("--eval_freq", type=int, default=int(5e3),
                        help="评估频率 (训练步数)")
    parser.add_argument("--save_freq", type=int, default=int(5e4),
                        help="保存频率 (训练步数)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="日志记录频率")

    # IQL-specific hyperparameters
    parser.add_argument("--expectile", type=float, default=0.7,
                        help="Expectile parameter for value function")
    parser.add_argument("--beta", type=float, default=3.0,
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

