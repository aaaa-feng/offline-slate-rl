"""
Conservative Q-Learning (CQL) for GeMS datasets with Dual-Stream E2E GRU
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2006.04779.pdf

Enhancements:
- Dual-Stream End-to-End GRU Architecture
- SwanLab logging support
- TrajectoryReplayBuffer for episode-based sampling
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
from config.offline.config import CQLConfig, auto_generate_paths, auto_generate_swanlab_config
from common.offline.buffer import TrajectoryReplayBuffer
from common.offline.utils import set_seed, soft_update
from common.offline.networks import TanhGaussianActor, Critic
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


class CQLAgent:
    """Conservative Q-Learning Agent with Dual-Stream E2E GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config: CQLConfig,
        ranker_params: Dict,  # 🔥 GeMS-aligned
        ranker=None,  # 🔥 Solution B: Accept ranker for real-time inference
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0
        self.total_it = 0

        # ========================================================================
        # 🔥 从 Ranker 参数中提取组件
        # ========================================================================

        # 0. Ranker for real-time action inference (Solution B)
        self.ranker = ranker

        # 1. Action Bounds
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        logging.info("=" * 80)
        logging.info("=== Action Bounds from GeMS ===")
        logging.info(f"  center mean: {self.action_center.mean().item():.6f}")
        logging.info(f"  scale mean: {self.action_scale.mean().item():.6f}")
        logging.info("=" * 80)

        # 2. Item Embeddings
        self.item_embeddings = ranker_params['item_embeddings']
        logging.info(f"Item embeddings from GeMS: {self.item_embeddings.num_items} items")

        # 3. Initialize Dual-Stream GRU
        logging.info("Initializing Dual-Stream GRU belief encoder...")
        input_dim = config.rec_size * (config.item_embedd_dim + 1)

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

        # 4. 双重冻结
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
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Optimizers (分离)
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.actor_lr)

        self.critic_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["critic"].parameters()},
            {'params': self.critic_1.parameters()},
            {'params': self.critic_2.parameters()}
        ], lr=config.critic_lr)

        logging.info("CQLAgent initialized with Dual-Stream E2E GRU")

    def train(self, batch) -> Dict[str, float]:
        """Train one step with CQL loss"""
        self.total_it += 1

        # Step 1: Dual-Stream GRU forward
        states, next_states = self.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic = states["critic"]
        ns_critic = next_states["critic"]

        # Step 2: Real-time action inference (Solution B)
        flat_slates = torch.cat(batch.obs["slate"], dim=0)
        flat_clicks = torch.cat(batch.obs["clicks"], dim=0)

        with torch.no_grad():
            true_actions, _ = self.ranker.run_inference(flat_slates, flat_clicks)
            true_actions = (true_actions - self.action_center) / self.action_scale

        rewards = torch.cat(batch.rewards, dim=0) if batch.rewards else None
        dones = torch.cat(batch.dones, dim=0) if batch.dones else None

        # Step 3: Critic Update (CQL Loss)
        with torch.no_grad():
            # Sample next actions
            next_actions, _ = self.actor(ns_critic, deterministic=False, need_log_prob=False)
            target_q1 = self.critic_1_target.q1(ns_critic, next_actions)
            target_q2 = self.critic_2_target.q1(ns_critic, next_actions)
            target_q = torch.min(target_q1, target_q2)

            if rewards is not None and dones is not None:
                target_q = rewards + (1 - dones) * self.config.gamma * target_q
            else:
                target_q = target_q * self.config.gamma

            # 🔧 HOT-FIX: Target Q Clamping (防止Q值爆炸)
            # 理论Q_max=63.4, CQL需要更大范围(100.0)支持保守惩罚机制
            target_q = torch.clamp(target_q, -10.0, 100.0)

        # Current Q values (detach s_critic to avoid gradient conflict)
        current_q1 = self.critic_1.q1(s_critic.detach(), true_actions)
        current_q2 = self.critic_2.q1(s_critic.detach(), true_actions)
        # 🔧 NUMERICAL STABILITY FIX: Clamp current Q-values to prevent explosion
        # CQL需要更大范围(100.0)支持保守惩罚机制，统一下界为-10.0
        current_q1 = torch.clamp(current_q1, -10.0, 100.0)
        current_q2 = torch.clamp(current_q2, -10.0, 100.0)

        # Bellman error
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # CQL penalty: sample random actions
        batch_size = s_critic.shape[0]
        random_actions = torch.FloatTensor(batch_size, self.action_dim).uniform_(-1, 1).to(self.device)

        # Q values for random actions (detach s_critic to avoid gradient conflict)
        random_q1 = self.critic_1.q1(s_critic.detach(), random_actions)
        random_q2 = self.critic_2.q1(s_critic.detach(), random_actions)
        # 🔧 NUMERICAL STABILITY FIX: Clamp random Q-values to prevent explosion
        random_q1 = torch.clamp(random_q1, -10.0, 100.0)
        random_q2 = torch.clamp(random_q2, -10.0, 100.0)

        # CQL loss
        cql_loss = (random_q1.mean() + random_q2.mean() - current_q1.mean() - current_q2.mean())
        total_critic_loss = critic_loss + self.config.alpha * cql_loss

        # Optimize critics
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        # 🔧 HOT-FIX: Critic Gradient Clipping (防止梯度爆炸)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["critic"].parameters()) +
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()),
            10.0
        )
        self.critic_optimizer.step()

        # Step 4: Actor Update
        # Sample actions from current policy
        pi, log_pi = self.actor(s_actor, deterministic=False, need_log_prob=True)

        # Q value (使用 detached s_critic)
        q_pi = self.critic_1.q1(s_critic.detach(), pi)

        # Actor loss
        actor_loss = -q_pi.mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 🔧 HOT-FIX: Actor Gradient Clipping (防止梯度爆炸)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["actor"].parameters()) + list(self.actor.parameters()),
            10.0
        )
        self.actor_optimizer.step()

        # Update target networks
        soft_update(self.critic_1_target, self.critic_1, self.config.tau)
        soft_update(self.critic_2_target, self.critic_2, self.config.tau)

        return {
            "critic_loss": critic_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_critic_loss": total_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_q1.mean().item(),
            "q1_value": current_q1.mean().item(),
            "q2_value": current_q2.mean().item(),
            "q_std": current_q1.std().item(),
            "random_q": random_q1.mean().item(),
            "target_q": target_q.mean().item(),
            # 🔧 HOT-FIX: 新增诊断指标
            "target_q_max": target_q.max().item(),
            "target_q_min": target_q.min().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
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

        # 🔥 使用 ranker 解码 latent action 为 slate
        if self.ranker is None:
            raise RuntimeError(
                "CQLAgent.act() requires a ranker for slate decoding. "
                "Please provide ranker during initialization."
            )

        # 🔧 确保设备一致性 - 使用 ranker 的设备
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
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
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
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, embedding_path: str, device: torch.device):
        """
        Load CQLAgent from checkpoint without requiring GeMS.

        Args:
            checkpoint_path: Path to saved agent checkpoint
            embedding_path: Path to item embeddings (.pt file)
            device: Device to load model on

        Returns:
            Loaded CQLAgent instance
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
        agent.total_it = checkpoint['total_it']

        logging.info(f"CQLAgent loaded from {checkpoint_path}")
        return agent


def train_cql(config: CQLConfig):
    """Train CQL with Dual-Stream E2E GRU"""
    timestamp = datetime.now().strftime("%Y%m%d")
    config = auto_generate_paths(config, timestamp)
    config = auto_generate_swanlab_config(config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    log_filename = f"{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_seed{config.seed}_{config.run_id}.log"
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
    logging.info("=== CQL Training Configuration ===")
    logging.info("=" * 80)
    logging.info(f"Environment: {config.env_name}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Alpha: {config.alpha}")
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

    # ========================================================================
    # 🔥 Load ranker dynamically using RankerFactory
    # ========================================================================
    from common.offline.ranker_factory import RankerFactory

    logging.info("=" * 80)
    logging.info("=== Loading Ranker via RankerFactory ===")
    logging.info(f"Ranker type: {config.ranker_type}")
    logging.info("=" * 80)

    ranker, action_dim, agent_embeddings = RankerFactory.create(
        ranker_type=config.ranker_type,
        config=config,
        device=config.device
    )

    # Get action dimension from ranker (unified interface)
    action_dim, _ = ranker.get_action_dim()

    logging.info(f"✅ Ranker loaded: {config.ranker_type}")
    logging.info(f"  Action dim: {action_dim}")
    logging.info("")

    # ========================================================================
    # 🔥 Solution B: Real-time Action Inference (No Pre-computed Actions)
    # ========================================================================
    logging.info("")
    logging.info("=" * 80)
    logging.info("✅ Using Real-time Action Inference (Solution B)")
    logging.info("=" * 80)
    logging.info("Strategy: Infer actions on-the-fly during training from slates/clicks")
    logging.info("Benefits: No preprocessing overhead, supports ranker fine-tuning")
    logging.info("=" * 80)
    logging.info("")

    # Action dim already obtained from RankerFactory.create()
    # No need to get it again

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
    # 🔥 Compute Action Normalization (TD3+BC approach)
    # ========================================================================
    logging.info("")
    logging.info("Computing action normalization parameters...")

    sample_size = min(10000, len(dataset['slates']))
    sample_indices = np.random.choice(len(dataset['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(dataset['slates'][sample_indices], device=config.device, dtype=torch.long)
    sample_clicks = torch.tensor(dataset['clicks'][sample_indices], device=config.device, dtype=torch.float)

    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, sample_clicks)

    action_min = sample_actions.min(dim=0)[0]
    action_max = sample_actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6

    logging.info(f"✅ Action normalization computed from {sample_size} samples")
    logging.info(f"  center mean: {action_center.mean().item():.6f}")
    logging.info(f"  scale mean: {action_scale.mean().item():.6f}")
    logging.info("")

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
    agent = CQLAgent(
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
    logging.info(f"Starting CQL training")
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
                "train/q1_value": metrics['q1_value'],
                "train/q2_value": metrics['q2_value'],
                "train/q_value_std": metrics['q_std'],
                "train/target_q_mean": metrics['target_q'],
                # 🔧 HOT-FIX: 新增诊断指标
                "train/target_q_max": metrics['target_q_max'],
                "train/target_q_min": metrics['target_q_min'],
                # CQL 特有指标
                "train/cql_loss": metrics['cql_loss'],
                "train/total_critic_loss": metrics['total_critic_loss'],
                "train/random_q": metrics['random_q'],
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
                f"cql_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Final save
    final_path = os.path.join(
        config.checkpoint_dir,
        f"cql_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_final.pt"
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
    logging.info(f"CQL training completed!")
    logging.info(f"{'='*80}")

    if swan_logger:
        swan_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CQL (Conservative Q-Learning) on offline datasets")

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
    parser.add_argument("--eval_freq", type=int, default=int(5e3),
                        help="评估频率 (训练步数)")
    parser.add_argument("--save_freq", type=int, default=int(5e4),
                        help="保存频率 (训练步数)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="日志记录频率")

    # CQL-specific hyperparameters
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="CQL alpha (conservative penalty weight)")
    parser.add_argument("--actor_lr", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate")

    # Network configuration
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--n_hidden", type=int, default=2,
                        help="隐藏层数量")

    # 🔥 Ranker configuration (解耦架构)
    parser.add_argument("--ranker", type=str, default="gems",
                        choices=["gems", "topk", "kheadargmax"],
                        help="Ranker类型: gems, topk, kheadargmax")

    # SwanLab configuration
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")

    args = parser.parse_args()

    config = CQLConfig(
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
        alpha=args.alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        use_swanlab=args.use_swanlab,
        ranker_type=args.ranker,  # 🔥 NEW: Ranker selection
    )

    train_cql(config)
