"""
Behavior Cloning (BC) for GeMS datasets
最简单的离线 RL baseline，用于验证数据加载和归一化
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass
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
from config.offline.config import BCConfig, auto_generate_paths, auto_generate_swanlab_config

from common.offline.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.offline.utils import set_seed, compute_mean_std
from common.offline.networks import (
    TanhGaussianActor,
    LOG_STD_MIN,
    LOG_STD_MAX,
)
from common.offline.eval_env import OfflineEvalEnv
from common.offline.checkpoint_utils import resolve_gems_checkpoint
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.item_embeddings import ItemEmbeddings

# SwanLab Logger
try:
    from common.offline.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available")


class BCAgent:
    """Behavior Cloning Agent with GeMS-aligned architecture"""

    def __init__(
        self,
        action_dim: int,
        config: BCConfig,
        ranker_params: Dict,  # 🔥 新增：接收 Ranker 参数
        ranker=None,  # 🔥 Solution B: Accept ranker for real-time inference
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim

        # ========================================================================
        # 🔥 关键：从 Ranker 参数中提取组件（复刻在线逻辑）
        # ========================================================================

        # 0. Ranker for real-time action inference (Solution B)
        self.ranker = ranker

        # 1. Action Bounds（直接使用 Ranker 的）
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        logging.info("=" * 80)
        logging.info("=== Action Bounds from GeMS ===")
        logging.info(f"  center shape: {self.action_center.shape}")
        logging.info(f"  center mean: {self.action_center.mean().item():.6f}")
        logging.info(f"  center std: {self.action_center.std().item():.6f}")
        logging.info(f"  scale shape: {self.action_scale.shape}")
        logging.info(f"  scale mean: {self.action_scale.mean().item():.6f}")
        logging.info(f"  scale std: {self.action_scale.std().item():.6f}")
        logging.info("=" * 80)

        # 2. Item Embeddings（使用 GeMS 训练后的）
        self.item_embeddings = ranker_params['item_embeddings']
        logging.info(f"Item embeddings from GeMS: {self.item_embeddings.num_items} items, "
                    f"{self.item_embeddings.embedd_dim} dims")

        # 3. 初始化 GRU belief encoder
        logging.info("Initializing GRU belief encoder...")
        input_dim = config.rec_size * (config.item_embedd_dim + 1)

        self.belief = GRUBelief(
            item_embeddings=self.item_embeddings,  # 🔥 传入 GeMS 的 Embeddings
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=self.device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=["actor"],  # BC 只需要 actor
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim  # 🔥 显式传入
        )

        # 4. 🔥 关键：双重保险 - 再次冻结 Embeddings
        # 即使 GRUBelief 内部 deepcopy，我们也确保副本是冻结的
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()
        logging.info("✅ Item embeddings frozen (double-checked)")

        # 5. Actor network - 使用与 IQL 相同的 TanhGaussianActor
        self.actor = TanhGaussianActor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_dim=config.hidden_dim,
            n_hidden=2,
        ).to(self.device)
        logging.info("✅ Using TanhGaussianActor (same as IQL)")

        # 6. Optimizer（只包含 GRU 和 Actor，不包含 Embeddings）
        self.optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.learning_rate)

        self.total_it = 0
        logging.info("✅ BCAgent initialized with GeMS-aligned architecture")

    def train(self, batch) -> Dict[str, float]:
        """
        训练一步 (端到端训练 GRU + Actor)

        Args:
            batch: TrajectoryBatch with obs (Dict with 'slate' and 'clicks' as List[Tensor])
        """
        self.total_it += 1

        # GRU forward on trajectories
        states, _ = self.belief.forward_batch(batch)
        state = states["actor"]  # [sum_seq_lens, belief_hidden_dim]

        # 🔥 Solution B: Real-time action inference from slates/clicks
        flat_slates = torch.cat(batch.obs["slate"], dim=0)  # [sum_seq_lens, rec_size]
        flat_clicks = torch.cat(batch.obs["clicks"], dim=0)  # [sum_seq_lens, rec_size]

        with torch.no_grad():
            true_actions, _ = self.ranker.run_inference(flat_slates, flat_clicks)
            # Normalize actions
            true_actions = (true_actions - self.action_center) / self.action_scale

        # ========================================================================
        # 🔥 BC Loss: 负对数似然 (Negative Log Likelihood)
        # ========================================================================
        log_prob = self.actor.log_prob(state, true_actions)
        bc_loss = -log_prob.mean()

        # 反向传播 (同时更新 GRU 和 Actor)
        self.optimizer.zero_grad()
        bc_loss.backward()

        # 计算梯度范数（用于监控）
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        gru_grad_norm = torch.nn.utils.clip_grad_norm_(self.belief.gru["actor"].parameters(), float('inf'))

        self.optimizer.step()

        # ========================================================================
        # 🔥 "死亡探头"：核心诊断指标（与 IQL 对齐）
        # ========================================================================
        with torch.no_grad():
            # 🔥 采样时序监控：batch 平均 reward
            if batch.rewards is not None:
                batch_reward_mean = torch.cat(batch.rewards, dim=0).mean().item()
            else:
                batch_reward_mean = 0.0

            # 1. 获取 Actor 内部状态
            hidden = self.actor.trunk(state)
            actor_mu = self.actor.mu(hidden)
            actor_log_std_raw = self.actor.log_std(hidden)
            actor_log_std = torch.clamp(actor_log_std_raw, min=LOG_STD_MIN, max=LOG_STD_MAX)

            # 2. Log Std 统计
            actor_log_std_raw_min = actor_log_std_raw.min().item()
            actor_log_std_raw_mean = actor_log_std_raw.mean().item()
            actor_log_std_raw_max = actor_log_std_raw.max().item()
            actor_log_std_min = actor_log_std.min().item()
            actor_log_std_mean = actor_log_std.mean().item()
            actor_log_std_max = actor_log_std.max().item()
            actor_log_std_floor_hit_rate = (actor_log_std_raw <= LOG_STD_MIN).float().mean().item()

            # 3. Tanh 饱和率
            mu_after_tanh = torch.tanh(actor_mu)
            tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()

            # 4. Policy Distance to Origin (BC 重力证据)
            policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

            # 5. Policy Entropy
            policy_entropy = -log_prob.mean().item()

            # 6. Action 统计（确定性输出）
            pred_actions_deterministic = self.actor(state, deterministic=True)[0]

            # 🔥 OOD Distance Probe: Actor 输出与数据集真实动作的距离
            ood_distances_det = torch.norm(pred_actions_deterministic - true_actions, dim=-1)
            ood_distance_mean_det = ood_distances_det.mean().item()
            ood_distance_max_det = ood_distances_det.max().item()

            # 采样动作的 OOD 距离
            pred_actions_samp, _ = self.actor(state, deterministic=False)
            ood_distances_samp = torch.norm(pred_actions_samp - true_actions, dim=-1)
            ood_distance_mean_samp = ood_distances_samp.mean().item()
            ood_distance_max_samp = ood_distances_samp.max().item()

        return {
            # 基础指标
            "bc_loss": bc_loss.item(),
            "action_mean": pred_actions_deterministic.mean().item(),
            "action_std": pred_actions_deterministic.std().item(),
            "action_min": pred_actions_deterministic.min().item(),
            "action_max": pred_actions_deterministic.max().item(),
            "target_action_mean": true_actions.mean().item(),
            "target_action_std": true_actions.std().item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "gru_grad_norm": gru_grad_norm.item(),
            # 🔥 "死亡探头"：核心诊断指标
            "actor_log_std_raw_min": actor_log_std_raw_min,
            "actor_log_std_raw_mean": actor_log_std_raw_mean,
            "actor_log_std_raw_max": actor_log_std_raw_max,
            "actor_log_std_min": actor_log_std_min,
            "actor_log_std_mean": actor_log_std_mean,
            "actor_log_std_max": actor_log_std_max,
            "actor_log_std_floor_hit_rate": actor_log_std_floor_hit_rate,
            "tanh_saturation_ratio": tanh_saturation_ratio,
            "policy_distance_to_origin": policy_distance_to_origin,
            "policy_entropy": policy_entropy,
            # 🔥 OOD Distance Probe
            "ood_distance_mean_det": ood_distance_mean_det,
            "ood_distance_max_det": ood_distance_max_det,
            "ood_distance_mean_samp": ood_distance_mean_samp,
            "ood_distance_max_samp": ood_distance_max_samp,
            # 🔥 采样时序监控探针
            "batch_reward_mean": batch_reward_mean,
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        Select action using Actor GRU and decode to slate.

        Args:
            obs: Dict with 'slate' and 'clicks' (torch.Tensor or numpy arrays)
            deterministic: 是否确定性选择 (BC 总是确定性的)

        Returns:
            slate: numpy array of shape [rec_size] containing item IDs
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # GRU 编码
        belief_state = self.belief.forward(obs_tensor, done=False)["actor"]

        # 🔥 FIX: Actor 预测 - TanhGaussianActor 返回 (action, log_prob) 元组
        # 必须传入 deterministic 标志并提取第一个元素 action
        raw_action, _ = self.actor(belief_state, deterministic=deterministic, need_log_prob=False)

        # 反归一化
        latent_action = raw_action * self.action_scale + self.action_center

        # 🔥 使用 ranker 解码 latent action 为 slate
        if self.ranker is None:
            raise RuntimeError(
                "BCAgent.act() requires a ranker for slate decoding. "
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
        """
        重置 GRU 隐藏状态 (在每个 episode 开始时调用)
        使用 dummy obs + done=True 来优雅地重置
        """
        dummy_obs = {
            "slate": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
            "clicks": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device)
        }
        self.belief.forward(dummy_obs, done=True)

    def save(self, filepath: str):
        """保存模型（包含所有必要信息，支持独立加载）"""
        torch.save({
            # 模型权重
            'belief_state_dict': self.belief.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'optimizer': self.optimizer.state_dict(),

            # Action Bounds
            'action_center': self.action_center,
            'action_scale': self.action_scale,

            # 🔥 新增：Embeddings 元数据（用于独立加载）
            'embeddings_meta': {
                'num_items': self.item_embeddings.num_items,
                'embedd_dim': self.item_embeddings.embedd_dim,
            },

            # 其他信息
            'action_dim': self.action_dim,
            'total_it': self.total_it,
            'config': self.config,
        }, filepath)
        logging.info(f"✅ Model saved to {filepath} (with embeddings_meta)")

    def load(self, filepath: str):
        """加载模型（需要先初始化 Agent）"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.belief.load_state_dict(checkpoint['belief_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """
        从 Checkpoint 独立加载 Agent，无需 GeMS（解决循环依赖）

        Args:
            checkpoint_path: Agent checkpoint 路径
            device: 设备

        Returns:
            BCAgent 实例
        """
        logging.info("=" * 80)
        logging.info("=== Loading BCAgent from Checkpoint (Standalone) ===")
        logging.info(f"Checkpoint: {checkpoint_path}")
        logging.info("=" * 80)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 1. 从 Checkpoint 恢复 Embeddings
        embeddings_meta = checkpoint['embeddings_meta']
        belief_state = checkpoint['belief_state_dict']

        # 提取 Embeddings 权重（从 belief state dict 中）
        embedding_weights = belief_state['item_embeddings.actor.embedd.weight']

        agent_embeddings = ItemEmbeddings(
            num_items=embeddings_meta['num_items'],
            item_embedd_dim=embeddings_meta['embedd_dim'],
            device=device,
            weights=embedding_weights
        )
        logging.info(f"✅ Embeddings restored: {embeddings_meta['num_items']} items, "
                    f"{embeddings_meta['embedd_dim']} dims")

        # 2. 构建 ranker_params
        ranker_params = {
            'item_embeddings': agent_embeddings,
            'action_center': checkpoint['action_center'],
            'action_scale': checkpoint['action_scale'],
            'num_items': embeddings_meta['num_items'],
            'item_embedd_dim': embeddings_meta['embedd_dim']
        }

        # 3. 创建 Agent
        agent = cls(
            action_dim=checkpoint['action_dim'],
            config=checkpoint['config'],
            ranker_params=ranker_params
        )

        # 4. 加载权重
        agent.belief.load_state_dict(belief_state)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.total_it = checkpoint['total_it']

        logging.info(f"✅ BCAgent loaded from {checkpoint_path} (standalone mode)")
        logging.info("=" * 80)
        return agent


def train_bc(config: BCConfig):
    """训练 BC"""
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d")

    # 自动生成路径配置
    config = auto_generate_paths(config, timestamp)

    # 自动生成 SwanLab 配置
    config = auto_generate_swanlab_config(config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 配置 logging（包含 ranker 信息）
    ranker_type = getattr(config, 'ranker_type', 'unknown')
    log_filename = f"{config.env_name}_{config.dataset_quality}_{ranker_type}_seed{config.seed}_{config.run_id}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    # 清除已有的 handlers 并重新配置
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

    # Set seed
    set_seed(config.seed)
    logging.info(f"Global seed set to {config.seed}")

    # 打印配置
    logging.info("=" * 80)
    logging.info("=== BC Training Configuration ===")
    logging.info("=" * 80)
    logging.info(f"Environment: {config.env_name}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Seed: {config.seed}")
    logging.info(f"Max timesteps: {config.max_timesteps}")
    logging.info(f"Batch size: {config.batch_size}")
    logging.info(f"Learning rate: {config.learning_rate}")
    logging.info(f"Log file: {log_filepath}")
    logging.info("=" * 80)

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
    # 加载数据集
    # ========================================================================
    logging.info(f"\nLoading dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)

    logging.info(f"Dataset statistics:")
    logging.info(f"  Slates shape: {dataset['slates'].shape}")
    logging.info(f"  Clicks shape: {dataset['clicks'].shape}")
    logging.info(f"  Next slates shape: {dataset['next_slates'].shape}")
    logging.info(f"  Next clicks shape: {dataset['next_clicks'].shape}")
    logging.info(f"  Total transitions: {len(dataset['slates'])}")

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

    logging.info(f"\nEnvironment info:")
    logging.info(f"  Action dim: {action_dim}")
    logging.info(f"  Rec size: {config.rec_size}")
    logging.info(f"  Belief hidden dim: {config.belief_hidden_dim}")

    # Create trajectory replay buffer
    replay_buffer = TrajectoryReplayBuffer(device=config.device)

    # 6. Load data (V4 format - no pre-computed actions)
    dataset_dict = {
        'episode_ids': dataset['episode_ids'],
        'slates': dataset['slates'],
        'clicks': dataset['clicks'],
        'next_slates': dataset['next_slates'],  # ✅ Required by TrajectoryReplayBuffer
        'next_clicks': dataset['next_clicks'],  # ✅ Required by TrajectoryReplayBuffer
        'rewards': dataset['rewards'],
        'terminals': dataset['terminals'],
    }

    replay_buffer.load_d4rl_dataset(dataset_dict)
    logging.info(f"✅ Buffer loaded successfully")

    # ========================================================================
    # 🔥 Compute Action Normalization (TD3+BC approach)
    # ========================================================================
    logging.info("")
    logging.info("Computing action normalization parameters...")

    # Sample a batch of slates/clicks to compute action statistics
    sample_size = min(10000, len(dataset['slates']))
    sample_indices = np.random.choice(len(dataset['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(dataset['slates'][sample_indices], device=config.device, dtype=torch.long)
    sample_clicks = torch.tensor(dataset['clicks'][sample_indices], device=config.device, dtype=torch.float)

    # Infer actions from samples
    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, sample_clicks)

    # Compute normalization parameters
    action_min = sample_actions.min(dim=0)[0]
    action_max = sample_actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6

    logging.info(f"✅ Action normalization computed from {sample_size} samples")
    logging.info(f"  center mean: {action_center.mean().item():.6f}")
    logging.info(f"  scale mean: {action_scale.mean().item():.6f}")
    logging.info("")

    # 🔥 Freeze embeddings
    for param in agent_embeddings.parameters():
        param.requires_grad = False
    logging.info("✅ Agent embeddings frozen")

    # Construct ranker_params
    ranker_params = {
        'item_embeddings': agent_embeddings,
        'action_center': action_center,
        'action_scale': action_scale,
        'num_items': agent_embeddings.num_items,
        'item_embedd_dim': config.item_embedd_dim
    }

    # Initialize BC agent (with GeMS-aligned architecture)
    agent = BCAgent(
        action_dim=action_dim,
        config=config,
        ranker_params=ranker_params,  # 🔥 传入 Ranker 参数
        ranker=ranker,  # 🔥 Solution B: Pass ranker for real-time inference
    )

    # Initialize SwanLab
    swan_logger = None
    if config.use_swanlab:
        if not SWANLAB_AVAILABLE:
            logging.warning("SwanLab not available, disabling SwanLab logging")
            config.use_swanlab = False
        else:
            try:
                swan_logger = SwanlabLogger(
                    project=config.swan_project,
                    experiment_name=config.run_name,
                    workspace=config.swan_workspace,
                    config=config.__dict__,
                    mode=config.swan_mode,
                    logdir=config.swan_logdir,
                )
                logging.info(f"SwanLab initialized: project={config.swan_project}, run={config.run_name}")
            except Exception as e:
                logging.warning(f"SwanLab initialization failed: {e}")
                config.use_swanlab = False

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
    logging.info(f"Starting BC training")
    logging.info(f"{'='*80}\n")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            # 🔥 构建分组的 SwanLab 指标字典（6 个类别）
            swanlab_metrics = {
                # 1️⃣ Loss 指标（1 个）
                "1_Loss/bc_loss": metrics['bc_loss'],

                # 2️⃣ Action 统计（6 个）
                "2_Action/mean": metrics['action_mean'],
                "2_Action/std": metrics['action_std'],
                "2_Action/min": metrics['action_min'],
                "2_Action/max": metrics['action_max'],
                "2_Action/target_mean": metrics['target_action_mean'],
                "2_Action/target_std": metrics['target_action_std'],

                # 3️⃣ 梯度范数（2 个）
                "3_Gradient/actor_norm": metrics['actor_grad_norm'],
                "3_Gradient/gru_norm": metrics['gru_grad_norm'],

                # 4️⃣ "死亡探头"：Log Std 统计（7 个）- 🔥 核心监控
                "4_LogStd/raw_min": metrics['actor_log_std_raw_min'],
                "4_LogStd/raw_mean": metrics['actor_log_std_raw_mean'],
                "4_LogStd/raw_max": metrics['actor_log_std_raw_max'],
                "4_LogStd/min": metrics['actor_log_std_min'],
                "4_LogStd/mean": metrics['actor_log_std_mean'],  # 🔥 关键：方差是否坍缩
                "4_LogStd/max": metrics['actor_log_std_max'],
                "4_LogStd/floor_hit_rate": metrics['actor_log_std_floor_hit_rate'],  # 🔥 关键：是否撞底

                # 5️⃣ 策略健康（3 个）
                "5_Policy/tanh_saturation_ratio": metrics['tanh_saturation_ratio'],
                "5_Policy/distance_to_origin": metrics['policy_distance_to_origin'],
                "5_Policy/entropy": metrics['policy_entropy'],
                "5_Policy/ood_distance_mean_det": metrics['ood_distance_mean_det'],
                "5_Policy/ood_distance_max_det": metrics['ood_distance_max_det'],
                "5_Policy/ood_distance_mean_samp": metrics['ood_distance_mean_samp'],
                "5_Policy/ood_distance_max_samp": metrics['ood_distance_max_samp'],

                # 6️⃣ 采样时序监控（1 个）- 🔥 数据顺序效应探针
                "Batch_Data/reward_mean": metrics['batch_reward_mean'],
            }

            # 完整分类日志记录
            progress_pct = (t + 1) / config.max_timesteps * 100
            logging.info(f"[Training] Step {t+1}/{config.max_timesteps} ({progress_pct:.1f}%)")

            # 1. Loss 指标
            logging.info(f"  [1] Loss: bc_loss={metrics['bc_loss']:.6f}")

            # 2. Action 统计
            logging.info(f"  [2] Action: mean={metrics['action_mean']:.6f}, std={metrics['action_std']:.6f}, min={metrics['action_min']:.6f}, max={metrics['action_max']:.6f}, target_mean={metrics['target_action_mean']:.6f}, target_std={metrics['target_action_std']:.6f}")

            # 3. 梯度范数
            logging.info(f"  [3] Gradient: actor_norm={metrics['actor_grad_norm']:.6f}, gru_norm={metrics['gru_grad_norm']:.6f}")

            # 4. Log Std 统计（"死亡探头"）
            logging.info(
                f"  [4] LogStd: raw(mean={metrics['actor_log_std_raw_mean']:.6f}, floor_hit={metrics['actor_log_std_floor_hit_rate']:.4f}), "
                f"clamped(mean={metrics['actor_log_std_mean']:.6f})"
            )

            # 5. 策略健康
            logging.info(
                f"  [5] Policy: tanh_saturation={metrics['tanh_saturation_ratio']:.4f}, "
                f"distance_to_origin={metrics['policy_distance_to_origin']:.6f}, "
                f"entropy={metrics['policy_entropy']:.6f}, "
                f"ood_det(mean={metrics['ood_distance_mean_det']:.4f}, max={metrics['ood_distance_max_det']:.4f}), "
                f"ood_samp(mean={metrics['ood_distance_mean_samp']:.4f}, max={metrics['ood_distance_max_samp']:.4f})"
            )
            logging.info("")  # 空行分隔

            if swan_logger:
                swan_logger.log_metrics(swanlab_metrics, step=t+1)

        # Evaluation
        if eval_env is not None and (t + 1) % config.eval_freq == 0:
            eval_metrics = eval_env.evaluate_policy(
                agent=agent,
                num_episodes=10,
                deterministic=True
            )

            # 简洁的 Evaluation 日志
            logging.info(f"[Evaluation] Step {t+1}/{config.max_timesteps}")
            logging.info(f"  Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} (min={eval_metrics.get('min_reward', 0):.2f}, max={eval_metrics.get('max_reward', 0):.2f})")
            logging.info(f"  Episode Length: {eval_metrics['mean_episode_length']:.1f}")
            logging.info("")  # 空行分隔

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
                f"bc_{config.env_name}_{config.dataset_quality}_lr{config.learning_rate}_seed{config.seed}_{config.run_id}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Final save
    final_path = os.path.join(
        config.checkpoint_dir,
        f"bc_{config.env_name}_{config.dataset_quality}_lr{config.learning_rate}_seed{config.seed}_{config.run_id}_final.pt"
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
    logging.info(f"BC 训练完成！")
    logging.info(f"{'='*80}")

    if swan_logger:
        swan_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BC (Behavior Cloning) on offline datasets")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment",
                        help="实验名称")
    parser.add_argument("--env_name", type=str, default="diffuse_mix",
                        help="环境名称")
    parser.add_argument("--dataset_quality", type=str, default="expert",
                        help="数据集质量 (旧 benchmark: random/medium/expert, 新 benchmark: v2_b3/v2_b5)")
    parser.add_argument("--seed", type=int, default=58407201,
                        help="随机种子")
    parser.add_argument("--run_id", type=str, default="",
                        help="唯一运行标识符 (格式：MMDD_HHMM, 如果为空则自动生成)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    # 数据集配置
    parser.add_argument("--dataset_path", type=str, default="",
                        help="数据集路径 (如果为空则自动生成)")

    # 训练配置
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
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")

    # 🔥 Ranker configuration (解耦架构)
    parser.add_argument("--ranker", type=str, default="gems",
                        choices=["gems", "topk", "kheadargmax", "wolpertinger", "wolpertinger_slate"],
                        help="Ranker 类型：gems, topk, kheadargmax, wolpertinger, wolpertinger_slate")

    # 🔥 Wolpertinger Ranker 参数
    parser.add_argument("--wolpertinger_k", type=int, default=50,
                        help="Wolpertinger kNN 候选数量")
    parser.add_argument("--wolpertinger_hidden_dims", type=int, nargs='+', default=[256, 128],
                        help="Wolpertinger Actor 隐层维度")

    # SwanLab 配置
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用 SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用 SwanLab")

    # 🔥 SwanLab 配置
    parser.add_argument("--swan_project", type=str, default="Offline_Slate_RL_202603",
                        help="SwanLab 项目名称")
    parser.add_argument("--swan_workspace", type=str, default="Cliff",
                        help="SwanLab 工作空间")
    parser.add_argument("--swan_mode", type=str, default="cloud",
                        choices=["cloud", "local", "offline"],
                        help="SwanLab 运行模式")
    parser.add_argument("--swan_logdir", type=str, default="experiments/swanlog",
                        help="SwanLab 本地日志目录")

    args = parser.parse_args()

    config = BCConfig(
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
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        use_swanlab=args.use_swanlab,
        # 🔥 SwanLab 配置
        swan_project=args.swan_project,
        swan_workspace=args.swan_workspace,
        swan_mode=args.swan_mode,
        swan_logdir=args.swan_logdir,
        # 🔥 Ranker configuration
        ranker_type=args.ranker,
        wolpertinger_k=args.wolpertinger_k,
        wolpertinger_hidden_dims=args.wolpertinger_hidden_dims,
    )

    train_bc(config)