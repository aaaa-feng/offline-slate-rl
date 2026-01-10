"""
TD3+BC for GeMS datasets (Enhanced Version with SwanLab)
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2106.06860.pdf

Enhancements:
- SwanLab logging support
- Simplified checkpoint/log structure
- Comprehensive metrics monitoring
- Reward normalization by default
"""
import copy
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入路径配置
sys.path.insert(0, str(PROJECT_ROOT.parent))
from config import paths
from config.offline_config import TD3BCConfig, auto_generate_paths, auto_generate_swanlab_config

from common.offline.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.offline.utils import set_seed, compute_mean_std, soft_update
from common.offline.networks import Actor, Critic
from common.offline.eval_env import OfflineEvalEnv
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.item_embeddings import ItemEmbeddings

# SwanLab Logger import (离线RL专用版本)
try:
    from common.offline.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    # 如果 swanlab 包不存在,使用 dummy logger
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available, using dummy logger")

    class SwanlabLogger:
        """Dummy logger when swanlab is not available"""
        def __init__(self, *args, **kwargs):
            pass

        def log_metrics(self, metrics, step=None):
            pass

        def log_hyperparams(self, params):
            pass

        @property
        def experiment(self):
            class DummyExperiment:
                def finish(self):
                    pass
            return DummyExperiment()

TensorBatch = List[torch.Tensor]


class TD3_BC:
    """TD3+BC algorithm with Dual-Stream End-to-End GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config: TD3BCConfig,
        ranker_params: Dict,  # 🔥 GeMS-aligned: 接收 Ranker 参数
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0  # 归一化后固定为 1.0

        # ========================================================================
        # 🔥 关键：从 Ranker 参数中提取组件（复刻 BC 逻辑）
        # ========================================================================

        # 1. Action Bounds（直接使用 Ranker 的）
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        logging.info("=" * 80)
        logging.info("=== Action Bounds from GeMS ===")
        logging.info(f"  center shape: {self.action_center.shape}")
        logging.info(f"  center mean: {self.action_center.mean().item():.6f}")
        logging.info(f"  scale shape: {self.action_scale.shape}")
        logging.info(f"  scale mean: {self.action_scale.mean().item():.6f}")
        logging.info("=" * 80)

        # 2. Item Embeddings（使用 GeMS 训练后的）
        self.item_embeddings = ranker_params['item_embeddings']
        logging.info(f"Item embeddings from GeMS: {self.item_embeddings.num_items} items, "
                    f"{self.item_embeddings.embedd_dim} dims")

        # 3. 初始化双流 GRU belief encoder (CRITICAL: beliefs=["actor", "critic"])
        logging.info("Initializing Dual-Stream GRU belief encoder...")
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
            beliefs=["actor", "critic"],  # DUAL-STREAM ARCHITECTURE
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim  # 🔥 显式传入
        )

        # 4. 🔥 关键：双重保险 - 再次冻结 Embeddings
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()
        logging.info("✅ Item embeddings frozen (double-checked)")

        # Initialize Actor network
        self.actor = Actor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=self.max_action,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Initialize Critic networks
        self.critic_1 = Critic(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # CRITICAL: 分离优化器 (Actor + Actor GRU) vs (Critics + Critic GRU)
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.actor_lr)

        self.critic_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["critic"].parameters()},
            {'params': self.critic_1.parameters()},
            {'params': self.critic_2.parameters()}
        ], lr=config.critic_lr)

        self.total_it = 0
        logging.info("TD3_BC initialized with Dual-Stream E2E GRU")

    def train(self, batch) -> Dict[str, float]:
        """
        训练一步 (双流端到端训练: Actor GRU + Critic GRU)

        Args:
            batch: TrajectoryBatch with obs, actions, rewards, dones
        """
        self.total_it += 1

        # Step 1: 双流 GRU 前向传播
        states, next_states = self.belief.forward_batch(batch)
        s_actor = states["actor"]  # [sum_seq_lens, belief_hidden_dim]
        s_critic = states["critic"]  # [sum_seq_lens, belief_hidden_dim]
        ns_actor = next_states["actor"]  # [sum_seq_lens, belief_hidden_dim]
        ns_critic = next_states["critic"]  # [sum_seq_lens, belief_hidden_dim]

        # Step 2: Concatenate trajectory data
        true_actions = torch.cat(batch.actions, dim=0)  # [sum_seq_lens, action_dim]
        rewards = torch.cat(batch.rewards, dim=0) if batch.rewards else None
        dones = torch.cat(batch.dones, dim=0) if batch.dones else None

        # Step 3: Critic Update (TD3 Loss)
        with torch.no_grad():
            # 使用 Actor GRU 的 next_state 生成 next_action
            noise = (torch.randn_like(true_actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(ns_actor) + noise).clamp(
                -self.max_action, self.max_action
            )

            # 使用 Critic GRU 的 next_state 计算 target Q
            target_q1 = self.critic_1_target.q1(ns_critic, next_action)
            target_q2 = self.critic_2_target.q1(ns_critic, next_action)
            target_q = torch.min(target_q1, target_q2)

            if rewards is not None and dones is not None:
                target_q = rewards + (1 - dones) * self.config.gamma * target_q
            else:
                # 如果没有 reward/done，使用简化版本
                target_q = target_q * self.config.gamma

            # 🔧 HOT-FIX: Target Q Clamping (防止Q值爆炸)
            # 由于reward已经缩放到 [-1, 1] 范围,Q值应该在 [-10, 10] 范围内
            target_q = torch.clamp(target_q, -10.0, 10.0)

        # 使用 Critic GRU 的 current_state 计算 current Q
        # Detach s_critic to avoid gradient conflict (Critic optimizer doesn't include GRU)
        current_q1 = self.critic_1.q1(s_critic.detach(), true_actions)
        current_q2 = self.critic_2.q1(s_critic.detach(), true_actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize Critic + Critic GRU
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 🔧 HOT-FIX: Critic Gradient Clipping (防止梯度爆炸)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            10.0  # 从 float('inf') 改为 10.0
        )
        self.critic_optimizer.step()

        # Step 4: Actor Update (TD3+BC Loss) - Delayed
        actor_loss = None
        bc_loss = None
        if self.total_it % self.config.policy_freq == 0:
            # 使用 Actor GRU 的 state 生成 action
            pi = self.actor(s_actor)

            # CRITICAL: 使用 detached Critic GRU state 计算 Q 值
            # 这样梯度不会流回 Critic GRU
            q = self.critic_1.q1(s_critic.detach(), pi)
            lmbda = self.config.alpha / q.abs().mean().detach()

            # BC loss (单独记录)
            bc_loss = F.mse_loss(pi, true_actions)

            # TD3+BC loss
            actor_loss = -lmbda * q.mean() + bc_loss

            # Optimize Actor + Actor GRU
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 🔧 HOT-FIX: Actor Gradient Clipping (防止梯度爆炸)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)  # 从 float('inf') 改为 100.0
            self.actor_optimizer.step()

            # Update target networks
            soft_update(self.actor_target, self.actor, self.config.tau)
            soft_update(self.critic_1_target, self.critic_1, self.config.tau)
            soft_update(self.critic_2_target, self.critic_2, self.config.tau)

        # 返回完整的监控指标
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
            "bc_loss": bc_loss.item() if bc_loss is not None else 0.0,
            "q_value": current_q1.mean().item(),
            "q1_value": current_q1.mean().item(),
            "q2_value": current_q2.mean().item(),
            "q_max": max(current_q1.max().item(), current_q2.max().item()),
            "q_min": min(current_q1.min().item(), current_q2.min().item()),
            "q_std": current_q1.std().item(),
            "target_q": target_q.mean().item(),
            # 🔧 HOT-FIX: 新增诊断指标
            "target_q_max": target_q.max().item(),
            "target_q_min": target_q.min().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item() if actor_loss is not None else 0.0,
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        选择动作 (使用 Actor GRU 编码 + Actor 预测 + 反归一化)

        Args:
            obs: Dict with 'slate' and 'clicks' (torch.Tensor or numpy arrays)
            deterministic: 是否确定性选择 (TD3+BC 总是确定性的)

        Returns:
            action: 反归一化后的动作
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # 只使用 Actor GRU 编码 (推理时不需要 Critic)
        belief_state = self.belief.forward(obs_tensor, done=False)["actor"]

        # Actor 预测
        raw_action = self.actor(belief_state)  # [1, action_dim]

        # 反归一化
        action = raw_action * self.action_scale + self.action_center
        action = action.cpu().numpy().flatten()

        return action

    def reset_hidden(self):
        """
        重置双流 GRU 隐藏状态 (在每个 episode 开始时调用)
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
            'belief_state_dict': self.belief.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'embeddings_meta': {
                'num_items': self.item_embeddings.num_items,
                'embedd_dim': self.item_embeddings.embedd_dim,
            },
            'action_dim': self.action_dim,
            'total_it': self.total_it,
            'config': self.config,
        }, filepath)
        logging.info(f"✅ Model saved to {filepath} (with embeddings_meta)")

    def load(self, filepath: str):
        """加载模型 (包含双流 GRU + Actor + Critics + 归一化参数)"""
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
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """从 Checkpoint 独立加载，无需 GeMS"""
        logging.info("=" * 80)
        logging.info("=== Loading TD3_BC from Checkpoint (Standalone) ===")
        logging.info(f"Checkpoint: {checkpoint_path}")
        logging.info("=" * 80)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 1. 恢复 Embeddings
        embeddings_meta = checkpoint['embeddings_meta']
        belief_state = checkpoint['belief_state_dict']
        embedding_weights = belief_state['item_embeddings.actor.embedd.weight']

        agent_embeddings = ItemEmbeddings(
            num_items=embeddings_meta['num_items'],
            item_embedd_dim=embeddings_meta['embedd_dim'],
            device=device,
            weights=embedding_weights
        )
        logging.info(f"✅ Embeddings restored: {embeddings_meta['num_items']} items")

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
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.total_it = checkpoint['total_it']

        logging.info(f"✅ TD3_BC loaded from {checkpoint_path} (standalone)")
        logging.info("=" * 80)
        return agent


def train_td3_bc(config: TD3BCConfig):
    """
    Train TD3+BC on GeMS dataset with SwanLab logging

    Args:
        config: Training configuration
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d")

    # 自动生成路径配置
    config = auto_generate_paths(config, timestamp)

    # 自动生成 SwanLab 配置
    config = auto_generate_swanlab_config(config)

    # 创建目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 生成日志文件名
    log_filename = f"{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_seed{config.seed}_{config.run_id}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    # 清除已有的handlers并重新配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置logging (输出到文件和stdout)
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
                    description=config.swan_description,
                    tags=config.swan_tags,
                    config=config.__dict__,
                    mode=config.swan_mode,
                    logdir=config.swan_logdir,
                )
                logging.info(f"SwanLab initialized: project={config.swan_project}, run={config.run_name}")
            except Exception as e:
                logging.warning(f"SwanLab initialization failed: {e}")
                config.use_swanlab = False

    # 打印完整命令 (参考在线算法)
    logging.info("=" * 80)
    logging.info("=== TD3+BC Training Configuration ===")
    logging.info("=" * 80)
    logging.info(f"Environment: {config.env_name}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Seed: {config.seed}")
    logging.info(f"Alpha (BC weight): {config.alpha}")
    logging.info(f"Discount (gamma): {config.gamma}")
    logging.info(f"Normalize rewards: {config.normalize_rewards}")
    logging.info(f"Max timesteps: {config.max_timesteps}")
    logging.info(f"Batch size: {config.batch_size}")
    logging.info(f"Learning rate: {config.learning_rate}")
    logging.info(f"Log file: {log_filepath}")
    logging.info(f"Checkpoint dir: {config.checkpoint_dir}")
    logging.info("=" * 80)

    # ========================================================================
    # 🔥 关键：加载 GeMS 并提取组件（复刻 BC 逻辑）
    # ========================================================================
    from rankers.gems.rankers import GeMS

    # 1. 构建 GeMS Checkpoint 路径
    gems_checkpoint_name = (
        f"GeMS_{config.env_name}_{config.dataset_quality}_"
        f"latent32_beta1.0_click0.5_seed58407201"
    )
    gems_path = (
        f"/data/liyuefeng/offline-slate-rl/checkpoints/gems/offline/"
        f"{gems_checkpoint_name}.ckpt"
    )

    logging.info("=" * 80)
    logging.info("=== Loading Pretrained GeMS ===")
    logging.info(f"Checkpoint: {gems_path}")

    # 2. 加载 GeMS Ranker
    temp_embeddings = ItemEmbeddings.from_pretrained(
        config.item_embedds_path,
        config.device
    )

    ranker = GeMS.load_from_checkpoint(
        gems_path,
        map_location=config.device,
        item_embeddings=temp_embeddings,
        device=config.device,
        rec_size=config.rec_size,
        item_embedd_dim=config.item_embedd_dim,
        num_items=config.num_items,
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
    ranker.eval()
    ranker.freeze()
    logging.info("✅ GeMS loaded and frozen")

    # 显式强制设备同步 (对标 eval_env.py 和 iql.py)
    ranker = ranker.to(config.device)
    logging.info(f"✅ GeMS moved to {config.device}")

    # 3. 提取 GeMS 训练后的 Embeddings
    gems_embedding_weights = ranker.item_embeddings.weight.data.clone()

    agent_embeddings = ItemEmbeddings(
        num_items=ranker.item_embeddings.num_embeddings,
        item_embedd_dim=ranker.item_embeddings.embedding_dim,
        device=config.device,
        weights=gems_embedding_weights
    )

    # 4. 提前冻结
    for param in agent_embeddings.parameters():
        param.requires_grad = False
    logging.info("✅ Agent embeddings created and frozen")

    # 5. 准备 Ranker 参数包
    ranker_params = {
        'item_embeddings': agent_embeddings,
        'action_center': ranker.action_center,
        'action_scale': ranker.action_scale,
        'num_items': ranker.num_items,
        'item_embedd_dim': ranker.item_embedd_dim
    }
    logging.info("=" * 80)

    # ========================================================================
    # 加载数据集
    # ========================================================================
    # Load dataset
    logging.info(f"\nLoading dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)

    logging.info(f"Dataset statistics:")
    logging.info(f"  Slates shape: {dataset['slates'].shape}")
    logging.info(f"  Clicks shape: {dataset['clicks'].shape}")
    logging.info(f"  Actions shape: {dataset['actions'].shape}")
    logging.info(f"  Total transitions: {len(dataset['slates'])}")

    # ========================================================================
    # 内存重打标 (In-Memory Action Relabeling) - Zero Trust Strategy
    # ========================================================================
    logging.info("")
    logging.info("=" * 80)
    logging.info("⚠️  IN-MEMORY ACTION RELABELING")
    logging.info("=" * 80)
    logging.info("Strategy: Zero Trust - Regenerate all actions using current GeMS")
    logging.info("Reason:   Ensure absolute consistency between training and inference")

    # 1. Extract raw discrete data
    raw_slates = torch.tensor(dataset['slates'], device=config.device, dtype=torch.long)
    raw_clicks = torch.tensor(dataset['clicks'], device=config.device, dtype=torch.float)
    total_samples = len(raw_slates)

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

            if (i + batch_size) % 100000 == 0 or (i + batch_size) >= total_samples:
                processed = min(i + batch_size, total_samples)
                logging.info(f"  Progress: {processed:,}/{total_samples:,}")

    new_actions = np.concatenate(new_actions_list, axis=0)

    # 3. Action statistics validation
    logging.info("Action Statistics (Primary Quality Indicator):")
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

    # Get dimensions
    action_dim = dataset['actions'].shape[1]

    logging.info(f"\nEnvironment info:")
    logging.info(f"  Action dim: {action_dim}")
    logging.info(f"  Rec size: {config.rec_size}")
    logging.info(f"  Belief hidden dim: {config.belief_hidden_dim}")

    # Create trajectory replay buffer
    replay_buffer = TrajectoryReplayBuffer(device=config.device)

    # 6. Load data with relabeled actions
    dataset_dict = {
        'episode_ids': dataset['episode_ids'],
        'slates': dataset['slates'],
        'clicks': dataset['clicks'],
        'actions': new_actions,  # Use relabeled actions!
    }

    # 可选字段
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
    logging.info(f"✅ Buffer loaded successfully")

    # 🔥 关键：从 Buffer 计算 Action Bounds
    logging.info("Calculating action bounds from buffer...")
    action_center, action_scale = replay_buffer.get_action_normalization_params()
    logging.info(f"✅ Action bounds calculated from buffer")

    # 更新 ranker_params 中的 action bounds
    ranker_params['action_center'] = action_center
    ranker_params['action_scale'] = action_scale

    # Initialize TD3+BC agent (with Dual-Stream E2E GRU)
    agent = TD3_BC(
        action_dim=action_dim,
        config=config,
        ranker_params=ranker_params,  # 🔥 传入 Ranker 参数
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
    logging.info(f"Starting TD3+BC training")
    logging.info(f"{'='*80}\n")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        train_metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            # 构建统一的 SwanLab 指标字典（带命名空间前缀）
            swanlab_metrics = {
                # 公共指标
                "train/actor_loss": train_metrics['actor_loss'],
                "train/actor_grad_norm": train_metrics['actor_grad_norm'],
                # Critic/Q 指标
                "train/critic_loss": train_metrics['critic_loss'],
                "train/critic_grad_norm": train_metrics['critic_grad_norm'],
                "train/q_value_mean": train_metrics['q_value'],
                "train/q1_value": train_metrics['q1_value'],
                "train/q2_value": train_metrics['q2_value'],
                "train/q_value_std": train_metrics['q_std'],
                "train/q_max": train_metrics['q_max'],
                "train/q_min": train_metrics['q_min'],
                "train/target_q_mean": train_metrics['target_q'],
                # 🔧 HOT-FIX: 新增诊断指标
                "train/target_q_max": train_metrics['target_q_max'],
                "train/target_q_min": train_metrics['target_q_min'],
                # TD3+BC 特有指标
                "train/bc_loss": train_metrics['bc_loss'],
            }

            # 全量本地日志记录（与 SwanLab 完全一致）
            log_parts = [f"Step {t+1}/{config.max_timesteps}:"]
            for key, value in swanlab_metrics.items():
                short_key = key.replace("train/", "")
                log_parts.append(f"{short_key}={value:.6f}")
            logging.info(", ".join(log_parts))

            if config.use_swanlab and swan_logger:
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
                f"td3bc_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(
        config.checkpoint_dir,
        f"td3bc_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_final.pt"
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
    logging.info(f"Training completed!")
    logging.info(f"{'='*80}\n")

    if config.use_swanlab and swan_logger:
        swan_logger.experiment.finish()

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TD3+BC (TD3 with Behavior Cloning) on offline datasets")

    # 实验配置
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

    # TD3+BC特定参数
    parser.add_argument("--alpha", type=float, default=2.5,
                        help="BC正则化系数")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="软更新系数")
    parser.add_argument("--policy_noise", type=float, default=0.2,
                        help="策略噪声")
    parser.add_argument("--noise_clip", type=float, default=0.5,
                        help="噪声裁剪")
    parser.add_argument("--policy_freq", type=int, default=2,
                        help="策略更新频率")

    # SwanLab配置
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")

    args = parser.parse_args()

    config = TD3BCConfig(
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
        alpha=args.alpha,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        use_swanlab=args.use_swanlab,
    )

    train_td3_bc(config)
