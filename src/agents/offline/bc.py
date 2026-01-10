"""
Behavior Cloning (BC) for GeMS datasets
最简单的离线 RL baseline,用于验证数据加载和归一化
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

# 导入路径配置
sys.path.insert(0, str(PROJECT_ROOT.parent))
from config import paths
from config.offline_config import BCConfig, auto_generate_paths, auto_generate_swanlab_config

from common.offline.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.offline.utils import set_seed, compute_mean_std
from common.offline.networks import Actor
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


class BCAgent:
    """Behavior Cloning Agent with GeMS-aligned architecture"""

    def __init__(
        self,
        action_dim: int,
        config: BCConfig,
        ranker_params: Dict,  # 🔥 新增：接收 Ranker 参数
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim

        # ========================================================================
        # 🔥 关键：从 Ranker 参数中提取组件（复刻在线逻辑）
        # ========================================================================

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

        # 5. Actor network
        self.actor = Actor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=1.0,  # 输出 [-1, 1]，后续会用 action_scale 反归一化
            hidden_dim=config.hidden_dim
        ).to(self.device)

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
                   and actions (List[Tensor])
        """
        self.total_it += 1

        # GRU forward on trajectories
        states, _ = self.belief.forward_batch(batch)
        state = states["actor"]  # [sum_seq_lens, belief_hidden_dim]

        # Concatenate actions
        true_actions = torch.cat(batch.actions, dim=0)  # [sum_seq_lens, action_dim]

        # Actor prediction
        pred_actions = self.actor(state)  # [sum_seq_lens, action_dim]

        # BC Loss (MSE)
        loss = F.mse_loss(pred_actions, true_actions)

        # 反向传播 (同时更新 GRU 和 Actor)
        self.optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数（用于监控）
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        gru_grad_norm = torch.nn.utils.clip_grad_norm_(self.belief.gru["actor"].parameters(), float('inf'))

        self.optimizer.step()

        return {
            "bc_loss": loss.item(),
            "action_mean": pred_actions.mean().item(),
            "action_std": pred_actions.std().item(),
            "action_min": pred_actions.min().item(),
            "action_max": pred_actions.max().item(),
            "target_action_mean": true_actions.mean().item(),
            "target_action_std": true_actions.std().item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "gru_grad_norm": gru_grad_norm.item(),
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        选择动作 (使用 GRU 编码 + Actor 预测 + 反归一化)

        Args:
            obs: Dict with 'slate' and 'clicks' (torch.Tensor or numpy arrays)
            deterministic: 是否确定性选择 (BC总是确定性的)

        Returns:
            action: 反归一化后的动作
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # GRU编码
        belief_state = self.belief.forward(obs_tensor, done=False)["actor"]

        # Actor预测
        raw_action = self.actor(belief_state)

        # 反归一化
        action = raw_action * self.action_scale + self.action_center
        action = action.cpu().numpy().flatten()

        return action

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

    # 配置 logging
    log_filename = f"{config.env_name}_{config.dataset_quality}_seed{config.seed}_{config.run_id}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    # 清除已有的handlers并重新配置
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
    # 🔥 关键：加载 GeMS 并提取组件（复刻在线逻辑）
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
    logging.info("=== Loading Pretrained GeMS (Replicating Online Logic) ===")
    logging.info("=" * 80)
    logging.info(f"Checkpoint: {gems_path}")

    # 2. 加载 GeMS Ranker
    # 🔥 关键：先创建临时 ItemEmbeddings 用于加载 GeMS
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
        latent_dim=32,  # 从 checkpoint 名称获取
        lambda_click=0.5,  # 从 checkpoint 名称获取
        lambda_KL=1.0,  # 从 checkpoint 名称获取
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

    # 4. 🔥 关键：提取 GeMS 训练后的 Embeddings
    gems_embedding_weights = ranker.item_embeddings.weight.data.clone()

    agent_embeddings = ItemEmbeddings(
        num_items=ranker.item_embeddings.num_embeddings,
        item_embedd_dim=ranker.item_embeddings.embedding_dim,
        device=config.device,
        weights=gems_embedding_weights
    )

    # 5. 🔥 关键：提前冻结（在传入 GRUBelief 前）
    for param in agent_embeddings.parameters():
        param.requires_grad = False
    logging.info("✅ Agent embeddings created and frozen")

    # 6. 准备 Ranker 参数包
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
        dataset_dict['rewards'] = dataset['rewards']
    if 'terminals' in dataset:
        dataset_dict['terminals'] = dataset['terminals']

    replay_buffer.load_d4rl_dataset(dataset_dict)
    logging.info(f"✅ Buffer loaded successfully")

    # 🔥 关键：从 Buffer 计算 Action Bounds（架构师方案）
    # 使用数据集的实际统计值，而不是 GeMS checkpoint 中的值
    logging.info("Calculating action bounds from buffer...")
    action_center, action_scale = replay_buffer.get_action_normalization_params()
    logging.info(f"✅ Action bounds calculated from buffer")
    logging.info(f"  center shape: {action_center.shape}")
    logging.info(f"  center mean: {action_center.mean().item():.6f}")
    logging.info(f"  scale shape: {action_scale.shape}")
    logging.info(f"  scale mean: {action_scale.mean().item():.6f}")

    # 更新 ranker_params 中的 action bounds
    ranker_params['action_center'] = action_center
    ranker_params['action_scale'] = action_scale

    # Initialize BC agent (with GeMS-aligned architecture)
    agent = BCAgent(
        action_dim=action_dim,
        config=config,
        ranker_params=ranker_params,  # 🔥 传入 Ranker 参数
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
            # 构建统一的 SwanLab 指标字典（带命名空间前缀）
            swanlab_metrics = {
                # BC 的 bc_loss 映射到统一的 actor_loss
                "train/actor_loss": metrics['bc_loss'],
                "train/actor_grad_norm": metrics['actor_grad_norm'],
                "train/action_mean": metrics['action_mean'],
                "train/action_std": metrics['action_std'],
                "train/action_min": metrics['action_min'],
                "train/action_max": metrics['action_max'],
                # BC 特有指标
                "train/target_action_mean": metrics['target_action_mean'],
                "train/target_action_std": metrics['target_action_std'],
                "train/gru_grad_norm": metrics['gru_grad_norm'],
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
    logging.info(f"BC training completed!")
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

    # SwanLab配置
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")

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
    )

    train_bc(config)
