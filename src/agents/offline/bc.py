"""
Behavior Cloning (BC) for GeMS datasets
最简单的离线 RL baseline,用于验证数据加载和归一化
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple
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

from common.offline.buffer import ReplayBuffer
from common.offline.utils import set_seed, compute_mean_std
from common.offline.networks import Actor
from common.offline.eval_env import OfflineEvalEnv

# SwanLab Logger
try:
    from common.offline.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available")


class BCAgent:
    """Behavior Cloning Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: BCConfig,
        action_center: torch.Tensor = None,
        action_scale: torch.Tensor = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # 归一化参数 (保存在 agent 中,不参与梯度更新)
        if action_center is not None:
            self.action_center = action_center.to(self.device)
            self.action_scale = action_scale.to(self.device)
        else:
            self.action_center = torch.zeros(action_dim, device=self.device)
            self.action_scale = torch.ones(action_dim, device=self.device)

        # Actor network (max_action=1.0 因为 action 已归一化)
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_dim=config.hidden_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )

        self.total_it = 0

    def train(self, batch) -> Dict[str, float]:
        """训练一步"""
        self.total_it += 1
        state, action, _, _, _ = batch

        # 前向传播
        pred_action = self.actor(state)

        # BC Loss (MSE)
        loss = F.mse_loss(pred_action, action)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "bc_loss": loss.item(),
            "action_mean": pred_action.mean().item(),
            "action_std": pred_action.std().item()
        }

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """选择动作 (带反归一化)"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().numpy().flatten()

        # 反归一化
        action_center_np = self.action_center.cpu().numpy()
        action_scale_np = self.action_scale.cpu().numpy()
        action = action * action_scale_np + action_center_np

        return action

    def save(self, filepath: str):
        """保存模型 (包含归一化参数)"""
        torch.save({
            'actor': self.actor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'total_it': self.total_it,
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")


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
    log_filename = f"{config.env_name}_{config.dataset_quality}_seed{config.seed}_{timestamp}.log"
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

    # Load dataset
    logging.info(f"\nLoading dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)

    logging.info(f"Dataset statistics:")
    logging.info(f"  Observations shape: {dataset['observations'].shape}")
    logging.info(f"  Actions shape: {dataset['actions'].shape}")
    logging.info(f"  Total transitions: {len(dataset['observations'])}")

    # Get dimensions
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    logging.info(f"\nEnvironment info:")
    logging.info(f"  State dim: {state_dim}")
    logging.info(f"  Action dim: {action_dim}")

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=len(dataset['observations']),
        device=config.device
    )

    # Load data
    dataset_dict = {
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'rewards': dataset['rewards'],
        'next_observations': dataset['next_observations'],
        'terminals': dataset['terminals'],
    }
    replay_buffer.load_d4rl_dataset(dataset_dict)

    # === 关键: 数据归一化 ===
    # 1. State normalization
    state_mean, state_std = 0.0, 1.0
    if config.normalize_states:
        state_mean, state_std = compute_mean_std(dataset['observations'])
        replay_buffer.normalize_states(state_mean, state_std)
        logging.info(f"States normalized")

    # 2. Action normalization (关键! 必须为True)
    if not config.normalize_actions:
        raise ValueError("normalize_actions must be True for offline RL!")
    action_center, action_scale = replay_buffer.normalize_actions()
    logging.info(f"Actions normalized to [-1, 1]")

    # Initialize BC agent
    agent = BCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        action_center=action_center,
        action_scale=action_scale,
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
            log_msg = (f"Step {t+1}/{config.max_timesteps}: "
                      f"bc_loss={metrics['bc_loss']:.6f}, "
                      f"action_mean={metrics['action_mean']:.4f}, "
                      f"action_std={metrics['action_std']:.4f}")
            logging.info(log_msg)

            if swan_logger:
                swan_logger.log_metrics(metrics, step=t+1)

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
                f"bc_{config.env_name}_{config.dataset_quality}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Final save
    final_path = os.path.join(
        config.checkpoint_dir,
        f"bc_{config.env_name}_{config.dataset_quality}_final.pt"
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
