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
from typing import Dict, List, Optional, Tuple
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

from common.offline.buffer import ReplayBuffer
from common.offline.utils import set_seed, compute_mean_std, soft_update
from common.offline.networks import Actor, Critic
from common.offline.eval_env import OfflineEvalEnv

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
    """TD3+BC algorithm with enhanced monitoring"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: TD3BCConfig,
        action_center: torch.Tensor = None,  # 新增
        action_scale: torch.Tensor = None,   # 新增
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.max_action = max_action

        # 归一化参数 (保存在 agent 中,不参与梯度更新)
        if action_center is not None:
            self.action_center = action_center.to(self.device)
            self.action_scale = action_scale.to(self.device)
        else:
            self.action_center = torch.zeros(action_dim, device=self.device)
            self.action_scale = torch.ones(action_dim, device=self.device)

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action, config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)

        self.critic_1 = Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.learning_rate)

        self.critic_2 = Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.learning_rate)

        self.total_it = 0

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Train one step with comprehensive metrics"""
        self.total_it += 1
        state, action, reward, next_state, done = batch

        # Critic training
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target.q1(next_state, next_action)
            target_q2 = self.critic_2_target.q1(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * target_q

        # Get current Q estimates
        current_q1 = self.critic_1.q1(state, action)
        current_q2 = self.critic_2.q1(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        actor_loss = None
        bc_loss = None
        if self.total_it % self.config.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1.q1(state, pi)
            lmbda = self.config.alpha / q.abs().mean().detach()

            # BC loss (单独记录)
            bc_loss = F.mse_loss(pi, action)
            actor_loss = -lmbda * q.mean() + bc_loss

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
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
        }

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def save(self, filepath: str):
        """Save model (包含归一化参数)"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'action_center': self.action_center,  # 新增
            'action_scale': self.action_scale,    # 新增
            'total_it': self.total_it,
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model (包含归一化参数)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.action_center = checkpoint['action_center']  # 新增
        self.action_scale = checkpoint['action_scale']    # 新增
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")


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
    log_filename = f"{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_seed{config.seed}_{timestamp}.log"
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

    # Load dataset
    logging.info(f"\nLoading GeMS dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)

    # Print dataset statistics
    logging.info(f"Dataset statistics:")
    logging.info(f"  Observations shape: {dataset['observations'].shape}")
    logging.info(f"  Actions shape: {dataset['actions'].shape}")
    logging.info(f"  Total transitions: {len(dataset['observations'])}")
    logging.info(f"  Num episodes: {dataset['terminals'].sum()}")
    logging.info(f"  Avg reward: {dataset['rewards'].mean():.4f}")
    logging.info(f"  Reward std: {dataset['rewards'].std():.4f}")

    # Get dimensions
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    max_action = 1.0  # 归一化后固定为 1.0 (关键修改!)

    logging.info(f"\nEnvironment info:")
    logging.info(f"  State dim: {state_dim}")
    logging.info(f"  Action dim: {action_dim}")
    logging.info(f"  Max action: {max_action} (normalized)")

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=len(dataset['observations']),
        device=config.device
    )

    # Load data into buffer
    dataset_dict = {
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'rewards': dataset['rewards'],
        'next_observations': dataset['next_observations'],
        'terminals': dataset['terminals'],
    }
    replay_buffer.load_d4rl_dataset(dataset_dict)

    # Normalize states if needed
    state_mean, state_std = 0.0, 1.0
    if config.normalize_states:
        state_mean, state_std = compute_mean_std(dataset['observations'])
        replay_buffer.normalize_states(state_mean, state_std)
        logging.info(f"States normalized")

    # Normalize rewards if needed
    if config.normalize_rewards:
        reward_mean = dataset['rewards'].mean()
        reward_std = dataset['rewards'].std()
        replay_buffer.normalize_rewards(reward_mean, reward_std)
        logging.info(f"Rewards normalized (mean={reward_mean:.4f}, std={reward_std:.4f})")

    # === 关键: Action normalization (必须为True!) ===
    if not config.normalize_actions:
        raise ValueError("normalize_actions must be True for offline RL!")
    action_center, action_scale = replay_buffer.normalize_actions()
    logging.info(f"Actions normalized to [-1, 1]")

    # Initialize TD3+BC
    agent = TD3_BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        config=config,
        action_center=action_center,  # 新增
        action_scale=action_scale,    # 新增
    )

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
    logging.info(f"Starting TD3+BC training")
    logging.info(f"{'='*80}\n")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        train_metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            log_msg = (f"Step {t+1}/{config.max_timesteps}: "
                      f"critic_loss={train_metrics['critic_loss']:.4f}, "
                      f"actor_loss={train_metrics['actor_loss']:.4f}, "
                      f"bc_loss={train_metrics['bc_loss']:.4f}, "
                      f"q_value={train_metrics['q_value']:.4f}")
            logging.info(log_msg)

            # SwanLab logging (完整指标)
            if config.use_swanlab and swan_logger:
                swan_logger.log_metrics({
                    # 训练Loss
                    "train/critic_loss": train_metrics['critic_loss'],
                    "train/actor_loss": train_metrics['actor_loss'],
                    "train/bc_loss": train_metrics['bc_loss'],
                    "train/q_value": train_metrics['q_value'],
                    "train/q1_value": train_metrics['q1_value'],
                    "train/q2_value": train_metrics['q2_value'],
                    # Q值监控
                    "train/q_max": train_metrics['q_max'],
                    "train/q_min": train_metrics['q_min'],
                    "train/q_std": train_metrics['q_std'],
                    # 训练状态
                    "train/step": t + 1,
                    "train/learning_rate": config.learning_rate,
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

        # Save checkpoint
        if (t + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"step_{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(
        config.checkpoint_dir,
        f"td3_bc_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_final.pt"
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
