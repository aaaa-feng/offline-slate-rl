#!/usr/bin/env python3
"""
通用Agent训练脚本

支持所有离线RL算法的训练:
- TD3+BC
- CQL (待实现)
- IQL (待实现)

使用方法:
    python scripts/train_agent.py --agent td3_bc --env_name diffuse_topdown --seed 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_rl_baselines.common.buffer import ReplayBuffer
from offline_rl_baselines.common.utils import set_seed, compute_mean_std


def get_agent(agent_type: str, state_dim: int, action_dim: int, device: str, **kwargs):
    """
    根据类型创建Agent

    Args:
        agent_type: 算法类型 (td3_bc, cql, iql)
        state_dim: 状态维度
        action_dim: 动作维度
        device: 计算设备

    Returns:
        agent: Agent实例
    """
    if agent_type == 'td3_bc':
        from offline_rl_baselines.agents.offline.td3_bc import TD3BCAgent, TD3BCConfig
        config = TD3BCConfig(
            alpha=kwargs.get('alpha', 2.5),
            hidden_dim=kwargs.get('hidden_dim', 256),
            learning_rate=kwargs.get('learning_rate', 3e-4),
            discount=kwargs.get('discount', 0.99),
        )
        return TD3BCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            config=config
        )
    elif agent_type == 'cql':
        raise NotImplementedError("CQL agent not yet implemented in new architecture")
    elif agent_type == 'iql':
        raise NotImplementedError("IQL agent not yet implemented in new architecture")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.agent}_{args.env_name}_seed{args.seed}_{timestamp}"
    log_dir = Path(args.log_dir) / exp_name
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件
    log_file = log_dir / "train.log"

    def log(msg: str):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log("=" * 60)
    log(f"Training {args.agent.upper()} on {args.env_name}")
    log("=" * 60)
    log(f"Seed: {args.seed}")
    log(f"Device: {args.device}")
    log(f"Max timesteps: {args.max_timesteps}")
    log(f"Batch size: {args.batch_size}")
    log("")

    # 加载数据集
    dataset_path = Path(PROJECT_ROOT) / "offline_datasets" / f"{args.env_name}_expert.npz"
    if not dataset_path.exists():
        # 尝试另一个路径格式
        dataset_path = Path(PROJECT_ROOT) / "offline_datasets" / args.env_name / "expert_data_d4rl.npz"

    log(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = np.load(dataset_path)

    # 打印数据集统计
    log(f"\nDataset statistics:")
    log(f"  Observations: {dataset['observations'].shape}")
    log(f"  Actions: {dataset['actions'].shape}")
    log(f"  Rewards: mean={dataset['rewards'].mean():.4f}, std={dataset['rewards'].std():.4f}")
    log(f"  Episodes: {dataset['terminals'].sum()}")
    log("")

    # 获取维度
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    log(f"State dim: {state_dim}")
    log(f"Action dim: {action_dim}")
    log("")

    # 创建ReplayBuffer
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=len(dataset['observations']) + 1000,
        device=args.device
    )

    # 加载数据到Buffer
    buffer.load_d4rl_dataset({
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'rewards': dataset['rewards'],
        'next_observations': dataset['next_observations'],
        'terminals': dataset['terminals'],
    })

    # 状态归一化
    state_mean, state_std = None, None
    if args.normalize:
        state_mean, state_std = compute_mean_std(dataset['observations'])
        buffer.normalize_states(state_mean, state_std)
        log("States normalized")
        log(f"  Mean: {state_mean[:5]}...")
        log(f"  Std: {state_std[:5]}...")
        log("")

    # Reward归一化（防止Q值爆炸）
    reward_mean, reward_std = None, None
    if args.normalize_reward:
        reward_mean, reward_std = buffer.normalize_rewards()
        log(f"Rewards normalized: mean={reward_mean:.4f}, std={reward_std:.4f}")
        log("")

    # 创建Agent
    agent = get_agent(
        agent_type=args.agent,
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        discount=args.discount,
    )

    log(f"Agent config: {agent.get_config()}")
    log("")

    # 训练循环
    log("=" * 60)
    log("Starting training...")
    log("=" * 60)

    best_q_value = float('-inf')

    for t in range(args.max_timesteps):
        # 采样batch
        batch = buffer.sample(args.batch_size)

        # 转换为字典格式
        batch_dict = {
            'states': batch[0],
            'actions': batch[1],
            'rewards': batch[2],
            'next_states': batch[3],
            'dones': batch[4],
        }

        # 训练一步
        metrics = agent.train(batch_dict)

        # 日志记录
        if (t + 1) % args.log_freq == 0:
            log_msg = f"Step {t+1}/{args.max_timesteps}"
            for k, v in metrics.items():
                log_msg += f" | {k}: {v:.4f}"
            log(log_msg)

        # 保存checkpoint
        if (t + 1) % args.save_freq == 0:
            save_path = checkpoint_dir / f"checkpoint_{t+1}"
            agent.save(str(save_path))
            log(f"Checkpoint saved: {save_path}")

        # 追踪最佳Q值
        if metrics.get('q_value', 0) > best_q_value:
            best_q_value = metrics['q_value']

    # 保存最终模型
    final_path = checkpoint_dir / "final"
    agent.save(str(final_path))
    log(f"\nFinal model saved: {final_path}")

    # 保存归一化参数
    if args.normalize and state_mean is not None:
        np.savez(
            checkpoint_dir / "normalization.npz",
            mean=state_mean,
            std=state_std
        )
        log(f"Normalization params saved")

    log("")
    log("=" * 60)
    log("Training completed!")
    log(f"Best Q value: {best_q_value:.4f}")
    log("=" * 60)

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train offline RL agent")

    # 基本参数
    parser.add_argument("--agent", type=str, default="td3_bc",
                        choices=["td3_bc", "cql", "iql"],
                        help="Agent type")
    parser.add_argument("--env_name", type=str, default="diffuse_topdown",
                        choices=["diffuse_topdown", "diffuse_mix", "diffuse_divpen"],
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")

    # 训练参数
    parser.add_argument("--max_timesteps", type=int, default=int(1e6),
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize states")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--normalize_reward", action="store_true", default=True,
                        help="Normalize rewards (prevents Q-value explosion)")
    parser.add_argument("--no_normalize_reward", dest="normalize_reward", action="store_false")

    # 算法参数
    parser.add_argument("--alpha", type=float, default=2.5,
                        help="BC weight for TD3+BC")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="Discount factor")

    # 日志参数
    parser.add_argument("--log_dir", type=str,
                        default="offline_rl_baselines/experiments/logs",
                        help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="offline_rl_baselines/experiments/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="Log frequency")
    parser.add_argument("--save_freq", type=int, default=100000,
                        help="Checkpoint save frequency")

    args = parser.parse_args()

    # 检查CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    train(args)


if __name__ == "__main__":
    main()
