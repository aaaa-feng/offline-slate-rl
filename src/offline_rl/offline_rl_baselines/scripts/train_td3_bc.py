#!/usr/bin/env python3
"""
训练TD3+BC的简单脚本
"""
import sys
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_rl_baselines.algorithms.td3_bc import TD3BCConfig, train_td3_bc


def main():
    parser = argparse.ArgumentParser(description="Train TD3+BC on GeMS dataset")

    # Dataset
    parser.add_argument("--env_name", type=str, default="diffuse_topdown",
                        choices=["diffuse_topdown", "diffuse_mix", "diffuse_divpen"],
                        help="Environment name")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path to dataset .npz file (if empty, will use default path)")

    # Training
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_timesteps", type=int, default=int(1e6),
                        help="Maximum training timesteps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--eval_freq", type=int, default=int(5e3),
                        help="Evaluation frequency")

    # TD3+BC parameters
    parser.add_argument("--alpha", type=float, default=2.5,
                        help="BC weight (higher = more BC)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2,
                        help="Noise added to target policy")
    parser.add_argument("--noise_clip", type=float, default=0.5,
                        help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", type=int, default=2,
                        help="Frequency of delayed policy updates")

    # Network
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")

    # Normalization
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize states")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false",
                        help="Don't normalize states")

    # Logging
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                        help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save_freq", type=int, default=int(1e5),
                        help="Checkpoint save frequency")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="GeMS-Offline-RL",
                        help="Wandb project name")
    parser.add_argument("--wandb_group", type=str, default="TD3_BC",
                        help="Wandb group name")
    parser.add_argument("--wandb_name", type=str, default="TD3_BC",
                        help="Wandb run name")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    # Set default dataset path if not provided
    if not args.dataset_path:
        args.dataset_path = str(PROJECT_ROOT / "offline_datasets" / f"{args.env_name}_expert.npz")

    # Create config
    config = TD3BCConfig(
        device=args.device,
        env_name=args.env_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        buffer_size=2_000_000,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        normalize=args.normalize,
        normalize_reward=False,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_freq=args.save_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
    )

    # Print configuration
    print("\n" + "="*60)
    print("TD3+BC Training Configuration")
    print("="*60)
    print(f"Environment: {config.env_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Seed: {config.seed}")
    print(f"Max timesteps: {config.max_timesteps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Alpha (BC weight): {config.alpha}")
    print(f"Normalize states: {config.normalize}")
    print(f"Device: {config.device}")
    print(f"Use wandb: {config.use_wandb}")
    print("="*60 + "\n")

    # Train
    train_td3_bc(config)


if __name__ == "__main__":
    main()
