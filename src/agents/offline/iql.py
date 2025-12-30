"""
Implicit Q-Learning (IQL) for GeMS datasets
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2110.06169.pdf

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
from typing import Dict, Tuple
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
from config.offline_config import IQLConfig, auto_generate_paths, auto_generate_swanlab_config

from common.offline.buffer import ReplayBuffer
from common.offline.utils import set_seed, compute_mean_std, soft_update
from common.offline.networks import TanhGaussianActor, Critic, ValueFunction
from common.offline.eval_env import OfflineEvalEnv

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
    """
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)


class IQLAgent:
    """Implicit Q-Learning Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: IQLConfig,
        action_center: torch.Tensor = None,
        action_scale: torch.Tensor = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.total_it = 0

        # Normalization parameters
        if action_center is not None:
            self.action_center = action_center.to(self.device)
            self.action_scale = action_scale.to(self.device)
        else:
            self.action_center = torch.zeros(action_dim, device=self.device)
            self.action_scale = torch.ones(action_dim, device=self.device)

        # Actor
        self.actor = TanhGaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
        ).to(self.device)

        # Critics (Twin Q)
        self.critic_1 = Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, config.hidden_dim).to(self.device)

        # Value function
        self.value = ValueFunction(state_dim, config.hidden_dim, config.n_hidden).to(self.device)

        # Target critics
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.value_lr)

    def train(self, batch) -> Dict[str, float]:
        """Train one step with IQL algorithm"""
        self.total_it += 1
        state, action, reward, next_state, done = batch

        # 1. Update Value function with Expectile Loss
        with torch.no_grad():
            target_q1, target_q2 = self.critic_1_target(state, action)
            target_q = torch.min(target_q1, target_q2)

        v = self.value(state)
        value_loss = expectile_loss(target_q - v, self.config.tau).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 2. Update Q-functions
        with torch.no_grad():
            next_v = self.value(next_state)
            target_q = reward + (1 - done) * self.config.gamma * next_v

        current_q1, current_q2 = self.critic_1(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 3. Update Actor with AWR
        actor_loss = torch.tensor(0.0)
        if self.total_it % 2 == 0:
            with torch.no_grad():
                v = self.value(state)
                q1, q2 = self.critic_1_target(state, action)
                q = torch.min(q1, q2)
                advantage = q - v
                exp_advantage = torch.exp(advantage * self.config.beta)
                exp_advantage = torch.clamp(exp_advantage, max=100.0)

            log_prob = self.actor.log_prob(state, action)
            actor_loss = -(exp_advantage * log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            soft_update(self.critic_1_target, self.critic_1, self.config.iql_tau)
            soft_update(self.critic_2_target, self.critic_2, self.config.iql_tau)

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "v_value": v.mean().item(),
        }

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor(state, deterministic=deterministic)
        action = action.cpu().numpy().flatten()

        # Denormalize
        action = action * self.action_scale.cpu().numpy() + self.action_center.cpu().numpy()
        return action

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'value': self.value.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'total_it': self.total_it,
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.value.load_state_dict(checkpoint['value'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")


def train_iql(config: IQLConfig):
    """Train IQL"""
    timestamp = datetime.now().strftime("%Y%m%d")
    config = auto_generate_paths(config, timestamp)
    config = auto_generate_swanlab_config(config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    log_filename = f"{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_seed{config.seed}_{timestamp}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    set_seed(config.seed)
    logging.info(f"Global seed set to {config.seed}")
def train_iql(config: IQLConfig):
    """Train IQL"""
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d")

    # Auto-generate paths
    config = auto_generate_paths(config, timestamp)
    config = auto_generate_swanlab_config(config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Configure logging
    log_filename = f"{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_seed{config.seed}_{timestamp}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    # Set seed
    set_seed(config.seed)
    logging.info(f"Global seed set to {config.seed}")

    # Print configuration
    logging.info(f"{'='*80}")
    logging.info(f"=== IQL Training Configuration ===")
    logging.info(f"{'='*80}")
    logging.info(f"Environment: {config.env_name}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Seed: {config.seed}")
    logging.info(f"Tau (Expectile): {config.tau}, Beta (AWR): {config.beta}")
    logging.info(f"Discount (gamma): {config.gamma}")
    logging.info(f"Normalize rewards: {config.normalize_rewards}")
    logging.info(f"Max timesteps: {config.max_timesteps}")
    logging.info(f"Batch size: {config.batch_size}")
    logging.info(f"Actor LR: {config.actor_lr}")
    logging.info(f"Critic LR: {config.critic_lr}")
    logging.info(f"Log file: {log_filepath}")
    logging.info(f"Checkpoint dir: {config.checkpoint_dir}")
    logging.info(f"{'='*80}")

    # Load dataset
    logging.info(f"\nLoading GeMS dataset from: {config.dataset_path}")
    dataset = np.load(config.dataset_path)
    
    logging.info(f"Dataset statistics:")
    logging.info(f"  Observations shape: {dataset['observations'].shape}")
    logging.info(f"  Actions shape: {dataset['actions'].shape}")
    logging.info(f"  Total transitions: {len(dataset['observations'])}")
    
    # Get dimensions
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    max_action = 1.0  # Normalized
    
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
    
    # Load data
    dataset_dict = {
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'next_observations': dataset['next_observations'],
        'rewards': dataset['rewards'],
        'terminals': dataset['terminals'],
    }
    replay_buffer.load_d4rl_dataset(dataset_dict)

    # Normalize states
    if config.normalize_states:
        state_mean, state_std = compute_mean_std(dataset['observations'])
        replay_buffer.normalize_states(state_mean, state_std)
        logging.info(f"States normalized")
    
    # Normalize rewards
    if config.normalize_rewards:
        reward_mean = dataset['rewards'].mean()
        reward_std = dataset['rewards'].std()
        replay_buffer.normalize_rewards(reward_mean, reward_std)
        logging.info(f"Rewards normalized (mean={reward_mean:.4f}, std={reward_std:.4f})")
    
    # Normalize actions (必须)
    if not config.normalize_actions:
        raise ValueError("normalize_actions must be True for offline RL!")
    action_center, action_scale = replay_buffer.normalize_actions()
    logging.info(f"Actions normalized to [-1, 1]")
    
    # Initialize IQL
    agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        config=config,
        action_center=action_center,
        action_scale=action_scale,
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
    logging.info(f"Starting IQL training")
    logging.info(f"{'='*80}\n")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)
        
        # Train
        metrics = agent.train(batch)
        
        # Logging
        if (t + 1) % config.log_freq == 0:
            log_msg = (f"Step {t+1}/{config.max_timesteps}: "
                      f"critic_loss={metrics['critic_loss']:.4f}, "
                      f"value_loss={metrics['value_loss']:.4f}, "
                      f"actor_loss={metrics['actor_loss']:.4f}, "
                      f"v_value={metrics['v_value']:.4f}")
            logging.info(log_msg)

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
        f"iql_{config.env_name}_{config.dataset_quality}_tau{config.tau}_beta{config.beta}_final.pt"
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

    logging.info(f"\n{'='*80}")
    logging.info(f"Training completed!")
    logging.info(f"{'='*80}\n")

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IQL (Implicit Q-Learning) on offline datasets")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment")
    parser.add_argument("--env_name", type=str, default="diffuse_mix")
    parser.add_argument("--dataset_quality", type=str, default="expert",
                        choices=["random", "medium", "expert"])
    parser.add_argument("--seed", type=int, default=58407201)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_path", type=str, default="")

    # 训练配置
    parser.add_argument("--max_timesteps", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq", type=int, default=int(5e3))
    parser.add_argument("--save_freq", type=int, default=int(5e4))
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)

    # IQL特定参数
    parser.add_argument("--tau", type=float, default=0.7,
                        help="Expectile for value loss")
    parser.add_argument("--beta", type=float, default=3.0,
                        help="Inverse temperature for AWR")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--iql_tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)

    # SwanLab配置
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab")

    args = parser.parse_args()

    config = IQLConfig(
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
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        tau=args.tau,
        beta=args.beta,
        gamma=args.gamma,
        iql_tau=args.iql_tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        use_swanlab=args.use_swanlab,
    )

    train_iql(config)
