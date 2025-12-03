"""
TD3+BC for GeMS datasets
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2106.06860.pdf
"""
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入路径配置
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))
from config import paths

from offline_rl_baselines.common.buffer import ReplayBuffer
from offline_rl_baselines.common.utils import set_seed, compute_mean_std, soft_update
from offline_rl_baselines.common.networks import Actor, Critic

TensorBatch = List[torch.Tensor]


@torch.no_grad()
def eval_actor(
    env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    state_mean: np.ndarray = 0.0,
    state_std: np.ndarray = 1.0,
) -> Tuple[float, float]:
    """
    评估actor在环境中的性能

    Args:
        env: 环境
        actor: Actor网络
        device: 设备
        n_episodes: 评估的episode数量
        seed: 随机种子
        state_mean: 状态均值（用于归一化）
        state_std: 状态标准差（用于归一化）

    Returns:
        平均回报, 标准差
    """
    env.seed(seed)
    actor.eval()
    episode_rewards = []

    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0

        while not done:
            # 归一化状态
            state_normalized = (state - state_mean) / state_std

            # 选择动作
            state_tensor = torch.FloatTensor(state_normalized.reshape(1, -1)).to(device)
            action = actor(state_tensor).cpu().data.numpy().flatten()

            # 执行动作
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    actor.train()
    return np.mean(episode_rewards), np.std(episode_rewards)


@dataclass
class TD3BCConfig:
    """TD3+BC configuration"""
    # Experiment
    device: str = "cuda"
    env_name: str = "diffuse_topdown"
    dataset_path: str = ""
    seed: int = 0

    # Training
    max_timesteps: int = int(1e6)
    batch_size: int = 256
    eval_freq: int = int(5e3)
    n_eval_episodes: int = 10

    # TD3+BC parameters
    buffer_size: int = 2_000_000
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5  # BC weight

    # Network
    hidden_dim: int = 256
    learning_rate: float = 3e-4

    # Normalization
    normalize: bool = True
    normalize_reward: bool = False

    # Logging
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"
    save_freq: int = int(1e5)

    # Wandb (optional)
    use_wandb: bool = False
    wandb_project: str = "GeMS-Offline-RL"
    wandb_group: str = "TD3_BC"
    wandb_name: str = "TD3_BC"


class TD3_BC:
    """TD3+BC algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: TD3BCConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.max_action = max_action

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
        """Train one step"""
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
            target_q1, target_q2 = self.critic_1_target(next_state, next_action), \
                                   self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic_1(state, action), self.critic_2(state, action)

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
        if self.total_it % self.config.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.config.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            soft_update(self.actor_target, self.actor, self.config.tau)
            soft_update(self.critic_1_target, self.critic_1, self.config.tau)
            soft_update(self.critic_2_target, self.critic_2, self.config.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
            "q_value": current_q1.mean().item(),
        }

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'total_it': self.total_it,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.total_it = checkpoint['total_it']
        print(f"Model loaded from {filepath}")


def train_td3_bc(config: TD3BCConfig):
    """
    Train TD3+BC on GeMS dataset

    Args:
        config: Training configuration
    """
    # Set seed
    set_seed(config.seed)

    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize wandb if needed
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                name=f"{config.wandb_name}-{config.env_name}-seed{config.seed}",
                config=config.__dict__,
            )
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            config.use_wandb = False

    # Load dataset
    print(f"\n{'='*50}")
    print(f"Loading GeMS dataset from: {config.dataset_path}")
    print(f"{'='*50}\n")

    dataset = np.load(config.dataset_path)

    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"  Observations shape: {dataset['observations'].shape}")
    print(f"  Actions shape: {dataset['actions'].shape}")
    print(f"  Total transitions: {len(dataset['observations'])}")
    print(f"  Num episodes: {dataset['terminals'].sum()}")
    print(f"  Avg reward: {dataset['rewards'].mean():.4f}")
    print(f"  Reward std: {dataset['rewards'].std():.4f}\n")

    # Get dimensions
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    max_action = 3.0  # GeMS action bounds

    print(f"Environment info:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Max action: {max_action}\n")

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
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
    if config.normalize:
        mean, std = compute_mean_std(dataset['observations'])
        replay_buffer.normalize_states(mean, std)
        print(f"States normalized\n")

    # Initialize TD3+BC
    agent = TD3_BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        config=config,
    )

    # Training loop
    print(f"{'='*50}")
    print(f"Starting TD3+BC training")
    print(f"{'='*50}\n")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        train_metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            print(f"Step {t+1}/{config.max_timesteps}: "
                  f"Critic Loss: {train_metrics['critic_loss']:.4f}, "
                  f"Actor Loss: {train_metrics['actor_loss']:.4f}, "
                  f"Q Value: {train_metrics['q_value']:.4f}")

            if config.use_wandb:
                wandb.log({
                    "train/critic_loss": train_metrics['critic_loss'],
                    "train/actor_loss": train_metrics['actor_loss'],
                    "train/q_value": train_metrics['q_value'],
                    "train/step": t + 1,
                })

        # Save checkpoint
        if (t + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"td3_bc_{config.env_name}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(
        config.checkpoint_dir,
        f"td3_bc_{config.env_name}_final.pt"
    )
    agent.save(final_path)

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"{'='*50}\n")

    if config.use_wandb:
        wandb.finish()

    return agent


if __name__ == "__main__":
    # Example usage
    dataset_path = paths.get_offline_dataset_dir("diffuse_topdown") / "expert_data_d4rl.npz"
    config = TD3BCConfig(
        env_name="diffuse_topdown",
        dataset_path=str(dataset_path),
        seed=0,
        max_timesteps=int(1e6),
        use_wandb=False,
    )

    train_td3_bc(config)
