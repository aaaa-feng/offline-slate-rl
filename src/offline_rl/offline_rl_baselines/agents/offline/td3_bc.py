"""
TD3+BC Agent - 适配GeMS离线RL框架

基于论文: "A Minimalist Approach to Offline Reinforcement Learning"
https://arxiv.org/pdf/2106.06860.pdf

核心思想: TD3 + Behavior Cloning约束
Loss = -λ * Q(s, π(s)) + MSE(π(s), a)
其中 λ = α / |Q(s, a)|.mean()
"""

import copy
from typing import Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_rl_baselines.agents.base_agent import BaseAgent
from offline_rl_baselines.common.utils import soft_update


# 单独的Q网络（不是Twin Q）
class SingleCritic(nn.Module):
    """Single Q-network for TD3"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SingleCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


# 确定性Actor
class DeterministicActor(nn.Module):
    """Deterministic actor for TD3+BC"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(DeterministicActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


@dataclass
class TD3BCConfig:
    """TD3+BC配置"""
    # 网络结构
    hidden_dim: int = 256
    learning_rate: float = 3e-4

    # TD3参数
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # BC参数
    alpha: float = 2.5  # BC权重，越大越接近行为策略

    # 动作范围
    max_action: float = 3.0

    # 稳定性参数（防止Q值爆炸）
    max_q_backup: float = 100.0  # Q值裁剪上限
    grad_clip: float = 1.0  # 梯度裁剪


class TD3BCAgent(BaseAgent):
    """
    TD3+BC Agent

    在潜空间中学习策略:
    - 输入: belief_state (state_dim维)
    - 输出: latent_action (action_dim维)
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 32,
        device: str = "cuda",
        config: Optional[TD3BCConfig] = None,
        **kwargs
    ):
        """
        初始化TD3+BC Agent

        Args:
            state_dim: 状态维度 (belief_state, 默认20)
            action_dim: 动作维度 (latent_action, 默认32)
            device: 计算设备
            config: TD3BC配置
        """
        super().__init__(state_dim, action_dim, device)

        self.config = config if config is not None else TD3BCConfig()
        self.max_action = self.config.max_action

        # Actor网络
        self.actor = DeterministicActor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=self.max_action,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )

        # Critic网络 (两个独立的Q网络)
        self.critic_1 = SingleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=self.config.learning_rate
        )

        self.critic_2 = SingleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=self.config.learning_rate
        )

        # 冻结target网络
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_1_target.parameters():
            param.requires_grad = False
        for param in self.critic_2_target.parameters():
            param.requires_grad = False

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        选择动作

        Args:
            state: belief_state, shape (state_dim,) 或 (batch_size, state_dim)
            deterministic: TD3+BC总是确定性的

        Returns:
            action: latent_action, shape (action_dim,) 或 (batch_size, action_dim)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action = self.actor(state)
            return action.cpu().numpy().squeeze()

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练一步

        Args:
            batch: 包含 states, actions, rewards, next_states, dones

        Returns:
            log_dict: 训练日志
        """
        self.total_it += 1

        # 解包batch
        state = batch['states']
        action = batch['actions']
        reward = batch['rewards'].unsqueeze(-1) if batch['rewards'].dim() == 1 else batch['rewards']
        next_state = batch['next_states']
        done = batch['dones'].unsqueeze(-1) if batch['dones'].dim() == 1 else batch['dones']

        # ========== Critic更新 ==========
        with torch.no_grad():
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # 目标Q值 (取两个Critic的最小值)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Q值裁剪（防止爆炸）
            target_q = torch.clamp(target_q, -self.config.max_q_backup, self.config.max_q_backup)

            target_q = reward + (1 - done) * self.config.discount * target_q

        # 当前Q值
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.config.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.config.grad_clip)

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ========== Actor更新 (延迟) ==========
        actor_loss_value = 0.0
        bc_loss_value = 0.0

        if self.total_it % self.config.policy_freq == 0:
            # Actor输出
            pi = self.actor(state)

            # Q值
            q = self.critic_1(state, pi)

            # TD3+BC的核心: λ = α / |Q|.mean()
            lmbda = self.config.alpha / q.abs().mean().detach()

            # Actor损失 = -λ * Q + BC损失
            bc_loss = F.mse_loss(pi, action)
            actor_loss = -lmbda * q.mean() + bc_loss

            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)

            self.actor_optimizer.step()

            # 软更新target网络
            soft_update(self.actor_target, self.actor, self.config.tau)
            soft_update(self.critic_1_target, self.critic_1, self.config.tau)
            soft_update(self.critic_2_target, self.critic_2, self.config.tau)

            actor_loss_value = actor_loss.item()
            bc_loss_value = bc_loss.item()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_value,
            'bc_loss': bc_loss_value,
            'q_value': current_q1.mean().item(),
            'q_std': current_q1.std().item(),
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'total_it': self.total_it,
            'config': self.config,
        }, f"{path}.pt")
        print(f"Model saved to {path}.pt")

    def load(self, path: str):
        """加载模型"""
        # 如果路径已经包含.pt后缀，不再添加
        if path.endswith('.pt'):
            load_path = path
        else:
            load_path = f"{path}.pt"
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.total_it = checkpoint['total_it']
        print(f"Model loaded from {path}.pt")

    def eval_mode(self):
        """切换到评估模式"""
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def train_mode(self):
        """切换到训练模式"""
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def get_config(self) -> Dict:
        """返回Agent配置"""
        base_config = super().get_config()
        base_config.update({
            'algorithm': 'TD3+BC',
            'hidden_dim': self.config.hidden_dim,
            'alpha': self.config.alpha,
            'discount': self.config.discount,
            'tau': self.config.tau,
            'policy_noise': self.config.policy_noise,
            'policy_freq': self.config.policy_freq,
        })
        return base_config
