"""
Neural network architectures for offline RL algorithms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Deterministic actor for TD3+BC"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """Twin Q-network for TD3+BC"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TanhGaussianActor(nn.Module):
    """Stochastic actor for CQL and IQL"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super(TanhGaussianActor, self).__init__()
        self.max_action = max_action
        self.action_dim = action_dim

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mu) * self.max_action
            log_prob = None
        else:
            dist = Normal(mu, std)
            z = dist.rsample()
            action = torch.tanh(z) * self.max_action

            if need_log_prob:
                log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
                # Enforcing action bounds
                log_prob -= torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6).sum(
                    dim=-1, keepdim=True
                )
            else:
                log_prob = None

        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given state"""
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        # Inverse tanh
        z = torch.atanh(torch.clamp(action / self.max_action, -0.999, 0.999))
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.max_action * (1 - (action / self.max_action).pow(2)) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        return log_prob


class ValueFunction(nn.Module):
    """Value function for IQL"""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super(ValueFunction, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class TwinQ(nn.Module):
    """Twin Q-network for CQL and IQL"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super(TwinQ, self).__init__()
        dims = [state_dim + action_dim, hidden_dim]
        dims += [hidden_dim] * (n_hidden - 1)

        self.q1 = self._build_network(dims)
        self.q2 = self._build_network(dims)

    def _build_network(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        return nn.Sequential(*layers)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))
