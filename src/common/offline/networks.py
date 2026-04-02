"""
Neural network architectures for offline RL algorithms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MIN = -5.0
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
                # FIX Phase 2: mean instead of sum — dimension-agnostic
                # sum scales with action_dim (32D→~160, 200D→~1000), causing gradient explosion
                log_prob = dist.log_prob(z).mean(dim=-1, keepdim=True)
                # Enforcing action bounds
                log_prob -= torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6).mean(
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
        # FIX: use safe_action for both atanh and Jacobian to avoid log(negative)
        safe_action = torch.clamp(action / self.max_action, -0.999999, 0.999999)
        z = torch.atanh(safe_action)
        log_prob = dist.log_prob(z).mean(dim=-1, keepdim=True)
        # FIX: Jacobian uses same safe_action, guarantees (1 - x^2) > 0
        jacobian_term = 1 - safe_action.pow(2)
        log_prob -= torch.log(self.max_action * jacobian_term + 1e-6).mean(
            dim=-1, keepdim=True
        )
        # FIX: relax clamp from -20 to -100 so gradient is non-zero even when
        # Actor prediction is far from true_actions
        # FIX Phase 1.5: remove max=0.0 — continuous Gaussian log_prob can be
        # positive (density > 1), clamping to 0 silently kills AWR loss
        log_prob = torch.clamp(log_prob, min=-100.0)
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


# ============================================================================
# 🔥 Actor Architecture Ablation: Variance Collapse Experiment
# ============================================================================

class DeterministicActor(nn.Module):
    """
    Deterministic actor (no stochasticity, no log_std)
    Only outputs mu (mean action), which is then passed through tanh
    
    Use case: Test if removing variance learning prevents collapse
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super(DeterministicActor, self).__init__()
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

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: tanh(mu) * max_action
            log_prob: None (deterministic policy has no log_prob)
        """
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        action = torch.tanh(mu) * self.max_action
        return action, None

    def get_mu(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw mu (before tanh) for loss computation"""
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        return mu


class FixedGaussianActor(nn.Module):
    """
    Gaussian actor with FIXED (non-learnable) log_std
    
    Architecture:
    - Network outputs mu (learnable)
    - log_std is a fixed constant tensor (NOT nn.Parameter)
    - Can still compute log_prob for AWR loss
    
    Use case: Test if fixing variance prevents collapse while keeping stochasticity
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        fixed_std: float = 0.1,  # Fixed standard deviation
    ):
        super(FixedGaussianActor, self).__init__()
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
        
        # 🔥 KEY: log_std is a FIXED buffer (not Parameter), won't be updated by optimizer
        fixed_log_std = torch.log(torch.tensor(fixed_std))
        self.register_buffer('log_std', fixed_log_std.repeat(action_dim))

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        
        # Expand log_std to match batch size
        log_std = self.log_std.unsqueeze(0).expand(mu.shape[0], -1)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mu) * self.max_action
            log_prob = None
        else:
            dist = Normal(mu, std)
            z = dist.rsample()
            action = torch.tanh(z) * self.max_action

            if need_log_prob:
                log_prob = dist.log_prob(z).mean(dim=-1, keepdim=True)
                log_prob -= torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6).mean(
                    dim=-1, keepdim=True
                )
            else:
                log_prob = None

        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given state"""
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        
        # Expand log_std to match batch size
        log_std = self.log_std.unsqueeze(0).expand(mu.shape[0], -1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        safe_action = torch.clamp(action / self.max_action, -0.999999, 0.999999)
        z = torch.atanh(safe_action)
        log_prob = dist.log_prob(z).mean(dim=-1, keepdim=True)
        jacobian_term = 1 - safe_action.pow(2)
        log_prob -= torch.log(self.max_action * jacobian_term + 1e-6).mean(
            dim=-1, keepdim=True
        )
        log_prob = torch.clamp(log_prob, min=-100.0)
        return log_prob


# Alias for backward compatibility
Critic = TwinQ
