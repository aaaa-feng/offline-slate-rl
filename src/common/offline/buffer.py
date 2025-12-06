"""
Replay Buffer for offline RL
不依赖d4rl，直接加载GeMS数据集
"""
import torch
import numpy as np
from typing import Dict, Tuple, List

class ReplayBuffer:
    """Replay buffer for offline RL training"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """
        加载D4RL格式的数据集（兼容CORL接口）

        Args:
            data: 包含observations, actions, rewards, next_observations, terminals的字典
        """
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                f"Replay buffer is smaller than the dataset you are trying to load! "
                f"Buffer size: {self._buffer_size}, Dataset size: {n_transitions}"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        采样一个batch的数据

        Returns:
            [states, actions, rewards, next_states, dones]
        """
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def normalize_states(self, mean: np.ndarray, std: np.ndarray):
        """
        对状态进行归一化

        Args:
            mean: 状态均值
            std: 状态标准差
        """
        mean = self._to_tensor(mean)
        std = self._to_tensor(std)
        self._states = (self._states - mean) / std
        self._next_states = (self._next_states - mean) / std
        print(f"States normalized with mean shape: {mean.shape}, std shape: {std.shape}")

    def normalize_rewards(self, mean: float = None, std: float = None):
        """
        对奖励进行归一化

        Args:
            mean: 奖励均值（如果为None，则自动计算）
            std: 奖励标准差（如果为None，则自动计算）
        """
        rewards = self._rewards[:self._size]
        if mean is None:
            mean = rewards.mean().item()
        if std is None:
            std = rewards.std().item()
            std = max(std, 1e-6)  # 防止除零

        self._rewards = (self._rewards - mean) / std
        print(f"Rewards normalized: mean={mean:.4f}, std={std:.4f}")
        return mean, std

    def scale_rewards(self, scale: float = 1.0):
        """
        缩放奖励

        Args:
            scale: 缩放因子
        """
        self._rewards = self._rewards * scale
        print(f"Rewards scaled by {scale}")
