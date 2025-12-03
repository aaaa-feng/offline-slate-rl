"""
BaseAgent - Agent基类

所有Agent都在潜空间中工作：
- 输入: belief_state (state_dim维)
- 输出: latent_action (action_dim维)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Agent基类，定义统一接口

    所有Agent（无论离线还是在线）都必须实现这些方法
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化Agent

        Args:
            state_dim: 状态维度 (belief_state维度，默认20)
            action_dim: 动作维度 (latent_action维度，默认32)
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.total_it = 0  # 训练步数计数器

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        选择动作

        Args:
            state: belief_state, shape (state_dim,) 或 (batch_size, state_dim)
            deterministic: 是否使用确定性策略（评估时为True）

        Returns:
            action: latent_action, shape (action_dim,) 或 (batch_size, action_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练一步

        Args:
            batch: 包含以下键的字典
                - 'states': (batch_size, state_dim)
                - 'actions': (batch_size, action_dim)
                - 'rewards': (batch_size,)
                - 'next_states': (batch_size, state_dim)
                - 'dones': (batch_size,)

        Returns:
            log_dict: 训练日志，如 {'critic_loss': ..., 'actor_loss': ..., 'q_value': ...}
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径（不含扩展名）
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径（不含扩展名）
        """
        raise NotImplementedError

    def eval_mode(self):
        """切换到评估模式"""
        pass

    def train_mode(self):
        """切换到训练模式"""
        pass

    def get_config(self) -> Dict:
        """返回Agent配置"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
        }
