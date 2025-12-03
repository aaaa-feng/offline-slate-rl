"""
BaseBeliefEncoder - Belief Encoder基类

Belief Encoder的作用：将原始observation编码为belief_state
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import torch


class BaseBeliefEncoder(ABC):
    """
    Belief Encoder基类，定义统一接口

    Belief Encoder负责将环境的原始observation编码为固定维度的belief_state
    """

    def __init__(
        self,
        state_dim: int,
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化Belief Encoder

        Args:
            state_dim: 输出的belief_state维度（默认20）
            device: 计算设备
        """
        self.state_dim = state_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode(self, obs: Any) -> np.ndarray:
        """
        编码observation为belief state

        Args:
            obs: 原始observation（格式取决于环境）

        Returns:
            belief_state: shape (state_dim,)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        重置内部状态

        对于RNN类编码器，需要重置hidden state
        """
        raise NotImplementedError

    def save(self, path: str):
        """保存模型"""
        pass

    def load(self, path: str):
        """加载模型"""
        pass

    def eval_mode(self):
        """切换到评估模式"""
        pass

    def train_mode(self):
        """切换到训练模式"""
        pass

    def get_config(self) -> Dict:
        """返回Encoder配置"""
        return {
            'state_dim': self.state_dim,
            'device': str(self.device),
        }
