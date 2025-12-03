"""
BaseRanker - Ranker基类

Ranker的作用：将latent_action解码为slate（推荐列表）
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np
import torch


class BaseRanker(ABC):
    """
    Ranker基类，定义统一接口

    Ranker负责将Agent输出的latent_action解码为具体的推荐slate
    """

    def __init__(
        self,
        action_dim: int,
        num_items: int,
        slate_size: int,
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化Ranker

        Args:
            action_dim: latent_action维度（默认32）
            num_items: 物品总数（默认1000）
            slate_size: 推荐列表大小（默认10）
            device: 计算设备
        """
        self.action_dim = action_dim
        self.num_items = num_items
        self.slate_size = slate_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def rank(
        self,
        latent_action: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将latent action解码为slate

        Args:
            latent_action: shape (action_dim,) 或 (batch_size, action_dim)

        Returns:
            slate: shape (slate_size,) 或 (batch_size, slate_size)
                   包含物品ID的推荐列表
        """
        raise NotImplementedError

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        训练ranker一步（如果需要训练）

        Args:
            batch: 包含以下键的字典
                - 'latent_actions': (batch_size, action_dim)
                - 'slates': (batch_size, slate_size)
                - 'clicks': (batch_size, slate_size) [可选]

        Returns:
            log_dict: 训练日志，如 {'loss': ...}，如果不需要训练则返回None
        """
        return None  # 默认不需要训练

    def save(self, path: str):
        """保存模型（如果需要）"""
        pass

    def load(self, path: str):
        """加载模型（如果需要）"""
        pass

    def eval_mode(self):
        """切换到评估模式"""
        pass

    def train_mode(self):
        """切换到训练模式"""
        pass

    def get_config(self) -> Dict:
        """返回Ranker配置"""
        return {
            'action_dim': self.action_dim,
            'num_items': self.num_items,
            'slate_size': self.slate_size,
            'device': str(self.device),
        }
