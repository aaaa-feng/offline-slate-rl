"""
Utility functions for offline RL baselines
"""
import torch
import numpy as np
import random
from typing import Tuple


def set_seed(seed: int, env=None):
    """
    设置随机种子以确保可复现性

    Args:
        seed: 随机种子
        env: 可选的gym环境
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算状态的均值和标准差（用于归一化）

    Args:
        states: 状态数组 (N, state_dim)
        eps: 防止除零的小常数

    Returns:
        mean, std
    """
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    归一化状态

    Args:
        states: 状态数组
        mean: 均值
        std: 标准差

    Returns:
        归一化后的状态
    """
    return (states - mean) / std


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """
    软更新目标网络

    Args:
        target: 目标网络
        source: 源网络
        tau: 更新系数
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """
    IQL使用的非对称L2损失

    Args:
        u: 输入张量
        tau: 分位数参数

    Returns:
        损失值
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)
