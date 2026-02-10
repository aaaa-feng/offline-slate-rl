#!/usr/bin/env python3
"""
离线RL算法配置模块
统一管理所有离线算法的参数配置
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os


@dataclass
class BaseOfflineConfig:
    """离线RL基础配置 (所有算法通用)"""

    # ============================================================================
    # 实验配置
    # ============================================================================
    experiment_name: str = "baseline_experiment"  # 实验名称

    # 环境名称 - 必须与数据集的环境参数匹配！
    # 可选值:
    #   - diffuse_mix_bt3_dp5: boredom=3, penalty=5.0 (对应 mix_divpen_v2_data_d4rl.npz)
    #   - diffuse_mix_bt5_dp5: boredom=5, penalty=5.0 (对应 mix_divpen_v2_b5_data_d4rl.npz)
    #   - diffuse_topdown_bt3_dp5: boredom=3, penalty=5.0 (对应 topdown_divpen_v2_data_d4rl.npz)
    #   - diffuse_topdown_bt5_dp5: boredom=5, penalty=5.0 (对应 topdown_divpen_v2_b5_data_d4rl.npz)
    env_name: str = "diffuse_mix_bt5_dp5"  # 默认使用 b5 环境

    dataset_quality: str = "expert"  # 数据集质量: random, medium, expert
    seed: int = 58407201  # 随机种子
    device: str = "cuda"  # 设备

    # ============================================================================
    # 数据集配置
    # ============================================================================
    dataset_path: str = ""  # 数据集路径 (如果为空则自动生成)
    normalize_states: bool = True  # 是否归一化状态
    normalize_actions: bool = True  # 是否归一化动作 (必须为True)
    normalize_rewards: bool = True  # 是否归一化奖励 (TD3+BC等算法需要)

    # ============================================================================
    # 训练配置
    # ============================================================================
    max_timesteps: int = int(1e6)  # 最大训练步数
    batch_size: int = 256  # 批次大小
    eval_freq: int = int(5e3)  # 评估频率 (训练步数)
    save_freq: int = int(5e4)  # 保存频率 (训练步数)
    log_freq: int = 1000  # 日志记录频率

    # ============================================================================
    # 网络配置
    # ============================================================================
    hidden_dim: int = 256  # 隐藏层维度
    learning_rate: float = 3e-4  # 学习率

    # ============================================================================
    # GRU & Embedding 配置 (用于端到端训练)
    # ============================================================================
    num_items: int = 1000  # 物品总数
    rec_size: int = 10  # 推荐列表大小
    item_embedd_dim: int = 20  # 物品嵌入维度
    belief_hidden_dim: int = 20  # GRU隐藏状态维度
    latent_dim: int = 32  # GeMS 潜在空间维度
    item_embedds_path: str = ""  # 物品嵌入路径 (如果为空则自动生成)

    # ============================================================================
    # 路径配置 (自动生成)
    # ============================================================================
    log_dir: str = ""  # 日志目录
    checkpoint_dir: str = ""  # 模型保存目录

    # ============================================================================
    # SwanLab配置
    # ============================================================================
    use_swanlab: bool = True  # 是否使用SwanLab
    swan_project: str = "Offline_Slate_RL_202601"  # SwanLab项目名
    swan_workspace: str = "Cliff"  # SwanLab工作空间
    swan_mode: str = "cloud"  # SwanLab模式: cloud, local, offline
    swan_logdir: str = "experiments/swanlog"  # SwanLab本地日志目录
    swan_tags: List[str] = field(default_factory=list)  # SwanLab标签
    swan_description: str = ""  # SwanLab描述
    run_name: str = ""  # 运行名称 (如果为空则自动生成)
    run_id: str = ""  # 唯一运行标识符 (格式: MMDD_HHMM, 如果为空则自动生成)


@dataclass
class BCConfig(BaseOfflineConfig):
    """Behavior Cloning 配置"""

    # 算法名称
    algo_name: str = "BC"

    # BC特定参数 (无额外参数)
    # BC只需要基础配置即可


@dataclass
class TD3BCConfig(BaseOfflineConfig):
    """TD3+BC 配置"""

    # 算法名称
    algo_name: str = "TD3_BC"

    # TD3+BC特定参数
    alpha: float = 2.5  # BC正则化系数
    policy_noise: float = 0.2  # 策略噪声
    noise_clip: float = 0.5  # 噪声裁剪
    policy_freq: int = 2  # 策略更新频率
    tau: float = 0.005  # 软更新系数
    gamma: float = 0.99  # 折扣因子

    # 网络配置
    actor_lr: float = 3e-4  # Actor学习率
    critic_lr: float = 3e-4  # Critic学习率

    # GRU架构配置
    use_shared_gru: bool = True  # 使用共享GRU（推荐）

    # GeMS配置（2026-01-30新增）
    gems_embedding_mode: str = "mf_fixed"  # GeMS embedding模式: 'default', 'mf_fixed', 'mf_scratch', 'epsilon-greedy'


@dataclass
class CQLConfig(BaseOfflineConfig):
    """Conservative Q-Learning 配置"""

    # 算法名称
    algo_name: str = "CQL"

    # CQL特定参数
    alpha: float = 1.0  # CQL正则化系数
    tau: float = 0.005  # 软更新系数
    gamma: float = 0.99  # 折扣因子

    # CQL特定超参数
    cql_n_actions: int = 10  # CQL采样动作数
    cql_importance_sample: bool = True  # 是否使用重要性采样
    cql_lagrange: bool = False  # 是否使用Lagrange约束
    cql_target_action_gap: float = -1.0  # 目标动作gap
    cql_temp: float = 1.0  # CQL温度参数
    cql_min_q_weight: float = 5.0  # CQL最小Q权重

    # 网络配置
    actor_lr: float = 3e-4  # Actor学习率
    critic_lr: float = 3e-4  # Critic学习率
    n_hidden: int = 2  # 隐藏层数量


@dataclass
class IQLConfig(BaseOfflineConfig):
    """Implicit Q-Learning 配置"""

    # 算法名称
    algo_name: str = "IQL"

    # IQL特定参数
    tau: float = 0.7  # 期望分位数
    beta: float = 3.0  # 优势加权系数
    gamma: float = 0.99  # 折扣因子

    # IQL特定超参数
    iql_tau: float = 0.005  # 软更新系数

    # 网络配置
    actor_lr: float = 3e-4  # Actor学习率
    critic_lr: float = 3e-4  # Critic学习率
    value_lr: float = 3e-4  # Value学习率
    n_hidden: int = 2  # 隐藏层数量


def get_config_by_algo(algo_name: str) -> BaseOfflineConfig:
    """根据算法名称获取配置类"""
    config_map = {
        "BC": BCConfig,
        "TD3_BC": TD3BCConfig,
        "CQL": CQLConfig,
        "IQL": IQLConfig,
    }

    if algo_name not in config_map:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(config_map.keys())}")

    return config_map[algo_name]()


def _get_key_params_str(config: BaseOfflineConfig) -> str:
    """获取关键参数字符串 (用于命名)"""
    if config.algo_name == "BC":
        return ""
    elif config.algo_name == "TD3_BC":
        return f"alpha{config.alpha}"
    elif config.algo_name == "CQL":
        return f"alpha{config.alpha}"
    elif config.algo_name == "IQL":
        return f"tau{config.tau}_beta{config.beta}"
    else:
        return ""


def auto_generate_paths(config: BaseOfflineConfig, timestamp: str) -> BaseOfflineConfig:
    """
    自动生成路径配置

    Args:
        config: 配置对象
        timestamp: 时间戳 (格式: YYYYMMDD, 仅用于日志文件名兼容性)

    Returns:
        更新后的配置对象
    """
    from datetime import datetime
    from config import paths
    from common.offline.checkpoint_utils import resolve_dataset_path

    # 0. 确保 run_id 存在
    if not config.run_id:
        config.run_id = datetime.now().strftime("%m%d_%H%M")

    # 1. 生成日志目录
    if not config.log_dir:
        config.log_dir = str(
            paths.LOGS_DIR / "offline" / f"log_{config.seed}" / config.algo_name
        )

    # 2. 生成checkpoint目录
    if not config.checkpoint_dir:
        config.checkpoint_dir = str(
            paths.get_offline_ckpt_dir(config.env_name, config.algo_name.lower())
        )

    # 3. 生成数据集路径
    if not config.dataset_path:
        # 使用统一函数解析数据集路径
        config.dataset_path = resolve_dataset_path(
            env_name=config.env_name,
            dataset_quality=config.dataset_quality
        )

    # 4. 生成物品嵌入路径 (根据环境名称自动推断)
    if not config.item_embedds_path:
        # 根据 env_name 推断 embedding 文件名
        # diffuse_* → item_embeddings_diffuse.pt
        # focused_* → item_embeddings_focused.pt
        if "diffuse" in config.env_name:
            embeddings_filename = "item_embeddings_diffuse.pt"
        elif "focused" in config.env_name:
            embeddings_filename = "item_embeddings_focused.pt"
        else:
            # 默认使用 diffuse
            embeddings_filename = "item_embeddings_diffuse.pt"

        config.item_embedds_path = str(paths.get_embeddings_path(embeddings_filename))

    return config


def auto_generate_swanlab_config(config: BaseOfflineConfig) -> BaseOfflineConfig:
    """
    自动生成 SwanLab 配置

    Args:
        config: 配置对象

    Returns:
        更新后的配置对象
    """
    from datetime import datetime

    # 0. 生成或使用 run_id
    if not config.run_id:
        config.run_id = datetime.now().strftime("%m%d_%H%M")

    # 1. 生成 run_name (添加 run_id 后缀)
    if not config.run_name:
        key_params = _get_key_params_str(config)
        if key_params:
            config.run_name = f"{config.algo_name}_{config.env_name}_{config.dataset_quality}_{key_params}_seed{config.seed}_{config.run_id}"
        else:
            config.run_name = f"{config.algo_name}_{config.env_name}_{config.dataset_quality}_seed{config.seed}_{config.run_id}"

    # 2. 生成 SwanLab tags
    if not config.swan_tags:
        config.swan_tags = [
            config.algo_name,
            config.env_name,
            config.dataset_quality,
            f"seed_{config.seed}",
        ]

    # 3. 生成 SwanLab description
    if not config.swan_description:
        config.swan_description = (
            f"{config.algo_name} Offline RL Training - "
            f"{config.env_name} - {config.dataset_quality} dataset - seed {config.seed}"
        )

    return config
