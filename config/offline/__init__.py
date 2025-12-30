"""
离线RL训练配置

此目录包含离线RL训练的配置模块。

模块:
- algorithms.py: 离线算法配置（BC, TD3+BC, CQL, IQL）
- env_loader.py: 环境参数加载器（从dataset_meta.json读取）

使用方式:
    from config.offline.algorithms import BCConfig, TD3BCConfig
    from config.offline.env_loader import get_env_config
"""

# 从子模块导出
from .algorithms import (
    BaseOfflineConfig,
    BCConfig,
    TD3BCConfig,
    CQLConfig,
    IQLConfig,
    get_config_by_algo,
    auto_generate_paths,
    auto_generate_swanlab_config,
)

from .env_loader import (
    get_env_config,
    list_available_envs,
    ENV_CONFIGS_FALLBACK,
)

__all__ = [
    # 算法配置
    'BaseOfflineConfig',
    'BCConfig',
    'TD3BCConfig',
    'CQLConfig',
    'IQLConfig',
    'get_config_by_algo',
    'auto_generate_paths',
    'auto_generate_swanlab_config',

    # 环境参数
    'get_env_config',
    'list_available_envs',
    'ENV_CONFIGS_FALLBACK',
]
