#!/usr/bin/env python3
"""
Config模块统一导出接口

此文件提供向后兼容的导入路径，确保现有代码无需修改即可正常工作。

目录结构:
- online/: 在线RL训练配置（历史参考，YAML格式）
- offline/: 离线RL训练配置（Python dataclass）
- paths.py: 全局路径管理
- env_params.py: 环境参数加载器（旧版本，保留兼容性）
- offline_config.py: 离线算法配置（旧版本，保留兼容性）

使用方式:
    # 方式1：直接从config导入（推荐）
    from config import paths, BCConfig, get_env_config

    # 方式2：从子模块导入（新代码推荐）
    from config.offline.algorithms import BCConfig
    from config.offline.env_loader import get_env_config

    # 方式3：从旧路径导入（向后兼容）
    from config.offline_config import BCConfig
    from config.env_params import get_env_config
"""

# ============================================================================
# 路径管理（全局通用）
# ============================================================================
from .paths import (
    # 主要目录
    PROJECT_ROOT,
    CONFIG_DIR,
    DATA_ROOT,
    DATASETS_ROOT,
    EXPERIMENTS_ROOT,
    RESULTS_ROOT,
    ONLINE_RL_RESULTS_DIR,
    OFFLINE_RL_RESULTS_DIR,

    # 数据子目录
    EMBEDDINGS_DIR,
    MF_EMBEDDINGS_DIR,
    CHECKPOINTS_DIR,
    ONLINE_RL_CKPT_DIR,
    OFFLINE_RL_CKPT_DIR,
    GEMS_CKPT_DIR,

    # 数据集目录
    ONLINE_DATASETS_DIR,
    OFFLINE_DATASETS_DIR,

    # 实验和日志
    LOGS_DIR,
    SWANLOG_DIR,
    BACKUPS_DIR,
    RAW_DATA_DIR,
    RAW_OFFLINE_DATA_DIR,

    # 辅助函数
    get_online_ckpt_dir,
    get_offline_ckpt_dir,
    get_offline_dataset_dir,
    get_log_dir,
    get_embeddings_path,
    get_mf_embeddings_path,
    get_offline_dataset_path,
    get_online_rl_checkpoint_path,
    get_offline_rl_checkpoint_path,
    get_online_dataset_path,
    get_gems_checkpoint_path,
    get_online_rl_results_dir,
    ensure_all_dirs,
    print_paths_info,
)

# ============================================================================
# 离线RL算法配置
# ============================================================================
from .offline_config import (
    # 配置类
    BaseOfflineConfig,
    BCConfig,
    TD3BCConfig,
    CQLConfig,
    IQLConfig,

    # 辅助函数
    get_config_by_algo,
    auto_generate_paths,
    auto_generate_swanlab_config,
)

# ============================================================================
# 环境参数加载器
# ============================================================================
from .env_params import (
    # 主要函数
    get_env_config,
    list_available_envs,

    # 常量
    PROJECT_ROOT as ENV_PARAMS_PROJECT_ROOT,
    DATASETS_ROOT as ENV_PARAMS_DATASETS_ROOT,
    ENV_CONFIGS_FALLBACK,
)

# ============================================================================
# 导出列表
# ============================================================================
__all__ = [
    # 路径管理
    'PROJECT_ROOT',
    'CONFIG_DIR',
    'DATA_ROOT',
    'DATASETS_ROOT',
    'EXPERIMENTS_ROOT',
    'RESULTS_ROOT',
    'ONLINE_RL_RESULTS_DIR',
    'OFFLINE_RL_RESULTS_DIR',
    'EMBEDDINGS_DIR',
    'MF_EMBEDDINGS_DIR',
    'CHECKPOINTS_DIR',
    'ONLINE_RL_CKPT_DIR',
    'OFFLINE_RL_CKPT_DIR',
    'GEMS_CKPT_DIR',
    'ONLINE_DATASETS_DIR',
    'OFFLINE_DATASETS_DIR',
    'LOGS_DIR',
    'SWANLOG_DIR',
    'BACKUPS_DIR',
    'RAW_DATA_DIR',
    'RAW_OFFLINE_DATA_DIR',
    'get_online_ckpt_dir',
    'get_offline_ckpt_dir',
    'get_offline_dataset_dir',
    'get_log_dir',
    'get_embeddings_path',
    'get_mf_embeddings_path',
    'get_offline_dataset_path',
    'get_online_rl_checkpoint_path',
    'get_offline_rl_checkpoint_path',
    'get_online_dataset_path',
    'get_gems_checkpoint_path',
    'get_online_rl_results_dir',
    'ensure_all_dirs',
    'print_paths_info',

    # 离线算法配置
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
