#!/usr/bin/env python3
"""
动态路径配置模块
消除所有硬编码路径，支持环境变量覆盖
"""
from pathlib import Path
import os

# ============================================================================
# 项目根目录（自动检测）
# ============================================================================

# config/ 目录（本文件所在位置的父目录）
CONFIG_DIR = Path(__file__).resolve().parent

# 项目根目录（config/ 的父目录）
PROJECT_ROOT = CONFIG_DIR.parent

# ============================================================================
# 主要目录（可通过环境变量覆盖）
# ============================================================================

# 数据根目录
DATA_ROOT = Path(os.getenv('OFFLINE_SLATE_RL_DATA_ROOT', PROJECT_ROOT / "data"))

# 数据集根目录（在线RL数据集在data/datasets/online/下）
DATASETS_ROOT = DATA_ROOT / "datasets"

# 实验根目录
EXPERIMENTS_ROOT = Path(os.getenv('OFFLINE_SLATE_RL_EXPERIMENTS_ROOT', PROJECT_ROOT / "experiments"))

# 结果根目录
RESULTS_ROOT = PROJECT_ROOT / "results"

# 在线RL结果目录
ONLINE_RL_RESULTS_DIR = RESULTS_ROOT / "online_rl"

# 离线RL结果目录
OFFLINE_RL_RESULTS_DIR = RESULTS_ROOT / "offline_rl"

# ============================================================================
# 数据子目录
# ============================================================================

# Embeddings目录
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"

# MF Embeddings目录
MF_EMBEDDINGS_DIR = DATA_ROOT / "mf_embeddings"

# Checkpoints根目录
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# 在线RL checkpoints
ONLINE_RL_CKPT_DIR = CHECKPOINTS_DIR / "online_rl"

# 离线RL checkpoints
OFFLINE_RL_CKPT_DIR = CHECKPOINTS_DIR / "offline_rl"

# GeMS checkpoints
GEMS_CKPT_DIR = CHECKPOINTS_DIR / "gems"

# ============================================================================
# 数据集子目录
# ============================================================================

# 在线RL数据集目录
ONLINE_DATASETS_DIR = DATASETS_ROOT / "online"

# 离线数据集目录
OFFLINE_DATASETS_DIR = DATASETS_ROOT / "offline"

# ============================================================================
# 实验和日志目录
# ============================================================================

# 训练日志目录
LOGS_DIR = EXPERIMENTS_ROOT / "logs"

# SwanLab日志目录
SWANLOG_DIR = EXPERIMENTS_ROOT / "swanlog"

# ============================================================================
# 辅助函数
# ============================================================================

def get_online_ckpt_dir(env_name: str) -> Path:
    """
    获取在线RL checkpoint目录

    Args:
        env_name: 环境名称（如 diffuse_topdown）

    Returns:
        checkpoint目录路径
    """
    ckpt_dir = ONLINE_RL_CKPT_DIR / env_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_offline_ckpt_dir(env_name: str, algo_name: str) -> Path:
    """
    获取离线RL checkpoint目录

    Args:
        env_name: 环境名称
        algo_name: 算法名称（如 td3_bc, cql, iql）

    Returns:
        checkpoint目录路径
    """
    ckpt_dir = OFFLINE_RL_CKPT_DIR / env_name / algo_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_offline_dataset_dir(env_name: str) -> Path:
    """
    获取离线数据集目录

    Args:
        env_name: 环境名称

    Returns:
        数据集目录路径
    """
    dataset_dir = OFFLINE_DATASETS_DIR / env_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_log_dir(experiment_name: str) -> Path:
    """
    获取实验日志目录

    Args:
        experiment_name: 实验名称

    Returns:
        日志目录路径
    """
    log_dir = LOGS_DIR / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_embeddings_path(embeddings_name: str) -> Path:
    """
    获取embeddings文件路径

    Args:
        embeddings_name: embeddings文件名（如 item_embeddings_diffuse.pt）

    Returns:
        embeddings文件路径
    """
    return EMBEDDINGS_DIR / embeddings_name


def get_mf_embeddings_path(mf_checkpoint: str) -> Path:
    """
    获取MF embeddings文件路径

    Args:
        mf_checkpoint: MF checkpoint名称（如 focused_topdown）

    Returns:
        MF embeddings文件路径
    """
    return MF_EMBEDDINGS_DIR / f"{mf_checkpoint}.pt"


def get_online_dataset_path(dataset_name: str) -> Path:
    """
    获取在线RL数据集路径

    Args:
        dataset_name: 数据集名称（如 focused_topdown）

    Returns:
        数据集文件路径
    """
    return ONLINE_DATASETS_DIR / f"{dataset_name}.pt"


def get_gems_checkpoint_path(checkpoint_name: str) -> Path:
    """
    获取GeMS checkpoint路径

    Args:
        checkpoint_name: checkpoint名称（不含.ckpt后缀）

    Returns:
        checkpoint文件路径
    """
    return GEMS_CKPT_DIR / f"{checkpoint_name}.ckpt"


def get_online_rl_results_dir(env_name: str) -> Path:
    """
    获取在线RL结果目录

    Args:
        env_name: 环境名称

    Returns:
        结果目录路径
    """
    results_dir = ONLINE_RL_RESULTS_DIR / env_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def ensure_all_dirs():
    """确保所有必要的目录都存在"""
    dirs_to_create = [
        DATA_ROOT,
        DATASETS_ROOT,
        EXPERIMENTS_ROOT,
        RESULTS_ROOT,
        EMBEDDINGS_DIR,
        MF_EMBEDDINGS_DIR,
        CHECKPOINTS_DIR,
        ONLINE_RL_CKPT_DIR,
        OFFLINE_RL_CKPT_DIR,
        GEMS_CKPT_DIR,
        ONLINE_DATASETS_DIR,
        OFFLINE_DATASETS_DIR,
        ONLINE_RL_RESULTS_DIR,
        OFFLINE_RL_RESULTS_DIR,
        LOGS_DIR,
        SWANLOG_DIR,
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_paths_info():
    """打印路径配置信息（用于调试）"""
    print("=" * 80)
    print("Offline Slate RL - 路径配置")
    print("=" * 80)
    print(f"\n项目根目录:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")

    print(f"\n主要目录:")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    print(f"  DATASETS_ROOT: {DATASETS_ROOT}")
    print(f"  EXPERIMENTS_ROOT: {EXPERIMENTS_ROOT}")
    print(f"  RESULTS_ROOT: {RESULTS_ROOT}")
    print(f"  CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")

    print(f"\n数据子目录:")
    print(f"  EMBEDDINGS_DIR: {EMBEDDINGS_DIR}")
    print(f"  MF_EMBEDDINGS_DIR: {MF_EMBEDDINGS_DIR}")
    print(f"  ONLINE_RL_CKPT_DIR: {ONLINE_RL_CKPT_DIR}")
    print(f"  OFFLINE_RL_CKPT_DIR: {OFFLINE_RL_CKPT_DIR}")
    print(f"  GEMS_CKPT_DIR: {GEMS_CKPT_DIR}")

    print(f"\n数据集目录:")
    print(f"  ONLINE_DATASETS_DIR: {ONLINE_DATASETS_DIR}")
    print(f"  OFFLINE_DATASETS_DIR: {OFFLINE_DATASETS_DIR}")

    print(f"\n结果目录:")
    print(f"  ONLINE_RL_RESULTS_DIR: {ONLINE_RL_RESULTS_DIR}")
    print(f"  OFFLINE_RL_RESULTS_DIR: {OFFLINE_RL_RESULTS_DIR}")

    print(f"\n实验和日志:")
    print(f"  LOGS_DIR: {LOGS_DIR}")
    print(f"  SWANLOG_DIR: {SWANLOG_DIR}")

    print(f"\n环境变量支持:")
    print(f"  OFFLINE_SLATE_RL_DATA_ROOT: {os.getenv('OFFLINE_SLATE_RL_DATA_ROOT', '(未设置)')}")
    print("=" * 80)


if __name__ == "__main__":
    # 测试路径配置
    print_paths_info()

    # 确保所有目录存在
    print("\n创建所有必要目录...")
    ensure_all_dirs()
    print("✅ 所有目录创建完成")

    # 测试辅助函数
    print("\n测试辅助函数:")
    print(f"  get_embeddings_path('item_embeddings_diffuse.pt'): {get_embeddings_path('item_embeddings_diffuse.pt')}")
    print(f"  get_mf_embeddings_path('focused_topdown'): {get_mf_embeddings_path('focused_topdown')}")
    print(f"  get_online_dataset_path('focused_topdown'): {get_online_dataset_path('focused_topdown')}")
    print(f"  get_gems_checkpoint_path('GeMS_focused_topdown_...'): {get_gems_checkpoint_path('GeMS_focused_topdown_test')}")
    print(f"  get_online_ckpt_dir('diffuse_topdown'): {get_online_ckpt_dir('diffuse_topdown')}")
    print(f"  get_online_rl_results_dir('diffuse_topdown'): {get_online_rl_results_dir('diffuse_topdown')}")
