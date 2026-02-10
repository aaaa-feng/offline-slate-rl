"""
统一的Checkpoint和数据集路径解析工具

采用单一事实来源原则，所有路径解析逻辑集中在此模块。
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

# Benchmark类型配置
BENCHMARK_CONFIG = {
    # 新benchmark（V4格式）
    'mix_divpen': {
        'type': 'new',
        'lambda_click': 1.0,
        'dataset_format': '{env_name}_{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click1.0_seed58407201.ckpt',
        # MF预训练格式（2026-02-01更新：使用正确的seed58407201）
        'checkpoint_format_mf_fixed': 'GeMS_{env_name}_{quality}_mf_fixed_latent32_beta1.0_click1.0_seed58407201.ckpt',
        'checkpoint_format_mf_scratch': 'GeMS_{env_name}_{quality}_scratch_latent32_beta1.0_click1.0_seed58407201.ckpt',
        # Epsilon-greedy特殊格式
        'checkpoint_format_epsilon': 'GeMS_{env_name}_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt'
    },
    'topdown_divpen': {
        'type': 'new',
        'lambda_click': 1.0,
        'dataset_format': '{env_name}_{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click1.0_seed58407201.ckpt',
        # MF预训练格式（2026-02-01更新：使用正确的seed58407201）
        'checkpoint_format_mf_fixed': 'GeMS_{env_name}_{quality}_mf_fixed_latent32_beta1.0_click1.0_seed58407201.ckpt',
        'checkpoint_format_mf_scratch': 'GeMS_{env_name}_{quality}_scratch_latent32_beta1.0_click1.0_seed58407201.ckpt',
        # Epsilon-greedy特殊格式
        'checkpoint_format_epsilon': 'GeMS_{env_name}_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt'
    },
    # 旧benchmark（V3格式）
    'diffuse_mix': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    },
    'diffuse_topdown': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    },
    'diffuse_divpen': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    },
    'focused_mix': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    },
    'focused_topdown': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    },
    'focused_divpen': {
        'type': 'old',
        'lambda_click': 0.5,
        'dataset_format': '{quality}_data_d4rl.npz',
        'checkpoint_format': 'GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt'
    }
}


def get_benchmark_config(env_name: str) -> Dict[str, Any]:
    """
    获取benchmark配置

    Args:
        env_name: 环境名称（如 'mix_divpen', 'diffuse_mix'）

    Returns:
        配置字典，包含type、lambda_click、格式等信息

    Raises:
        ValueError: 如果env_name不在支持列表中
    """
    if env_name not in BENCHMARK_CONFIG:
        available = ', '.join(BENCHMARK_CONFIG.keys())
        raise ValueError(
            f"不支持的环境名称: '{env_name}'\n"
            f"支持的环境: {available}"
        )
    return BENCHMARK_CONFIG[env_name]


def resolve_gems_checkpoint(
    env_name: str,
    dataset_quality: str,
    base_dir: str = "/data/liyuefeng/offline-slate-rl/checkpoints/gems/offline/",
    gems_embedding_mode: str = "default"
) -> Tuple[str, float]:
    """
    解析GeMS checkpoint路径和lambda_click参数

    Args:
        env_name: 环境名称
        dataset_quality: 数据集质量（新benchmark用'v2_b3'/'v2_b5'，旧benchmark用'expert'/'medium'）
        base_dir: checkpoint基础目录
        gems_embedding_mode: GeMS embedding模式 ('default', 'mf_fixed', 'mf_scratch', 'epsilon-greedy')

    Returns:
        (checkpoint_path, lambda_click_value)

    Raises:
        FileNotFoundError: 如果checkpoint不存在
    """
    config = get_benchmark_config(env_name)

    # 🔥 根据gems_embedding_mode选择checkpoint格式
    if gems_embedding_mode == 'mf_fixed' and 'checkpoint_format_mf_fixed' in config:
        checkpoint_name = config['checkpoint_format_mf_fixed'].format(
            env_name=env_name,
            quality=dataset_quality
        )
    elif gems_embedding_mode == 'mf_scratch' and 'checkpoint_format_mf_scratch' in config:
        checkpoint_name = config['checkpoint_format_mf_scratch'].format(
            env_name=env_name,
            quality=dataset_quality
        )
    elif dataset_quality == 'epsilon-greedy' and 'checkpoint_format_epsilon' in config:
        checkpoint_name = config['checkpoint_format_epsilon'].format(env_name=env_name)
    else:
        # 默认格式
        checkpoint_name = config['checkpoint_format'].format(
            env_name=env_name,
            quality=dataset_quality
        )
    checkpoint_path = os.path.join(base_dir, checkpoint_name)

    # 验证文件存在
    if not os.path.exists(checkpoint_path):
        # 列出可用的checkpoint
        available = list(Path(base_dir).glob(f"GeMS_{env_name}*.ckpt"))
        error_msg = (
            f"GeMS checkpoint不存在: {checkpoint_path}\n"
            f"环境: {env_name}, 质量: {dataset_quality}, embedding模式: {gems_embedding_mode}\n"
        )
        if available:
            error_msg += "可用的checkpoints:\n" + "\n".join(f"  - {c.name}" for c in available)
        else:
            error_msg += f"目录 {base_dir} 中没有找到 {env_name} 的任何checkpoint"
        raise FileNotFoundError(error_msg)

    lambda_click = config['lambda_click']
    logging.info(f"[checkpoint_utils.py] GeMS: {checkpoint_name}, λ_click={lambda_click}, embedding_mode={gems_embedding_mode}")

    return checkpoint_path, lambda_click


def resolve_dataset_path(
    env_name: str,
    dataset_quality: str,
    base_dir: str = "/data/liyuefeng/offline-slate-rl/data/datasets/offline/"
) -> str:
    """
    解析数据集路径

    Args:
        env_name: 环境名称
        dataset_quality: 数据集质量
        base_dir: 数据集基础目录

    Returns:
        数据集完整路径

    Raises:
        FileNotFoundError: 如果数据集不存在
    """
    config = get_benchmark_config(env_name)

    # 构建数据集文件名
    dataset_filename = config['dataset_format'].format(
        env_name=env_name,
        quality=dataset_quality
    )
    dataset_path = os.path.join(base_dir, env_name, dataset_filename)

    # 验证文件存在
    if not os.path.exists(dataset_path):
        # 列出可用的数据集
        env_dir = os.path.join(base_dir, env_name)
        if os.path.exists(env_dir):
            available = list(Path(env_dir).glob("*_data_d4rl.npz"))
            error_msg = (
                f"数据集不存在: {dataset_path}\n"
                f"环境: {env_name}, 质量: {dataset_quality}\n"
            )
            if available:
                error_msg += "可用的数据集:\n" + "\n".join(f"  - {d.name}" for d in available)
            else:
                error_msg += f"目录 {env_dir} 中没有找到任何数据集"
        else:
            error_msg = f"环境目录不存在: {env_dir}"
        raise FileNotFoundError(error_msg)

    logging.info(f"✓ 解析数据集路径: {dataset_filename}")
    return dataset_path

def extract_boredom_threshold(dataset_quality: str, env_name: str) -> Optional[int]:
    """
    从 dataset_quality 中提取 boredom threshold（仅新 benchmark）

    Args:
        dataset_quality: 数据集标识，如 'v2_b3', 'v2_b5', 'random', 'medium', 'expert'
        env_name: 环境名称

    Returns:
        boredom threshold 值（3或5），如果不是新 benchmark 则返回 None

    Examples:
        extract_boredom_threshold('v2_b3', 'mix_divpen') -> 3
        extract_boredom_threshold('v2_b5', 'topdown_divpen') -> 5
        extract_boredom_threshold('random', 'diffuse_mix') -> None
    """
    if env_name not in ['mix_divpen', 'topdown_divpen']:
        return None

    import re
    match = re.search(r'v2_b(\d+)', dataset_quality)
    if match:
        return int(match.group(1))
    return None
