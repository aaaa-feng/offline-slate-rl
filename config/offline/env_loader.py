"""
环境参数配置映射

此文件存储每个环境的完整参数配置,确保离线RL评估时使用与数据收集时一致的环境参数。

🔥 新版本 (2025-12-28):
- 优先从 dataset_meta.json 读取参数（数据集护照）
- 保留硬编码配置作为 fallback 机制
- 参数来源: data/datasets/offline/{env_name}/expert_data_meta.json
"""

import json
from pathlib import Path
from typing import Dict, Any
import warnings

# 项目根目录 (config/offline/env_loader.py -> config/offline -> config -> 项目根)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据集根目录
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets" / "offline"

# 6个环境的完整配置（作为 fallback）
ENV_CONFIGS_FALLBACK: Dict[str, Dict[str, Any]] = {
    "diffuse_topdown": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "diffuse_topdown",

        # Embeddings配置
        "item_embeddings": "item_embeddings_diffuse.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "tdPBM",
        "diversity_penalty": 1.0,

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "diffuse_topdown",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "diffuse_topdown",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/diffuse_topdown/SAC+GeMS_GeMS_diffuse_topdown_latentdim32_beta0.5_lambdaclick0.2_lambdaprior0.0_scratch_seed58407201_agentseed58407201_gamma0.8.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/diffuse_topdown/GeMS_diffuse_topdown_latentdim32_beta0.5_lambdaclick0.2_lambdaprior0.0_scratch_seed58407201.ckpt",
    },

    "diffuse_mix": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "diffuse_mix",

        # Embeddings配置
        "item_embeddings": "item_embeddings_diffuse.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "mixPBM",
        "diversity_penalty": 1.0,

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "diffuse_mix",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "diffuse_mix",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/diffuse_mix/SAC+GeMS_Medium_GeMS_diffuse_mix_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201_agentseed58407201_gamma0.8_best-v1.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/diffuse_mix/GeMS_diffuse_mix_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201.ckpt",
    },

    "diffuse_divpen": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "diffuse_divpen",

        # Embeddings配置
        "item_embeddings": "item_embeddings_diffuse.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "mixPBM",
        "diversity_penalty": 3.0,  # 注意: 这个环境的diversity_penalty是3.0

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "diffuse_divpen",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "diffuse_divpen",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/diffuse_divpen/SAC+GeMS_Medium_GeMS_diffuse_divpen_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201_agentseed58407201_gamma0.8_best-v1.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/diffuse_divpen/GeMS_diffuse_divpen_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201.ckpt",
    },

    "focused_topdown": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "focused_topdown",

        # Embeddings配置
        "item_embeddings": "item_embeddings_focused.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "tdPBM",
        "diversity_penalty": 1.0,

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "focused_topdown",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "focused_topdown",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/focused_topdown/SAC+GeMS_beta0.5_lambdaclick0.2_seed58407201_gamma0.8.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/focused_topdown/GeMS_beta0.5_lambdaclick0.2_latentdim32_seed58407201.ckpt",
    },

    "focused_mix": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "focused_mix",

        # Embeddings配置
        "item_embeddings": "item_embeddings_focused.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "mixPBM",
        "diversity_penalty": 1.0,

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "focused_mix",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "focused_mix",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/focused_mix/SAC+GeMS_beta0.5_lambdaclick0.2_seed58407201_gamma0.8.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/focused_mix/GeMS_beta0.5_lambdaclick0.2_latentdim32_seed58407201.ckpt",
    },

    "focused_divpen": {
        # 环境标识
        "env_name": "topics",
        "dataset_name": "focused_divpen",

        # Embeddings配置
        "item_embeddings": "item_embeddings_focused.pt",
        "item_embedds": "scratch",

        # Click model配置
        "click_model": "mixPBM",
        "diversity_penalty": 3.0,  # 注意: 这个环境的diversity_penalty是3.0

        # Ranker配置
        "ranker": "GeMS",
        "ranker_dataset": "focused_divpen",
        "ranker_embedds": "scratch",
        "ranker_sample": False,
        "latent_dim": 32,

        # Belief配置
        "belief": "GRU",
        "belief_state_dim": 20,

        # MF checkpoint
        "MF_checkpoint": "focused_divpen",

        # 环境参数
        "num_items": 1000,
        "item_embedd_dim": 20,
        "episode_length": 100,
        "boredom_threshold": 5,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "env_omega": 0.9,
        "short_term_boost": 1.0,
        "env_offset": 0.28,
        "env_slope": 100,
        "diversity_threshold": 4,
        "topic_size": 2,
        "num_topics": 10,

        # Checkpoint路径 (用于评估)
        "sac_gems_checkpoint": "expert/sac_gems_models/focused_divpen/SAC+GeMS_beta0.5_lambdaclick0.2_seed58407201_gamma0.8.ckpt",
        "gems_checkpoint": "expert/gems_checkpoints/focused_divpen/GeMS_beta0.5_lambdaclick0.2_latentdim32_seed58407201.ckpt",
    },
}


def _load_from_dataset_meta(env_name: str) -> Dict[str, Any]:
    """
    从 dataset_meta.json 加载环境配置

    Args:
        env_name: 环境名称

    Returns:
        环境配置字典，如果加载失败则返回 None
    """
    # 构建 expert_data_meta.json 的路径
    meta_path = DATASETS_ROOT / env_name / "expert_data_meta.json"

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # 提取 env_config 和 model_config
        env_config = meta.get("env_config", {})
        model_config = meta.get("model_config", {})
        source_model = meta.get("source_model", {})

        # 构建完整的配置字典
        config = {
            # 环境标识
            "env_name": env_config.get("env_name", "topics"),
            "dataset_name": env_config.get("dataset_name", env_name),

            # Embeddings配置
            "item_embeddings": env_config.get("env_embedds", "item_embeddings_diffuse.pt"),
            "item_embedds": "scratch",

            # Click model配置
            "click_model": env_config.get("click_model", "tdPBM"),
            "diversity_penalty": env_config.get("diversity_penalty", 1.0),

            # Ranker配置
            "ranker": model_config.get("ranker_type", "GeMS"),
            "ranker_dataset": env_name,
            "ranker_embedds": "scratch",
            "ranker_sample": False,
            "latent_dim": model_config.get("latent_dim", 32),

            # Belief配置
            "belief": model_config.get("belief_type", "GRU"),
            "belief_state_dim": model_config.get("belief_state_dim", 20),

            # MF checkpoint
            "MF_checkpoint": env_name,

            # 环境参数（从 env_config 提取）
            "num_items": env_config.get("num_items", 1000),
            "item_embedd_dim": env_config.get("item_embedd_dim", 20),
            "episode_length": env_config.get("episode_length", 100),
            "boredom_threshold": env_config.get("boredom_threshold", 5),
            "recent_items_maxlen": env_config.get("recent_items_maxlen", 10),
            "boredom_moving_window": env_config.get("boredom_moving_window", 5),
            "env_omega": env_config.get("env_omega", 0.9),
            "short_term_boost": env_config.get("short_term_boost", 1.0),
            "env_offset": env_config.get("env_offset", 0.28),
            "env_slope": env_config.get("env_slope", 100),
            "diversity_threshold": env_config.get("diversity_threshold", 4),
            "topic_size": env_config.get("topic_size", 2),
            "num_topics": env_config.get("num_topics", 10),

            # 🔥 关键：从 source_model 提取 checkpoint 路径
            "sac_gems_checkpoint": source_model.get("sac_gems_checkpoint", ""),
            "gems_checkpoint": source_model.get("gems_checkpoint", ""),

            # 🔥 额外信息：记录配置来源
            "_config_source": "dataset_meta.json",
            "_meta_path": str(meta_path),
        }

        return config

    except Exception as e:
        warnings.warn(f"Failed to load dataset_meta.json for {env_name}: {e}")
        return None


def get_env_config(env_name: str) -> Dict[str, Any]:
    """
    获取指定环境的配置

    🔥 新版本逻辑:
    1. 优先从 dataset_meta.json 读取（数据集护照）
    2. 如果读取失败，回退到硬编码配置

    Args:
        env_name: 环境名称 (如 diffuse_mix)

    Returns:
        环境配置字典

    Raises:
        ValueError: 如果环境名称不存在
    """
    # 尝试从 dataset_meta.json 加载
    config = _load_from_dataset_meta(env_name)

    if config is not None:
        # 成功从 dataset_meta.json 加载
        return config

    # 回退到硬编码配置
    if env_name not in ENV_CONFIGS_FALLBACK:
        available_envs = list(ENV_CONFIGS_FALLBACK.keys())
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available environments: {available_envs}"
        )

    # 添加配置来源标记
    fallback_config = ENV_CONFIGS_FALLBACK[env_name].copy()
    fallback_config["_config_source"] = "hardcoded_fallback"

    warnings.warn(
        f"Using fallback config for {env_name}. "
        f"dataset_meta.json not found at {DATASETS_ROOT / env_name / 'expert_data_meta.json'}"
    )

    return fallback_config


def list_available_envs() -> list:
    """返回所有可用的环境名称"""
    return list(ENV_CONFIGS_FALLBACK.keys())
