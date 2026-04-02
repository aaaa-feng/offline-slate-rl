"""
RankerFactory: 统一的Ranker加载工厂类

实现Agent-Ranker解耦，支持通过配置参数动态加载不同类型的ranker。

支持的ranker类型：
- gems: GeMS (VAE-based ranker)
- topk: TopKRanker (retrieval-based ranker)
- kheadargmax: kHeadArgmaxRanker (position-independent ranker)
- wolpertinger: WolpertingerRanker (single item kNN ranker)
- wolpertinger_slate: WolpertingerSlateRanker (multi-position kNN ranker)
- greedy: GreedySlateRanker (iterative greedy ranker)

核心设计：
- 返回"三个支柱": (ranker, action_dim, item_embeddings)
- GeMS特有参数封装在factory内部
- 非GeMS ranker不依赖GeMS checkpoint
"""

import logging
import torch
from typing import Tuple

from rankers.gems.item_embeddings import ItemEmbeddings
from rankers.gems.rankers import (
    GeMS, TopKRanker, kHeadArgmaxRanker,
    WolpertingerRanker, WolpertingerSlateRanker, GreedySlateRanker
)
from common.offline.checkpoint_utils import resolve_gems_checkpoint


class RankerFactory:
    """
    Ranker工厂类，负责根据配置动态创建ranker实例

    核心功能：
    1. 封装所有ranker的加载逻辑
    2. 提供统一的创建接口
    3. 处理不同ranker的初始化差异
    4. 确保返回格式一致
    """

    @staticmethod
    def create(ranker_type: str, config, device) -> Tuple:
        """
        根据ranker_type创建ranker实例

        Args:
            ranker_type: ranker类型 ("gems", "topk", "kheadargmax", "wolpertinger", "wolpertinger_slate", "greedy")
            config: 配置对象（包含所有必要参数）
            device: 设备（cpu/cuda）

        Returns:
            Tuple[Ranker, int, ItemEmbeddings]:
                - ranker: Ranker实例（实现了rank()和run_inference()）
                - action_dim: Action维度（int，非tuple）
                - item_embeddings: ItemEmbeddings对象（用于GRUBelief）

        Raises:
            ValueError: 不支持的ranker类型
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"RankerFactory: Creating ranker of type '{ranker_type}'")
        logging.info(f"{'='*80}")

        if ranker_type == "gems":
            return RankerFactory._create_gems(config, device)
        elif ranker_type == "topk":
            return RankerFactory._create_topk(config, device)
        elif ranker_type == "kheadargmax":
            return RankerFactory._create_kheadargmax(config, device)
        elif ranker_type == "wolpertinger":
            return RankerFactory._create_wolpertinger(config, device)
        elif ranker_type == "wolpertinger_slate":
            return RankerFactory._create_wolpertinger_slate(config, device)
        elif ranker_type == "greedy":
            return RankerFactory._create_greedy(config, device)
        else:
            raise ValueError(
                f"Unsupported ranker type: {ranker_type}. "
                f"Supported types: gems, topk, kheadargmax, wolpertinger, wolpertinger_slate, greedy"
            )

    @staticmethod
    def _create_gems(config, device) -> Tuple:
        """
        创建GeMS ranker

        GeMS特有逻辑：
        - 需要调用 resolve_gems_checkpoint() 解析checkpoint路径
        - 需要加载预训练的GeMS模型
        - lambda_click等参数封装在此方法内部

        Returns:
            (ranker, action_dim=32, item_embeddings)
        """
        logging.info("Loading GeMS checkpoint and extracting embeddings")

        # 解析GeMS checkpoint路径（仅GeMS需要）
        gems_path, lambda_click = resolve_gems_checkpoint(
            env_name=config.env_name,
            dataset_quality=config.dataset_quality
        )
        logging.info(f"GeMS checkpoint: {gems_path}")
        logging.info(f"lambda_click: {lambda_click}")

        # 加载临时embeddings用于GeMS初始化
        temp_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 加载GeMS checkpoint
        ranker = GeMS.load_from_checkpoint(
            gems_path,
            map_location=device,
            item_embeddings=temp_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            device=device,
            rec_size=config.rec_size,
            latent_dim=32,
            lambda_click=lambda_click,
            lambda_KL=1.0,
            lambda_prior=1.0,
            ranker_lr=3e-3,
            fixed_embedds="scratch",
            ranker_sample=False,
            hidden_layers_infer=[512, 256],
            hidden_layers_decoder=[256, 512]
        )
        ranker.freeze()
        ranker = ranker.to(device)
        logging.info(f"✅ GeMS loaded and moved to {device}")

        # 提取GeMS训练的embeddings
        gems_embedding_weights = ranker.item_embeddings.weight.data.clone()
        logging.info(f"✅ Extracted embeddings: shape={gems_embedding_weights.shape}")

        # 创建新的ItemEmbeddings对象（用于IQL内部）
        item_embeddings = ItemEmbeddings(
            num_items=ranker.num_items,
            item_embedd_dim=config.item_embedd_dim,
            device=device,
            weights=gems_embedding_weights
        )

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ GeMS action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

    @staticmethod
    def _create_topk(config, device) -> Tuple:
        """
        创建TopKRanker

        TopKRanker特点：
        - 不需要预训练checkpoint
        - 只需要原始item embeddings
        - action_dim = item_embedd_dim (20)

        Returns:
            (ranker, action_dim=20, item_embeddings)
        """
        logging.info("Creating TopKRanker (no checkpoint needed)")
        logging.info(f"Item embeddings: {config.item_embedds_path}")

        # 加载原始item embeddings
        item_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 初始化TopKRanker
        ranker = TopKRanker(
            item_embeddings=item_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            device=device
        )
        ranker = ranker.to(device)
        logging.info(f"✅ TopKRanker created and moved to {device}")

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ TopKRanker action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

    @staticmethod
    def _create_kheadargmax(config, device) -> Tuple:
        """
        创建kHeadArgmaxRanker

        kHeadArgmaxRanker特点：
        - 不需要预训练checkpoint
        - 只需要原始item embeddings
        - action_dim = item_embedd_dim * rec_size (200)

        Returns:
            (ranker, action_dim=200, item_embeddings)
        """
        logging.info("Creating kHeadArgmaxRanker (no checkpoint needed)")
        logging.info(f"Item embeddings: {config.item_embedds_path}")

        # 加载原始item embeddings
        item_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 初始化kHeadArgmaxRanker
        ranker = kHeadArgmaxRanker(
            item_embeddings=item_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            device=device
        )
        ranker = ranker.to(device)
        logging.info(f"✅ kHeadArgmaxRanker created and moved to {device}")

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ kHeadArgmaxRanker action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

    @staticmethod
    def _create_wolpertinger(config, device) -> Tuple:
        """
        创建WolpertingerRanker (单 item kNN)

        WolpertingerRanker特点：
        - 不需要预训练checkpoint
        - 使用kNN搜索 + Actor网络
        - action_dim = item_embedd_dim (20)

        Returns:
            (ranker, action_dim=20, item_embeddings)
        """
        logging.info("Creating WolpertingerRanker (single item kNN)")
        logging.info(f"Item embeddings: {config.item_embedds_path}")
        logging.info(f"kNN k: {config.wolpertinger_k}")

        # 加载原始item embeddings
        item_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 初始化WolpertingerRanker
        ranker = WolpertingerRanker(
            item_embeddings=item_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            device=device,
            k=config.wolpertinger_k,
            actor_hidden_dims=config.wolpertinger_hidden_dims,
            state_dim=config.belief_state_dim
        )
        ranker = ranker.to(device)
        logging.info(f"✅ WolpertingerRanker created and moved to {device}")

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ WolpertingerRanker action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

    @staticmethod
    def _create_wolpertinger_slate(config, device) -> Tuple:
        """
        创建WolpertingerSlateRanker (多位置 kNN)

        WolpertingerSlateRanker特点：
        - 不需要预训练checkpoint
        - 每个位置独立kNN搜索
        - action_dim = item_embedd_dim * rec_size (200)

        Returns:
            (ranker, action_dim=200, item_embeddings)
        """
        logging.info("Creating WolpertingerSlateRanker (multi-position kNN)")
        logging.info(f"Item embeddings: {config.item_embedds_path}")
        logging.info(f"kNN k: {config.wolpertinger_k}")

        # 加载原始item embeddings
        item_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 初始化WolpertingerSlateRanker
        ranker = WolpertingerSlateRanker(
            item_embeddings=item_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            device=device,
            k=config.wolpertinger_k,
            actor_hidden_dims=config.wolpertinger_hidden_dims,
            state_dim=config.belief_state_dim
        )
        ranker = ranker.to(device)
        logging.info(f"✅ WolpertingerSlateRanker created and moved to {device}")

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ WolpertingerSlateRanker action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

    @staticmethod
    def _create_greedy(config, device) -> Tuple:
        """
        创建GreedySlateRanker

        GreedySlateRanker特点：
        - 不需要预训练checkpoint
        - 迭代贪心选择，考虑累积效应
        - action_dim = item_embedd_dim (20)

        Returns:
            (ranker, action_dim=20, item_embeddings)
        """
        logging.info("Creating GreedySlateRanker (iterative greedy)")
        logging.info(f"Item embeddings: {config.item_embedds_path}")
        logging.info(f"s_no_click: {config.greedy_s_no_click}")

        # 加载原始item embeddings
        item_embeddings = ItemEmbeddings.from_pretrained(
            config.item_embedds_path, device
        )

        # 初始化GreedySlateRanker
        ranker = GreedySlateRanker(
            item_embeddings=item_embeddings,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            device=device,
            s_no_click=config.greedy_s_no_click
        )
        ranker = ranker.to(device)
        logging.info(f"✅ GreedySlateRanker created and moved to {device}")

        # 获取action维度
        action_dim, _ = ranker.get_action_dim()
        logging.info(f"✅ GreedySlateRanker action_dim: {action_dim}")

        return ranker, action_dim, item_embeddings

