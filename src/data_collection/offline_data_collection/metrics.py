#!/usr/bin/env python3
"""
Slate推荐的指标计算工具
包括多样性分数、覆盖率分数等
"""
import torch
import numpy as np
from typing import List, Set, Dict, Any
import math

class SlateMetrics:
    """Slate推荐指标计算器"""
    
    def __init__(self, item_embeddings: torch.Tensor, total_items: int):
        """
        初始化指标计算器
        
        Args:
            item_embeddings: 物品embedding矩阵 (num_items, embed_dim)
            total_items: 总物品数量
        """
        self.item_embeddings = item_embeddings
        self.total_items = total_items
        self.recommended_items_global: Set[int] = set()  # 全局推荐过的物品
    
    def calculate_diversity_score(self, slate: List[int]) -> float:
        """
        计算slate的多样性分数
        基于物品embedding的余弦相似度
        
        Args:
            slate: 物品ID列表
            
        Returns:
            diversity_score: 多样性分数 (0-1, 越高越多样)
        """
        if len(slate) <= 1:
            return 1.0
        
        try:
            # 获取slate中物品的embeddings
            slate_embeddings = self.item_embeddings[slate]  # (slate_size, embed_dim)
            
            # 计算两两相似度矩阵
            similarities = torch.cosine_similarity(
                slate_embeddings.unsqueeze(1), 
                slate_embeddings.unsqueeze(0), 
                dim=2
            )
            
            # 去除对角线（自相似度=1）
            mask = ~torch.eye(len(slate), dtype=bool, device=similarities.device)
            avg_similarity = similarities[mask].mean()
            
            # 多样性 = 1 - 平均相似度
            diversity_score = 1.0 - avg_similarity.item()
            
            # 确保在[0,1]范围内
            return max(0.0, min(1.0, diversity_score))
            
        except Exception as e:
            print(f"计算多样性分数时出错: {e}")
            return 0.5  # 返回中性值
    
    def calculate_intra_list_diversity(self, slate: List[int]) -> float:
        """
        计算列表内多样性 (ILD - Intra-List Diversity)
        使用Jaccard距离的变体
        """
        if len(slate) <= 1:
            return 1.0
        
        try:
            slate_embeddings = self.item_embeddings[slate]
            
            # 计算所有物品对之间的距离
            distances = []
            for i in range(len(slate)):
                for j in range(i + 1, len(slate)):
                    # 使用欧几里得距离
                    dist = torch.dist(slate_embeddings[i], slate_embeddings[j]).item()
                    distances.append(dist)
            
            # 平均距离作为多样性指标
            avg_distance = np.mean(distances)
            
            # 归一化到[0,1]范围（这里使用简单的sigmoid函数）
            diversity = 1.0 / (1.0 + math.exp(-avg_distance))
            
            return diversity
            
        except Exception as e:
            print(f"计算列表内多样性时出错: {e}")
            return 0.5
    
    def calculate_coverage_score(self, slate: List[int], episode_slates: List[List[int]] = None) -> float:
        """
        计算覆盖率分数
        
        Args:
            slate: 当前slate
            episode_slates: 当前episode的所有slates（用于计算episode覆盖率）
            
        Returns:
            coverage_score: 覆盖率分数
        """
        if episode_slates is None:
            episode_slates = [slate]
        
        # 统计episode中推荐过的唯一物品
        episode_items = set()
        for s in episode_slates:
            # 确保s是列表格式，如果是字典则提取slate字段
            if isinstance(s, dict):
                if 'slate' in s:
                    episode_items.update(s['slate'])
                else:
                    continue
            else:
                episode_items.update(s)
        
        # Episode覆盖率
        episode_coverage = len(episode_items) / self.total_items
        
        return episode_coverage
    
    def calculate_global_coverage(self, slate: List[int]) -> float:
        """
        计算全局覆盖率（整个数据收集过程中的覆盖率）
        """
        # 更新全局推荐物品集合
        self.recommended_items_global.update(slate)
        
        # 全局覆盖率
        global_coverage = len(self.recommended_items_global) / self.total_items
        
        return global_coverage
    
    def calculate_novelty_score(self, slate: List[int], item_popularity: Dict[int, float]) -> float:
        """
        计算新颖性分数
        基于物品的流行度，推荐不流行的物品得分更高
        
        Args:
            slate: 物品ID列表
            item_popularity: 物品流行度字典 {item_id: popularity}
            
        Returns:
            novelty_score: 新颖性分数
        """
        if not slate:
            return 0.0
        
        novelty_scores = []
        for item_id in slate:
            popularity = item_popularity.get(item_id, 0.0)
            # 新颖性 = 1 - 流行度
            novelty = 1.0 - popularity
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def calculate_serendipity_score(self, slate: List[int], user_profile: torch.Tensor = None) -> float:
        """
        计算意外性分数
        推荐与用户历史偏好不同但质量高的物品
        
        Args:
            slate: 物品ID列表
            user_profile: 用户偏好向量
            
        Returns:
            serendipity_score: 意外性分数
        """
        if user_profile is None:
            # 如果没有用户画像，返回多样性分数作为替代
            return self.calculate_diversity_score(slate)
        
        try:
            slate_embeddings = self.item_embeddings[slate]
            
            # 计算slate与用户偏好的相似度
            similarities = torch.cosine_similarity(
                slate_embeddings, 
                user_profile.unsqueeze(0), 
                dim=1
            )
            
            # 意外性 = 1 - 平均相似度
            serendipity = 1.0 - similarities.mean().item()
            
            return max(0.0, min(1.0, serendipity))
            
        except Exception as e:
            print(f"计算意外性分数时出错: {e}")
            return 0.5
    
    def calculate_click_through_rate(self, slate: List[int], clicks: torch.Tensor) -> float:
        """
        计算点击率
        
        Args:
            slate: 物品ID列表
            clicks: 点击向量
            
        Returns:
            ctr: 点击率
        """
        if len(slate) == 0:
            return 0.0
        
        total_clicks = clicks.sum().item()
        ctr = total_clicks / len(slate)
        
        return ctr
    
    def calculate_position_bias_metrics(self, slate: List[int], clicks: torch.Tensor) -> Dict[str, float]:
        """
        计算位置偏差相关指标
        
        Args:
            slate: 物品ID列表
            clicks: 点击向量
            
        Returns:
            position_metrics: 位置相关指标字典
        """
        metrics = {}
        
        # 各位置的点击率
        position_ctrs = []
        for i, click in enumerate(clicks):
            position_ctrs.append(click.item())
        
        metrics['position_ctrs'] = position_ctrs
        metrics['top3_ctr'] = np.mean(position_ctrs[:3]) if len(position_ctrs) >= 3 else 0.0
        metrics['bottom3_ctr'] = np.mean(position_ctrs[-3:]) if len(position_ctrs) >= 3 else 0.0
        
        # 位置偏差强度（top位置与bottom位置的CTR差异）
        if len(position_ctrs) >= 6:
            metrics['position_bias'] = metrics['top3_ctr'] - metrics['bottom3_ctr']
        else:
            metrics['position_bias'] = 0.0
        
        return metrics
    
    def calculate_comprehensive_metrics(self, slate: List[int], clicks: torch.Tensor, 
                                      episode_slates: List[List[int]] = None,
                                      item_popularity: Dict[int, float] = None,
                                      user_profile: torch.Tensor = None) -> Dict[str, float]:
        """
        计算综合指标
        
        Returns:
            metrics: 所有指标的字典
        """
        metrics = {}
        
        # 基础指标
        metrics['diversity_score'] = self.calculate_diversity_score(slate)
        metrics['coverage_score'] = self.calculate_coverage_score(slate, episode_slates)
        metrics['global_coverage'] = self.calculate_global_coverage(slate)
        metrics['click_through_rate'] = self.calculate_click_through_rate(slate, clicks)
        
        # 高级指标
        metrics['intra_list_diversity'] = self.calculate_intra_list_diversity(slate)
        
        if item_popularity is not None:
            metrics['novelty_score'] = self.calculate_novelty_score(slate, item_popularity)
        
        if user_profile is not None:
            metrics['serendipity_score'] = self.calculate_serendipity_score(slate, user_profile)
        
        # 位置偏差指标
        position_metrics = self.calculate_position_bias_metrics(slate, clicks)
        metrics.update(position_metrics)
        
        return metrics

def create_item_popularity_dict(all_slates: List[List[int]], total_items: int) -> Dict[int, float]:
    """
    从历史推荐数据创建物品流行度字典
    
    Args:
        all_slates: 所有历史slates
        total_items: 总物品数
        
    Returns:
        item_popularity: 物品流行度字典
    """
    item_counts = {}
    total_recommendations = 0
    
    for slate in all_slates:
        for item_id in slate:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
            total_recommendations += 1
    
    # 计算流行度（推荐频率）
    item_popularity = {}
    for item_id in range(total_items):
        count = item_counts.get(item_id, 0)
        popularity = count / total_recommendations if total_recommendations > 0 else 0.0
        item_popularity[item_id] = popularity
    
    return item_popularity

if __name__ == "__main__":
    # 测试指标计算
    print("测试Slate指标计算...")
    
    # 创建测试数据
    num_items = 1000
    embed_dim = 64
    item_embeddings = torch.randn(num_items, embed_dim)
    
    # 初始化指标计算器
    metrics_calculator = SlateMetrics(item_embeddings, num_items)
    
    # 测试slate
    test_slate = [1, 5, 10, 50, 100, 200, 300, 400, 500, 600]
    test_clicks = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    
    # 计算各种指标
    diversity = metrics_calculator.calculate_diversity_score(test_slate)
    coverage = metrics_calculator.calculate_coverage_score(test_slate)
    ctr = metrics_calculator.calculate_click_through_rate(test_slate, test_clicks)
    
    print(f"多样性分数: {diversity:.4f}")
    print(f"覆盖率分数: {coverage:.4f}")
    print(f"点击率: {ctr:.4f}")
    
    # 测试综合指标
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
        test_slate, test_clicks
    )
    
    print("\n综合指标:")
    for key, value in comprehensive_metrics.items():
        if isinstance(value, list):
            print(f"  {key}: {[f'{v:.4f}' for v in value]}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("✅ 指标计算测试完成!")
