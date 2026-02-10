"""
数据质量分析工具
用于分析数据集的质量指标（Overlap率、唯一物品数、Top10覆盖率等）
"""
import numpy as np
from typing import Dict
from collections import Counter


def analyze_dataset_quality(dataset_path: str) -> Dict:
    """
    分析数据集质量指标

    指标包括：
    - Overlap率（连续slate重叠率）
    - 唯一物品数
    - Top10覆盖率
    - 平均回报
    - 回报标准差
    """
    data = np.load(dataset_path)

    slates = data['slates']
    rewards = data['rewards']

    # 1. 计算overlap率
    overlap_rate = calculate_consecutive_overlap(slates)

    # 2. 计算唯一物品数
    unique_items = len(np.unique(slates))

    # 3. 计算top10覆盖率
    top10_coverage = calculate_top10_coverage(slates)

    # 4. 计算回报统计
    avg_return = np.mean(rewards)
    std_return = np.std(rewards)
    min_return = np.min(rewards)
    max_return = np.max(rewards)

    # 5. 计算episode级别的回报
    episode_ids = data['episode_ids']
    episode_returns = []
    for ep_id in np.unique(episode_ids):
        ep_mask = episode_ids == ep_id
        episode_returns.append(np.sum(rewards[ep_mask]))

    avg_episode_return = np.mean(episode_returns)
    std_episode_return = np.std(episode_returns)

    # 输出分析结果
    print(f"\n数据集路径: {dataset_path}")
    print(f"总转换数: {len(rewards):,}")
    print(f"总Episodes: {len(episode_returns):,}")
    print("\n" + "-" * 80)
    print("马太效应指标:")
    print(f"  Overlap率: {overlap_rate:.2%} (目标: <50%)")
    print(f"  Top10覆盖率: {top10_coverage:.2%} (目标: <60%)")
    print("\n探索性指标:")
    print(f"  唯一物品数: {unique_items} / 1000 ({unique_items/10:.1f}%)")
    print("\n性能指标:")
    print(f"  平均Step回报: {avg_return:.2f} ± {std_return:.2f}")
    print(f"  Step回报范围: [{min_return:.2f}, {max_return:.2f}]")
    print(f"  平均Episode回报: {avg_episode_return:.2f} ± {std_episode_return:.2f}")
    print("-" * 80)

    # 评估数据质量
    print("\n质量评估:")
    if overlap_rate < 0.50:
        print("  ✅ Overlap率达标")
    elif overlap_rate < 0.60:
        print("  ⚠️  Overlap率接近目标")
    else:
        print("  ❌ Overlap率过高")

    if unique_items > 700:
        print("  ✅ 物品覆盖度达标")
    elif unique_items > 600:
        print("  ⚠️  物品覆盖度接近目标")
    else:
        print("  ❌ 物品覆盖度不足")

    if top10_coverage < 0.60:
        print("  ✅ Top10覆盖率达标")
    elif top10_coverage < 0.70:
        print("  ⚠️  Top10覆盖率接近目标")
    else:
        print("  ❌ Top10覆盖率过高")

    return {
        'overlap_rate': overlap_rate,
        'unique_items': unique_items,
        'top10_coverage': top10_coverage,
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_episode_return': avg_episode_return,
        'std_episode_return': std_episode_return
    }


def calculate_consecutive_overlap(slates: np.ndarray) -> float:
    """计算连续slate的重叠率"""
    overlaps = []
    for i in range(len(slates) - 1):
        slate1 = set(slates[i])
        slate2 = set(slates[i + 1])
        overlap = len(slate1 & slate2) / len(slate1)
        overlaps.append(overlap)
    return np.mean(overlaps)


def calculate_top10_coverage(slates: np.ndarray) -> float:
    """计算top10物品的覆盖率"""
    item_counts = Counter(slates.flatten())
    top10_items = [item for item, _ in item_counts.most_common(10)]
    top10_count = sum(count for item, count in item_counts.items() if item in top10_items)
    return top10_count / slates.size
