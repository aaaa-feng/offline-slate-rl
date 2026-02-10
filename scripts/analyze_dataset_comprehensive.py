#!/usr/bin/env python3
"""
综合数据集分析脚本
分析奖励分布、状态-动作覆盖、轨迹质量等
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

def analyze_reward_distribution(dataset_path):
    """分析奖励分布"""
    print("="*80)
    print("1. 奖励分布分析")
    print("="*80)

    data = np.load(dataset_path)
    rewards = data['rewards']

    print(f"样本数: {len(rewards)}")
    print(f"最小值: {rewards.min():.4f}")
    print(f"最大值: {rewards.max():.4f}")
    print(f"均值: {rewards.mean():.4f}")
    print(f"标准差: {rewards.std():.4f}")
    print(f"中位数: {np.median(rewards):.4f}")

    print(f"\n百分位数:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}%: {np.percentile(rewards, p):.4f}")

    # 检查异常值
    outliers_low = (rewards < 0).sum()
    outliers_high = (rewards > 10).sum()
    print(f"\n异常值:")
    print(f"  < 0: {outliers_low} ({100*outliers_low/len(rewards):.2f}%)")
    print(f"  > 10: {outliers_high} ({100*outliers_high/len(rewards):.2f}%)")

    # 奖励值分布（离散化）
    unique_rewards, counts = np.unique(rewards, return_counts=True)
    print(f"\n奖励值分布 (前10个最常见的值):")
    sorted_idx = np.argsort(-counts)[:10]
    for idx in sorted_idx:
        r, c = unique_rewards[idx], counts[idx]
        print(f"  reward={r:.1f}: {c} 次 ({100*c/len(rewards):.2f}%)")

    return rewards

def analyze_episode_structure(dataset_path):
    """分析episode结构"""
    print("\n" + "="*80)
    print("2. Episode结构分析")
    print("="*80)

    data = np.load(dataset_path)
    episode_ids = data['episode_ids']
    timesteps = data['timesteps']
    terminals = data['terminals']
    rewards = data['rewards']

    unique_episodes = np.unique(episode_ids)
    print(f"Episode总数: {len(unique_episodes)}")

    # 计算每个episode的长度和回报
    episode_lengths = []
    episode_returns = []

    for ep_id in unique_episodes[:1000]:  # 只分析前1000个episode
        mask = (episode_ids == ep_id)
        ep_len = mask.sum()
        ep_return = rewards[mask].sum()
        episode_lengths.append(ep_len)
        episode_returns.append(ep_return)

    episode_lengths = np.array(episode_lengths)
    episode_returns = np.array(episode_returns)

    print(f"\nEpisode长度统计 (前1000个):")
    print(f"  均值: {episode_lengths.mean():.2f}")
    print(f"  标准差: {episode_lengths.std():.2f}")
    print(f"  最小值: {episode_lengths.min()}")
    print(f"  最大值: {episode_lengths.max()}")
    print(f"  中位数: {np.median(episode_lengths):.0f}")

    print(f"\nEpisode回报统计 (前1000个):")
    print(f"  均值: {episode_returns.mean():.2f}")
    print(f"  标准差: {episode_returns.std():.2f}")
    print(f"  最小值: {episode_returns.min():.2f}")
    print(f"  最大值: {episode_returns.max():.2f}")
    print(f"  中位数: {np.median(episode_returns):.2f}")

    return episode_lengths, episode_returns

def analyze_slate_action_coverage(dataset_path):
    """分析slate和action的覆盖度"""
    print("\n" + "="*80)
    print("3. Slate-Action覆盖度分析")
    print("="*80)

    data = np.load(dataset_path)
    slates = data['slates']  # (N, 10)
    clicks = data['clicks']  # (N, 10)

    print(f"Slate维度: {slates.shape}")
    print(f"Clicks维度: {clicks.shape}")

    # 分析item ID的分布
    all_items = slates.flatten()
    unique_items = np.unique(all_items)
    print(f"\n唯一item数量: {len(unique_items)}")
    print(f"Item ID范围: [{all_items.min()}, {all_items.max()}]")

    # Item出现频率
    item_counts = Counter(all_items)
    most_common = item_counts.most_common(10)
    print(f"\n最常见的10个items:")
    for item_id, count in most_common:
        print(f"  Item {item_id}: {count} 次 ({100*count/len(all_items):.2f}%)")

    # 分析clicks的分布
    click_rates = clicks.mean(axis=1)  # 每个slate的平均点击率
    print(f"\n点击率统计:")
    print(f"  均值: {click_rates.mean():.4f}")
    print(f"  标准差: {click_rates.std():.4f}")
    print(f"  最小值: {click_rates.min():.4f}")
    print(f"  最大值: {click_rates.max():.4f}")

    # 每个slate的总点击数分布
    total_clicks = clicks.sum(axis=1)
    print(f"\n每个slate的总点击数分布:")
    for i in range(11):  # 0-10次点击
        count = (total_clicks == i).sum()
        print(f"  {i}次点击: {count} ({100*count/len(total_clicks):.2f}%)")

    return slates, clicks

def main():
    dataset_path = 'data/datasets/offline/mix_divpen/mix_divpen_v2_b3_data_d4rl.npz'

    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return

    print(f"分析数据集: {dataset_path}\n")

    # 1. 奖励分布分析
    rewards = analyze_reward_distribution(dataset_path)

    # 2. Episode结构分析
    episode_lengths, episode_returns = analyze_episode_structure(dataset_path)

    # 3. Slate-Action覆盖度分析
    slates, clicks = analyze_slate_action_coverage(dataset_path)

    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)

if __name__ == "__main__":
    main()
