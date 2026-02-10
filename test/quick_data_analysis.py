#!/usr/bin/env python3
"""
快速数据集分析脚本
检查GeMS训练数据的关键指标,判断是否存在偏见
"""

import torch
import numpy as np
from pathlib import Path
from collections import Counter
import sys

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))

def calculate_gini_coefficient(frequencies):
    """计算基尼系数 (0=完全均匀, 1=完全不均)"""
    sorted_freq = np.sort(frequencies)
    n = len(sorted_freq)
    cumsum = np.cumsum(sorted_freq)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_freq)) / (n * cumsum[-1]) - (n + 1) / n

def analyze_dataset(dataset_path, sample_size=None, num_items=1000):
    """
    快速分析数据集

    Args:
        dataset_path: 数据集路径
        sample_size: 采样大小(None=全部数据)
        num_items: 物品总数
    """
    print(f"\n{'='*80}")
    print(f"分析数据集: {Path(dataset_path).name}")
    print(f"{'='*80}")

    # 加载数据
    print("加载数据...")
    data = torch.load(dataset_path, map_location='cpu')

    total_sessions = len(data)
    print(f"总会话数: {total_sessions}")

    # 采样(如果数据太大)
    if sample_size and sample_size < total_sessions:
        print(f"采样 {sample_size} 个会话进行分析...")
        sample_keys = np.random.choice(list(data.keys()), sample_size, replace=False)
        data_sample = {k: data[k] for k in sample_keys}
    else:
        data_sample = data
        sample_size = total_sessions

    # 初始化统计变量
    all_items = []
    all_clicks = []
    episode_returns = []
    slate_sizes = []

    print("统计中...")

    # 遍历采样数据
    for sess_id, session in data_sample.items():
        slates = session["slate"]  # (T, rec_size)
        clicks = session["clicks"]  # (T, rec_size)

        # 收集所有推荐的物品
        all_items.extend(slates.flatten().tolist())

        # 收集所有点击
        all_clicks.extend(clicks.flatten().tolist())

        # 计算episode return (点击总数)
        episode_return = clicks.sum().item()
        episode_returns.append(episode_return)

        # 记录slate大小
        slate_sizes.append(slates.shape[0])  # episode长度

    # 转换为numpy数组
    all_items = np.array(all_items)
    all_clicks = np.array(all_clicks)
    episode_returns = np.array(episode_returns)

    print(f"✓ 统计完成 (分析了 {sample_size} 个会话)")

    return {
        'all_items': all_items,
        'all_clicks': all_clicks,
        'episode_returns': episode_returns,
        'slate_sizes': slate_sizes,
        'num_items': num_items,
        'sample_size': sample_size,
        'total_sessions': total_sessions
    }


def print_report(stats):
    """打印分析报告"""
    all_items = stats['all_items']
    all_clicks = stats['all_clicks']
    episode_returns = stats['episode_returns']
    num_items = stats['num_items']

    print(f"\n{'='*80}")
    print("📊 数据集分析报告")
    print(f"{'='*80}")

    # 1. 基本统计
    print(f"\n【1. 基本统计】")
    print(f"  总推荐次数: {len(all_items):,}")
    print(f"  平均Episode长度: {np.mean(stats['slate_sizes']):.1f}")
    print(f"  平均Episode Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"  Return范围: [{np.min(episode_returns):.0f}, {np.max(episode_returns):.0f}]")

    # 2. 点击率
    click_rate = np.mean(all_clicks)
    print(f"\n【2. 点击率】")
    print(f"  平均点击率: {click_rate:.4f} ({click_rate*100:.2f}%)")
    if click_rate > 0.15:
        print(f"  ⚠️  点击率较高 → 可能是Expert数据 → Zero-Action陷阱风险!")
    elif click_rate < 0.05:
        print(f"  ✓ 点击率较低 → 可能是Random数据 → 有利于探索")
    else:
        print(f"  ℹ️  点击率中等 → 可能是Mixed数据")

    # 3. 物品覆盖率
    unique_items = np.unique(all_items)
    coverage = len(unique_items) / num_items
    print(f"\n【3. 物品覆盖率】")
    print(f"  唯一物品数: {len(unique_items)} / {num_items}")
    print(f"  覆盖率: {coverage:.2%}")
    if coverage < 0.5:
        print(f"  ⚠️  覆盖率过低 → VAE会有盲区 → 探索能力受限!")
    elif coverage > 0.95:
        print(f"  ✓ 覆盖率很高 → VAE能学到所有物品")
    else:
        print(f"  ℹ️  覆盖率中等")

    # 4. 物品频率分布
    item_counts = Counter(all_items)
    frequencies = np.array(list(item_counts.values()))
    gini = calculate_gini_coefficient(frequencies)

    print(f"\n【4. 物品频率分布】")
    print(f"  基尼系数: {gini:.4f}")
    if gini > 0.7:
        print(f"  ⚠️  严重不均 → 热门物品垄断 → 潜在空间偏置!")
    elif gini < 0.3:
        print(f"  ✓ 分布均匀 → 有利于VAE学习")
    else:
        print(f"  ℹ️  中等不均")

    # Top-10物品占比
    top_k = 10
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_k_ratio = sum([count for _, count in top_items]) / len(all_items)
    print(f"  Top-{top_k}物品占比: {top_k_ratio:.2%}")
    if top_k_ratio > 0.3:
        print(f"  ⚠️  Top-{top_k}占比过高 → 热门物品主导!")

    # 显示最热门的5个物品
    print(f"  最热门5个物品: {[item_id for item_id, _ in top_items[:5]]}")

    # 5. 诊断总结
    print(f"\n{'='*80}")
    print("🔍 诊断总结")
    print(f"{'='*80}")

    issues = []
    if click_rate > 0.15:
        issues.append("❌ 高点击率 → Zero-Action陷阱风险")
    if coverage < 0.5:
        issues.append("❌ 低覆盖率 → VAE盲区")
    if gini > 0.7:
        issues.append("❌ 高基尼系数 → 潜在空间偏置")
    if top_k_ratio > 0.3:
        issues.append("❌ 热门物品垄断")

    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n建议: 考虑使用更随机的数据重新训练GeMS VAE")
    else:
        print("\n✓ 数据质量良好,适合训练VAE")

    print(f"{'='*80}\n")


def main():
    """主函数"""
    # 数据集路径
    datasets = [
        "/data/liyuefeng/offline-slate-rl/data/test_data/oracle_aug_mix_eps0.5.pt",
    ]

    # 采样大小(为了快速分析,只采样部分数据)
    SAMPLE_SIZE = None  # 分析全部数据(只有30个会话)

    print("\n" + "="*80)
    print("Oracle-Augmented策略数据分析")
    print("="*80)
    print(f"采样大小: 全部数据")
    print("="*80)

    # 分析每个数据集
    for dataset_path in datasets:
        if not Path(dataset_path).exists():
            print(f"\n⚠️  数据集不存在: {dataset_path}")
            continue

        try:
            stats = analyze_dataset(dataset_path, sample_size=SAMPLE_SIZE)
            print_report(stats)
        except Exception as e:
            print(f"\n❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
