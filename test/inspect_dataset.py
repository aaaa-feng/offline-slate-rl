#!/usr/bin/env python3
"""
数据尸检脚本 (Data Autopsy Script)
用于检查离线数据集的详细内容和结构
"""
import numpy as np
import sys
from pathlib import Path

def inspect_dataset(dataset_path: str):
    """详细检查数据集内容"""
    print("=" * 80)
    print("数据尸检报告 (Data Autopsy Report)")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print()

    # 加载数据集
    try:
        dataset = np.load(dataset_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print("✅ 数据集加载成功")
    print()

    # ========================================================================
    # Part 1: 所有Keys列表
    # ========================================================================
    print("=" * 80)
    print("Part 1: 数据集包含的所有Keys")
    print("=" * 80)
    keys = list(dataset.keys())
    for i, key in enumerate(keys, 1):
        print(f"{i}. {key}")
    print()

    # ========================================================================
    # Part 2: 每个Key的Shape和Data Type
    # ========================================================================
    print("=" * 80)
    print("Part 2: 每个Key的Shape和Data Type")
    print("=" * 80)
    for key in keys:
        data = dataset[key]
        print(f"{key}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size: {data.size:,} elements")
        print()

    # ========================================================================
    # Part 3: Slates样本内容
    # ========================================================================
    print("=" * 80)
    print("Part 3: Slates 前2条样本 (验证是否为Raw Item IDs)")
    print("=" * 80)
    if 'slates' in dataset:
        slates = dataset['slates']
        print(f"Slates shape: {slates.shape}")
        print(f"Slates dtype: {slates.dtype}")
        print()
        print("Sample 1 (前10个物品ID):")
        print(slates[0])
        print()
        print("Sample 2 (前10个物品ID):")
        print(slates[1])
        print()
        print(f"Slates值域: min={slates.min()}, max={slates.max()}")
        print(f"是否为整数: {np.issubdtype(slates.dtype, np.integer)}")
    else:
        print("❌ 数据集中没有'slates'字段")
    print()

    # ========================================================================
    # Part 4: Actions样本内容
    # ========================================================================
    print("=" * 80)
    print("Part 4: Actions 前2条样本 (验证是否为Latent Vectors)")
    print("=" * 80)
    if 'actions' in dataset:
        actions = dataset['actions']
        print(f"Actions shape: {actions.shape}")
        print(f"Actions dtype: {actions.dtype}")
        print()
        print("Sample 1 (32维向量):")
        print(actions[0])
        print()
        print("Sample 2 (32维向量):")
        print(actions[1])
        print()
    else:
        print("❌ 数据集中没有'actions'字段")
    print()

    # ========================================================================
    # Part 5: Clicks样本内容
    # ========================================================================
    print("=" * 80)
    print("Part 5: Clicks 前2条样本")
    print("=" * 80)
    if 'clicks' in dataset:
        clicks = dataset['clicks']
        print(f"Clicks shape: {clicks.shape}")
        print(f"Clicks dtype: {clicks.dtype}")
        print()
        print("Sample 1 (10个点击反馈):")
        print(clicks[0])
        print()
        print("Sample 2 (10个点击反馈):")
        print(clicks[1])
        print()
        print(f"Clicks值域: min={clicks.min()}, max={clicks.max()}")
        print(f"点击率: {clicks.mean():.4f}")
    else:
        print("❌ 数据集中没有'clicks'字段")
    print()

    # ========================================================================
    # Part 6: Actions统计信息
    # ========================================================================
    print("=" * 80)
    print("Part 6: Actions 统计信息 (判断数据分布)")
    print("=" * 80)
    if 'actions' in dataset:
        actions = dataset['actions']
        print(f"Actions shape: {actions.shape}")
        print(f"Total samples: {actions.shape[0]:,}")
        print(f"Action dimension: {actions.shape[1]}")
        print()
        print("统计量 (全局):")
        print(f"  Min:  {actions.min():.6f}")
        print(f"  Max:  {actions.max():.6f}")
        print(f"  Mean: {actions.mean():.6f}")
        print(f"  Std:  {actions.std():.6f}")
        print()
        print("统计量 (每个维度):")
        for dim in range(min(5, actions.shape[1])):  # 只显示前5维
            print(f"  Dim {dim}: min={actions[:, dim].min():.4f}, "
                  f"max={actions[:, dim].max():.4f}, "
                  f"mean={actions[:, dim].mean():.4f}, "
                  f"std={actions[:, dim].std():.4f}")
        if actions.shape[1] > 5:
            print(f"  ... (剩余 {actions.shape[1] - 5} 维)")
    else:
        print("❌ 数据集中没有'actions'字段")
    print()

    # ========================================================================
    # Part 7: 关键判断
    # ========================================================================
    print("=" * 80)
    print("Part 7: 关键判断")
    print("=" * 80)

    if 'slates' in dataset and 'actions' in dataset:
        slates = dataset['slates']
        actions = dataset['actions']

        print("✅ 数据集同时包含 slates 和 actions")
        print()
        print("判断1: Slates是否为Raw Item IDs?")
        if np.issubdtype(slates.dtype, np.integer) and slates.min() >= 0:
            print("  ✅ 是! Slates是整数类型,值域合理 (0-999)")
            print("  → 这是原始的物品ID")
        else:
            print("  ❌ 否! Slates不是整数或值域异常")
        print()

        print("判断2: Actions是否为Latent Vectors?")
        if actions.dtype == np.float32 or actions.dtype == np.float64:
            print("  ✅ 是! Actions是浮点数类型")
            print(f"  → 维度: {actions.shape[1]}")
            print(f"  → 值域: [{actions.min():.2f}, {actions.max():.2f}]")
            print("  → 这很可能是GeMS的Latent Space表示")
        else:
            print("  ❌ 否! Actions不是浮点数类型")
        print()

        print("判断3: 数据是否支持Raw Action算法?")
        print("  ✅ 支持! 因为数据集包含完整的slates")
        print("  → 可以用slates作为监督信号训练Raw Action算法")
        print("  → 不需要重新收集数据")
    else:
        print("❌ 数据集缺少关键字段")

    print()
    print("=" * 80)
    print("数据尸检完成")
    print("=" * 80)


if __name__ == "__main__":
    dataset_path = "/data/liyuefeng/offline-slate-rl/data/datasets/offline/diffuse_mix/medium_data_d4rl.npz"
    inspect_dataset(dataset_path)
