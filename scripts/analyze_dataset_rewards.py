#!/usr/bin/env python3
"""
Dataset Reward Distribution Analysis
Experiment 1: Analyze reward distribution to identify potential data issues
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def analyze_rewards(dataset_path):
    """Analyze reward distribution in the dataset"""

    print("="*80)
    print("Dataset Reward Analysis - Experiment 1")
    print("="*80)
    print(f"Loading dataset from: {dataset_path}")

    # Load dataset
    dataset = np.load(dataset_path)
    rewards = dataset['rewards']

    print(f"\n{'='*80}")
    print("Basic Statistics")
    print("="*80)
    print(f"Shape: {rewards.shape}")
    print(f"Total samples: {len(rewards)}")
    print(f"Min: {rewards.min():.4f}")
    print(f"Max: {rewards.max():.4f}")
    print(f"Mean: {rewards.mean():.4f}")
    print(f"Std: {rewards.std():.4f}")
    print(f"Median: {np.median(rewards):.4f}")

    print(f"\n{'='*80}")
    print("Percentile Analysis")
    print("="*80)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(rewards, p)
        print(f"  {p:3d}%: {value:8.4f}")

    print(f"\n{'='*80}")
    print("Outlier Detection")
    print("="*80)

    # Check for outliers outside expected range [0, 100]
    outliers_low = (rewards < 0).sum()
    outliers_high = (rewards > 100).sum()
    total = len(rewards)

    print(f"Expected range: [0, 100]")
    print(f"  < 0:    {outliers_low:8d} ({100*outliers_low/total:6.2f}%)")
    print(f"  > 100:  {outliers_high:8d} ({100*outliers_high/total:6.2f}%)")
    print(f"  Valid:  {total - outliers_low - outliers_high:8d} ({100*(total - outliers_low - outliers_high)/total:6.2f}%)")

    # Check for extreme outliers
    q1 = np.percentile(rewards, 25)
    q3 = np.percentile(rewards, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    extreme_outliers = ((rewards < lower_bound) | (rewards > upper_bound)).sum()
    print(f"\nExtreme outliers (3×IQR method):")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"  Count: {extreme_outliers} ({100*extreme_outliers/total:.2f}%)")

    return rewards

def plot_distribution(rewards, output_path):
    """Create distribution plots"""

    print(f"\n{'='*80}")
    print("Creating Distribution Plots")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram
    axes[0, 0].hist(rewards, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].axvline(rewards.mean(), color='red', linestyle='--', label=f'Mean: {rewards.mean():.2f}')
    axes[0, 0].axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cumulative distribution
    axes[0, 1].hist(rewards, bins=100, cumulative=True, density=True,
                    edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Box plot
    axes[1, 0].boxplot(rewards, vert=True)
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Box Plot (Outlier Detection)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Zoomed histogram (focus on main distribution)
    q1, q99 = np.percentile(rewards, [1, 99])
    mask = (rewards >= q1) & (rewards <= q99)
    axes[1, 1].hist(rewards[mask], bins=100, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Zoomed Distribution (1%-99% percentile)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    return fig

if __name__ == "__main__":
    # Dataset path
    dataset_path = 'data/datasets/offline/mix_divpen/mix_divpen_v2_b3_data_d4rl.npz'
    output_path = 'test/offlinetest/reward_distribution_analysis.png'

    # Analyze rewards
    rewards = analyze_rewards(dataset_path)

    # Create plots
    plot_distribution(rewards, output_path)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  - Mean reward: {rewards.mean():.2f}")
    print(f"  - Std deviation: {rewards.std():.2f}")

    # Interpretation
    print(f"\nInterpretation:")
    if rewards.min() < 0:
        print(f"  ⚠️  WARNING: Found negative rewards (min={rewards.min():.2f})")
        print(f"      This suggests diversity penalty is creating negative values")
    if rewards.max() > 100:
        print(f"  ⚠️  WARNING: Found rewards > 100 (max={rewards.max():.2f})")
        print(f"      This suggests reward normalization may be incorrect")
    if rewards.min() >= 0 and rewards.max() <= 100:
        print(f"  ✅ Rewards are within expected range [0, 100]")
        print(f"      Normalization to [0, 1.0] should work correctly")
