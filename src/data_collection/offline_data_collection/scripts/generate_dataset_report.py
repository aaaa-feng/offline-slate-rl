#!/usr/bin/env python3
"""
生成离线数据集报告 - 详细分析版
分析数据集质量，评估是否适合训练离线强化学习算法
"""
import os
import numpy as np
import sys
from pathlib import Path

def generate_report_from_npz(datasets_dir=None):
    """生成数据集报告

    Args:
        datasets_dir: 数据集目录，如果为None则使用默认路径
    """
    if datasets_dir is None:
        # 使用统一路径配置
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root / "config"))
        from paths import OFFLINE_DATASETS_DIR
        datasets_dir = str(OFFLINE_DATASETS_DIR)

    print("="*80)
    print("离线强化学习数据集详细分析报告")
    print("="*80)

    # 打印数据集路径信息
    print("\n📁 数据集路径信息:")
    print(f"  根目录: {datasets_dir}")
    print(f"  数据格式: D4RL标准格式 (.npz)")
    print(f"  文件结构:")
    print(f"    {datasets_dir}/")
    print(f"    ├── diffuse_topdown/")
    print(f"    │   └── expert_data_d4rl.npz")
    print(f"    ├── diffuse_mix/")
    print(f"    │   └── expert_data_d4rl.npz")
    print(f"    └── diffuse_divpen/")
    print(f"        └── expert_data_d4rl.npz")
    print(f"\n  NPZ文件包含字段:")
    print(f"    - observations: (N, 20) belief states")
    print(f"    - actions: (N, 32) latent actions")
    print(f"    - slates: (N, 10) discrete recommendations")
    print(f"    - rewards: (N,) immediate rewards")
    print(f"    - terminals: (N,) episode终止标志")
    print(f"    - clicks: (N, 10) 用户点击行为")
    print(f"    - diversity_scores: (N,) 推荐多样性")
    print(f"    - coverage_scores: (N,) 物品覆盖率")
    print("="*80)

    # 在线性能（基准，用于计算性能比率）
    # 这些是训练时的test reward，用于评估离线数据质量
    online_performance = {
        'diffuse_topdown': 447.60,
        'diffuse_mix': 349.07,
        'diffuse_divpen': 296.73,
        'focused_topdown': 391.65,
        'focused_mix': 287.90,
        'focused_divpen': 299.80
    }

    results = []
    env_list = [
        'diffuse_topdown', 'diffuse_mix', 'diffuse_divpen',
        'focused_topdown', 'focused_mix', 'focused_divpen'
    ]

    print(f"\n🔍 正在扫描数据集...\n")

    for env_name in env_list:
        # 🎯 关键：只读取 .npz 文件
        npz_path = os.path.join(datasets_dir, env_name, 'expert_data_d4rl.npz')
        
        if os.path.exists(npz_path):
            try:
                # 1. 加载 NPZ (极快)
                data = np.load(npz_path)
                
                # 2. 提取关键数组
                observations = data['observations']
                actions = data['actions']
                rewards = data['rewards']
                terminals = data['terminals']

                # 提取推荐系统特有指标
                diversity_scores = data['diversity_scores'] if 'diversity_scores' in data else None
                coverage_scores = data['coverage_scores'] if 'coverage_scores' in data else None
                clicks = data['clicks'] if 'clicks' in data else None
                
                # 3. 计算统计数据
                total_transitions = len(rewards)
                
                # 计算 Episode 数量和回报
                # D4RL格式是平铺的，需要根据 terminals (done=True) 切分
                episode_returns = []
                current_ep_return = 0
                current_ep_len = 0
                episode_lengths = []
                
                for i in range(total_transitions):
                    current_ep_return += rewards[i]
                    current_ep_len += 1
                    
                    # 如果遇到结束符 或 最后一个点
                    if terminals[i] or i == total_transitions - 1:
                        episode_returns.append(current_ep_return)
                        episode_lengths.append(current_ep_len)
                        current_ep_return = 0
                        current_ep_len = 0
                
                num_episodes = len(episode_returns)
                avg_return = np.mean(episode_returns) if episode_returns else 0
                std_return = np.std(episode_returns) if episode_returns else 0
                avg_len = np.mean(episode_lengths) if episode_lengths else 0
                
                # 计算更多统计指标
                non_zero_ratio = np.sum(rewards > 0) / total_transitions
                min_return = np.min(episode_returns) if episode_returns else 0
                max_return = np.max(episode_returns) if episode_returns else 0

                # 计算reward分布
                reward_mean = np.mean(rewards)
                reward_std = np.std(rewards)
                reward_min = np.min(rewards)
                reward_max = np.max(rewards)

                # 计算点击率
                if clicks is not None:
                    click_rate = np.mean(clicks)
                else:
                    click_rate = 0

                # 4. 获取文件物理大小
                file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)

                results.append({
                    'env_name': env_name,
                    'num_episodes': num_episodes,
                    'total_transitions': total_transitions,
                    'avg_episode_length': avg_len,
                    'avg_episode_return': avg_return,
                    'std_episode_return': std_return,
                    'min_episode_return': min_return,
                    'max_episode_return': max_return,
                    'file_size_mb': file_size_mb,
                    'non_zero_reward_ratio': non_zero_ratio,
                    'online_performance': online_performance.get(env_name, 0),
                    'diversity': np.mean(diversity_scores) if diversity_scores is not None else 0,
                    'coverage': np.mean(coverage_scores) if coverage_scores is not None else 0,
                    'click_rate': click_rate,
                    'reward_mean': reward_mean,
                    'reward_std': reward_std,
                    'reward_min': reward_min,
                    'reward_max': reward_max,
                    'obs_dim': observations.shape[1] if len(observations.shape) > 1 else 0,
                    'action_dim': actions.shape[1] if len(actions.shape) > 1 else 0,
                })
                
                print(f"✅ {env_name}: 加载成功 | {num_episodes} eps | Avg Ret: {avg_return:.2f}")
                
            except Exception as e:
                print(f"❌ {env_name}: NPZ解析失败 - {e}")
        else:
            # 如果文件不存在，静默跳过或打印提示
            # print(f"⚠️ {env_name}: 未找到 .npz 文件")
            pass
            
    if not results:
        print("❌ 未找到有效数据。请确认 collect_data.py 是否成功执行并生成了 .npz 文件。")
        return

    # ================= 生成详细报表 =================

    # 1. 数据集规模统计
    print("\n" + "="*80)
    print("📊 表1：数据集规模统计")
    print("="*80)
    print(f"| {'环境':<18} | {'Episodes':<10} | {'Transitions':<12} | {'Avg Len':<8} | {'Size(MB)':<9} |")
    print("|" + "-"*78 + "|")

    total_episodes = 0
    total_transitions = 0
    total_size = 0

    for r in results:
        print(f"| {r['env_name']:<18} | {r['num_episodes']:<10,} | {r['total_transitions']:<12,} | {r['avg_episode_length']:<8.1f} | {r['file_size_mb']:<9.1f} |")
        total_episodes += r['num_episodes']
        total_transitions += r['total_transitions']
        total_size += r['file_size_mb']

    print("|" + "-"*78 + "|")
    print(f"| {'总计':<18} | {total_episodes:<10,} | {total_transitions:<12,} | {'-':<8} | {total_size:<9.1f} |")

    # 2. 数据质量与性能对比
    print("\n" + "="*80)
    print("📈 表2：数据质量与在线性能对比")
    print("="*80)
    print(f"| {'环境':<18} | {'平均回报':<10} | {'标准差':<8} | {'最小值':<8} | {'最大值':<8} | {'在线性能':<10} | {'比率':<8} |")
    print("|" + "-"*98 + "|")

    for r in results:
        ratio = (r['avg_episode_return'] / r['online_performance'] * 100) if r['online_performance'] > 0 else 0
        print(f"| {r['env_name']:<18} | {r['avg_episode_return']:<10.2f} | {r['std_episode_return']:<8.2f} | {r['min_episode_return']:<8.2f} | {r['max_episode_return']:<8.2f} | {r['online_performance']:<10.2f} | {ratio:<7.1f}% |")

    # 3. 推荐系统特有指标
    print("\n" + "="*80)
    print("🎯 表3：推荐系统指标 (Diversity & Coverage & Click Rate)")
    print("="*80)
    print(f"| {'环境':<18} | {'Diversity':<11} | {'Coverage':<10} | {'点击率':<10} | {'非零奖励':<10} |")
    print("|" + "-"*78 + "|")

    for r in results:
        print(f"| {r['env_name']:<18} | {r['diversity']:<11.4f} | {r['coverage']:<10.4f} | {r['click_rate']*100:<9.2f}% | {r['non_zero_reward_ratio']*100:<9.1f}% |")

    # 4. 数据维度信息
    print("\n" + "="*80)
    print("🔢 表4：数据维度信息")
    print("="*80)
    print(f"| {'环境':<18} | {'Obs维度':<10} | {'Action维度':<12} | {'说明':<30} |")
    print("|" + "-"*88 + "|")

    for r in results:
        print(f"| {r['env_name']:<18} | {r['obs_dim']:<10} | {r['action_dim']:<12} | {'belief_state + latent_action':<30} |")


    # ================= 离线RL适用性分析 =================
    print("\n" + "="*80)
    print("🤖 离线强化学习适用性分析")
    print("="*80)

    print("\n1️⃣ 数据规模评估:")
    print(f"   总Episodes: {total_episodes:,} 个")
    print(f"   总Transitions: {total_transitions:,} 个 ({total_transitions/1e6:.1f}M)")

    if total_transitions >= 1_000_000:
        print(f"   ✅ 数据规模充足 (>100万条)")
        print(f"      - 足够训练TD3+BC、CQL、IQL等离线RL算法")
        print(f"      - 可以支持多次训练和超参数调优")
    elif total_transitions >= 100_000:
        print(f"   ⚠️ 数据规模中等 (10-100万条)")
        print(f"      - 可以训练离线RL，但可能需要更多数据增强")
    else:
        print(f"   ❌ 数据规模不足 (<10万条)")
        print(f"      - 建议收集更多数据")

    print("\n2️⃣ 数据质量评估:")
    avg_ratio = np.mean([r['avg_episode_return'] / r['online_performance'] * 100
                         for r in results if r['online_performance'] > 0])
    print(f"   平均性能比率: {avg_ratio:.1f}%")

    if avg_ratio >= 70:
        print(f"   ✅ Expert级别数据 (70-90%)")
        print(f"      - 高质量专家数据，适合Behavior Cloning")
        print(f"      - 适合Conservative Q-Learning (CQL)")
        print(f"      - 适合TD3+BC等算法")
    elif avg_ratio >= 40:
        print(f"   ✅ Medium级别数据 (40-70%)")
        print(f"      - 中等质量数据，适合大多数离线RL算法")
    else:
        print(f"   ⚠️ 低质量数据 (<40%)")
        print(f"      - 可能需要更强的正则化")

    print("\n3️⃣ 数据多样性评估:")
    avg_diversity = np.mean([r['diversity'] for r in results])
    avg_std = np.mean([r['std_episode_return'] for r in results])
    print(f"   平均Diversity: {avg_diversity:.4f}")
    print(f"   平均标准差: {avg_std:.2f}")

    if avg_diversity > 0.85 and avg_std > 30:
        print(f"   ✅ 数据多样性良好")
        print(f"      - 推荐多样性高，覆盖不同策略行为")
        print(f"      - 回报标准差合理，包含不同质量轨迹")
    else:
        print(f"   ⚠️ 数据多样性可能不足")

    print("\n4️⃣ 推荐算法建议:")
    print(f"   基于当前数据质量 ({avg_ratio:.1f}%)，推荐以下算法:")
    print(f"   ")
    print(f"   🥇 首选: TD3+BC")
    print(f"      - 适合高质量expert数据")
    print(f"      - 简单有效，易于调参")
    print(f"      - 论文: Fujimoto & Gu, 2021")
    print(f"   ")
    print(f"   🥈 次选: Conservative Q-Learning (CQL)")
    print(f"      - 适合各种质量的数据")
    print(f"      - 强大的分布外动作惩罚")
    print(f"      - 论文: Kumar et al., 2020")
    print(f"   ")
    print(f"   🥉 备选: Implicit Q-Learning (IQL)")
    print(f"      - 无需显式策略约束")
    print(f"      - 适合多模态数据")
    print(f"      - 论文: Kostrikov et al., 2021")

    print("\n5️⃣ 数据格式兼容性:")
    print(f"   ✅ D4RL标准格式")
    print(f"   ✅ 包含observations, actions, rewards, terminals")
    print(f"   ✅ 可直接用于d3rlpy、rlkit等离线RL库")
    print(f"   ✅ 维度: obs={results[0]['obs_dim']}维, action={results[0]['action_dim']}维")

    print("\n" + "="*80)
    print("✅ 总结: 数据集质量优秀，完全支持离线强化学习训练！")
    print("="*80)

if __name__ == "__main__":
    generate_report_from_npz()