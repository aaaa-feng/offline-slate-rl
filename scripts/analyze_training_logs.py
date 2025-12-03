#!/usr/bin/env python3
"""
分析训练日志，提取关键信息
"""
import os
import re
import sys
from pathlib import Path

# 添加项目路径
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from config import paths

def analyze_log(log_path):
    """分析单个日志文件"""
    result = {
        'filename': os.path.basename(log_path),
        'completed': False,
        'progress': 0,
        'last_step': 0,
        'max_step': 100001,
        'best_val_reward': None,
        'final_val_reward': None,
        'test_reward': None,
        'val_rewards': []
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # 查找所有训练步数
        train_steps = re.findall(r'\[Training Step (\d+)/(\d+)\]', content)
        if train_steps:
            last_step, max_step = train_steps[-1]
            result['last_step'] = int(last_step)
            result['max_step'] = int(max_step)
            result['progress'] = (result['last_step'] / result['max_step']) * 100

        # 查找所有验证奖励
        val_rewards = re.findall(r'Mean Reward:\s+([\d.]+)', content)
        if val_rewards:
            result['val_rewards'] = [float(r) for r in val_rewards]
            result['best_val_reward'] = max(result['val_rewards'])
            result['final_val_reward'] = result['val_rewards'][-1]

        # 查找test_reward
        test_rewards = re.findall(r'test_reward[:\s]+([\d.]+)', content)
        if test_rewards:
            result['test_reward'] = float(test_rewards[-1])
            result['completed'] = True

        # 如果没有test_reward但进度100%，也算完成
        if result['progress'] >= 99.9:
            result['completed'] = True

    except Exception as e:
        result['error'] = str(e)

    return result

def main():
    # 使用动态路径配置
    log_dir = paths.get_log_dir("log_58407201") / "SAC_GeMS"

    print("="*100)
    print("SAC+GeMS 训练日志分析报告")
    print("="*100)
    print()

    # 分析所有日志
    results = []
    for log_file in sorted(log_dir.glob("*.log")):
        result = analyze_log(log_file)
        results.append(result)

    # 按环境分组
    envs = {}
    for result in results:
        # 提取环境名称和参数
        filename = result['filename']
        parts = filename.replace('.log', '').split('_')

        if len(parts) >= 4:
            env_type = parts[0]  # diffuse/focused
            env_name = parts[1]  # topdown/mix/divpen
            full_env = f"{env_type}_{env_name}"
            kl = parts[2]  # KL0.5/KL1.0
            click = parts[3]  # click0.2/click0.5

            if full_env not in envs:
                envs[full_env] = []

            envs[full_env].append({
                'params': f"{kl}_{click}",
                'result': result
            })

    # 打印结果
    for env_name in sorted(envs.keys()):
        print(f"\n{'='*100}")
        print(f"环境: {env_name}")
        print(f"{'='*100}")

        configs = envs[env_name]

        # 表头
        print(f"{'参数配置':<25} {'状态':<10} {'进度':<10} {'最佳Val':<12} {'最终Val':<12} {'Test':<12}")
        print("-"*100)

        for config in configs:
            params = config['params']
            r = config['result']

            status = "✅ 完成" if r['completed'] else "⏳ 运行中"
            progress = f"{r['progress']:.1f}%"
            best_val = f"{r['best_val_reward']:.2f}" if r['best_val_reward'] else "N/A"
            final_val = f"{r['final_val_reward']:.2f}" if r['final_val_reward'] else "N/A"
            test = f"{r['test_reward']:.2f}" if r['test_reward'] else "N/A"

            print(f"{params:<25} {status:<10} {progress:<10} {best_val:<12} {final_val:<12} {test:<12}")

    # 总结
    print(f"\n{'='*100}")
    print("总结")
    print(f"{'='*100}")

    completed = sum(1 for r in results if r['completed'])
    running = len(results) - completed

    print(f"总任务数: {len(results)}")
    print(f"已完成: {completed}")
    print(f"运行中: {running}")

    # 找出最佳配置
    print(f"\n{'='*100}")
    print("最佳配置推荐（基于验证性能）")
    print(f"{'='*100}")

    for env_name in sorted(envs.keys()):
        configs = envs[env_name]

        # 找出该环境下最佳的配置
        best_config = None
        best_reward = -float('inf')

        for config in configs:
            r = config['result']
            if r['best_val_reward'] and r['best_val_reward'] > best_reward:
                best_reward = r['best_val_reward']
                best_config = config

        if best_config:
            print(f"\n{env_name}:")
            print(f"  最佳配置: {best_config['params']}")
            print(f"  最佳验证奖励: {best_config['result']['best_val_reward']:.2f}")
            if best_config['result']['test_reward']:
                print(f"  测试奖励: {best_config['result']['test_reward']:.2f}")
            print(f"  状态: {'✅ 已完成' if best_config['result']['completed'] else '⏳ 运行中'}")

    # 原文标准对比
    print(f"\n{'='*100}")
    print("与原文标准对比")
    print(f"{'='*100}")
    print("\n原文中SAC+GeMS的性能（Table 2）:")
    print("  diffuse_topdown: ~450 (估计)")
    print("  diffuse_mix: ~350 (估计)")
    print("  diffuse_divpen: ~300 (估计)")
    print("\n注: 原文没有直接给出SAC+GeMS的具体数值，以上是根据图表估计")

if __name__ == "__main__":
    main()
