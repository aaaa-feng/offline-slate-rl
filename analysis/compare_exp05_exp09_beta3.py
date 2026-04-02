#!/usr/bin/env python3
"""
对比分析 exp05 (tau=0.7, beta=3.0) vs exp09 (tau=0.9, beta=3.0) 实验结果

实验配置:
- exp05: expectile=0.7, beta=3.0, 3 seeds
- exp09: expectile=0.9, beta=3.0, 3 seeds
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import statistics

@dataclass
class EvalPoint:
    step: int
    mean_reward: float
    std_reward: float
    median_reward: float
    iqm_reward: float
    min_reward: float
    max_reward: float

@dataclass
class ExperimentResult:
    seed: int
    log_file: str
    expectile: float
    beta: float
    final_eval: Optional[Dict] = None
    best_iqm: float = 0.0
    best_iqm_step: int = 0
    eval_points: List[EvalPoint] = None
    
    def __post_init__(self):
        if self.eval_points is None:
            self.eval_points = []

def parse_log_file(log_path: str) -> ExperimentResult:
    """解析单个日志文件"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取实验配置
    expectile_match = re.search(r'Expectile:\s*([\d.]+)', content)
    beta_match = re.search(r'Beta:\s*([\d.]+)', content)
    seed_match = re.search(r'seed(\d+)_\d{8}_\d{6}', log_path)
    
    expectile = float(expectile_match.group(1)) if expectile_match else 0.7
    beta = float(beta_match.group(1)) if beta_match else 1.0
    seed = int(seed_match.group(1)) if seed_match else 0
    
    result = ExperimentResult(
        seed=seed,
        log_file=log_path,
        expectile=expectile,
        beta=beta
    )
    
    # 提取评估结果 (支持两种格式)
    # 格式 1: [Evaluation] Step X/20000\n  Mean Reward: Y ± Z (min=A, max=B)\n  Median Reward: M | IQM: I
    # 格式 2 (带时间戳): [2026-03-03 07:58:12,203] INFO: [Evaluation] Step 500/20000
    #                   [2026-03-03 07:58:12,204] INFO:   Mean Reward: 62.58 ± 23.51 (min=18.00, max=115.00)
    #                   [2026-03-03 07:58:12,204] INFO:   Median Reward: 58.50 | IQM: 60.16
    
    # 多行匹配模式 (带时间戳格式)
    eval_pattern = r'\[Evaluation\] Step (\d+)/\d+\n.*?Mean Reward:\s*([\d.]+)\s*±\s*([\d.]+)\s*\(min=([\d.]+),\s*max=([\d.]+)\)\n.*?Median Reward:\s*([\d.]+)\s*\|\s*IQM:\s*([\d.]+)'
    
    for match in re.finditer(eval_pattern, content, re.DOTALL):
        step = int(match.group(1))
        mean = float(match.group(2))
        std = float(match.group(3))
        min_r = float(match.group(4))
        max_r = float(match.group(5))
        median = float(match.group(6))
        iqm = float(match.group(7))
        
        eval_point = EvalPoint(
            step=step,
            mean_reward=mean,
            std_reward=std,
            median_reward=median,
            iqm_reward=iqm,
            min_reward=min_r,
            max_reward=max_r
        )
        result.eval_points.append(eval_point)
        
        # 追踪最佳 IQM
        if iqm > result.best_iqm:
            result.best_iqm = iqm
            result.best_iqm_step = step
    
    # 提取最终评估结果 (Final Evaluation)
    final_pattern = r'Final Evaluation.*?Mean Reward:\s*([\d.]+)\s*±\s*([\d.]+).*?Median Reward:\s*([\d.]+).*?IQM:\s*([\d.]+)'
    final_match = re.search(final_pattern, content, re.DOTALL)
    if final_match:
        result.final_eval = {
            'mean': float(final_match.group(1)),
            'std': float(final_match.group(2)),
            'median': float(final_match.group(3)),
            'iqm': float(final_match.group(4))
        }
    
    # 如果没有找到 Final Evaluation，使用最后一个评估点
    if not result.final_eval and result.eval_points:
        last = result.eval_points[-1]
        result.final_eval = {
            'mean': last.mean_reward,
            'std': last.std_reward,
            'median': last.median_reward,
            'iqm': last.iqm_reward
        }
    
    return result

def compute_group_stats(results: List[ExperimentResult]) -> Dict:
    """计算一组实验的统计信息"""
    if not results:
        return {}
    
    final_iqms = [r.final_eval['iqm'] for r in results if r.final_eval]
    best_iqms = [r.best_iqm for r in results]
    
    return {
        'num_seeds': len(results),
        'final_iqm_mean': statistics.mean(final_iqms) if final_iqms else 0,
        'final_iqm_std': statistics.stdev(final_iqms) if len(final_iqms) > 1 else 0,
        'final_iqm_min': min(final_iqms) if final_iqms else 0,
        'final_iqm_max': max(final_iqms) if final_iqms else 0,
        'best_iqm_mean': statistics.mean(best_iqms) if best_iqms else 0,
        'best_iqm_std': statistics.stdev(best_iqms) if len(best_iqms) > 1 else 0,
        'best_iqm_max': max(best_iqms) if best_iqms else 0,
    }

def main():
    # 定义日志文件路径
    log_files = {
        'exp05_beta3': [
            'offline-slate-rl/test/offlinetest/iql/iql_exp05_beta3_iqm_seed58407201_20260303_075203.log',
            'offline-slate-rl/test/offlinetest/iql/iql_exp05_beta3_iqm_seed12345_20260303_075212.log',
            'offline-slate-rl/test/offlinetest/iql/iql_exp05_beta3_iqm_seed42_20260303_075220.log',
        ],
        'exp09_tau09_beta3': [
            'offline-slate-rl/test/offlinetest/iql/iql_exp09_tau0.9_beta3.0_seed58407201_20260303_085902.log',
            'offline-slate-rl/test/offlinetest/iql/iql_exp09_tau0.9_beta3.0_seed12345_20260303_085937.log',
            'offline-slate-rl/test/offlinetest/iql/iql_exp09_tau0.9_beta3.0_seed42_20260303_085951.log',
        ]
    }
    
    # 解析所有日志
    all_results = {}
    for group_name, files in log_files.items():
        results = []
        for f in files:
            try:
                result = parse_log_file(f)
                results.append(result)
                print(f"✓ 解析成功：{f}")
                print(f"  Seed={result.seed}, Expectile={result.expectile}, Beta={result.beta}")
                print(f"  评估点数：{len(result.eval_points)}, 最佳 IQM={result.best_iqm:.2f} @ step {result.best_iqm_step}")
            except Exception as e:
                print(f"✗ 解析失败：{f} - {e}")
        all_results[group_name] = results
    
    # 计算统计信息
    print("\n" + "="*80)
    print("实验对比报告")
    print("="*80)
    
    for group_name, results in all_results.items():
        stats = compute_group_stats(results)
        print(f"\n{group_name}:")
        print(f"  配置：expectile={results[0].expectile if results else 'N/A'}, beta={results[0].beta if results else 'N/A'}")
        print(f"  种子数：{stats.get('num_seeds', 0)}")
        print(f"  Final IQM: {stats.get('final_iqm_mean', 0):.2f} ± {stats.get('final_iqm_std', 0):.2f}")
        print(f"  Final IQM 范围：[{stats.get('final_iqm_min', 0):.2f}, {stats.get('final_iqm_max', 0):.2f}]")
        print(f"  Best IQM: {stats.get('best_iqm_mean', 0):.2f} ± {stats.get('best_iqm_std', 0):.2f}")
        print(f"  Best IQM 最大值：{stats.get('best_iqm_max', 0):.2f}")
    
    # 跨组对比
    print("\n" + "="*80)
    print("跨组对比")
    print("="*80)
    
    exp05_stats = compute_group_stats(all_results.get('exp05_beta3', []))
    exp09_stats = compute_group_stats(all_results.get('exp09_tau09_beta3', []))
    
    print(f"\n{'指标':<25} {'exp05 (τ=0.7, β=3.0)':<25} {'exp09 (τ=0.9, β=3.0)':<25}")
    print("-"*75)
    print(f"{'Final IQM Mean':<25} {exp05_stats.get('final_iqm_mean', 0):>10.2f}{'':>14} {exp09_stats.get('final_iqm_mean', 0):>10.2f}")
    print(f"{'Final IQM Std':<25} {exp05_stats.get('final_iqm_std', 0):>10.2f}{'':>14} {exp09_stats.get('final_iqm_std', 0):>10.2f}")
    print(f"{'Best IQM Mean':<25} {exp05_stats.get('best_iqm_mean', 0):>10.2f}{'':>14} {exp09_stats.get('best_iqm_mean', 0):>10.2f}")
    print(f"{'Best IQM Max':<25} {exp05_stats.get('best_iqm_max', 0):>10.2f}{'':>14} {exp09_stats.get('best_iqm_max', 0):>10.2f}")
    
    # 保存详细结果到 JSON
    output = {
        'exp05_beta3': {
            'config': {'expectile': 0.7, 'beta': 3.0},
            'seeds': [asdict(r) for r in all_results.get('exp05_beta3', [])],
            'stats': exp05_stats
        },
        'exp09_tau09_beta3': {
            'config': {'expectile': 0.9, 'beta': 3.0},
            'seeds': [asdict(r) for r in all_results.get('exp09_tau09_beta3', [])],
            'stats': exp09_stats
        }
    }
    
    # 移除 eval_points 以简化 JSON 输出
    for group in output.values():
        for seed_data in group['seeds']:
            eval_pts = seed_data.get('eval_points', [])
            if eval_pts and isinstance(eval_pts[0], dict):
                seed_data['eval_points'] = [{'step': p.get('step', 0), 'iqm': p.get('iqm_reward', 0), 'mean': p.get('mean_reward', 0)} for p in eval_pts]
            else:
                seed_data['eval_points'] = [{'step': p.step, 'iqm': p.iqm_reward, 'mean': p.mean_reward} for p in eval_pts] if eval_pts else []
    
    output_path = 'offline-slate-rl/analysis/exp05_exp09_beta3_comparison.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细结果已保存到：{output_path}")


if __name__ == '__main__':
    main()
