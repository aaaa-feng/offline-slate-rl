#!/usr/bin/env python3
"""
全面数据收集测试结果分析脚本
分析 Expert-Hard, Expert-ε, Random-Uniform, Random-Hard 四种方式的数据质量
"""
import os
import re
import pandas as pd
import numpy as np

def parse_log_file(log_path):
    """解析日志文件，提取关键指标"""
    metrics = {
        'avg_return': None,
        'std_return': None,
        'min_return': None,
        'max_return': None,
        'reward_std': None,
        'consecutive_overlap': None,
        'top10_coverage': None
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

            # 提取平均回报
            match = re.search(r'avg_episode_return[:\s]+([0-9.]+)', content)
            if match:
                metrics['avg_return'] = float(match.group(1))

            # 提取标准差
            match = re.search(r'std_episode_return[:\s]+([0-9.]+)', content)
            if match:
                metrics['std_return'] = float(match.group(1))

            # 提取最小/最大回报
            match = re.search(r'min_episode_return[:\s]+([0-9.]+)', content)
            if match:
                metrics['min_return'] = float(match.group(1))

            match = re.search(r'max_episode_return[:\s]+([0-9.]+)', content)
            if match:
                metrics['max_return'] = float(match.group(1))

            # 提取数据质量指标
            match = re.search(r'Reward.*标准差.*?([0-9.]+)', content)
            if match:
                metrics['reward_std'] = float(match.group(1))

            match = re.search(r'连续 Slate 重叠率[:\s]+([0-9.]+)%', content)
            if match:
                metrics['consecutive_overlap'] = float(match.group(1))

            match = re.search(r'Top-10% 物品覆盖率[:\s]+([0-9.]+)%', content)
            if match:
                metrics['top10_coverage'] = float(match.group(1))

    except Exception as e:
        print(f"解析 {log_path} 失败: {e}")

    return metrics
