"""
数据集合并工具
用于将多个子数据集合并为单个混合数据集
"""
import numpy as np
from pathlib import Path
from typing import List


def merge_datasets(dataset_paths: List[str], output_name: str, env_name: str) -> str:
    """
    合并多个数据集为单个混合数据集

    Args:
        dataset_paths: 数据集路径列表
        output_name: 输出数据集名称
        env_name: 环境名称

    Returns:
        合并后的数据集路径
    """
    print(f"合并 {len(dataset_paths)} 个子数据集...")

    # 初始化数据容器
    all_data = {
        'slates': [],
        'clicks': [],
        'rewards': [],
        'next_slates': [],
        'next_clicks': [],
        'terminals': [],
        'episode_ids': [],
        'timesteps': []
    }

    all_oracle = {
        'item_relevances': [],
        'user_states': [],
        'user_bored': [],
        'episode_ids': []
    }

    # 读取并合并所有子数据集
    episode_id_offset = 0
    for i, path in enumerate(dataset_paths):
        print(f"  [{i+1}/{len(dataset_paths)}] 加载: {path}")

        # 加载核心数据
        data = np.load(path)
        for key in all_data.keys():
            if key == 'episode_ids':
                # 调整episode_ids避免冲突
                adjusted_ids = data[key] + episode_id_offset
                all_data[key].append(adjusted_ids)
                episode_id_offset = adjusted_ids.max() + 1
            else:
                all_data[key].append(data[key])

        # 加载Oracle数据
        oracle_path = path.replace('_data_d4rl.npz', '_oracle.npz')
        if Path(oracle_path).exists():
            oracle = np.load(oracle_path)
            for key in all_oracle.keys():
                if key == 'episode_ids':
                    all_oracle[key].append(data['episode_ids'] + episode_id_offset - data['episode_ids'].max() - 1)
                else:
                    all_oracle[key].append(oracle[key])

    # 合并数组
    print("  合并数组...")
    merged_data = {key: np.concatenate(arrays) for key, arrays in all_data.items()}
    merged_oracle = {key: np.concatenate(arrays) for key, arrays in all_oracle.items() if arrays}

    # 保存合并后的数据集
    output_dir = f"data/datasets/offline/{env_name}"
    output_path = f"{output_dir}/{output_name}_data_d4rl.npz"
    oracle_output_path = f"{output_dir}/{output_name}_oracle.npz"

    print(f"  保存核心数据: {output_path}")
    np.savez_compressed(output_path, **merged_data)

    if merged_oracle:
        print(f"  保存Oracle数据: {oracle_output_path}")
        np.savez_compressed(oracle_output_path, **merged_oracle)

    print(f"  ✅ 合并完成，总转换数: {len(merged_data['rewards'])}")

    return output_path
