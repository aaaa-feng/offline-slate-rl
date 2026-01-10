import numpy as np
import os

# 替换为你实际的 V3 数据集路径 (任选一个 expert 数据集)
data_path = "/data/liyuefeng/offline-slate-rl/data/datasets/offline/diffuse_mix/expert_data_d4rl.npz"


if not os.path.exists(data_path):
    print(f"❌ 文件不存在: {data_path}")
else:
    data = np.load(data_path, allow_pickle=True)
    print(f"=== 数据集文件: {os.path.basename(data_path)} ===")
    print(f"包含的 Keys: {list(data.keys())}")
    
    # 检查决定轨迹切分的关键字段
    print("\n--- 轨迹切分相关字段检查 ---")
    if 'episode_ids' in data:
        eid = data['episode_ids']
        print(f"✅ 发现 'episode_ids'")
        print(f"  Shape: {eid.shape}")
        print(f"  前 20 个值: {eid[:20]}")
        print(f"  Unique IDs 数量: {len(np.unique(eid))}")
    else:
        print("❌ 未发现 'episode_ids'")

    if 'terminals' in data:
        term = data['terminals']
        print(f"✅ 发现 'terminals'")
        print(f"  Shape: {term.shape}")
        print(f"  True 的数量 (Episode 结束次数): {np.sum(term)}")
    
    if 'timeouts' in data:
        time = data['timeouts']
        print(f"✅ 发现 'timeouts'")
        print(f"  True 的数量: {np.sum(time)}")

    # 检查 GRU 需要的字段
    print("\n--- GRU 输入字段检查 ---")
    for field in ['slates', 'clicks']:
        if field in data:
            print(f"✅ {field}: Shape={data[field].shape}, Type={data[field].dtype}")
        else:
            print(f"❌ 缺失 {field}")