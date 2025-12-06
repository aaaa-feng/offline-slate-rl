# Medium 数据集收集操作指南

**日期**: 2025-12-06
**目的**: 使用训练到 50,000 步的 SAC+GeMS 模型收集 Medium 质量的离线数据集

---

## 一、训练阶段（已完成）

### 1.1 训练配置

**训练脚本**: `scripts/batch_runs/run_medium_collection_training.sh`

**关键参数**:
- `--max_steps=100000` - 训练 100k 步
- `--save_every_n_steps=50000` - 在第 50,000 步保存模型
- `--check_val_every_n_epoch=1000` - 每 1000 个 episode 验证一次
- `--lambda_KL=1.0`, `--lambda_click=0.5` - 论文官方参数

**训练环境** (6个):
1. `diffuse_topdown` - Diffuse 用户 + Top-down 点击模型
2. `diffuse_mix` - Diffuse 用户 + Mix 点击模型
3. `diffuse_divpen` - Diffuse 用户 + Mix 点击模型 + 高多样性惩罚
4. `focused_topdown` - Focused 用户 + Top-down 点击模型
5. `focused_mix` - Focused 用户 + Mix 点击模型
6. `focused_divpen` - Focused 用户 + Mix 点击模型 + 高多样性惩罚

**模型保存位置**:
```
checkpoints/online_rl/{env_name}/
├── SAC+GeMS_Medium_GeMS_{env}_..._best.ckpt      # 最佳模型
├── SAC+GeMS_Medium_GeMS_{env}_..._step50000.ckpt # 50k步模型 ⭐ 用于数据收集
└── SAC+GeMS_Medium_GeMS_{env}_..._last.ckpt      # 最终模型
```

### 1.2 训练监控

**查看训练进度**:
```bash
# 查看所有环境的最新进度
grep 'Training Step' experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log | tail -20

# 查看验证结果
grep 'VALIDATION' experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log | tail -20

# 实时监控某个环境
tail -f experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/diffuse_topdown_KL1.0_click0.5_20251206.log
```

**确认训练完成**:
```bash
# 检查是否到达 50,000 步并保存了模型
ls -lh checkpoints/online_rl/*/SAC+GeMS_Medium_*_step50000.ckpt
```

预期输出：6 个环境各有一个 `*_step50000.ckpt` 文件。

---

## 二、数据收集阶段（待执行）

### 2.1 哲学：训练归训练，存储归存储，决策在人

**核心原则**:
1. **训练归训练**: 所有模型统一保存到 `checkpoints/online_rl/{env}/`，带有明确的步数标记
2. **存储归存储**: 数据收集脚本从标准位置读取模型，收集的数据保存到标准位置
3. **决策在人**: 人工检查模型质量，决定是否使用该模型收集数据

### 2.2 模型质量检查（必须步骤）

在开始数据收集前，**必须**检查模型质量：

#### 步骤 1: 查看训练曲线

访问 SwanLab 项目查看训练曲线：
```
https://swanlab.cn/@Cliff/GeMS_RL_Training_202512
```

**检查指标**:
- `train_reward`: 训练 reward 是否稳定上升
- `val_reward`: 验证 reward 是否达到合理水平
- 对比 6 个环境的表现，确认没有异常

#### 步骤 2: 查看日志中的验证结果

```bash
# 查看 Step 49999 附近的验证结果
grep -A 5 "VALIDATION @ Step 49999" experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log
```

**预期结果**:
- `Mean Reward`: 应该显著高于初始值（Step 0 的 reward）
- 不同环境的 reward 范围：
  - Diffuse 环境: 通常较低（用户兴趣分散）
  - Focused 环境: 通常较高（用户兴趣集中）

#### 步骤 3: 确认模型文件完整性

```bash
# 检查所有 50k 步模型是否存在且大小合理
for env in diffuse_topdown diffuse_mix diffuse_divpen focused_topdown focused_mix focused_divpen; do
    echo "=== $env ==="
    ls -lh checkpoints/online_rl/$env/*_step50000.ckpt
done
```

**预期**: 每个文件大小应该相似（约几百 MB），如果某个文件明显偏小或为 0，说明保存失败。

### 2.3 数据收集脚本准备

#### 创建数据收集脚本

创建文件：`scripts/batch_runs/collect_medium_data.sh`

```bash
#!/bin/bash

# =================================================================
# Medium 数据集收集脚本
# =================================================================
# 功能：
# 1. 使用训练到 50k 步的模型收集 Medium 质量数据
# 2. 为 6 个环境各收集 10,000 条轨迹
# 3. 数据保存到 data/offline_datasets/medium/
# =================================================================

# 0. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gems

# 1. 基础配置
GPU_IDS=(1 2 3)
SEED=58407201
NUM_TRAJECTORIES=10000  # 每个环境收集的轨迹数
EPISODE_LENGTH=100

# 2. 定义环境列表
ENVS=(
    "diffuse_topdown"
    "diffuse_mix"
    "diffuse_divpen"
    "focused_topdown"
    "focused_mix"
    "focused_divpen"
)

# 3. 数据保存目录
DATA_BASE_DIR="/data/liyuefeng/offline-slate-rl/data/offline_datasets/medium"
mkdir -p ${DATA_BASE_DIR}

# 4. 日志目录
LOG_BASE_DIR="/data/liyuefeng/offline-slate-rl/experiments/logs/data_collection/medium_$(date +%Y%m%d)"
mkdir -p ${LOG_BASE_DIR}

echo "=== 开始收集 Medium 数据集 ==="
echo "=== 数据将保存到: ${DATA_BASE_DIR}/ ==="
echo "=== 日志将保存到: ${LOG_BASE_DIR}/ ==="
echo ""

# 5. 循环收集数据
for i in "${!ENVS[@]}"; do
    ENV=${ENVS[$i]}

    # 自动分配 GPU
    GPU_IDX=$((i % 3))
    GPU_ID=${GPU_IDS[$GPU_IDX]}

    # 确定 Click Model
    if [[ "$ENV" == *"topdown"* ]]; then
        CLICK_MODEL="tdPBM"
    else
        CLICK_MODEL="mixPBM"
    fi

    # 确定 Diversity Penalty
    if [[ "$ENV" == *"divpen"* ]]; then
        DIV_PENALTY=3.0
    else
        DIV_PENALTY=1.0
    fi

    # 确定 Environment Embeddings
    if [[ "$ENV" == *"diffuse"* ]]; then
        ENV_EMBEDDS="item_embeddings_diffuse.pt"
    else
        ENV_EMBEDDS="item_embeddings_focused.pt"
    fi

    # 模型路径（50k 步模型）
    MODEL_PATH="checkpoints/online_rl/${ENV}/SAC+GeMS_Medium_GeMS_${ENV}_agentseed${SEED}_gamma0.8_step50000.ckpt"

    # 数据保存路径
    DATA_OUTPUT="${DATA_BASE_DIR}/${ENV}_medium_${NUM_TRAJECTORIES}traj.pkl"

    # 日志文件
    LOG_FILE="${LOG_BASE_DIR}/${ENV}_collection.log"

    echo "----------------------------------------------------------------"
    echo "收集数据: ${ENV}"
    echo "  - GPU: ${GPU_ID}"
    echo "  - Model: ${MODEL_PATH}"
    echo "  - Output: ${DATA_OUTPUT}"
    echo "  - Trajectories: ${NUM_TRAJECTORIES}"
    echo "  - Log: ${LOG_FILE}"
    echo "----------------------------------------------------------------"

    # 执行数据收集命令
    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u scripts/collect_offline_data.py \
        --agent=SAC \
        --belief=GRU \
        --ranker=GeMS \
        --item_embedds=scratch \
        --env_name=topics \
        --device=cuda \
        --seed=${SEED} \
        --ranker_seed=${SEED} \
        --model_checkpoint=${MODEL_PATH} \
        --num_trajectories=${NUM_TRAJECTORIES} \
        --episode_length=${EPISODE_LENGTH} \
        --output_path=${DATA_OUTPUT} \
        --latent_dim=32 \
        --lambda_KL=1.0 \
        --lambda_click=0.5 \
        --lambda_prior=0.0 \
        --ranker_embedds=scratch \
        --ranker_sample=False \
        --ranker_dataset=${ENV} \
        --click_model=${CLICK_MODEL} \
        --env_embedds=${ENV_EMBEDDS} \
        --diversity_penalty=${DIV_PENALTY} \
        --belief_state_dim=20 \
        --item_embedd_dim=20 \
        --num_items=1000 \
        --boredom_threshold=5 \
        --recent_items_maxlen=10 \
        --boredom_moving_window=5 \
        --env_omega=0.9 \
        --short_term_boost=1.0 \
        --env_offset=0.28 \
        --env_slope=100 \
        --diversity_threshold=4 \
        --topic_size=2 \
        --num_topics=10 \
        --beliefs actor critic \
        > "${LOG_FILE}" 2>&1 &

    sleep 2
done

echo ""
echo "🎉 所有数据收集任务已启动!"
echo "📁 数据目录: ${DATA_BASE_DIR}/"
echo "📁 日志目录: ${LOG_BASE_DIR}/"
echo ""
echo "监控命令:"
echo "  - tail -f ${LOG_BASE_DIR}/*.log          # 查看收集日志"
echo "  - ls -lh ${DATA_BASE_DIR}/               # 查看已收集的数据文件"
echo ""
```

**注意**:
1. 脚本中的 `MODEL_PATH` 需要根据实际的文件名格式调整
2. `scripts/collect_offline_data.py` 需要确认是否存在，如果不存在需要创建

### 2.4 执行数据收集

#### 步骤 1: 赋予脚本执行权限

```bash
chmod +x scripts/batch_runs/collect_medium_data.sh
```

#### 步骤 2: 启动数据收集

```bash
cd /data/liyuefeng/offline-slate-rl
bash scripts/batch_runs/collect_medium_data.sh
```

#### 步骤 3: 监控数据收集进度

```bash
# 查看实时日志
tail -f experiments/logs/data_collection/medium_20251206/*.log

# 查看已收集的数据文件
ls -lh data/offline_datasets/medium/

# 检查收集进度（如果日志中有进度信息）
grep -i "progress\|trajectory\|collected" experiments/logs/data_collection/medium_20251206/*.log
```

### 2.5 数据质量验证

数据收集完成后，**必须**验证数据质量：

#### 步骤 1: 检查数据文件完整性

```bash
# 检查所有数据文件是否存在
for env in diffuse_topdown diffuse_mix diffuse_divpen focused_topdown focused_mix focused_divpen; do
    echo "=== $env ==="
    ls -lh data/offline_datasets/medium/${env}_medium_10000traj.pkl
done
```

**预期**: 每个文件大小应该相似且合理（取决于轨迹长度和特征维度）。

#### 步骤 2: 加载并检查数据内容

创建验证脚本 `scripts/verify_medium_data.py`:

```python
import pickle
import numpy as np
from pathlib import Path

def verify_dataset(data_path):
    """验证数据集的完整性和质量"""
    print(f"\n{'='*60}")
    print(f"验证数据集: {data_path.name}")
    print(f"{'='*60}")

    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 检查数据结构
    print(f"数据类型: {type(data)}")

    if isinstance(data, dict):
        print(f"数据字段: {list(data.keys())}")

        # 检查轨迹数量
        if 'observations' in data:
            num_traj = len(data['observations'])
            print(f"轨迹数量: {num_traj}")

        # 检查 reward 分布
        if 'rewards' in data:
            rewards = np.concatenate(data['rewards'])
            print(f"Reward 统计:")
            print(f"  - Mean: {rewards.mean():.4f}")
            print(f"  - Std: {rewards.std():.4f}")
            print(f"  - Min: {rewards.min():.4f}")
            print(f"  - Max: {rewards.max():.4f}")

        # 检查轨迹长度
        if 'observations' in data:
            traj_lengths = [len(obs) for obs in data['observations']]
            print(f"轨迹长度统计:")
            print(f"  - Mean: {np.mean(traj_lengths):.2f}")
            print(f"  - Min: {np.min(traj_lengths)}")
            print(f"  - Max: {np.max(traj_lengths)}")

    print(f"✅ 数据集验证完成")
    return True

if __name__ == "__main__":
    data_dir = Path("data/offline_datasets/medium")

    envs = [
        "diffuse_topdown",
        "diffuse_mix",
        "diffuse_divpen",
        "focused_topdown",
        "focused_mix",
        "focused_divpen"
    ]

    for env in envs:
        data_path = data_dir / f"{env}_medium_10000traj.pkl"
        if data_path.exists():
            verify_dataset(data_path)
        else:
            print(f"❌ 数据文件不存在: {data_path}")
```

运行验证：
```bash
python scripts/verify_medium_data.py
```

#### 步骤 3: 对比 Medium 数据与 Random 数据

如果已有 Random 数据，对比 reward 分布：

```bash
# Medium 数据的 reward 应该显著高于 Random 数据
# 可以通过验证脚本输出的统计信息进行对比
```

---

## 三、数据组织与归档

### 3.1 数据目录结构

```
data/offline_datasets/
├── random/                          # Random 策略数据
│   ├── diffuse_topdown_random_10000traj.pkl
│   └── ...
├── medium/                          # Medium 策略数据 ⭐ 新收集
│   ├── diffuse_topdown_medium_10000traj.pkl
│   ├── diffuse_mix_medium_10000traj.pkl
│   ├── diffuse_divpen_medium_10000traj.pkl
│   ├── focused_topdown_medium_10000traj.pkl
│   ├── focused_mix_medium_10000traj.pkl
│   └── focused_divpen_medium_10000traj.pkl
└── expert/                          # Expert 策略数据（未来）
    └── ...
```

### 3.2 元数据记录

创建 `data/offline_datasets/medium/README.md`:

```markdown
# Medium Dataset

**收集日期**: 2025-12-06
**模型**: SAC+GeMS trained to 50,000 steps
**种子**: 58407201
**轨迹数量**: 10,000 per environment
**Episode 长度**: 100 steps

## 环境列表

| 环境 | 用户类型 | 点击模型 | 多样性惩罚 | 数据文件 |
|------|---------|---------|-----------|---------|
| diffuse_topdown | Diffuse | tdPBM | 1.0 | diffuse_topdown_medium_10000traj.pkl |
| diffuse_mix | Diffuse | mixPBM | 1.0 | diffuse_mix_medium_10000traj.pkl |
| diffuse_divpen | Diffuse | mixPBM | 3.0 | diffuse_divpen_medium_10000traj.pkl |
| focused_topdown | Focused | tdPBM | 1.0 | focused_topdown_medium_10000traj.pkl |
| focused_mix | Focused | mixPBM | 1.0 | focused_mix_medium_10000traj.pkl |
| focused_divpen | Focused | mixPBM | 3.0 | focused_divpen_medium_10000traj.pkl |

## 模型来源

所有数据使用以下模型收集：
```
checkpoints/online_rl/{env}/SAC+GeMS_Medium_GeMS_{env}_agentseed58407201_gamma0.8_step50000.ckpt
```

## 训练配置

- lambda_KL: 1.0
- lambda_click: 0.5
- gamma: 0.8
- 训练步数: 50,000 steps
- 验证频率: 每 1000 episodes

## 数据质量指标

[在数据收集完成后填写]

| 环境 | Mean Reward | Std Reward | Min Reward | Max Reward |
|------|------------|-----------|-----------|-----------|
| diffuse_topdown | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... |
```

---

## 四、后续步骤

### 4.1 使用 Medium 数据训练离线 RL 算法

数据收集完成后，可以使用这些数据训练离线 RL 算法（如 CQL, IQL, BCQ 等）。

### 4.2 收集 Expert 数据（可选）

如果需要更高质量的数据，可以：
1. 继续训练模型到 100,000 步
2. 使用 100k 步的模型收集 Expert 数据
3. 重复本文档的数据收集流程

### 4.3 数据混合实验（可选）

可以尝试混合不同质量的数据：
- Random + Medium
- Medium + Expert
- Random + Medium + Expert

---

## 五、故障排查

### 5.1 模型加载失败

**问题**: 数据收集时提示找不到模型文件

**解决**:
```bash
# 检查模型文件是否存在
ls checkpoints/online_rl/*/SAC+GeMS_Medium_*_step50000.ckpt

# 如果文件名不匹配，更新脚本中的 MODEL_PATH
```

### 5.2 数据收集速度慢

**问题**: 数据收集进度缓慢

**解决**:
- 减少 `NUM_TRAJECTORIES`（如改为 5000）
- 增加 GPU 数量，并行收集更多环境
- 检查 GPU 利用率：`nvidia-smi`

### 5.3 数据文件损坏

**问题**: 数据文件无法加载或大小异常

**解决**:
```bash
# 删除损坏的文件
rm data/offline_datasets/medium/{env}_medium_10000traj.pkl

# 重新收集该环境的数据
# 修改脚本只收集特定环境
```

---

## 六、检查清单

在开始下一步之前，确认以下所有项目：

### 训练阶段
- [ ] 6 个环境的训练都已完成（到达 50,000 步）
- [ ] 所有 `*_step50000.ckpt` 文件都已生成
- [ ] SwanLab 上的训练曲线正常
- [ ] 验证 reward 达到合理水平

### 数据收集阶段
- [ ] 数据收集脚本已创建并测试
- [ ] 模型路径正确配置
- [ ] 数据保存目录已创建
- [ ] 6 个环境的数据都已收集完成
- [ ] 数据文件大小合理且完整

### 数据验证阶段
- [ ] 所有数据文件都可以正常加载
- [ ] Reward 分布合理（高于 Random 数据）
- [ ] 轨迹数量正确（10,000 per environment）
- [ ] 元数据文档已创建

---

## 七、参考信息

### 相关文件路径

**训练脚本**: `scripts/batch_runs/run_medium_collection_training.sh`
**数据收集脚本**: `scripts/batch_runs/collect_medium_data.sh` (待创建)
**验证脚本**: `scripts/verify_medium_data.py` (待创建)
**模型目录**: `checkpoints/online_rl/`
**数据目录**: `data/offline_datasets/medium/`
**日志目录**: `experiments/logs/`

### SwanLab 项目

**项目链接**: https://swanlab.cn/@Cliff/GeMS_RL_Training_202512
**实验标签**: `medium_collection`, `50k_steps`, `seed_58407201`

### 联系方式

如有问题，请查看：
- 项目 README
- SwanLab 实验记录
- 训练日志文件

---

**文档版本**: v1.0
**最后更新**: 2025-12-06
**作者**: Claude Code
