# Random级别模型目录

## 状态
🔄 **待收集** - 此目录预留给Random级别的数据收集模型

## 定义
Random级别模型是指使用随机策略或早期训练checkpoint收集的数据，提供高探索性的轨迹。

## 特点
- **训练进度**: 随机策略或约10%训练完成的checkpoint
- **回报水平**: 低回报（约为Expert的20-40%）
- **动作多样性**: 非常高（接近100%）
- **探索性**: 高度探索，广泛的状态空间覆盖
- **用途**: 提供基线数据、边界情况、帮助算法学习避免低质量行为

## 目录结构（待创建）

```
random/
├── sac_gems_models/
│   ├── diffuse_topdown/
│   ├── diffuse_mix/
│   ├── diffuse_divpen/
│   ├── focused_topdown/
│   ├── focused_mix/
│   └── focused_divpen/
├── gems_checkpoints/
│   └── (同上6个环境)
├── model_info.json
└── README.md (本文件)
```

## 收集计划

### 模型来源
有两种选择：

1. **纯随机策略**
   - 直接使用随机动作采样
   - 不需要训练模型
   - 最高探索性

2. **早期Checkpoint**
   - 使用训练进度10%左右的checkpoint
   - 建议选择test reward约为最终性能20-40%的checkpoint
   - 保留一定的策略结构

### 数据收集目标
- **Episodes数量**: 每个环境5,000 episodes
- **预期回报**:
  - Diffuse环境: 50-100
  - Focused环境: 50-100
- **预期动作多样性**: 90-100%

### 使用方法（未来）

```python
from pathlib import Path

MODELS_DIR = Path("/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models")
RANDOM_SAC_GEMS_DIR = MODELS_DIR / "random/sac_gems_models"
RANDOM_GEMS_CKPT_DIR = MODELS_DIR / "random/gems_checkpoints"

# 选项1: 加载早期checkpoint
env_name = 'focused_topdown'
params = 'beta1.0_lambdaclick0.5'
sac_gems_path = RANDOM_SAC_GEMS_DIR / env_name / f"SAC+GeMS_{params}_seed58407201_gamma0.8_random.ckpt"
gems_ckpt_path = RANDOM_GEMS_CKPT_DIR / env_name / f"GeMS_{params}_latentdim32_seed58407201.ckpt"

# 选项2: 使用纯随机策略（不需要加载模型）
# 直接在环境中采样随机动作
```

## 注意事项

1. **策略选择**:
   - 纯随机策略：最简单，但可能产生很多无意义的轨迹
   - 早期checkpoint：保留一定策略结构，数据质量稍好

2. **命名规则**: 建议在文件名中添加 `_random` 或 `_10pct` 标识

3. **GeMS Checkpoint**:
   - 如果使用早期SAC+GeMS checkpoint，仍然使用完全训练好的GeMS checkpoint
   - 如果使用纯随机策略，可能不需要GeMS checkpoint

4. **数据价值**:
   - Random数据对某些离线RL算法（如CQL）很重要
   - 帮助算法学习Q函数的下界
   - 提供OOD（Out-of-Distribution）样本

5. **收集效率**: Random数据收集速度最快，因为不需要复杂的模型推理

## 相关文档

- [总体说明](../README.md)
- [Expert模型](../expert/README.md)
- [Medium模型](../medium/README.md)
