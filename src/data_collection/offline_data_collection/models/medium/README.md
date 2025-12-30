# Medium级别模型目录

## 状态
🔄 **待收集** - 此目录预留给Medium级别的数据收集模型

## 定义
Medium级别模型是指使用训练中期checkpoint收集的数据，提供中等质量的轨迹。

## 特点
- **训练进度**: 约50%训练完成的checkpoint
- **回报水平**: 中等回报（约为Expert的60-80%）
- **动作多样性**: 60-80%
- **探索性**: 探索与利用平衡
- **用途**: 提供更多样化的状态-动作覆盖，帮助离线RL算法学习更鲁棒的策略

## 目录结构（待创建）

```
medium/
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
- 使用训练过程中保存的中期checkpoint
- 建议选择训练进度50%左右的checkpoint
- 确保该checkpoint的test reward约为最终性能的60-80%

### 数据收集目标
- **Episodes数量**: 每个环境5,000-10,000 episodes
- **预期回报**:
  - Diffuse环境: 120-200
  - Focused环境: 120-200
- **预期动作多样性**: 60-80%

### 使用方法（未来）

```python
from pathlib import Path

MODELS_DIR = Path("/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models")
MEDIUM_SAC_GEMS_DIR = MODELS_DIR / "medium/sac_gems_models"
MEDIUM_GEMS_CKPT_DIR = MODELS_DIR / "medium/gems_checkpoints"

# 加载模型
env_name = 'focused_topdown'
params = 'beta1.0_lambdaclick0.5'
sac_gems_path = MEDIUM_SAC_GEMS_DIR / env_name / f"SAC+GeMS_{params}_seed58407201_gamma0.8_medium.ckpt"
gems_ckpt_path = MEDIUM_GEMS_CKPT_DIR / env_name / f"GeMS_{params}_latentdim32_seed58407201.ckpt"
```

## 注意事项

1. **Checkpoint选择**: 需要从训练日志中确定合适的中期checkpoint
2. **命名规则**: 建议在文件名中添加 `_medium` 或 `_50pct` 标识
3. **GeMS Checkpoint**: 仍然使用完全训练好的GeMS checkpoint（与Expert相同）
4. **数据质量验证**: 收集后需要验证数据质量是否符合预期

## 相关文档

- [总体说明](../README.md)
- [Expert模型](../expert/README.md)
- [Random模型](../random/README.md)
