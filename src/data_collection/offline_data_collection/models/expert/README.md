# 数据收集专用模型目录

## 目录结构

```
models_for_data_collection/
├── sac_gems_models/          # SAC+GeMS完整模型
│   ├── focused_topdown/
│   ├── focused_mix/
│   └── focused_divpen/
├── gems_checkpoints/         # GeMS预训练模型
│   ├── focused_topdown/
│   ├── focused_mix/
│   └── focused_divpen/
├── model_info.json          # 模型详细信息
└── README.md                # 本文件
```

## 模型命名规则

### SAC+GeMS模型
格式: `SAC+GeMS_{params}_seed{seed}_gamma{gamma}.ckpt`

示例: `SAC+GeMS_beta1.0_lambdaclick0.5_seed58407201_gamma0.8.ckpt`

### GeMS Checkpoint
格式: `GeMS_{params}_latentdim{dim}_seed{seed}.ckpt`

示例: `GeMS_beta1.0_lambdaclick0.5_latentdim32_seed58407201.ckpt`

## 使用方法

在数据收集脚本中:

```python
MODEL_BASE_DIR = '/data/liyuefeng/offline-slate-rl/models_for_data_collection'
SAC_GEMS_DIR = f'{MODEL_BASE_DIR}/sac_gems_models'
GEMS_CKPT_DIR = f'{MODEL_BASE_DIR}/gems_checkpoints'

# 加载模型
env_name = 'focused_topdown'
params = 'beta1.0_lambdaclick0.5'
sac_gems_path = f'{SAC_GEMS_DIR}/{env_name}/SAC+GeMS_{params}_seed58407201_gamma0.8.ckpt'
gems_ckpt_path = f'{GEMS_CKPT_DIR}/{env_name}/GeMS_{params}_latentdim32_seed58407201.ckpt'
```

## 模型信息

### Diffuse环境（批次1 - 旧项目）

| 环境 | 参数组 | Test Reward | 训练日期 |
|------|--------|-------------|----------|
| diffuse_topdown | beta1.0_lambdaclick0.5 | 324.54 | 2025-11-28 |
| diffuse_mix | beta1.0_lambdaclick0.5 | 255.49 | 2025-11-28 |
| diffuse_divpen | beta1.0_lambdaclick0.5 | 194.81 | 2025-11-28 |

### Focused环境（批次2 - 新项目复现）

| 环境 | 参数组 | Test Reward | 训练日期 |
|------|--------|-------------|----------|
| focused_topdown | beta0.5_lambdaclick0.2 | 321.57 | 2025-11-29 |
| focused_topdown | beta1.0_lambdaclick0.5 | 303.18 | 2025-11-29 |
| focused_mix | beta0.5_lambdaclick0.2 | 215.42 | 2025-11-29 |
| focused_mix | beta1.0_lambdaclick0.5 | 218.24 | 2025-11-29 |
| focused_divpen | beta0.5_lambdaclick0.2 | 204.11 | 2025-11-29 |
| focused_divpen | beta1.0_lambdaclick0.5 | 224.13 | 2025-11-29 |

## 注意事项

1. **Diffuse环境**：来自批次1（旧项目，2025-11-28），已用于数据收集
2. **Focused环境**：来自批次2（新项目，2025-11-29），复现验证成功
3. 所有模型已迁移到新项目，完全独立，不依赖旧项目路径
4. 模型文件名包含关键参数，便于识别和使用
5. 详细信息请查看 `model_info.json`
