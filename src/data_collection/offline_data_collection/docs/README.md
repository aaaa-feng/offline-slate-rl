# SAC+GeMS 离线数据收集系统

## 📋 目录结构

```
offline_data_collection/
├── collect_data.py              # 主数据收集脚本
├── data_formats.py              # 数据格式定义（支持D4RL格式）
├── environment_factory.py       # 环境工厂
├── model_loader.py              # 模型加载器（支持SAC+GeMS）
├── metrics.py                   # 指标计算
├── generate_dataset_report.py  # 数据集报告生成
├── test.py                      # 完整交互测试脚本
├── README.md                    # 本文档
├── README_SAC_GEMS.md          # 详细使用文档
└── sac_gems_models/            # SAC+GeMS模型目录
    ├── diffuse_topdown/        # diffuse_topdown环境模型
    ├── diffuse_mix/            # diffuse_mix环境模型
    └── diffuse_divpen/         # diffuse_divpen环境模型
```

## 🚀 快速开始

### 1. 运行测试

```bash
cd /data/liyuefeng/gems/gems_official/official_code
python offline_data_collection/test.py
```

这将展示完整的交互过程，包括：
- 模型加载（SAC+GeMS）
- 环境初始化
- Belief state编码
- Latent action生成（32维）
- Slate解码（10个物品）
- 用户交互
- 数据保存

### 2. 收集测试数据（100 episodes）

```bash
python offline_data_collection/collect_data.py \
    --env_name diffuse_topdown \
    --episodes 100 \
    --output_dir ./offline_datasets_test
```

### 3. 收集完整数据集（10000 episodes）

```bash
python offline_data_collection/collect_data.py \
    --env_name all \
    --episodes 10000 \
    --output_dir ./offline_datasets
```

## 📊 数据格式

### D4RL标准格式

数据保存为`.npz`格式，包含以下字段：

| 字段 | 维度 | 说明 |
|------|------|------|
| **observations** | (N, 20) | Belief states |
| **actions** | (N, 32) | **Latent actions** (用于TD3+BC) |
| **rewards** | (N,) | 即时奖励 |
| **next_observations** | (N, 20) | 下一个belief states |
| **terminals** | (N,) | 终止标志 |
| **slates** | (N, 10) | 推荐的物品列表 |
| **clicks** | (N, 10) | 用户点击 |

**关键**：`actions`字段保存的是32维的latent_action，可直接用于TD3+BC和Decision Diffuser训练。

## 🎯 模型配置

### SAC+GeMS模型参数

- **Latent dim**: 32
- **Beta (λ_KL)**: 1.0
- **Lambda_click**: 0.5
- **Gamma**: 0.8
- **Action bounds**: center=0, scale=3.0
- **Embeddings**: scratch (不使用特权信息)

### 性能指标

| 环境 | 训练日志 | 测试性能 |
|------|---------|---------|
| diffuse_topdown | 317.75 | ~250-320 |
| diffuse_mix | ~300-320 | TBD |
| diffuse_divpen | ~300-320 | TBD |

## 📚 详细文档

查看 [README_SAC_GEMS.md](README_SAC_GEMS.md) 获取：
- 完整的模型加载链路
- 参数详细说明
- 故障排除指南
- 数据格式详解

## ✅ 验证清单

数据收集前请确认：

- [ ] 测试脚本运行成功
- [ ] 模型加载显示32维latent空间
- [ ] 环境交互正常
- [ ] 性能在合理范围内（~250-320分）
- [ ] 数据格式正确（actions是32维）

## 🔧 核心文件说明

### collect_data.py
主数据收集脚本，支持：
- 多环境并行收集
- Expert/Medium/Random三种质量数据
- 自动保存为Pickle和D4RL格式

### model_loader.py
模型加载器，支持：
- SAC+GeMS统一加载
- GeMS预训练权重加载
- 动态action bounds设置

### data_formats.py
数据格式定义，支持：
- SlateDataset/SlateTrajectory/SlateTransition
- D4RL格式转换
- 优先保存latent_action

### test.py
完整交互测试，展示：
- 每一步的详细过程
- 所有中间变量
- 数据流转过程

## 📞 支持

如有问题，请查看：
1. [README_SAC_GEMS.md](README_SAC_GEMS.md) - 详细文档
2. `test.py` - 运行测试查看详细输出
3. 对话记录 - `document/conversation_2025-11-29_session1.md`

---

**最后更新**: 2025-11-30
**状态**: ✅ 已清理整理，可以开始数据收集
