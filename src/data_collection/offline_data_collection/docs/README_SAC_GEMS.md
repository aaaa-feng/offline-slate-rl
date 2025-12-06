# SAC+GeMS 数据收集系统

## 📋 概述

本文档说明如何使用正确的 SAC+GeMS 模型收集离线强化学习数据。

---

## ✅ 已完成的工作

### 1. 代码迁移
- ✅ 将 `offline_data_collection` 目录迁移到 `official_code/`
- ✅ 修改所有硬编码路径为动态路径
- ✅ 确保代码可移植性

### 2. 模型准备
- ✅ 从 `/data/liyuefeng/gems/data/checkpoints/` 复制成功训练的 SAC+GeMS 模型
- ✅ 存放位置：`offline_data_collection/sac_gems_models/`
- ✅ 包含3个环境：diffuse_topdown, diffuse_mix, diffuse_divpen

### 3. 代码修改
- ✅ 修改 `model_loader.py` 的 `load_diffuse_models()` 方法
- ✅ 现在加载 SAC+GeMS 模型（32维latent空间）
- ✅ 不再使用错误的 TopK 模型（20维item空间）

---

## 📂 目录结构

```
offline_data_collection/
├── sac_gems_models/              # SAC+GeMS模型存储
│   ├── diffuse_topdown/
│   │   └── SAC_GeMS_scratch_diffuse_topdown_seed58407201_*.ckpt
│   ├── diffuse_mix/
│   │   └── SAC_GeMS_scratch_diffuse_mix_seed58407201_*.ckpt
│   └── diffuse_divpen/
│       └── SAC_GeMS_scratch_diffuse_divpen_seed58407201_*.ckpt
├── best_models_for_data_collection/  # 旧的TopK模型（不使用）
├── collect_data.py               # 主数据收集脚本
├── model_loader.py               # 模型加载器（已修改）
├── environment_factory.py        # 环境工厂
├── data_formats.py               # 数据格式定义
├── metrics.py                    # 指标计算
├── test_sac_gems_loading.py     # 测试脚本（新增）
└── README_SAC_GEMS.md           # 本文档
```

---

## 🔑 关键修改说明

### 1. `model_loader.py` 的修改

#### 修改前（错误）：
```python
def load_diffuse_models(self):
    # 加载 TopK + ideal embeddings
    agent, ranker, belief_encoder = self.load_agent(
        env_name=env_name,
        agent_type="SAC",
        ranker_type="TopK",      # ❌ 错误：20维
        embedding_type="ideal"   # ❌ 使用特权信息
    )
```

#### 修改后（正确）：
```python
def load_diffuse_models(self):
    # 加载 SAC+GeMS
    sac_gems_models_dir = Path(__file__).resolve().parent / "sac_gems_models"
    self.models_dir = str(sac_gems_models_dir / env_name)

    agent, ranker, belief_encoder = self.load_agent(
        env_name=env_name,
        agent_type="SAC",
        ranker_type="GeMS",      # ✅ 正确：32维latent空间
        embedding_type="scratch" # ✅ 不使用特权信息
    )
```

### 2. 动作空间对比

| 模型类型 | 动作空间维度 | 语义 | 是否可用 |
|---------|------------|------|---------|
| SAC+TopK (ideal) | 20维 | item embedding空间 | ❌ 不可用 |
| SAC+GeMS | 32维 | GeMS latent空间 | ✅ 可用 |

---

## 🚀 使用方法

### 步骤1：测试模型加载和性能

运行测试脚本验证模型是否正确加载：

```bash
cd /data/liyuefeng/gems/gems_official/official_code
python offline_data_collection/test_sac_gems_loading.py
```

**预期输出**：
```
测试1：SAC+GeMS模型加载
  ✅ 模型加载成功!
  Agent动作维度: 32
  Ranker类型: GeMS
  Ranker latent_dim: 32

测试2：模型推理测试
  SAC输出latent_action: shape=(32,)
  GeMS输出slate: shape=10
  ✅ 模型推理测试通过!

测试3：环境交互测试（5个episodes）
  Episode 1: return=315.23, length=100
  Episode 2: return=320.45, length=100
  ...
  平均回报: 317.75 ± 2.34
  训练日志test_reward: 317.75
  ✅ 性能接近训练日志（差异<20）
```

### 步骤2：收集小规模测试数据

先收集少量数据测试流程：

```bash
cd /data/liyuefeng/gems/gems_official/official_code
python offline_data_collection/collect_data.py \
    --env_name diffuse_topdown \
    --episodes 100 \
    --output_dir ./offline_datasets_test
```

### 步骤3：收集完整数据集

确认测试通过后，收集完整数据：

```bash
cd /data/liyuefeng/gems/gems_official/official_code
python offline_data_collection/collect_data.py \
    --env_name all \
    --episodes 10000 \
    --output_dir ./offline_datasets
```

---

## 📊 模型性能参考

根据训练日志 `/data/liyuefeng/gems/logs/logs_baseline_2025/diffuse_topdown/SAC_GeMS_scratch_diffuse_topdown_seed58407201_gpu7.log`：

| 环境 | 训练步数 | Test Reward | Episode Length |
|-----|---------|-------------|----------------|
| diffuse_topdown | 100,000 | 317.75 | 100 |
| diffuse_mix | - | ~300-320 | 100 |
| diffuse_divpen | - | ~300-320 | 100 |

**注意**：如果测试时性能与训练日志差异较大（>20），可能需要：
1. 检查模型是否正确加载
2. 检查环境配置是否一致
3. 检查随机种子设置

---

## 🔍 数据格式

收集的数据将保存为两种格式：

### 1. Pickle格式 (`.pkl`)
- 完整的轨迹数据
- 包含所有元信息
- 用于详细分析

### 2. D4RL格式 (`.npz`)
- 标准的离线RL数据格式
- 包含：observations, actions, rewards, next_observations, terminals
- **关键**：actions 是 32维的 latent_action（不是slate）

### 数据结构示例

```python
# D4RL格式
data = np.load('expert_data_d4rl.npz')
print(data['observations'].shape)      # (N, 20) - belief states
print(data['actions'].shape)           # (N, 32) - latent actions ✅
print(data['rewards'].shape)           # (N,)
print(data['next_observations'].shape) # (N, 20)
print(data['terminals'].shape)         # (N,)
```

---

## ⚠️ 重要注意事项

### 1. 不要使用 `best_models_for_data_collection/` 中的模型
- 这些是 SAC+TopK 模型
- 动作空间是 20维（错误）
- 使用特权信息（ideal embeddings）
- **仅保留用于对比实验**

### 2. 确保使用 `sac_gems_models/` 中的模型
- 这些是 SAC+GeMS 模型
- 动作空间是 32维（正确）
- 不使用特权信息
- **这是论文的核心方法**

### 3. 数据收集策略
根据 D4RL 标准，应该收集三种质量的数据：
- **Expert**: 使用训练收敛的 SAC+GeMS（~317分）
- **Medium**: 使用训练中途的 SAC+GeMS（~150-200分）
- **Random**: 使用随机初始化的 SAC+GeMS（~50-100分）

---

## 🐛 故障排除

### 问题1：模型加载失败
```
❌ 模型加载失败: FileNotFoundError
```
**解决方案**：
- 检查 `sac_gems_models/` 目录是否存在
- 检查模型文件是否已复制
- 运行：`ls -lh offline_data_collection/sac_gems_models/*/`

### 问题2：动作维度错误
```
AssertionError: latent_action维度错误: 20, 应该是32
```
**解决方案**：
- 确认使用的是 `load_diffuse_models()` 而不是 `load_diffuse_models_topk()`
- 检查 `model_loader.py` 中的 ranker_type 是否为 "GeMS"

### 问题3：性能与训练日志差异大
```
⚠️ 性能与训练日志有较大差异
```
**解决方案**：
- 检查环境配置参数是否一致
- 检查模型是否正确加载（查看加载日志）
- 尝试多运行几次取平均

---

## 📚 相关文件

- 训练日志：`/data/liyuefeng/gems/logs/logs_baseline_2025/diffuse_topdown/SAC_GeMS_scratch_diffuse_topdown_seed58407201_gpu7.log`
- 原始模型：`/data/liyuefeng/gems/data/checkpoints/diffuse_*/SAC_GeMS_*.ckpt`
- 复制的模型：`offline_data_collection/sac_gems_models/`

---

## ✅ 验证清单

在开始大规模数据收集前，请确认：

- [ ] 测试脚本运行成功
- [ ] 模型加载正确（32维latent空间）
- [ ] 模型推理正常（输出32维latent_action）
- [ ] 环境交互正常（平均回报接近训练日志）
- [ ] 小规模测试数据收集成功
- [ ] 数据格式正确（actions是32维）

---

## 📞 联系方式

如有问题，请查看：
- 对话记录：`document/conversation_2025-11-29_session1.md`
- 代码迁移报告：本目录下的其他文档

---

**最后更新**: 2025-11-29
**状态**: ✅ 准备就绪，可以开始数据收集
