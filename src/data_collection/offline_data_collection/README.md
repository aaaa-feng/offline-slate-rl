# SAC+GeMS 离线数据收集系统

完整的离线强化学习数据收集系统，用于从训练好的 SAC+GeMS 模型收集高质量的推荐系统轨迹数据。

---

## 📋 目录结构

```
offline_data_collection/
├── collect_data.py              # 主数据收集脚本
├── core/                        # 核心模块
│   ├── model_loader.py          # 模型加载器（支持SAC+GeMS）
│   ├── environment_factory.py   # 环境工厂
│   ├── data_formats.py          # 数据格式定义（支持D4RL格式）
│   └── metrics.py               # 指标计算
├── models/                      # 模型存储目录
│   ├── expert/                  # Expert级别模型（高质量数据）
│   │   ├── sac_gems_models/     # SAC+GeMS完整模型
│   │   ├── gems_checkpoints/    # GeMS预训练模型
│   │   └── README.md            # Expert模型说明
│   ├── medium/                  # Medium级别模型（中等质量数据）
│   ├── random/                  # Random级别模型（随机策略数据）
│   └── model_tests/             # 模型测试工具
│       ├── test.py              # 完整交互测试脚本
│       ├── pre_collection_test.py  # 数据收集前验证脚本
│       ├── verify_performance.py   # 性能验证脚本
│       └── model_test_logs/     # 测试日志
├── logs/                        # 数据收集日志
├── shell/                       # Shell脚本
└── README.md                    # 本文档
```

---

## 🚀 快速开始

### 1. 运行完整交互测试

测试脚本展示从模型加载到数据收集的完整流程：

```bash
cd /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/model_tests

# 测试 expert 级别的 focused_topdown 环境
python test.py --quality expert --env focused_topdown --episodes 1 --verbose

# 测试 medium 级别的 diffuse_mix 环境
python test.py --quality medium --env diffuse_mix --episodes 1
```

**测试脚本会展示：**
- ✅ 模型加载（SAC+GeMS）
- ✅ 环境初始化
- ✅ Belief state 编码
- ✅ Latent action 生成（32维）
- ✅ Slate 解码（10个物品）
- ✅ 用户交互和点击
- ✅ 数据保存格式

### 2. 数据收集前验证

在收集大规模数据前，运行验证脚本确保配置正确：

```bash
cd /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/model_tests

python pre_collection_test.py --env diffuse_topdown --quality expert
```

**验证内容：**
- ✅ 模型加载成功
- ✅ 环境参数正确
- ✅ 性能在合理范围内（~250-320分）
- ✅ 数据格式正确（actions是32维）

### 3. 收集测试数据（100 episodes）

```bash
cd /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection

python collect_data.py \
    --env_name diffuse_topdown \
    --quality expert \
    --episodes 100 \
    --output_dir /data/liyuefeng/offline-slate-rl/data/datasets/offline_test
```

### 4. 收集完整数据集（10,000 episodes）

```bash
# 收集单个环境的 expert 数据
python collect_data.py \
    --env_name diffuse_topdown \
    --quality expert \
    --episodes 10000 \
    --output_dir /data/liyuefeng/offline-slate-rl/data/datasets/offline

# 批量收集所有环境的数据
python collect_data.py \
    --env_name all \
    --quality expert \
    --episodes 10000 \
    --output_dir /data/liyuefeng/offline-slate-rl/data/datasets/offline
```

---

## 📊 数据格式

### D4RL 标准格式

数据保存为 `.npz` 格式，包含以下字段：

| 字段 | 维度 | 说明 |
|------|------|------|
| **observations** | (N, 20) | Belief states（GRU编码的用户状态） |
| **actions** | (N, 32) | **Latent actions**（用于TD3+BC/Diffuser） |
| **rewards** | (N,) | 即时奖励 |
| **next_observations** | (N, 20) | 下一个 belief states |
| **terminals** | (N,) | 终止标志 |
| **timeouts** | (N,) | 超时标志 |
| **slates** | (N, 10) | 推荐的物品列表（物品ID） |
| **clicks** | (N, 10) | 用户点击（0/1） |
| **diversity_scores** | (N,) | 推荐多样性分数 |
| **coverage_scores** | (N,) | 物品覆盖率分数 |
| **episode_ids** | (N,) | Episode ID |
| **timesteps** | (N,) | 时间步 |

**关键**：`actions` 字段保存的是 **32维的 latent_action**，可直接用于 TD3+BC 和 Decision Diffuser 训练。

### 数据示例

```python
import numpy as np

# 加载数据
data = np.load('diffuse_topdown_expert_data_d4rl.npz')

print(f"Observations shape: {data['observations'].shape}")  # (1000000, 20)
print(f"Actions shape: {data['actions'].shape}")            # (1000000, 32)
print(f"Slates shape: {data['slates'].shape}")              # (1000000, 10)
print(f"Clicks shape: {data['clicks'].shape}")              # (1000000, 10)
```

---

## 🎯 模型配置

### 数据质量级别

#### Expert 级别 ✅
- **定义**: 使用完全训练好的高性能模型收集的数据
- **特点**: 高回报、高动作多样性、接近最优策略
- **用途**: 作为离线RL算法的主要训练数据
- **状态**: 已完成（6个环境）

#### Medium 级别 🔄
- **定义**: 使用训练中期的模型收集的数据
- **特点**: 中等回报、探索与利用平衡
- **用途**: 提供更多样化的状态-动作覆盖
- **状态**: 待收集

#### Random 级别 🔄
- **定义**: 使用随机策略或早期训练模型收集的数据
- **特点**: 低回报、高探索性、广泛的状态覆盖
- **用途**: 提供基线数据和边界情况
- **状态**: 待收集

### 支持的环境

所有级别都支持以下 6 个环境：

| 环境名称 | 用户模型 | 奖励函数 | Diversity Penalty |
|---------|---------|---------|-------------------|
| **diffuse_topdown** | Diffuse | Top-down | 1.0 |
| **diffuse_mix** | Diffuse | Mixed | 1.0 |
| **diffuse_divpen** | Diffuse | Diversity Penalty | 3.0 |
| **focused_topdown** | Focused | Top-down | 1.0 |
| **focused_mix** | Focused | Mixed | 1.0 |
| **focused_divpen** | Focused | Diversity Penalty | 3.0 |

### SAC+GeMS 模型参数

- **Latent dim**: 32
- **Beta (λ_KL)**: 1.0
- **Lambda_click**: 0.5
- **Gamma**: 0.8
- **Action bounds**: center=0, scale=3.0
- **Embeddings**: scratch（不使用特权信息）

### 性能指标

| 环境 | Expert 性能 | Medium 性能 | Random 性能 |
|------|------------|------------|------------|
| diffuse_topdown | ~250-320 | TBD | TBD |
| diffuse_mix | ~300-320 | TBD | TBD |
| diffuse_divpen | ~300-320 | TBD | TBD |
| focused_topdown | ~250-320 | TBD | TBD |
| focused_mix | ~300-320 | TBD | TBD |
| focused_divpen | ~300-320 | TBD | TBD |

---

## 🔧 核心模块说明

### collect_data.py
主数据收集脚本，支持：
- ✅ 多环境并行收集
- ✅ Expert/Medium/Random 三种质量数据
- ✅ 自动保存为 Pickle 和 D4RL 格式
- ✅ 实时指标计算（多样性、覆盖率）
- ✅ 进度条显示

**主要参数：**
```bash
--env_name        # 环境名称（diffuse_topdown/all）
--quality         # 数据质量（expert/medium/random）
--episodes        # 收集的 episode 数量
--output_dir      # 输出目录
--seed            # 随机种子（可选）
```

### core/model_loader.py
模型加载器，支持：
- ✅ SAC+GeMS 统一加载
- ✅ GeMS 预训练权重加载
- ✅ 动态 action bounds 设置
- ✅ 自动设备选择（GPU/CPU）

**关键方法：**
```python
loader = ModelLoader()
agent, ranker, belief_encoder = loader.load_model(
    env_name='diffuse_topdown',
    quality='expert'
)
```

### core/environment_factory.py
环境工厂，支持：
- ✅ 6 个推荐环境创建
- ✅ 环境参数自动配置
- ✅ 与训练代码参数一致

**关键方法：**
```python
factory = EnvironmentFactory()
env = factory.create_environment('diffuse_topdown')
```

### core/data_formats.py
数据格式定义，支持：
- ✅ SlateDataset/SlateTrajectory/SlateTransition
- ✅ D4RL 格式转换
- ✅ 优先保存 latent_action

**数据结构：**
- `SlateObservation`: 观察数据（belief state）
- `SlateAction`: 动作数据（latent action + slate）
- `SlateInfo`: 额外信息（clicks, diversity, coverage）
- `SlateTransition`: 单步转移
- `SlateTrajectory`: 完整轨迹
- `SlateDataset`: 数据集

### core/metrics.py
指标计算，支持：
- ✅ 推荐多样性计算
- ✅ 物品覆盖率计算
- ✅ 点击率统计
- ✅ Episode 回报统计

---

## 📁 模型目录结构

```
models/
├── expert/                      # Expert 级别模型
│   ├── sac_gems_models/         # SAC+GeMS 完整模型
│   │   ├── diffuse_topdown/
│   │   ├── diffuse_mix/
│   │   ├── diffuse_divpen/
│   │   ├── focused_topdown/
│   │   ├── focused_mix/
│   │   └── focused_divpen/
│   └── gems_checkpoints/        # GeMS 预训练模型
│       ├── diffuse_topdown/
│       ├── diffuse_mix/
│       ├── diffuse_divpen/
│       ├── focused_topdown/
│       ├── focused_mix/
│       └── focused_divpen/
├── medium/                      # Medium 级别模型（待添加）
└── random/                      # Random 级别模型（待添加）
```

### 模型命名规则

**SAC+GeMS 模型：**
```
SAC+GeMS_{params}_seed{seed}_gamma{gamma}.ckpt
示例: SAC+GeMS_beta1.0_lambdaclick0.5_seed58407201_gamma0.8.ckpt
```

**GeMS Checkpoint：**
```
GeMS_{params}_latentdim{dim}_seed{seed}.ckpt
示例: GeMS_beta1.0_lambdaclick0.5_latentdim32_seed58407201.ckpt
```

---

## ✅ 数据收集前验证清单

在开始大规模数据收集前，请确认：

- [ ] 测试脚本运行成功（`test.py`）
- [ ] 模型加载显示 32 维 latent 空间
- [ ] 环境交互正常，无报错
- [ ] 性能在合理范围内（~250-320分）
- [ ] 数据格式正确（actions 是 32 维）
- [ ] 输出目录有足够的磁盘空间（每个环境约 10GB）

---

## 🐛 常见问题

### 1. 模型加载失败
**问题**: `FileNotFoundError: Model checkpoint not found`

**解决方案**:
- 检查模型路径是否正确
- 确认 SAC+GeMS 模型和 GeMS checkpoint 都存在
- 使用 `--quality expert` 参数指定正确的质量级别

### 2. 性能异常低
**问题**: Episode 回报远低于预期（<100）

**解决方案**:
- 确认加载了正确的 GeMS checkpoint
- 检查环境参数是否与训练时一致
- 运行 `pre_collection_test.py` 验证配置

### 3. 数据格式错误
**问题**: Actions 维度不是 32

**解决方案**:
- 确认使用了 `latent_action` 而不是 `slate`
- 检查 `data_formats.py` 中的 `to_d4rl_format()` 方法

### 4. 内存不足
**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减少 batch size（如果使用批量收集）
- 使用 CPU 模式：`--device cpu`
- 分批收集数据

---

## 📚 相关文档

- [Expert 模型详情](models/expert/README.md)
- [数据分析工具](/data/liyuefeng/offline-slate-rl/data/data_analysis/)
- [模型测试工具](models/model_tests/)

---

## 📞 支持

如有问题，请：
1. 查看本文档的常见问题部分
2. 运行 `test.py` 查看详细输出
3. 检查 `logs/` 目录中的日志文件

---

**最后更新**: 2025-12-25
**状态**: ✅ 已完成重构，Expert 数据收集系统就绪
**维护者**: liyuefeng
