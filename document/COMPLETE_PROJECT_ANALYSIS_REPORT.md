# Offline-Slate-RL 项目完整分析报告

**生成日期**: 2025-12-04
**项目路径**: `/data/liyuefeng/offline-slate-rl`

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构分析](#2-目录结构分析)
3. [核心模块详解](#3-核心模块详解)
4. [算法运行指南](#4-算法运行指南)
5. [数据流程说明](#5-数据流程说明)
6. [当前问题与待解决事项](#6-当前问题与待解决事项)

---

## 1. 项目概述

### 1.1 项目目标

这是一个**推荐系统强化学习**项目，目标是：
- 实现在线RL算法（SAC、SlateQ、REINFORCE等）用于Slate推荐
- 实现离线RL算法（TD3-BC、CQL、IQL）用于从离线数据学习
- 使用GeMS（Generative Model for Slate）作为动作空间映射器

### 1.2 核心技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch + PyTorch Lightning |
| 环境模拟 | RecSim (推荐系统模拟器) |
| 日志记录 | SwanLab |
| 数据格式 | D4RL标准格式 (.npz) |

### 1.3 支持的算法

**在线RL算法**:
- SAC (Soft Actor-Critic)
- SAC+GeMS (本项目核心方法)
- SAC+TopK (baseline)
- SAC+WkNN (Wolpertinger方法)
- SlateQ
- REINFORCE+SoftMax
- Random (随机策略)
- Short-term Oracle (短期最优)

**离线RL算法**:
- TD3+BC
- CQL (Conservative Q-Learning)
- IQL (Implicit Q-Learning)

---

## 2. 目录结构分析

### 2.1 当前项目结构

```
/data/liyuefeng/offline-slate-rl/
│
├── src/                          # 源代码主目录
│   ├── agents/                   # RL智能体
│   │   ├── online.py            # 在线RL算法 (987行)
│   │   └── offline/             # 离线RL算法
│   │       ├── cql.py           # CQL (993行)
│   │       ├── iql.py           # IQL (649行)
│   │       └── td3_bc.py        # TD3+BC (399行)
│   │
│   ├── belief_encoders/          # 信念编码器
│   │   └── gru_belief.py        # GRU编码器 (171行)
│   │
│   ├── rankers/                  # Slate生成器
│   │   └── gems/                # GeMS模块
│   │       ├── rankers.py       # 排序器 (386行)
│   │       ├── item_embeddings.py
│   │       └── matrix_factorization/
│   │
│   ├── envs/                     # 环境
│   │   └── RecSim/              # RecSim模拟器
│   │       └── simulators.py    # 环境实现 (441行)
│   │
│   ├── training/                 # 训练循环
│   │   └── online_loops.py      # 在线训练循环 (660行)
│   │
│   ├── common/                   # 通用工具
│   │   ├── data_utils.py        # 数据工具 (189行)
│   │   ├── argument_parser.py   # 参数解析
│   │   └── logger.py            # 日志工具
│   │
│   ├── data_collection/          # 离线数据收集
│   │   └── offline_data_collection/
│   │       ├── core/            # 核心模块
│   │       ├── scripts/         # 收集脚本
│   │       ├── shell/           # Shell脚本
│   │       └── models/          # SAC+GeMS模型
│   │
│   └── offline_rl/               # 离线RL基线
│       └── offline_rl_baselines/
│
├── scripts/                      # 训练脚本
│   ├── train_agent.py           # 主训练脚本 (326行)
│   └── train_online_rl.py       # 在线RL训练
│
├── data/                         # 数据目录
│   ├── datasets/                # 数据集
│   ├── embeddings/              # Item embeddings
│   └── mf_embeddings/           # MF embeddings
│
├── checkpoints/                  # 模型检查点
│   ├── online_rl/               # 在线RL模型
│   ├── offline_rl/              # 离线RL模型
│   ├── gems/                    # GeMS模型
│   └── expert/                  # 专家策略
│
├── experiments/                  # 实验日志
│   └── logs/                    # 训练日志
│
├── datasets/                     # 离线数据集
│   └── offline_datasets/        # 收集的离线数据
│
└── document/                     # 文档
```

### 2.2 关键路径说明

| 路径 | 用途 |
|------|------|
| `data/datasets/` | 预训练数据集 (focused_*.pt, diffuse_*.pt) |
| `data/embeddings/` | Item embeddings (focused/diffuse) |
| `data/mf_embeddings/` | MF预训练的embeddings |
| `checkpoints/gems/` | 预训练的GeMS模型 |
| `checkpoints/online_rl/` | 在线RL训练的模型 |
| `datasets/offline_datasets/` | 离线数据收集的输出 |

---

## 3. 核心模块详解

### 3.1 智能体模块 (agents/)

#### 在线RL智能体 (`agents/online.py`)

| 类名 | 说明 | 动作空间 |
|------|------|----------|
| `SAC` | Soft Actor-Critic | 连续 |
| `WolpertingerSAC` | Wolpertinger方法的SAC | 连续→离散 |
| `SlateQ` | Slate Q-Learning | 离散 |
| `REINFORCE` | 策略梯度 | 连续 |
| `REINFORCESlate` | Slate版REINFORCE | 离散 |
| `RandomSlate` | 随机策略 | 离散 |
| `STOracleSlate` | 短期最优 | 离散 |
| `EpsGreedyOracle` | ε-贪心预言机 | 离散 |

#### 离线RL智能体 (`agents/offline/`)

| 文件 | 算法 | 说明 |
|------|------|------|
| `td3_bc.py` | TD3+BC | TD3 + 行为克隆正则化 |
| `cql.py` | CQL | 保守Q学习 |
| `iql.py` | IQL | 隐式Q学习 |

### 3.2 排序器模块 (rankers/gems/)

GeMS (Generative Model for Slate) 是本项目的核心创新：

```
连续动作 (latent_dim维) → GeMS解码器 → 离散物品列表 (slate_size个物品)
```

| 类名 | 说明 |
|------|------|
| `Ranker` | 抽象基类 |
| `TopKRanker` | Top-K排序 (需要item embeddings) |
| `kHeadArgmaxRanker` | K-Head Argmax |
| `GeMS` | 生成式模型 (VAE结构) |

### 3.3 信念编码器 (belief_encoders/)

用于POMDP环境，将观察历史编码为信念状态：

```
观察序列 [o_1, o_2, ..., o_t] → GRU → 信念状态 b_t
```

### 3.4 环境模块 (envs/RecSim/)

RecSim模拟器支持6种环境配置：

| 环境名 | 用户行为 | 点击模型 | 多样性惩罚 |
|--------|----------|----------|------------|
| `focused_topdown` | 聚焦 | tdPBM | 无 |
| `focused_mix` | 聚焦 | mixPBM | 1.0 |
| `focused_divpen` | 聚焦 | mixPBM | 3.0 |
| `diffuse_topdown` | 分散 | tdPBM | 无 |
| `diffuse_mix` | 分散 | mixPBM | 1.0 |
| `diffuse_divpen` | 分散 | mixPBM | 3.0 |

---

## 4. 算法运行指南

### 4.0 两个训练脚本的区别 (重要!)

项目中有**两个**训练脚本，它们使用不同的导入方式：

| 脚本 | 导入方式 | 路径配置 | 状态 |
|------|----------|----------|------|
| `scripts/train_agent.py` | 旧式 (`modules.agents`) | 硬编码 `data_dir` | ⚠️ 需要修复 |
| `scripts/train_online_rl.py` | 新式 (`agents.online`) | 使用 `config/paths.py` | ✅ 推荐使用 |

#### train_agent.py (旧版)
```python
# 导入方式 - 依赖 src/online_rl/ 下的旧结构
sys.path.insert(0, str(CODE_ROOT / "src" / "online_rl"))
from modules.agents import SAC, SlateQ, ...
from GeMS.modules.rankers import GeMS, TopKRanker, ...
```

#### train_online_rl.py (新版 - 推荐)
```python
# 导入方式 - 使用重构后的 src/ 结构
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from agents.online import SAC, SlateQ, ...
from rankers.gems.rankers import GeMS, TopKRanker, ...
from paths import get_embeddings_path, get_gems_checkpoint_path, ...
```

**建议**: 使用 `train_online_rl.py`，它已经适配了新的项目结构。

### 4.1 运行前提条件

#### 必需的数据文件

```bash
# 检查数据文件
ls data/embeddings/item_embeddings_focused.pt
ls data/embeddings/item_embeddings_diffuse.pt
ls data/datasets/focused_*.pt
ls data/datasets/diffuse_*.pt
ls data/mf_embeddings/*.pt
ls checkpoints/gems/*.ckpt
```

#### 环境设置

```bash
cd /data/liyuefeng/offline-slate-rl
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### 4.2 八种算法的运行命令

---

#### 1️⃣ Short-term Oracle (短期最优)

**特点**: 使用环境的真实信息，选择短期最优动作（上界参考）

```bash
python scripts/train_agent.py \
    --agent="STOracleSlate" \
    --belief="none" \
    --ranker="none" \
    --item_embedds="none" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --test_size=500 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --env_embedds="item_embeddings_focused.pt" \
    --name="STOracle"
```

---

#### 2️⃣ Random (随机策略)

**特点**: 随机选择物品，作为下界参考

```bash
python scripts/train_agent.py \
    --agent="RandomSlate" \
    --belief="none" \
    --ranker="none" \
    --item_embedds="none" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --test_size=500 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --env_embedds="item_embeddings_focused.pt" \
    --name="Random"
```

---

#### 3️⃣ SAC+TopK (ideal) - 使用特权信息

**特点**: 使用环境真实的item embeddings（特权信息），性能上界

```bash
python scripts/train_agent.py \
    --agent="SAC" \
    --belief="GRU" \
    --beliefs actor critic \
    --ranker="topk" \
    --item_embedds="ideal" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet 256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --pi_lr=0.003 \
    --hidden_layers_pinet 256 \
    --gamma=0.8 \
    --auto_entropy="True" \
    --alpha=0.2 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --name="SAC+TopK(ideal)"
```

---

#### 4️⃣ SAC+TopK (MF) - 使用MF预训练embeddings

**特点**: 使用矩阵分解预训练的embeddings

```bash
python scripts/train_agent.py \
    --agent="SAC" \
    --belief="GRU" \
    --beliefs actor critic \
    --ranker="topk" \
    --item_embedds="mf" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet 256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --pi_lr=0.003 \
    --hidden_layers_pinet 256 \
    --gamma=0.8 \
    --auto_entropy="True" \
    --alpha=0.2 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --MF_checkpoint="focused_topdown" \
    --name="SAC+TopK(MF)"
```

---

#### 5️⃣ SlateQ

**特点**: 基于Q-learning的Slate推荐方法

```bash
python scripts/train_agent.py \
    --agent="SlateQ" \
    --belief="GRU" \
    --beliefs critic \
    --ranker="none" \
    --item_embedds="scratch" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet 256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --gamma=0.8 \
    --opt_method="topk" \
    --epsilon_start=1.0 \
    --epsilon_end=0.01 \
    --epsilon_decay=1000 \
    --gradient_steps=1 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --name="SlateQ"
```

---

#### 6️⃣ REINFORCE+SoftMax

**特点**: 策略梯度方法，on-policy算法

**注意**: capacity=1, batch_size=1 (on-policy)

```bash
python scripts/train_agent.py \
    --agent="REINFORCESlate" \
    --belief="GRU" \
    --beliefs actor \
    --ranker="none" \
    --item_embedds="scratch" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=1 \
    --batch_size=1 \
    --pi_lr=0.003 \
    --hidden_layers_pinet 256 \
    --gamma=0.8 \
    --sigma_explo=0.29 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --name="REINFORCE+SoftMax"
```

---

#### 7️⃣ SAC+WkNN (Wolpertinger)

**特点**: 使用Wolpertinger方法处理大动作空间

```bash
python scripts/train_agent.py \
    --agent="WolpertingerSAC" \
    --belief="GRU" \
    --beliefs actor critic \
    --ranker="none" \
    --item_embedds="mf" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet 256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --pi_lr=0.003 \
    --hidden_layers_pinet 256 \
    --gamma=0.8 \
    --auto_entropy="True" \
    --alpha=0.2 \
    --full_slate=True \
    --wolpertinger_k=10 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --MF_checkpoint="focused_topdown" \
    --name="SAC+WkNN"
```

---

#### 8️⃣ SAC+GeMS (本项目核心方法)

**特点**: 使用GeMS生成式模型作为动作空间映射器

**前提**: 需要预训练的GeMS模型

```bash
python scripts/train_agent.py \
    --agent="SAC" \
    --belief="GRU" \
    --beliefs actor critic \
    --ranker="GeMS" \
    --item_embedds="scratch" \
    --env_name="topics" \
    --device="cuda" \
    --seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet 256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --pi_lr=0.003 \
    --hidden_layers_pinet 256 \
    --gamma=0.8 \
    --auto_entropy="True" \
    --alpha=0.2 \
    --latent_dim=32 \
    --lambda_KL=1.0 \
    --lambda_click=0.5 \
    --lambda_prior=0.0 \
    --ranker_embedds="scratch" \
    --ranker_sample="False" \
    --ranker_dataset="focused_topdown" \
    --ranker_seed=58407201 \
    --num_items=1000 \
    --episode_length=100 \
    --click_model="tdPBM" \
    --diversity_penalty=1.0 \
    --env_embedds="item_embeddings_focused.pt" \
    --name="SAC+GeMS"
```

---

### 4.3 算法对比总结

| 算法 | Agent | Belief | Ranker | Item Embedds | 特点 |
|------|-------|--------|--------|--------------|------|
| Short-term Oracle | STOracleSlate | none | none | none | 上界参考 |
| Random | RandomSlate | none | none | none | 下界参考 |
| SAC+TopK (ideal) | SAC | GRU | topk | ideal | 特权信息 |
| SAC+TopK (MF) | SAC | GRU | topk | mf | MF embeddings |
| SlateQ | SlateQ | GRU | none | scratch | Q-learning |
| REINFORCE+SoftMax | REINFORCESlate | GRU | none | scratch | 策略梯度 |
| SAC+WkNN | WolpertingerSAC | GRU | none | mf | Wolpertinger |
| **SAC+GeMS** | SAC | GRU | GeMS | scratch | **本文方法** |

### 4.4 关键参数差异

| 参数 | SAC系列 | REINFORCE | SlateQ |
|------|---------|-----------|--------|
| `--beliefs` | actor critic | actor | critic |
| `--capacity` | 10000 | **1** | 10000 |
| `--batch_size` | 20 | **1** | 20 |

---

## 5. 数据流程说明

### 5.1 完整实验流程

```
步骤1: 生成日志数据
    └── RecSim/generate_dataset.py
    └── 输出: data/datasets/{env_name}.pt

步骤2: 训练MF (仅baseline需要)
    └── GeMS/train_MF.py
    └── 输出: data/mf_embeddings/{env_name}.pt

步骤3: 预训练GeMS (仅SAC+GeMS需要)
    └── GeMS/pretrain_ranker.py
    └── 输出: checkpoints/gems/GeMS_{env_name}_*.ckpt

步骤4: 训练RL Agent
    └── scripts/train_agent.py
    └── 输出: checkpoints/online_rl/{env_name}/*.ckpt
```

### 5.2 离线数据收集流程

```
训练好的SAC+GeMS模型
    └── 加载模型
    └── 与环境交互收集数据
    └── 保存为D4RL格式
    └── 输出: datasets/offline_datasets/{env_name}_expert.npz
```

---

## 6. 当前问题与待解决事项

### 6.1 已完成

- [x] 在线RL算法实现 (SAC, SlateQ, REINFORCE等)
- [x] GeMS排序器实现
- [x] RecSim环境配置
- [x] 离线数据收集框架
- [x] 离线RL算法实现 (TD3-BC, CQL, IQL)

### 6.2 待解决

- [ ] **路径配置混乱**: `scripts/train_agent.py` 中的路径仍使用旧结构
- [ ] **离线RL集成**: 离线RL算法尚未与主训练脚本集成
- [ ] **focused环境数据收集**: 需要完成focused环境的离线数据收集
- [ ] **统一入口**: 需要一个统一的训练入口脚本

### 6.3 路径问题详情

当前 `scripts/train_agent.py` 中的路径配置：

```python
# 第188行 - MF数据集路径
dataset_path = args.data_dir + "datasets/" + args.MF_dataset

# 第192行 - MF embeddings路径
item_embeddings = ItemEmbeddings.from_pretrained(args.data_dir + "MF_embeddings/" + ...)

# 第212行 - GeMS checkpoint路径
ranker = ranker_class.load_from_checkpoint(args.data_dir + "GeMS/checkpoints/" + ...)

# 第279行 - checkpoint保存目录
ckpt_dir = args.data_dir + "checkpoints/" + checkpoint_dir_name + "/"
```

**问题**: 这些路径假设 `data_dir` 指向 `code/data/`，但重构后应该指向项目根目录。

---

## 附录A: 环境参数配置

### 所有环境共享的参数

```bash
--num_items=1000
--boredom_threshold=5
--recent_items_maxlen=10
--boredom_moving_window=5
--env_omega=0.9
--short_term_boost=1.0
--episode_length=100
--env_offset=0.28
--env_slope=100
--diversity_threshold=4
--topic_size=2
--num_topics=10
```

### 环境特定参数

| 环境 | click_model | diversity_penalty | env_embedds |
|------|-------------|-------------------|-------------|
| focused_topdown | tdPBM | 1.0 | item_embeddings_focused.pt |
| focused_mix | mixPBM | 1.0 | item_embeddings_focused.pt |
| focused_divpen | mixPBM | 3.0 | item_embeddings_focused.pt |
| diffuse_topdown | tdPBM | 1.0 | item_embeddings_diffuse.pt |
| diffuse_mix | mixPBM | 1.0 | item_embeddings_diffuse.pt |
| diffuse_divpen | mixPBM | 3.0 | item_embeddings_diffuse.pt |

---

## 附录B: 论文性能参考

### Focused环境预期性能 (论文Table 2)

| 方法 | TopDown | Mixed | DivPen |
|------|---------|-------|--------|
| SAC+TopK (ideal) | 429.0 | 384.1 | 386.3 |
| **SAC+GeMS** | **~400** | **~350** | **~360** |
| SAC+TopK (MF) | 254.4 | 232.7 | 242.2 |
| REINFORCE+SoftMax | 248.1 | 233.5 | 249.1 |
| SAC+WkNN | ~100 | ~100 | ~100 |
| SlateQ | ~150 | ~150 | ~150 |

---

*报告生成完成*
