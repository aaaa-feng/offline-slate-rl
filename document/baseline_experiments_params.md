# Baseline实验完整配置文档

## 📋 目录
1. [实验概述](#实验概述)
2. [前置条件检查](#前置条件检查)
3. [通用参数配置](#通用参数配置)
4. [四个Baseline方法详细配置](#四个baseline方法详细配置)
5. [环境特定参数](#环境特定参数)
6. [日志和SwanLab命名规范](#日志和swanlab命名规范)
7. [参数对比总结](#参数对比总结)

---

## 实验概述

### 🎯 实验目标
在focused环境的3个配置上运行4个baseline方法，每个方法只跑1个种子(58407201)

### 📊 实验矩阵
- **方法数量**: 4个 (SAC+WkNN, REINFORCE+SoftMax, SAC+TopK(ideal), SlateQ)
- **环境数量**: 3个 (focused_topdown, focused_mix, focused_divpen)
- **种子数量**: 1个 (58407201)
- **总实验数**: 4 × 3 = 12个实验

### 🔑 关键要求
1. ✅ 不涉及GeMS的争议参数（KL、click等）
2. ✅ SwanLab的run_name只显示方法名和环境名
3. ✅ 日志文件命名不包含争议参数
4. ✅ 所有参数都要明确列出（通用参数 vs 独有参数）

---

## 前置条件检查

### ✅ 数据文件（全部已准备好）

#### 1. Item Embeddings
```
data/RecSim/embeddings/item_embeddings_focused.pt  ✅ 存在
```

#### 2. 预训练数据集
```
data/RecSim/datasets/focused_topdown.pt   ✅ 存在 (1.6GB)
data/RecSim/datasets/focused_mix.pt       ✅ 存在 (1.6GB)
data/RecSim/datasets/focused_divpen.pt    ✅ 存在 (1.6GB)
```

#### 3. MF Embeddings（SAC+WkNN需要）
```
data/MF_embeddings/focused_topdown.pt     ✅ 存在 (80KB)
data/MF_embeddings/focused_mix.pt         ✅ 存在 (80KB)
data/MF_embeddings/focused_divpen.pt      ✅ 存在 (80KB)
```

### ✅ 日志目录结构
```
logs/log_58407201/
├── SAC_WkNN/
├── REINFORCE_SoftMax/
├── SAC_TopK_ideal/
└── SlateQ/
```

---

## 通用参数配置

### 所有4个方法共享的参数

```bash
# ==================== 环境参数 ====================
--env_name="topics"
--device="cuda"
--seed=58407201

# ==================== 训练参数 ====================
--max_steps=100000
--check_val_every_n_epoch=1000
--val_step_length=200
--test_size=500
--random_steps=2000

# ==================== Belief参数 ====================
--belief="GRU"
--belief_state_dim=20
--item_embedd_dim=20
# 注意: --beliefs 参数根据agent不同而不同，见下文

# ==================== 环境特定参数 ====================
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

# ==================== SwanLab参数 ====================
--swan_project="GeMS_RL_Training_202512"
--swan_mode="cloud"
--swan_workspace="Cliff"
# 注意: --run_name, --swan_tags, --swan_description 根据方法和环境不同而不同
```

---

## 四个Baseline方法详细配置

---

## 1️⃣ SAC+WkNN (WolpertingerSAC)

### 核心配置
```bash
--agent="WolpertingerSAC"
--belief="GRU"
--beliefs actor critic          # ⚠️ 注意是 actor critic
--ranker="none"
--item_embedds="mf"              # ⚠️ 使用MF embeddings
--name="SAC+WkNN"
```

### Replay Buffer参数
```bash
--capacity=10000
--batch_size=20
```

### Wolpertinger特定参数
```bash
--full_slate=True
--wolpertinger_k=10
```

### SAC参数（继承自SAC）
```bash
# Q-Network
--q_lr=0.001
--hidden_layers_qnet 256
--target_update_frequency=1
--tau=0.002

# Policy Network
--pi_lr=0.003
--hidden_layers_pinet 256

# RL参数
--gamma=0.8
--auto_entropy="True"
--alpha=0.2
--alpha_lr=0.001
--l2_reg=0.001
```

### 环境特定参数（3个环境）

#### focused_topdown
```bash
--click_model="tdPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
--MF_checkpoint="focused_topdown"
```

#### focused_mix
```bash
--click_model="mixPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
--MF_checkpoint="focused_mix"
```

#### focused_divpen
```bash
--click_model="mixPBM"
--diversity_penalty=3.0
--env_embedds="item_embeddings_focused.pt"
--MF_checkpoint="focused_divpen"
```

### SwanLab配置

#### focused_topdown
```bash
--run_name="SAC_WkNN_focused_topdown_seed58407201"
--swan_tags "SAC_WkNN" "focused_topdown" "seed_58407201"
--swan_description="SAC+WkNN - focused_topdown - seed 58407201"
```

#### focused_mix
```bash
--run_name="SAC_WkNN_focused_mix_seed58407201"
--swan_tags "SAC_WkNN" "focused_mix" "seed_58407201"
--swan_description="SAC+WkNN - focused_mix - seed 58407201"
```

#### focused_divpen
```bash
--run_name="SAC_WkNN_focused_divpen_seed58407201"
--swan_tags "SAC_WkNN" "focused_divpen" "seed_58407201"
--swan_description="SAC+WkNN - focused_divpen - seed 58407201"
```

### 日志文件路径
```
logs/log_58407201/SAC_WkNN/focused_topdown_20251129.log
logs/log_58407201/SAC_WkNN/focused_mix_20251129.log
logs/log_58407201/SAC_WkNN/focused_divpen_20251129.log
```

---

## 2️⃣ REINFORCE+SoftMax (REINFORCESlate)

### 核心配置
```bash
--agent="REINFORCESlate"
--belief="GRU"
--beliefs actor                  # ⚠️ 注意只有 actor
--ranker="none"
--item_embedds="scratch"         # ⚠️ 使用scratch embeddings
--name="REINFORCE+SoftMax"
```

### Replay Buffer参数 ⚠️ 重要！
```bash
--capacity=1                     # ⚠️ 不是10000！REINFORCE是on-policy
--batch_size=1                   # ⚠️ 不是20！
```

### REINFORCE特定参数
```bash
--pi_lr=0.003
--hidden_layers_pinet 256
--gamma=0.8
--sigma_explo=0.29
```

### 环境特定参数（3个环境）

#### focused_topdown
```bash
--click_model="tdPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_mix
```bash
--click_model="mixPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_divpen
```bash
--click_model="mixPBM"
--diversity_penalty=3.0
--env_embedds="item_embeddings_focused.pt"
```

### SwanLab配置

#### focused_topdown
```bash
--run_name="REINFORCE_SoftMax_focused_topdown_seed58407201"
--swan_tags "REINFORCE_SoftMax" "focused_topdown" "seed_58407201"
--swan_description="REINFORCE+SoftMax - focused_topdown - seed 58407201"
```

#### focused_mix
```bash
--run_name="REINFORCE_SoftMax_focused_mix_seed58407201"
--swan_tags "REINFORCE_SoftMax" "focused_mix" "seed_58407201"
--swan_description="REINFORCE+SoftMax - focused_mix - seed 58407201"
```

#### focused_divpen
```bash
--run_name="REINFORCE_SoftMax_focused_divpen_seed58407201"
--swan_tags "REINFORCE_SoftMax" "focused_divpen" "seed_58407201"
--swan_description="REINFORCE+SoftMax - focused_divpen - seed 58407201"
```

### 日志文件路径
```
logs/log_58407201/REINFORCE_SoftMax/focused_topdown_20251129.log
logs/log_58407201/REINFORCE_SoftMax/focused_mix_20251129.log
logs/log_58407201/REINFORCE_SoftMax/focused_divpen_20251129.log
```

---

## 3️⃣ SAC+TopK (ideal)

### 核心配置
```bash
--agent="SAC"
--belief="GRU"
--beliefs actor critic           # ⚠️ 注意是 actor critic
--ranker="topk"                  # ⚠️ 使用topk ranker
--item_embedds="ideal"           # ⚠️ 使用ideal embeddings（特权信息）
--name="SAC+TopK(ideal)"
```

### Replay Buffer参数
```bash
--capacity=10000
--batch_size=20
```

### SAC参数
```bash
# Q-Network
--q_lr=0.001
--hidden_layers_qnet 256
--target_update_frequency=1
--tau=0.002

# Policy Network
--pi_lr=0.003
--hidden_layers_pinet 256

# RL参数
--gamma=0.8
--auto_entropy="True"
--alpha=0.2
--alpha_lr=0.001
--l2_reg=0.001
```

### 环境特定参数（3个环境）

#### focused_topdown
```bash
--click_model="tdPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_mix
```bash
--click_model="mixPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_divpen
```bash
--click_model="mixPBM"
--diversity_penalty=3.0
--env_embedds="item_embeddings_focused.pt"
```

### SwanLab配置

#### focused_topdown
```bash
--run_name="SAC_TopK_ideal_focused_topdown_seed58407201"
--swan_tags "SAC_TopK_ideal" "focused_topdown" "seed_58407201"
--swan_description="SAC+TopK(ideal) - focused_topdown - seed 58407201"
```

#### focused_mix
```bash
--run_name="SAC_TopK_ideal_focused_mix_seed58407201"
--swan_tags "SAC_TopK_ideal" "focused_mix" "seed_58407201"
--swan_description="SAC+TopK(ideal) - focused_mix - seed 58407201"
```

#### focused_divpen
```bash
--run_name="SAC_TopK_ideal_focused_divpen_seed58407201"
--swan_tags "SAC_TopK_ideal" "focused_divpen" "seed_58407201"
--swan_description="SAC+TopK(ideal) - focused_divpen - seed 58407201"
```

### 日志文件路径
```
logs/log_58407201/SAC_TopK_ideal/focused_topdown_20251129.log
logs/log_58407201/SAC_TopK_ideal/focused_mix_20251129.log
logs/log_58407201/SAC_TopK_ideal/focused_divpen_20251129.log
```

---

## 4️⃣ SlateQ

### 核心配置
```bash
--agent="SlateQ"
--belief="GRU"
--beliefs critic                 # ⚠️ 注意只有 critic
--ranker="none"
--item_embedds="scratch"         # ⚠️ 使用scratch embeddings
--name="SlateQ"
```

### Replay Buffer参数
```bash
--capacity=10000
--batch_size=20
```

### SlateQ特定参数
```bash
--opt_method="topk"
```

### DQN参数（SlateQ继承自DQN）
```bash
# Q-Network
--q_lr=0.001
--hidden_layers_qnet 256
--target_update_frequency=1
--tau=0.002

# RL参数
--gamma=0.8

# Epsilon-greedy探索
--epsilon_start=1.0
--epsilon_end=0.01
--epsilon_decay=1000
--gradient_steps=1
```

### 环境特定参数（3个环境）

#### focused_topdown
```bash
--click_model="tdPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_mix
```bash
--click_model="mixPBM"
--diversity_penalty=1.0
--env_embedds="item_embeddings_focused.pt"
```

#### focused_divpen
```bash
--click_model="mixPBM"
--diversity_penalty=3.0
--env_embedds="item_embeddings_focused.pt"
```

### SwanLab配置

#### focused_topdown
```bash
--run_name="SlateQ_focused_topdown_seed58407201"
--swan_tags "SlateQ" "focused_topdown" "seed_58407201"
--swan_description="SlateQ - focused_topdown - seed 58407201"
```

#### focused_mix
```bash
--run_name="SlateQ_focused_mix_seed58407201"
--swan_tags "SlateQ" "focused_mix" "seed_58407201"
--swan_description="SlateQ - focused_mix - seed 58407201"
```

#### focused_divpen
```bash
--run_name="SlateQ_focused_divpen_seed58407201"
--swan_tags "SlateQ" "focused_divpen" "seed_58407201"
--swan_description="SlateQ - focused_divpen - seed 58407201"
```

### 日志文件路径
```
logs/log_58407201/SlateQ/focused_topdown_20251129.log
logs/log_58407201/SlateQ/focused_mix_20251129.log
logs/log_58407201/SlateQ/focused_divpen_20251129.log
```

---

## 环境特定参数

### 三个环境的配置差异

| 环境 | Click Model | Diversity Penalty | 说明 |
|------|-------------|-------------------|------|
| **focused_topdown** | tdPBM | 1.0 | 纯自上而下浏览 |
| **focused_mix** | mixPBM | 1.0 | 混合浏览模式 |
| **focused_divpen** | mixPBM | 3.0 | 混合+多样性惩罚 |

### 所有环境共享
```bash
--env_embedds="item_embeddings_focused.pt"
```

---

## 日志和SwanLab命名规范

### 📁 日志文件命名规范

#### 格式
```
logs/log_58407201/{METHOD}/{ENV}_YYYYMMDD.log
```

#### 完整示例（12个日志文件）
```
logs/log_58407201/SAC_WkNN/focused_topdown_20251129.log
logs/log_58407201/SAC_WkNN/focused_mix_20251129.log
logs/log_58407201/SAC_WkNN/focused_divpen_20251129.log

logs/log_58407201/REINFORCE_SoftMax/focused_topdown_20251129.log
logs/log_58407201/REINFORCE_SoftMax/focused_mix_20251129.log
logs/log_58407201/REINFORCE_SoftMax/focused_divpen_20251129.log

logs/log_58407201/SAC_TopK_ideal/focused_topdown_20251129.log
logs/log_58407201/SAC_TopK_ideal/focused_mix_20251129.log
logs/log_58407201/SAC_TopK_ideal/focused_divpen_20251129.log

logs/log_58407201/SlateQ/focused_topdown_20251129.log
logs/log_58407201/SlateQ/focused_mix_20251129.log
logs/log_58down_20251129.log
```

### 🏷️ SwanLab命名规范

#### run_name格式
```
{METHOD}_{ENV}_seed{SEED}
```

#### tags格式
```bash
--swan_tags "{METHOD}" "{ENV}" "seed_{SEED}"
```

#### description格式
```bash
--swan_description="{METHOD} - {ENV} - seed {SEED}"
```

#### 完整示例（12个实验）
```
SAC_WkNN_focused_topdown_seed58407201
SAC_WkNN_focused_mix_seed58407201
SAC_WkNN_focused_divpen_seed58407201

REINFORCE_SoftMax_focused_topdown_seed58407201
REINFORCE_SoftMax_focused_mix_seed58407201
REINFORCE_SoftMax_focused_divpen_seed58407201

SAC_TopK_ideal_focused_topdown_seed58407201
SAC_TopK_ideal_focused_mix_seed58407201
SAC_TopK_ideal_focused_divpen_seed58407201

SlateQ_focused_topdown_seed58407201
SlateQ_focused_mix_seed58407201
SlateQ_focused_divpen_seed58407201
```

---

## 参数对比总结

### 🔑 核心差异对比表

| 参数类别 | SAC+WkNN | REINFORCE+SoftMax | SAC+TopK(ideal) | SlateQ |
|---------|----------|-------------------|-----------------|--------|
| **Agent** | WolpertingerSAC | REINFORCESlate | SAC | SlateQ |
| **Beliefs** | actor critic | actor | actor critic | critic |
| **Ranker** | none | none | topk | none |
| **Item Embedds** | mf | scratch | ideal | scratch |
| **Capacity** | 10000 | 1 ⚠️ | 10000 | 10000 |
| **Batch Size** | 20 | 1 ⚠️ | 20 | 20 |
| **需要MF** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **特权信息** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **优化器** | SAC (Q+Pi) | REINFORCE (Pi) | SAC (Q+Pi) | DQN (Q) |

### 📊 独有参数总结

#### SAC+WkNN独有
```bash
--full_slate=True
--wolpertinger_k=10
--MF_checkpoint="..."
--alpha_lr=0.001
--l2_reg=0.001
```

#### REINFORCE+SoftMax独有
```bash
--sigma_explo=0.29
--capacity=1
--batch_size=1
```

#### SAC+TopK(ideal)独有
```bash
--ranker="topk"
--item_embedds="ideal"
--alpha_lr=0.001
--l2_reg=0.001
```

#### SlateQ独有
```bash
--opt_method="topk"
--epsilon_start=1.0
--epsilon_end=0.01
--epsilon_decay=1000
--gradient_steps=1
```

### 📋 通用参数（所有方法共享）

```bash
# 环境和训练
--env_name="topics"
--device="cuda"
--seed=58407201
--max_steps=100000
--check_val_every_n_epoch=1000
--val_step_length=200
--test_size=500
--random_steps=2000

# Belief
--belief="GRU"
--belief_state_dim=20
--item_embedd_dim=20

# 环境特定
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

# SwanLab
--swan_project="GeMS_RL_Training_202512"
--swan_mode="cloud"
--swan_workspace="Cliff"
```

---

## 运行注意事项

### ✅ 执行前检查清单

1. **创建日志目录**
```bash
mkdir -p logs/log_58407201/SAC_WkNN
mkdir -p logs/log_58407201/REINFORCE_SoftMax
mkdir -p logs/log_58407201/SAC_TopK_ideal
mkdir -p logs/log_58407201/SlateQ
```

2. **验证数据文件存在**
```bash
ls data/RecSim/embeddings/item_embeddings_focused.pt
ls data/RecSim/datasets/focused_*.pt
ls data/MF_embeddings/focused_*.pt
```

3. **GPU分配建议**
- 每个实验约占用2.3-2.4 GB显存
- 可以在不同GPU上并行运行
- 使用 `CUDA_VISIBLE_DEVICES` 指定GPU

4. **使用nohup运行**
```bash
nohup python -u train_agent.py [参数...] > logs/xxx.log 2>&1 &
```

### ⚠️ 关键注意事项

1. **REINFORCE的buffer大小**: capacity=1, batch_size=1（不是10000和20）
2. **Beliefs参数**: 不同agent使用不同的beliefs配置
3. **MF_checkpoint**: SAC+WkNN需要根据环境选择对应的MF checkpoint
4. **日志命名**: 不包含KL、click等争议参数
5. **SwanLab命名**: 只包含方法名、环境名和种子

---

## 实验完成后的验证

### 检查项目

1. **日志文件**: 12个日志文件都已生成
2. **SwanLab记录**: 12个实验都已上传到SwanLab
3. **命令记录**: 每个日志文件开头都有完整命令
4. **训练完成**: 所有实验都达到100000步
5. **验证结果**: 每个实验都有validation和test结果

---

## 附录：论文中的性能参考

### Focused环境的预期性能（论文Table 2）

#### TopDown环境
- SAC+TopK (ideal): 429.0 (特权信息)
- SAC+TopK (MF): 254.4
- REINFORCE+SoftMax: 248.1
- SAC+WkNN: ~95-107

#### Mixed环境
- SAC+TopK (ideal): 384.1 (特权信息)
- REINFORCE+SoftMax: 233.5
- SAC+TopK (MF): 232.7
- SAC+WkNN: ~95-107

#### DivPen环境
- SAC+TopK (ideal): 386.3 (特权信息)
- REINFORCE+SoftMax: 249.1
- SAC+TopK (MF): 242.2
- SAC+WkNN: ~95-107

**注意**: 这些是论文中10个种子的平均结果，单个种子的结果可能有波动。
