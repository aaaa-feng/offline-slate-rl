# GeMS项目数据文件和工作流程详解

## 📚 目录
1. [原作者提供的数据](#1-原作者提供的数据)
2. [各类文件的作用和关系](#2-各类文件的作用和关系)
3. [Embedding到底是什么](#3-embedding到底是什么)
4. [完整工作流程](#4-完整工作流程)
5. [MF训练脚本的作用](#5-mf训练脚本的作用)

---

## 1. 原作者提供的数据

### 📦 gems-master.zip 中包含的数据文件

根据原作者的README和zip文件内容，作者**只提供了以下数据**：

```
gems-master/data/
├── RecSim/embeddings/
│   ├── item_embeddings_focused.pt    (80KB)
│   └── item_embeddings_diffuse.pt    (80KB)
└── MF_embeddings/
    ├── focused_topdown.pt            (80KB)
    ├── focused_mix.pt                (80KB)
    ├── focused_divpen.pt             (80KB)
    ├── diffuse_topdown.pt            (80KB)
    ├── diffuse_mix.pt                (80KB)
    └── diffuse_divpen.pt             (80KB)
```

### ⚠️ 原作者**没有**提供的数据

原作者**没有提供**预训练数据集（logged data），需要自己生成：
- ❌ `data/RecSim/datasets/focused_topdown.pt` (1.6GB)
- ❌ `data/RecSim/datasets/focused_mix.pt` (1.6GB)
- ❌ `data/RecSim/datasets/focused_divpen.pt` (1.6GB)
- ❌ `data/RecSim/datasets/diffuse_topdown.pt` (1.6GB)
- ❌ `data/RecSim/datasets/diffuse_mix.pt` (1.6GB)
- ❌ `data/RecSim/datasets/diffuse_divpen.pt` (1.6GB)

**你已经生成了这些数据集**，所以你的环境比原作者提供的更完整！

---

## 2. 各类文件的作用和关系

### 2.1 Item Embeddings（物品嵌入向量）

**文件：**
- `item_embeddings_focused.pt` (1000个物品 × 20维)
- `item_embeddings_diffuse.pt` (1000个物品 × 20维)

**作用：**
这是**RecSim模拟器内部使用的真实物品表示**，用于：
1. **模拟器计算用户兴趣和物品的相关性**
2. **生成用户点击行为**
3. **作为环境的"ground truth"**

**与RecSim的关系：**
```python
# RecSim模拟器内部使用这些embeddings
class TopicRec:
    def __init__(self, env_embedds="item_embeddings_focused.pt"):
        # 加载物品embeddings
        self.item_embeddings = torch.load(env_embedds)

    def compute_relevance(self, user_state, item):
        # 用item_embeddings计算用户对物品的兴趣
        relevance = user_state @ self.item_embeddings[item]
        return relevance
```

**Focused vs Diffuse的区别：**
- **Focused**: 峰度较高（embeddings平方后重新归一化），用户兴趣更集中
- **Diffuse**: 峰度较低（原始分布），用户兴趣更分散

---

### 2.2 预训练数据集（Logged Data）

**文件：**
- `focused_topdown.pt` (1.6GB, 100K trajectories)
- `focused_mix.pt` (1.6GB, 100K trajectories)
- `focused_divpen.pt` (1.6GB, 100K trajectories)
- `diffuse_topdown.pt` (1.6GB, 100K trajectories)
- `diffuse_mix.pt` (1.6GB, 100K trajectories)
- `diffuse_divpen.pt` (1.6GB, 100K trajectories)

**作用：**
这是**用ε-greedy oracle策略在RecSim环境中收集的历史交互数据**，包含：
- 用户状态序列
- 推荐的slate序列
- 用户点击行为
- 奖励信号

**生成方式：**
```bash
python RecSim/generate_dataset.py \
    --n_sess=100000 \
    --epsilon_pol=0.5 \
    --env_embedds="item_embeddings_focused.pt" \
    --click_model="tdPBM" \
    --path="data/RecSim/datasets/focused_topdown"
```

**用途：**
1. **训练GeMS的VAE模型**（学习slate的生成分布）
2. **训练MF模型**（学习物品的协同过滤表示）

**与RecSim的关系：**
- 这些数据是**从RecSim环境中采样出来的**
- 记录了在特定环境配置下的用户行为模式

---

### 2.3 MF Embeddings（矩阵分解嵌入）

**文件：**
- `MF_embeddings/focused_topdown.pt` (80KB, 1000个物品 × 20维)
- `MF_embeddings/focused_mix.pt`
- `MF_embeddings/focused_divpen.pt`
- `MF_embeddings/diffuse_topdown.pt`
- `MF_embeddings/diffuse_mix.pt`
- `MF_embeddings/diffuse_divpen.pt`

**作用：**
这是**从logged data中学习到的物品协同过滤表示**，用于：
1. **SAC+WkNN baseline**：定义连续动作空间
2. **SAC+TopK (MF) baseline**：作为物品的特征表示

**生成方式：**
```bash
python GeMS/train_MF.py --MF_dataset="focused_topdown.pt"
```

**与其他embeddings的区别：**
- **Item Embeddings (ideal)**：环境内部的真实表示（特权信息）
- **MF Embeddings**：从用户行为数据中学习的表示（实际可用）
- **Scratch Embeddings**：随机初始化，训练过程中学习

---

## 3. Embedding到底是什么

### 3.1 什么是Embedding？

**Embedding = 向量表示**

在推荐系统中，embedding是将离散的物品ID映射到连续的向量空间：
```
物品ID: 0, 1, 2, ..., 999
       ↓
Embedding: [0.1, -0.3, 0.5, ..., 0.2]  (20维向量)
```

### 3.2 为什么需要Embedding？

1. **计算相似度**：向量可以计算距离和相似度
2. **神经网络输入**：神经网络需要连续的数值输入
3. **降维表示**：1000个物品用20维向量表示，更紧凑

### 3.3 三种Embedding的区别

| Embedding类型 | 来源 | 用途 | 是否特权信息 |
|--------------|------|------|-------------|
| **Item Embeddings (ideal)** | RecSim环境内部 | 计算真实相关性 | ✅ 是 |
| **MF Embeddings** | 从logged data学习 | SAC+WkNN, SAC+TopK(MF) | ❌ 否 |
| **Scratch Embeddings** | 随机初始化 | REINFORCE, SlateQ | ❌ 否 |

### 3.4 Embedding在不同方法中的使用

```python
# 1. SAC+TopK (ideal) - 使用环境的真实embeddings
item_embedds = env.get_item_embeddings()  # 特权信息！

# 2. SAC+WkNN - 使用MF学习的embeddings
item_embedds = torch.load("MF_embeddings/focused_topdown.pt")
# 在embedding空间中选择连续动作
action = policy_net(state)  # 输出: 10×20维向量
# 找到最近的k个物品
slate = knn_search(action, item_embedds, k=10)

# 3. REINFORCE+SoftMax - 使用scratch embeddings
item_embedds = nn.Embedding(1000, 20)  # 随机初始化
# 训练过程中学习
```

---

## 4. 完整工作流程

### 阶段1: 数据准备（你已完成✅）

```
1. Item Embeddings (原作者提供)
   item_embeddings_focused.pt
   item_embeddings_diffuse.pt

2. 生成Logged Data (你已生成)
   RecSim/generate_dataset.py
   → focused_topdown.pt (1.6GB)
   → focused_mix.pt (1.6GB)
   → focused_divpen.pt (1.6GB)
   → diffuse_topdown.pt (1.6GB)
   → diffuse_mix.pt (1.6GB)
   → diffuse_divpen.pt (1.6GB)

3. 训练MF Embeddings (原作者提供)
   GeMS/train_MF.py
   → MF_embeddings/focused_topdown.pt
   → MF_embeddings/focused_mix.pt
   → ...
```

### 阶段2: 预训练GeMS（你已完成✅）

```
GeMS/pretrain_ranker.py
输入: focused_topdown.pt (logged data)
输出: GeMS_focused_topdown_...beta0.5_lambdaclick0.2.pt
```

### 阶段3: RL训练（进行中🚀）

```
train_agent.py
输入:
  - GeMS checkpoint (如果用SAC+GeMS)
  - MF embeddings (如果用SAC+WkNN)
  - Item embeddings (如果用ideal)
  - RecSim环境配置
输出:
  - 训练好的RL agent
  - 验证和测试结果
```

---

## 5. MF训练脚本的作用

### 5.1 train_MF.py 是什么？

**文件路径：** `GeMS/train_MF.py`

**作用：** 从logged data中训练Matrix Factorization模型，学习物品的协同过滤表示

### 5.2 为什么需要MF？

**问题：** SAC+WkNN需要在连续的embedding空间中选择动作，但是：
- ❌ 不能用`item_embeddings_focused.pt`（这是特权信息）
- ❌ 不能随机初始化（需要有意义的物品表示）

**解决方案：** 从用户行为数据中学习物品表示
```
Logged Data (用户点击历史)
    ↓
Matrix Factorization (BPR loss)
    ↓
MF Embeddings (物品的协同过滤表示)
```

### 5.3 MF训练过程

```python
# train_MF.py 的核心逻辑
item_embeddings = MFEmbeddings(num_items=1000, embedd_dim=20)

# 从logged data加载用户-物品交互
dataset = torch.load("data/RecSim/datasets/focused_topdown.pt")

# 训练MF模型（BPR loss）
for epoch in range(epochs):
    for user, positive_item in dataset:
        # 采样负样本
        negative_item = sample_negative()

        # BPR loss: 正样本得分 > 负样本得分
        loss = -log(sigmoid(score(user, positive_item) -
                           score(user, negative_item)))

        # 更新embeddings
        optimizer.step()

# 保存学习到的物品embeddings
torch.save(item_embeddings.weight, "MF_embeddings/focused_topdown.pt")
```

### 5.4 MF是否需要预训练？

**答案：是的，需要预训练！**

**原因：**
1. **原作者已经提供了预训练好的MF embeddings**（在gems-master.zip中）
2. **你不需要重新训练**，直接使用即可
3. **如果要重新训练**，需要运行：
   ```bash
   python GeMS/train_MF.py --MF_dataset="focused_topdown.pt"
   ```

**你的情况：**
- ✅ 你已经有了原作者提供的MF embeddings
- ✅ 可以直接用于SAC+WkNN实验
- ❌ 不需要重新训练（除非你想验证结果）

---

## 6. 数据流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    RecSim Environment                        │
│  (使用 item_embeddings_focused.pt 计算真实相关性)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ ε-greedy oracle采样
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Logged Data                               │
│  focused_topdown.pt (100K trajectories, 1.6GB)              │
│  包含: states, slates, clicks, rewards                       │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ↓                       ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│   训练GeMS VAE           │  │   训练MF Embeddings      │
│   (pretrain_ranker.py)   │  │   (train_MF.py)          │
└──────────────────────────┘  └──────────────────────────┘
                ↓                       ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│  GeMS Checkpoint         │  │  MF Embeddings           │
│  (用于SAC+GeMS)          │  │  (用于SAC+WkNN)          │
└──────────────────────────┘  └──────────────────────────┘
                │                       │
                └───────────┬───────────┘
                            ↓
                ┌─────────────────────┐
                │   RL Training       │
                │  (train_agent.py)   │
                └─────────────────────┘
                            ↓
                ┌─────────────────────┐
                │  Trained RL Agent   │
                └─────────────────────┘
```

---

## 7. 总结：你需要知道的关键点

### ✅ 你已经有的数据
1. **Item Embeddings** (原作者提供)
2. **Logged Data** (你已生成)
3. **MF Embeddings** (原作者提供)
4. **GeMS Checkpoints** (你已训练)

### 🎯 各类数据的用途
- **Item Embeddings**: RecSim环境内部使用，计算真实相关性
- **Logged Data**: 训练GeMS和MF的原始数据
- **MF Embeddings**: SAC+WkNN baseline使用
- **GeMS Checkpoints**: SAC+GeMS方法使用

### 🔑 关键理解
1. **Embedding = 物品的向量表示**
2. **不同方法使用不同的embeddings**
3. **Ideal embeddings是特权信息，实际方法不能用**
4. **MF embeddings是从用户行为中学习的，实际可用**
5. **原作者已经提供了MF embeddings，不需要重新训练**

### 🚀 下一步
你现在可以直接运行4个baseline实验：
1. **SAC+WkNN** - 使用MF embeddings ✅
2. **REINFORCE+SoftMax** - 使用scratch embeddings ✅
3. **SAC+TopK (ideal)** - 使用ideal embeddings ✅
4. **SlateQ** - 使用scratch embeddings ✅

所有前置条件都已满足！
