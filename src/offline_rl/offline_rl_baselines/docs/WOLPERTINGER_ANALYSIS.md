# Wolpertinger算法分析报告

**日期**: 2025-12-01
**目的**: 分析rl_wolpertinger和rl_slate_wolpertinger两个项目，评估作为GeMS离线RL baseline的可行性

---

## 1. 项目概述

### 1.1 论文来源

**论文**: "Scalable Deep Q-learning for Session-Based Slate Recommendation"
**发表**: ACM RecSys 2023
**作者**: Aayush Singha Roy, Edoardo D'Amico (Insight Centre)

### 1.2 两个项目的关系

| 项目 | 描述 | 核心算法 |
|------|------|----------|
| `rl_wolpertinger` | 单物品推荐版本 | Wolpertinger Actor + DQN Critic |
| `rl_slate_wolpertinger` | Slate推荐版本 | Wolpertinger Slate Actor + DQN Critic |

**关键区别**:
- `rl_wolpertinger`: Actor输出单个proto-item (20维)
- `rl_slate_wolpertinger`: Actor输出proto-slate (slate_size × 20维 = 100维)

---

## 2. 强化学习建模分析

### 2.1 状态空间 (State)

```python
# 状态定义
user_state: torch.Tensor  # shape: (num_user_features,) = (20,)
```

**状态来源**: `user_modeling/user_state.py`

```python
class ObservedUserState:
    """用户状态模型"""
    def __init__(self, num_user_features=20):
        self.user_state = np.random.randn(num_user_features)

    def update_state(self, selected_doc_feature):
        # 用户状态根据选择的文档更新
        # 简单的加权平均更新
        self.user_state = alpha * self.user_state + (1-alpha) * selected_doc_feature
```

**与GeMS对比**:
| 特性 | Wolpertinger | GeMS |
|------|--------------|------|
| 状态维度 | 20维 | 20维 (belief_state) |
| 状态来源 | 直接用户特征 | GRU编码的belief state |
| 历史建模 | 简单加权平均 | RNN序列编码 |
| 复杂度 | 低 | 高 |

### 2.2 动作空间 (Action)

#### rl_wolpertinger (单物品)

```python
# Actor输出
proto_action: torch.Tensor  # shape: (20,) - 一个proto-item

# 动作选择流程
1. Actor网络输出proto_action (20维连续向量)
2. k-NN搜索找到最近的k个候选物品
3. Critic评估这k个物品的Q值
4. 选择Q值最高的物品
```

#### rl_slate_wolpertinger (Slate)

```python
# Actor输出
proto_slate: torch.Tensor  # shape: (slate_size * 20,) = (100,)

# 动作选择流程
1. Actor网络输出proto_slate (100维 = 5个proto-item)
2. 对每个proto-item，k-NN搜索找到最近的k个候选
3. 组合成slate
4. Critic评估slate的Q值
```

**与GeMS对比**:
| 特性 | Wolpertinger | GeMS |
|------|--------------|------|
| 动作类型 | Proto-action + k-NN | Latent action + Ranker |
| 动作维度 | 20维(单物品)/100维(slate) | 32维 (latent) |
| 解码方式 | k-NN最近邻搜索 | VAE解码器 |
| 可微性 | 不可微（k-NN） | 可微（VAE） |

### 2.3 奖励函数 (Reward)

```python
# response_model.py
class WeightedDotProductResponseModel:
    def generate_response(self, user_state, doc_feature, doc_quality):
        # 奖励 = (1-alpha) * 相关性 + alpha * 质量
        relevance = np.dot(user_state, doc_feature)
        reward = (1 - self.alpha) * relevance + self.alpha * doc_quality
        return reward * self.amp_factor
```

**奖励组成**:
- **相关性**: 用户状态与文档特征的点积
- **质量**: 文档固有质量分数
- **权重**: alpha控制两者比例 (默认0.25)

**与GeMS对比**:
| 特性 | Wolpertinger | GeMS |
|------|--------------|------|
| 奖励来源 | 点积相关性 + 质量 | 用户点击反馈 |
| 奖励范围 | 连续值 | 离散/连续 |
| 复杂度 | 简单线性 | 复杂用户模型 |

### 2.4 状态转移 (Transition)

```python
# environment.py - SlateGym
def step(self, slate):
    # 1. 用户选择模型选择slate中的一个物品
    selected_doc_idx = self.curr_user.choice_model.choose_document()

    # 2. 生成奖励
    response = self.curr_user.response_model.generate_response(...)

    # 3. 更新用户状态
    self.curr_user.state_model.update_state(selected_doc_feature)

    # 4. 更新用户预算
    self.curr_user.update_budget(response, doc_length)

    # 5. 检查是否终止
    is_terminal = self.curr_user.is_terminal()

    return selected_doc_feature, doc_quality, response, is_terminal, info
```

---

## 3. 算法实现分析

### 3.1 网络架构

#### Actor网络 (Wolpertinger)

```python
# wp_agent.py
class WolpertingerActor(nn.Module):
    def __init__(self, nn_dim, k, input_dim=20):
        # 输入: user_state (20维)
        # 输出: proto_action (20维)
        layers = []
        for i, dim in enumerate(nn_dim):
            if i == 0:
                layers.append(nn.Linear(input_dim, dim))
            elif i == len(nn_dim) - 1:
                layers.append(nn.Linear(dim, 20))  # 输出20维
            else:
                layers.append(nn.Linear(dim, dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x
```

#### Actor网络 (Slate版本)

```python
# wp_slate_agent.py
class WolpertingerActorSlate(nn.Module):
    def __init__(self, nn_dim, k, input_dim=20, slate_size=5):
        # 输入: user_state (20维)
        # 输出: proto_slate (slate_size * 20 = 100维)
        layers = []
        for i, dim in enumerate(nn_dim):
            if i == 0:
                layers.append(nn.Linear(input_dim, dim))
            elif i == len(nn_dim) - 1:
                layers.append(nn.Linear(nn_dim[i-1], slate_size * 20))
            else:
                layers.append(nn.Linear(nn_dim[i-1], dim))
```

#### Critic网络 (DQN)

```python
# dqn_agent.py
class DQNnet(nn.Module):
    def __init__(self, input_size, hidden_dims=[40,20,10,5], output_size=1):
        # 输入: state + action (40维)
        # 输出: Q值 (1维)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x)
        return x

class DQNAgent:
    def compute_q_values(self, state, candidate_docs_repr, use_policy_net=True):
        # 拼接state和action
        input1 = torch.cat([state, candidate_docs_repr], dim=1)
        q_val = self.policy_net(input1)
        return q_val
```

### 3.2 训练流程

#### SlateQ训练 (topic_run_simulation.py)

```python
def optimize_model(batch):
    # 1. 计算当前Q值
    q_val = agent.compute_q_values(state_batch, selected_doc_feat_batch)

    # 2. 计算目标Q值 (考虑用户选择概率)
    for b in range(batch_size):
        # 获取候选物品的Q值
        cand_qtgt = agent.compute_q_values(next_state, candidates)

        # 用户选择概率 (softmax)
        choice_model.score_documents(next_state, candidates)
        scores = torch.softmax(scores, dim=0)

        # 加权Q值 (考虑用户选择概率)
        topk = torch.topk((cand_qtgt * scores), k=SLATE_SIZE)
        curr_q_tgt = torch.sum(topk.values / p_sum)

    # 3. TD目标
    expected_q_values = q_tgt * GAMMA + reward

    # 4. 损失函数
    loss = criterion(q_val, expected_q_values)
    loss.backward()
    optimizer.step()
```

#### Wolpertinger训练 (topic_wa_run_simulation.py)

```python
def optimize_model(batch):
    # 1. Critic更新 (同SlateQ)
    loss = criterion(q_val, expected_q_values)
    loss.backward()
    optimizer.step()

    # 2. Actor更新 (DDPG风格)
    actor_loss = -agent.compute_q_values(
        state_batch,
        actor.compute_proto_item(state_batch),  # Actor输出
        use_policy_net=True
    ).mean()
    actor_loss.backward()
    actor_optimizer.step()
```

### 3.3 k-NN动作选择

```python
# wp_agent.py
def k_nearest(self, input_state, candidate_docs, use_actor_policy_net):
    # 1. 计算proto_action
    proto_action = self.compute_proto_item(input_state)

    # 2. 计算与所有候选物品的距离
    distances = torch.linalg.norm(candidate_docs - proto_action, axis=1)

    # 3. 选择k个最近的
    indices = torch.argsort(distances, dim=0)[:self.k]
    candidates_subset = candidate_docs[indices]

    return candidates_subset, indices
```

---

## 4. 测试环境分析

### 4.1 环境配置 (config.yaml)

```yaml
parameters:
  # 用户相关
  num_users: 5000
  num_user_features: 20
  sess_budget: 200  # 每个session的预算

  # 环境相关
  slate_size: 5
  num_candidates: 300  # 候选物品数量
  num_item_features: 20
  alpha_response: 0.25  # 奖励中质量的权重

  # 训练相关
  replay_memory_capacity: 10000
  batch_size: 30
  gamma: 1.0  # 折扣因子
  tau: 0.0001  # 软更新率
  lr: 1e-4
  num_episodes: 10000

  # Wolpertinger特有
  nearest_neighbours: 15  # k-NN的k值
```

### 4.2 环境组件

```
SlateGym环境
├── UserSampler (用户采样器)
│   ├── UserFeaturesGenerator (用户特征生成)
│   ├── StateModel (状态模型)
│   ├── ChoiceModel (选择模型)
│   └── ResponseModel (响应模型)
│
├── DocumentSampler (文档采样器)
│   └── 随机生成候选文档特征
│
└── SlateGenerator (Slate生成器)
    └── TopKSlateGenerator
```

### 4.3 与GeMS/RecSim环境对比

| 特性 | Wolpertinger环境 | GeMS/RecSim环境 |
|------|------------------|-----------------|
| 用户模型 | 简单点积模型 | 复杂兴趣演化模型 |
| 物品表示 | 随机生成特征 | 预训练embeddings |
| 候选集 | 每episode随机采样 | 固定物品库 |
| 状态转移 | 简单加权平均 | 复杂用户动态 |
| 终止条件 | 预算耗尽 | 用户离开 |
| 复杂度 | 低 | 高 |

---

## 5. 数据集分析

### 5.1 Wolpertinger的数据生成

**Wolpertinger不使用预收集的离线数据集**，而是：

```python
# 在线交互生成数据
for i_episode in range(NUM_EPISODES):
    env.reset()  # 采样新用户和候选文档

    while not is_terminal:
        # 1. 选择动作
        slate = agent.get_action(scores, q_val)

        # 2. 环境交互
        response, is_terminal = env.step(slate)

        # 3. 存入ReplayBuffer
        replay_memory_dataset.push(Transition(
            state, selected_doc_feat, candidates, response, next_state
        ))

    # 4. 从Buffer采样训练
    batch = next(iter(replay_memory_dataloader))
    optimize_model(batch)
```

### 5.2 Transition数据结构

```python
class Transition(NamedTuple):
    state: torch.Tensor           # (20,) 用户状态
    selected_doc_feat: torch.Tensor  # (20,) 选中的文档特征
    candidate_docs: torch.Tensor  # (num_candidates, 20) 候选文档
    reward: torch.Tensor          # (1,) 奖励
    next_state: torch.Tensor      # (20,) 下一状态
```

### 5.3 与GeMS离线数据集对比

| 特性 | Wolpertinger | GeMS离线数据集 |
|------|--------------|----------------|
| 数据来源 | 在线交互生成 | 预收集的离线数据 |
| 状态表示 | 原始用户特征 | GRU编码的belief |
| 动作表示 | 选中的物品特征 | latent action |
| 候选集 | 每step变化 | 固定 |
| 数据量 | 动态生成 | 1M transitions |

---

## 6. 迁移到GeMS项目的可行性分析

### 6.1 核心差异总结

```
Wolpertinger架构:
user_state (20维) → Actor → proto_action (20维) → k-NN → item_id → Env

GeMS架构:
belief_state (20维) → Agent → latent_action (32维) → Ranker → slate → Env
```

**关键差异**:
1. **动作空间**: Wolpertinger在物品embedding空间，GeMS在latent空间
2. **解码方式**: Wolpertinger用k-NN，GeMS用VAE
3. **训练方式**: Wolpertinger在线训练，GeMS需要离线训练
4. **状态编码**: Wolpertinger直接用特征，GeMS用RNN编码

### 6.2 迁移方案

#### 方案A: 直接迁移Wolpertinger到GeMS环境

**思路**: 保持Wolpertinger的Actor-Critic结构，但适配GeMS的状态和动作空间

```python
# 修改后的Wolpertinger for GeMS
class WolpertingerActorGeMS(nn.Module):
    def __init__(self, input_dim=20, output_dim=32):  # 适配GeMS维度
        # 输入: belief_state (20维)
        # 输出: latent_action (32维)
        ...

    def forward(self, belief_state):
        return latent_action  # 32维

# 使用GeMS的Ranker替代k-NN
class WolpertingerAgentGeMS:
    def select_action(self, belief_state, ranker):
        latent_action = self.actor(belief_state)
        slate = ranker.rank(latent_action)  # 用GeMS ranker解码
        return slate
```

**优点**:
- 保持Wolpertinger的Actor-Critic结构
- 可以与GeMS的Ranker组合

**缺点**:
- 需要修改网络维度
- k-NN的优势（可解释性）丢失

#### 方案B: 保持Wolpertinger原始设计，适配GeMS数据

**思路**: 保持Wolpertinger的k-NN设计，但使用GeMS的物品embeddings

```python
# 使用GeMS的物品embeddings作为候选集
item_embeddings = load_gems_item_embeddings()  # (1000, 20)

class WolpertingerAgentOriginal:
    def select_action(self, belief_state, item_embeddings):
        proto_action = self.actor(belief_state)  # (20,)

        # k-NN在物品embedding空间搜索
        distances = torch.norm(item_embeddings - proto_action, dim=1)
        topk_indices = torch.topk(-distances, k=self.k).indices

        # Critic评估
        q_values = self.critic(belief_state, item_embeddings[topk_indices])
        best_idx = topk_indices[q_values.argmax()]

        return best_idx
```

**优点**:
- 保持k-NN的可解释性
- 实现简单

**缺点**:
- 只能推荐单个物品，不是slate
- 与GeMS的latent action设计不兼容

#### 方案C: 混合方案（推荐）

**思路**: 将Wolpertinger作为一种特殊的Ranker，与GeMS的Agent解耦

```python
# Wolpertinger作为Ranker
class WolpertingerRanker(BaseRanker):
    """
    将latent_action通过k-NN映射到slate
    """
    def __init__(self, item_embeddings, k=10, slate_size=10):
        self.item_embeddings = item_embeddings  # (num_items, embed_dim)
        self.k = k
        self.slate_size = slate_size

    def rank(self, latent_action):
        # 将latent_action投影到物品embedding空间
        proto_items = self.project(latent_action)  # (slate_size, embed_dim)

        slate = []
        for proto_item in proto_items:
            # k-NN搜索
            distances = torch.norm(self.item_embeddings - proto_item, dim=1)
            nearest_idx = distances.argmin()
            slate.append(nearest_idx)

        return torch.tensor(slate)
```

**优点**:
- 与GeMS架构完全兼容
- 可以作为Ranker的一种选择
- 保持k-NN的可解释性

**缺点**:
- 需要额外的投影层

### 6.3 离线训练适配

**核心问题**: Wolpertinger是在线算法，如何用离线数据训练？

```python
# 离线版Wolpertinger
class OfflineWolpertinger:
    def train(self, offline_dataset):
        """
        使用GeMS离线数据集训练

        数据集格式:
        - observations: belief_states (1M, 20)
        - actions: latent_actions (1M, 32)
        - rewards: (1M,)
        """
        for batch in dataloader:
            states, actions, rewards, next_states, dones = batch

            # Critic更新 (标准DQN)
            q_val = self.critic(states, actions)
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                q_next = self.critic_target(next_states, next_actions)
                q_target = rewards + gamma * (1 - dones) * q_next

            critic_loss = F.mse_loss(q_val, q_target)

            # Actor更新 (DDPG风格)
            actor_loss = -self.critic(states, self.actor(states)).mean()

            # 可选: 添加BC约束 (类似TD3+BC)
            bc_loss = F.mse_loss(self.actor(states), actions)
            actor_loss = actor_loss + alpha * bc_loss
```

---

## 7. 作为Baseline的实施建议

### 7.1 建议的Baseline列表

基于分析，建议将以下算法作为baseline：

| 算法 | 类型 | 来源 | 优先级 |
|------|------|------|--------|
| TD3+BC | 离线RL | CORL | 高 |
| CQL | 离线RL | CORL | 高 |
| IQL | 离线RL | CORL | 高 |
| SAC (离线) | 在线→离线 | GeMS | 中 |
| **Wolpertinger (离线)** | 在线→离线 | 本项目 | 中 |
| **SlateQ** | 在线 | 本项目 | 低 |

### 7.2 Wolpertinger实施步骤

#### 步骤1: 创建离线版Wolpertinger Agent

```
agents/online/wolpertinger.py
├── WolpertingerActor (网络)
├── WolpertingerCritic (网络)
└── WolpertingerAgent (算法)
    ├── select_action()
    ├── train()  # 离线训练
    ├── save()
    └── load()
```

#### 步骤2: 创建Wolpertinger Ranker（可选）

```
rankers/wolpertinger_ranker.py
├── WolpertingerRanker
    ├── __init__(item_embeddings, k)
    ├── rank(latent_action) -> slate
    └── project(latent_action) -> proto_items
```

#### 步骤3: 适配训练脚本

```python
# scripts/train_wolpertinger.py
def train_wolpertinger(config):
    # 1. 加载离线数据
    dataset = np.load(config.dataset_path)
    buffer = ReplayBuffer(...)
    buffer.load_d4rl_dataset(dataset)

    # 2. 初始化Agent
    agent = WolpertingerAgent(
        state_dim=20,
        action_dim=32,  # 适配GeMS
        ...
    )

    # 3. 训练循环
    for step in range(max_steps):
        batch = buffer.sample(batch_size)
        log_dict = agent.train(batch)
```

### 7.3 实验设计

#### 实验1: Agent对比（固定GeMS Ranker）

| Agent | Ranker | 预期结果 |
|-------|--------|----------|
| TD3+BC | GeMS | 高性能 |
| CQL | GeMS | 高性能 |
| IQL | GeMS | 高性能 |
| Wolpertinger (离线) | GeMS | 中等性能 |
| SAC (离线) | GeMS | 低性能 |

#### 实验2: Ranker对比（固定TD3+BC Agent）

| Agent | Ranker | 预期结果 |
|-------|--------|----------|
| TD3+BC | GeMS | 高性能 |
| TD3+BC | WkNN | 中等性能 |
| TD3+BC | Wolpertinger | 中等性能 |
| TD3+BC | Softmax | 低性能 |

---

## 8. 关键技术挑战

### 8.1 动作空间不匹配

**问题**: Wolpertinger在物品embedding空间(20维)，GeMS在latent空间(32维)

**解决方案**:
```python
# 方案1: 修改Wolpertinger输出维度
actor_output_dim = 32  # 匹配GeMS

# 方案2: 添加投影层
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=20, output_dim=32):
        self.proj = nn.Linear(input_dim, output_dim)
```

### 8.2 k-NN在离线设置下的问题

**问题**: k-NN需要候选集，但离线数据中候选集可能不完整

**解决方案**:
```python
# 使用固定的物品embedding库
item_embeddings = load_all_item_embeddings()  # (1000, 20)

# 在整个物品库上做k-NN
def k_nearest_global(proto_action, item_embeddings, k):
    distances = torch.norm(item_embeddings - proto_action, dim=1)
    return torch.topk(-distances, k=k).indices
```

### 8.3 Slate生成的组合爆炸

**问题**: Wolpertinger Slate版本需要为每个proto-item做k-NN，计算量大

**解决方案**:
```python
# 批量k-NN
def batch_k_nearest(proto_slate, item_embeddings, k):
    # proto_slate: (slate_size, embed_dim)
    # 使用FAISS加速
    import faiss
    index = faiss.IndexFlatL2(embed_dim)
    index.add(item_embeddings.numpy())

    D, I = index.search(proto_slate.numpy(), k)
    return I  # (slate_size, k)
```

---

## 9. 总结与建议

### 9.1 可行性评估

| 方面 | 评估 | 说明 |
|------|------|------|
| 算法迁移 | ✅ 可行 | 需要适配维度和训练方式 |
| 离线训练 | ✅ 可行 | 类似DDPG+BC |
| 与GeMS兼容 | ⚠️ 部分兼容 | 需要投影层或修改维度 |
| 作为Baseline | ✅ 有价值 | 提供不同视角的对比 |

### 9.2 实施优先级

1. **高优先级**: 先完成TD3+BC/CQL/IQL的实现和测试
2. **中优先级**: 实现离线版Wolpertinger作为额外baseline
3. **低优先级**: 实现Wolpertinger Ranker作为Ranker选项

### 9.3 预期贡献

将Wolpertinger作为baseline可以提供：
1. **方法论对比**: k-NN vs VAE解码
2. **可解释性**: Wolpertinger的proto-action更直观
3. **计算效率**: k-NN可能比VAE更快
4. **泛化性测试**: 不同解码方式的泛化能力

---

## 附录: 关键代码位置

### rl_wolpertinger

| 文件 | 功能 |
|------|------|
| `src/rl_recsys/agent_modeling/wp_agent.py` | Wolpertinger Actor |
| `src/rl_recsys/agent_modeling/dqn_agent.py` | DQN Critic |
| `src/scripts/simulation/topic_wa_run_simulation.py` | 训练脚本 |
| `src/rl_recsys/simulation_environment/environment.py` | 环境 |
| `src/scripts/config.yaml` | 配置文件 |

### rl_slate_wolpertinger

| 文件 | 功能 |
|------|------|
| `src/rl_recsys/agent_modeling/wp_slate_agent.py` | Slate版Actor |
| `src/scripts/simulation/topic_wa_slate_simulation.py` | Slate版训练脚本 |

---

**文档版本**: v1.0
**最后更新**: 2025-12-01
