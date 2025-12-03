# GeMS离线RL框架重构计划 - 最终确认版

**日期**: 2025-12-01
**状态**: 需求已确认，待执行
**目标**: 建立完全解耦的模块化架构，支持Agent/Ranker/BeliefEncoder的自由组合

---

## 🎯 核心设计理念

### 1. 完全解耦的三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Belief Encoder Layer (可替换)                               │
│  - GRU Belief (当前)                                         │
│  - LSTM Belief (未来)                                        │
│  - Transformer Belief (未来)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    belief_state (20维)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Agent Layer (可替换)                                         │
│  - 离线算法: TD3+BC, CQL, IQL, Decision Diffuser            │
│  - 在线算法: SAC, Reinforce (用离线数据训练)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    latent_action (32维)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Ranker Layer (可替换)                                        │
│  - GeMS Ranker (VAE-based)                                   │
│  - WkNN Ranker (k-NN based)                                  │
│  - Softmax Ranker (直接softmax)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    slate (10个物品ID)
                            ↓
                    Environment → reward
```

### 2. 关键约束与假设

#### 约束1: 潜空间训练
- **所有Agent都在潜空间中学习**
- Actor输入: belief_state (20维)
- Actor输出: latent_action (32维)
- Critic输入: (belief_state, latent_action)
- **训练时不需要Ranker，不需要Environment**

#### 约束2: Reward的归属
- 数据集中的reward是通过`SAC + GeMS Ranker`产生的
- 训练新Agent时，**必须完全信任数据集中的reward**
- Agent学习的是: `(belief_state, latent_action) -> reward`的映射
- Ranker的逻辑隐含在reward中

#### 约束3: 接口维度可配置
- 默认: state_dim=20, action_dim=32
- 但应该在Config中可配置，为未来扩展做准备

---

## 📁 目录结构设计

```
offline_rl_baselines/
│
├── agents/                          # Agent层（潜空间策略学习）
│   ├── __init__.py
│   ├── base_agent.py                # 基类，定义统一接口
│   │
│   ├── offline/                     # 离线RL算法
│   │   ├── __init__.py
│   │   ├── td3_bc.py                # TD3+BC
│   │   ├── cql.py                   # Conservative Q-Learning
│   │   ├── iql.py                   # Implicit Q-Learning
│   │   └── decision_diffuser.py     # Decision Diffuser (未来)
│   │
│   └── online/                      # 在线算法（用离线数据训练）
│       ├── __init__.py
│       ├── sac.py                   # Soft Actor-Critic
│       └── reinforce.py             # REINFORCE
│
├── rankers/                         # Ranker层（潜空间→slate解码）
│   ├── __init__.py
│   ├── base_ranker.py               # 基类，定义统一接口
│   ├── gems_ranker.py               # GeMS VAE ranker
│   ├── wknn_ranker.py               # k近邻ranker
│   └── softmax_ranker.py            # Softmax ranker
│
├── belief_encoders/                 # Belief Encoder层（obs→belief_state）
│   ├── __init__.py
│   ├── base_encoder.py              # 基类
│   ├── gru_belief.py                # GRU编码器（当前）
│   └── lstm_belief.py               # LSTM编码器（未来）
│
├── envs/                            # 环境包装
│   ├── __init__.py
│   └── gems_env.py                  # 完整环境（用于在线评估）
│
├── common/                          # 通用组件
│   ├── __init__.py
│   ├── buffer.py                    # ReplayBuffer
│   ├── networks.py                  # 神经网络模块
│   └── utils.py                     # 工具函数
│
├── configs/                         # 配置文件
│   ├── agents/                      # Agent配置
│   │   ├── td3_bc.yaml
│   │   ├── cql.yaml
│   │   └── sac.yaml
│   ├── rankers/                     # Ranker配置
│   │   ├── gems.yaml
│   │   └── wknn.yaml
│   └── experiments/                 # 实验配置
│       └── baseline_comparison.yaml
│
├── scripts/                         # 训练脚本
│   ├── train_agent.py               # 通用Agent训练脚本
│   ├── train_ranker.py              # Ranker训练脚本
│   └── evaluate.py                  # 评估脚本
│
├── experiments/                     # 实验结果
│   ├── logs/
│   ├── checkpoints/
│   └── results/
│
└── docs/                            # 文档
    ├── REFACTORING_PLAN_FINAL.md    # 本文档
    ├── API_REFERENCE.md             # API文档
    └── EXPERIMENT_GUIDE.md          # 实验指南
```

---

## 🔌 接口设计

### 1. BaseAgent接口

```python
# agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import torch

class BaseAgent(ABC):
    """
    Agent基类，定义统一接口

    所有Agent都在潜空间中工作：
    - 输入: belief_state (state_dim维)
    - 输出: latent_action (action_dim维)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def select_action(
        self,
        belief_state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        选择动作

        Args:
            belief_state: (state_dim,) belief state
            deterministic: 是否使用确定性策略

        Returns:
            latent_action: (action_dim,) latent action
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练一步

        Args:
            batch: {
                'states': (batch_size, state_dim),
                'actions': (batch_size, action_dim),
                'rewards': (batch_size,),
                'next_states': (batch_size, state_dim),
                'dones': (batch_size,)
            }

        Returns:
            log_dict: {'loss': ..., 'q_value': ..., ...}
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError
```

### 2. BaseRanker接口

```python
# rankers/base_ranker.py

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch

class BaseRanker(ABC):
    """
    Ranker基类，定义统一接口

    Ranker的作用：将latent_action解码为slate
    """

    def __init__(
        self,
        action_dim: int,
        num_items: int,
        slate_size: int,
        device: str = "cuda"
    ):
        self.action_dim = action_dim
        self.num_items = num_items
        self.slate_size = slate_size
        self.device = device

    @abstractmethod
    def rank(self, latent_action: np.ndarray) -> np.ndarray:
        """
        将latent action解码为slate

        Args:
            latent_action: (action_dim,) 或 (batch_size, action_dim)

        Returns:
            slate: (slate_size,) 或 (batch_size, slate_size)
                   物品ID列表
        """
        raise NotImplementedError

    def train(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        训练ranker（如果需要）

        Args:
            batch: {
                'latent_actions': (batch_size, action_dim),
                'slates': (batch_size, slate_size),
                'clicks': (batch_size, slate_size)  # optional
            }

        Returns:
            log_dict: {'loss': ..., ...} 或 None（如果不需要训练）
        """
        return None  # 默认不需要训练

    def save(self, path: str):
        """保存模型（如果需要）"""
        pass

    def load(self, path: str):
        """加载模型（如果需要）"""
        pass
```

### 3. BaseBeliefEncoder接口

```python
# belief_encoders/base_encoder.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import torch

class BaseBeliefEncoder(ABC):
    """
    Belief Encoder基类，定义统一接口

    Belief Encoder的作用：将原始observation编码为belief_state
    """

    def __init__(
        self,
        state_dim: int,
        device: str = "cuda"
    ):
        self.state_dim = state_dim
        self.device = device

    @abstractmethod
    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        编码observation为belief state

        Args:
            obs: 原始observation（RecSim格式）

        Returns:
            belief_state: (state_dim,) belief state
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """重置内部状态（如RNN的hidden state）"""
        raise NotImplementedError

    def save(self, path: str):
        """保存模型"""
        pass

    def load(self, path: str):
        """加载模型"""
        pass
```

---

## 🔄 训练流程设计

### Phase 1: Ranker训练（可选）

```bash
# 如果需要训练新的GeMS Ranker
python scripts/train_ranker.py \
    --ranker_type gems \
    --dataset_path offline_datasets/diffuse_topdown_expert.npz \
    --output_dir experiments/rankers/gems_v1 \
    --epochs 100
```

**数据来源**:
- 使用离线数据集中的`(latent_action, slate, clicks)`
- 混合Expert/Medium/Random数据以保证多样性

**训练目标**:
- GeMS Ranker: 学习`latent_action -> slate`的VAE映射
- 预测clicks作为监督信号

### Phase 2: Agent训练（核心）

```bash
# 训练TD3+BC Agent
python scripts/train_agent.py \
    --agent_type td3_bc \
    --dataset_path offline_datasets/diffuse_topdown_expert.npz \
    --state_dim 20 \
    --action_dim 32 \
    --max_timesteps 1000000 \
    --output_dir experiments/agents/td3_bc_v1
```

**关键点**:
- **不需要Ranker**，不需要Environment
- 只需要数据集中的`(belief_state, latent_action, reward)`
- 完全在潜空间中训练

### Phase 3: 在线评估

```bash
# 评估TD3+BC + GeMS Ranker
python scripts/evaluate.py \
    --agent_path experiments/agents/td3_bc_v1/final.pth \
    --agent_type td3_bc \
    --ranker_type gems \
    --ranker_path experiments/rankers/gems_v1/final.pth \
    --env_name diffuse_topdown \
    --n_episodes 100
```

**评估流程**:
```
1. 加载Agent
2. 加载Ranker
3. 加载Belief Encoder
4. 创建Environment
5. For each episode:
   - obs = env.reset()
   - belief_state = belief_encoder.encode(obs)
   - latent_action = agent.select_action(belief_state)
   - slate = ranker.rank(latent_action)
   - next_obs, reward, done = env.step(slate)
   - 累积reward
```

---

## 🧪 实验设计

### 实验1: Agent对比（主要实验）

**固定Ranker = GeMS**，对比不同Agent:

| Agent | Ranker | 数据集 | 预期结果 |
|-------|--------|--------|----------|
| TD3+BC | GeMS | Expert | 高性能（有BC约束） |
| CQL | GeMS | Expert | 高性能（保守估计） |
| IQL | GeMS | Expert | 高性能（隐式Q学习） |
| SAC (离线) | GeMS | Expert | 低性能（OOD问题） |
| Reinforce (离线) | GeMS | Expert | 低性能（方差大） |

**目的**: 证明离线RL算法的必要性

### 实验2: Ranker泛化性（次要实验）

**固定Agent = TD3+BC**，对比不同Ranker:

| Agent | Ranker | 数据集 | 预期结果 |
|-------|--------|--------|----------|
| TD3+BC | GeMS | Expert | 高性能（训练匹配） |
| TD3+BC | WkNN | Expert | 中等性能（OOD解码） |
| TD3+BC | Softmax | Expert | 低性能（简单解码） |

**目的**: 验证潜空间表示的鲁棒性

### 实验3: 组合矩阵（完整实验）

**所有Agent × 所有Ranker**:

```
         GeMS    WkNN    Softmax
TD3+BC    ✓       ✓        ✓
CQL       ✓       ✓        ✓
IQL       ✓       ✓        ✓
SAC       ✓       ✓        ✓
```

**目的**: 全面对比

---

## 📊 评估指标

### 主要指标（Primary Metrics）

使用**GeMS Ranker**评估（最公平）:

1. **Average Return**: 平均episode回报
2. **Success Rate**: 成功率（如果有定义）
3. **Training Stability**: 训练曲线的稳定性

### 次要指标（Secondary Metrics）

使用**WkNN/Softmax Ranker**评估（泛化性）:

1. **Cross-Ranker Performance**: 跨Ranker性能
2. **Robustness**: 鲁棒性

### 离线指标（Offline Metrics）

训练过程中的指标（不需要环境）:

1. **Q-value**: Q函数估计
2. **Policy Loss**: 策略损失
3. **BC Loss**: 行为克隆损失（如果有）

---

## 🚀 实施计划

### 阶段1: 基础架构（1-2天）

**任务**:
1. ✅ 创建目录结构
2. ✅ 实现BaseAgent接口
3. ✅ 实现BaseRanker接口
4. ✅ 实现BaseBeliefEncoder接口

**验证**:
- 接口定义清晰
- 类型注解完整
- 文档字符串完整

### 阶段2: Agent实现（2-3天）

**任务**:
1. ✅ 迁移TD3+BC到新架构
   - 清理冗余代码
   - 适配BaseAgent接口
   - 移除d4rl依赖
2. ✅ 迁移CQL到新架构
3. ✅ 迁移IQL到新架构
4. ✅ 实现SAC（在线转离线）
5. ✅ 实现Reinforce（在线转离线）

**验证**:
- 所有Agent继承BaseAgent
- 可以加载离线数据训练
- 保存/加载功能正常

### 阶段3: Ranker实现（1-2天）

**任务**:
1. ✅ 包装GeMS Ranker
   - 从原GeMS代码中提取
   - 适配BaseRanker接口
2. ✅ 实现WkNN Ranker
3. ✅ 实现Softmax Ranker
4. ✅ 实现Ranker训练脚本

**验证**:
- 所有Ranker继承BaseRanker
- rank()方法正常工作
- GeMS Ranker可以训练

### 阶段4: 训练脚本（1天）

**任务**:
1. ✅ 实现train_agent.py
   - 支持所有Agent类型
   - 使用argparse配置
   - 日志和checkpoint
2. ✅ 实现train_ranker.py
   - 支持GeMS Ranker训练
3. ✅ 实现evaluate.py
   - Agent + Ranker组合评估

**验证**:
- 可以训练任意Agent
- 可以训练GeMS Ranker
- 可以评估任意组合

### 阶段5: 测试与验证（1-2天）

**任务**:
1. ✅ 测试TD3+BC训练
   - 短时间训练（10K steps）
   - 验证loss下降
2. ✅ 测试完整流程
   - 训练Agent
   - 在线评估
3. ✅ 运行对比实验
   - 至少3个Agent
   - 至少2个Ranker

**验证**:
- TD3+BC性能合理
- 在线算法性能较差（符合预期）
- 不同Ranker有差异

---

## ⚠️ 关键技术问题与解决方案

### 问题1: Reward归属问题

**问题**: 数据集中的reward是通过GeMS Ranker得到的，训练新Agent时能直接用吗？

**解决方案**:
- ✅ **可以直接用**
- Agent学习的是`(belief_state, latent_action) -> reward`的映射
- Ranker的逻辑隐含在reward中
- 这是离线RL的标准假设

### 问题2: 评估公平性问题

**问题**: 如果用WkNN Ranker评估，但训练数据是GeMS Ranker产生的，公平吗？

**解决方案**:
- ✅ **主要评估用GeMS Ranker**（最公平）
- ✅ **次要评估用WkNN/Softmax**（测试泛化性）
- 明确区分两种评估的目的

### 问题3: Ranker训练的多样性问题

**问题**: 如果只用Expert数据训练Ranker，潜空间覆盖不够怎么办？

**解决方案**:
- ✅ **混合Expert/Medium/Random数据**
- 数据收集时已经包含了这些数据
- 训练Ranker时使用混合数据集

### 问题4: 在线算法转离线的实现

**问题**: SAC/Reinforce怎么用离线数据训练？

**解决方案**:
- ✅ **最小修改**：去掉环境交互，从buffer采样
- ✅ **保留原始算法逻辑**（不加BC约束）
- 预期会失败，这正是我们想展示的

---

## 📝 代码规范

### 1. 命名规范

- 类名: `PascalCase` (例如: `TD3BCAgent`)
- 函数名: `snake_case` (例如: `select_action`)
- 常量: `UPPER_CASE` (例如: `MAX_TIMESTEPS`)
- 私有方法: `_snake_case` (例如: `_update_target`)

### 2. 类型注解

```python
def select_action(
    self,
    belief_state: np.ndarray,
    deterministic: bool = False
) -> np.ndarray:
    ...
```

### 3. 文档字符串

```python
def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    训练一步

    Args:
        batch: 包含states, actions, rewards等的字典

    Returns:
        log_dict: 包含loss等训练信息的字典
    """
    ...
```

### 4. 配置管理

- 使用dataclass定义配置
- 支持从YAML文件加载
- 支持命令行覆盖

```python
@dataclass
class TD3BCConfig:
    state_dim: int = 20
    action_dim: int = 32
    alpha: float = 2.5
    ...
```

---

## ✅ 最终确认清单

在开始实施前，请确认以下所有项目：

- [ ] 目录结构清晰，符合需求
- [ ] 接口设计合理，支持扩展
- [ ] 训练流程明确，分为3个Phase
- [ ] 实验设计完整，包含对比实验
- [ ] 技术问题已识别，有解决方案
- [ ] 实施计划可行，时间合理
- [ ] 代码规范明确，易于维护

---

## 🎯 成功标准

项目成功的标准：

1. **架构清晰**:
   - ✅ 三层完全解耦
   - ✅ 接口统一
   - ✅ 易于扩展

2. **功能完整**:
   - ✅ 至少3个离线Agent可用（TD3+BC, CQL, IQL）
   - ✅ 至少2个在线Agent可用（SAC, Reinforce）
   - ✅ 至少2个Ranker可用（GeMS, WkNN）

3. **实验可复现**:
   - ✅ 训练脚本完整
   - ✅ 评估脚本完整
   - ✅ 配置文件完整

4. **性能合理**:
   - ✅ TD3+BC性能 > SAC (离线)
   - ✅ GeMS Ranker评估 > WkNN Ranker评估

---

**准备好开始实施了吗？请确认后我们开始动手！**
