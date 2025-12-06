# GeMS离线RL Baseline项目审阅文档

**日期**: 2025-12-01 (更新: 2025-12-05)
**状态**: 代码重构完成，在线/离线RL模块物理隔离
**作者**: Claude Code

---

## 📋 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [整体技术思路](#2-整体技术思路)
3. [代码架构与文件结构](#3-代码架构与文件结构)
4. [关键代码实现](#4-关键代码实现)
5. [数据收集与验证](#5-数据收集与验证)
6. [测试结果](#6-测试结果)
7. [当前状态与后续工作](#7-当前状态与后续工作)
8. [重构记录 (2025-12-05)](#8-重构记录-2025-12-05)

---

## 1. 项目背景与目标

### 1.1 研究背景

**GeMS (Generative Model for Slate Recommendation)** 是一个推荐系统框架，使用以下架构：
- **SAC (Soft Actor-Critic)**: 在线强化学习智能体
- **GeMS Ranker**: 将连续latent action解码为推荐slate
- **Belief Encoder**: 将用户历史编码为belief state

原始GeMS通过与RecSim环境交互进行在线训练。

### 1.2 项目目标

本项目的核心目标是：

1. **收集离线数据集**: 使用训练好的SAC+GeMS模型与环境交互，收集高质量的离线轨迹数据
2. **建立Baseline框架**: 实现主流离线RL算法（TD3+BC, CQL, IQL）作为baseline
3. **为Decision Diffuser做准备**: 这些baseline将作为后续Decision Diffuser算法的性能对比基准

### 1.3 关键约束

- **零依赖**: 不能修改现有的gems conda环境，不安装新依赖
- **代码隔离**: 与GeMS原始代码完全隔离，不污染原有项目
- **快速验证**: 短期内（1周）完成baseline验证
- **数据兼容**: 数据格式必须兼容D4RL标准，便于算法移植

---

## 2. 整体技术思路

### 2.1 数据流程

```
┌─────────────────────────────────────────────────────────────┐
│  阶段1: 数据收集 (已完成)                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
    训练好的SAC Agent + GeMS Ranker + Belief Encoder
                            ↓
              与RecSim环境交互 (10K episodes)
                            ↓
        收集轨迹: (belief_state, latent_action, reward, ...)
                            ↓
              保存为D4RL格式 (.npz文件)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  离线数据集                                                    │
│  - observations: (1M, 20)  # 20维belief states              │
│  - actions: (1M, 32)       # 32维连续latent actions         │
│  - rewards: (1M,)          # 即时奖励                        │
│  - next_observations: (1M, 20)                              │
│  - terminals: (1M,)        # 终止标志                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段2: 离线RL训练 (当前阶段)                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
              加载离线数据到ReplayBuffer
                            ↓
        训练离线RL算法 (TD3+BC / CQL / IQL)
                            ↓
              保存训练好的策略模型
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段3: 性能评估与对比                                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键技术点

#### 2.2.1 状态空间设计

**原始GeMS环境**:
- 观测空间: RecSim的复杂字典结构（用户状态、物品特征等）
- 需要belief encoder将历史编码为固定维度向量

**离线RL适配**:
- 状态空间: 20维belief state（已编码）
- 优势: 降维后的表示，便于离线学习
- 数据来源: 数据收集时已经通过belief encoder处理

#### 2.2.2 动作空间设计

**原始GeMS环境**:
- 动作空间: 离散slate（从候选集中选择10个物品）
- 组合爆炸: 候选集很大，直接学习困难

**GeMS的解决方案**:
- 使用32维连续latent action
- SAC学习latent action策略
- GeMS ranker将latent action解码为slate

**离线RL适配**:
- 动作空间: 32维连续latent action
- 优势: 连续动作空间，适合TD3/SAC等算法
- 数据来源: 数据收集时保存的是latent action，不是slate

#### 2.2.3 在线RL vs 离线RL的核心差异

| 特性 | 在线RL | 离线RL |
|------|--------|--------|
| 框架 | PyTorch Lightning | 纯PyTorch |
| ReplayBuffer | 动态交互，deque实现 | 静态D4RL格式，tensor预分配 |
| 网络定义 | Agent类内联构建 | 独立networks.py |
| 参数配置 | argparse (MyParser) | @dataclass |
| 日志系统 | SwanLab | WandB (待迁移) |

---

## 3. 代码架构与文件结构

### 3.1 重构后的目录树 (2025-12-05更新)

```
offline-slate-rl/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── online.py                    # 在线RL算法 (PyTorch Lightning)
│   │   │                                # DQN, SAC, SlateQ, REINFORCE等
│   │   └── offline/                     # 离线RL算法 (纯PyTorch)
│   │       ├── __init__.py
│   │       ├── td3_bc.py                # TD3+BC算法 ✅
│   │       ├── cql.py                   # CQL算法 ⚠️
│   │       └── iql.py                   # IQL算法 ⚠️
│   │
│   ├── common/                          # ← 重构核心
│   │   ├── __init__.py                  # 延迟导入模式
│   │   ├── logger.py                    # 共享：SwanLab日志
│   │   │
│   │   ├── online/                      # 在线RL专用
│   │   │   ├── __init__.py
│   │   │   ├── buffer.py                # 动态ReplayBuffer (deque)
│   │   │   ├── data_module.py           # BufferDataModule (Lightning)
│   │   │   ├── env_wrapper.py           # EnvWrapper
│   │   │   └── argument_parser.py       # MyParser, MainParser
│   │   │
│   │   └── offline/                     # 离线RL专用
│   │       ├── __init__.py
│   │       ├── buffer.py                # D4RL格式ReplayBuffer (tensor)
│   │       ├── networks.py              # Actor, Critic, TwinQ等
│   │       └── utils.py                 # set_seed, compute_mean_std等
│   │
│   ├── rankers/gems/                    # GeMS Ranker
│   ├── belief_encoders/                 # Belief Encoder
│   ├── envs/RecSim/                     # RecSim环境
│   ├── training/                        # 训练循环
│   └── data_collection/                 # 数据收集工具
│
├── scripts/
│   ├── train_online_rl.py               # 在线RL训练入口 ✅
│   └── train_offline_rl.py              # 离线RL训练入口 (待创建)
│
├── config/
│   └── paths.py                         # 路径配置
│
├── data/
│   └── offline_datasets/                # 离线数据集
│       ├── diffuse_topdown_expert.npz
│       ├── diffuse_mix_expert.npz
│       └── diffuse_divpen_expert.npz
│
└── document/
    ├── PROJECT_REVIEW_20251201.md       # 本文档
    └── REFACTORING_FEASIBILITY_ANALYSIS_20251204.md  # 重构分析
```

### 3.2 核心模块说明

#### 3.2.1 `common/online/` - 在线RL工具

| 文件 | 内容 | 用途 |
|------|------|------|
| `buffer.py` | `ReplayBuffer`, `Trajectory` | 动态经验回放，支持环境交互 |
| `data_module.py` | `BufferDataset`, `BufferDataModule` | PyTorch Lightning数据模块 |
| `env_wrapper.py` | `EnvWrapper`, `get_file_name` | 环境包装器 |
| `argument_parser.py` | `MyParser`, `MainParser` | 命令行参数解析 |

#### 3.2.2 `common/offline/` - 离线RL工具

| 文件 | 内容 | 用途 |
|------|------|------|
| `buffer.py` | `ReplayBuffer` | D4RL格式静态buffer，tensor预分配 |
| `networks.py` | `Actor`, `Critic`, `TwinQ`, `TanhGaussianActor`, `ValueFunction` | 神经网络架构 |
| `utils.py` | `set_seed`, `compute_mean_std`, `soft_update`, `asymmetric_l2_loss` | 工具函数 |

#### 3.2.3 `agents/offline/` - 离线RL算法

| 文件 | 算法 | 状态 | 说明 |
|------|------|------|------|
| `td3_bc.py` | TD3+BC | ✅ 可用 | 确定性策略 + 行为克隆 |
| `cql.py` | CQL | ⚠️ 待处理 | 需要pyrallis/d4rl/wandb依赖 |
| `iql.py` | IQL | ⚠️ 待处理 | 需要pyrallis/d4rl/wandb依赖 |

### 3.3 导入路径变更

重构后的导入方式：

```python
# 在线RL
from common.online.buffer import ReplayBuffer, Trajectory
from common.online.data_module import BufferDataModule
from common.online.env_wrapper import EnvWrapper
from common.online.argument_parser import MainParser, MyParser

# 离线RL
from common.offline.buffer import ReplayBuffer
from common.offline.networks import Actor, Critic, TwinQ
from common.offline.utils import set_seed, compute_mean_std

# 共享
from common.logger import SwanlabLogger
```

---

## 4. 关键代码实现

### 4.1 两种ReplayBuffer对比

#### 在线RL Buffer (`common/online/buffer.py`)

```python
from collections import deque
from recordclass import recordclass

Trajectory = recordclass("Trajectory", ("obs", "action", "reward", "next_obs", "done"))

class ReplayBuffer():
    """动态经验回放，支持环境交互"""
    def __init__(self, offline_data: List[Trajectory], capacity: int):
        self.buffer_env = deque(offline_data, maxlen=capacity)
        self.buffer_model = deque([], maxlen=capacity)

    def push(self, buffer_type: str, *args):
        """动态添加经验"""
        if buffer_type == "env":
            self.buffer_env.append(Trajectory(*args))
        elif buffer_type == "model":
            self.buffer_model.append(Trajectory(*args))

    def sample(self, batch_size: int, from_data: bool = False):
        return random.sample(self.buffer_env + self.buffer_model, batch_size)
```

#### 离线RL Buffer (`common/offline/buffer.py`)

```python
class ReplayBuffer:
    """D4RL格式静态buffer，tensor预分配"""
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str):
        self._states = torch.zeros((buffer_size, state_dim), device=device)
        self._actions = torch.zeros((buffer_size, action_dim), device=device)
        self._rewards = torch.zeros((buffer_size, 1), device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), device=device)
        self._dones = torch.zeros((buffer_size, 1), device=device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """一次性加载整个数据集"""
        n_transitions = data["observations"].shape[0]
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        # ...

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        indices = np.random.randint(0, self._size, size=batch_size)
        return [self._states[indices], self._actions[indices], ...]
```

### 4.2 TD3+BC核心训练逻辑

```python
class TD3_BC:
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch

        # 1. 更新Critic
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-0.5, 0.5)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 2. 更新Actor (延迟更新)
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic.q1(states, pi)

            # TD3+BC: Q-learning + Behavior Cloning
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)

            # 软更新目标网络
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
```

---

## 5. 数据收集与验证

### 5.1 数据收集配置

**环境列表**:
- diffuse_topdown
- diffuse_mix
- diffuse_divpen

**数据规模**:
- 每个环境: 10,000 episodes
- 每个环境: 1,000,000 transitions
- 总数据量: 3M transitions

**数据收集时间**:
- 开始时间: 2025-11-30 08:44
- 完成时间: 2025-11-30 12:21
- 总耗时: 约3.6小时

### 5.2 数据格式

```python
# D4RL标准格式
dataset = {
    'observations': np.ndarray,      # (N, 20) belief states
    'actions': np.ndarray,           # (N, 32) latent actions
    'rewards': np.ndarray,           # (N,) rewards
    'next_observations': np.ndarray, # (N, 20) next belief states
    'terminals': np.ndarray,         # (N,) done flags
}
```

---

## 6. 测试结果

### 6.1 重构后的验证测试 (2025-12-05)

#### 在线RL模块测试

```bash
$ python scripts/train_online_rl.py --help
usage: train_online_rl.py [-h] --agent
                          {DQN,SAC,WolpertingerSAC,SlateQ,REINFORCE,...}
                          --belief {none,GRU} --ranker {none,topk,kargmax,GeMS}
                          --item_embedds {none,scratch,mf,ideal} --env_name ENV_NAME
```
**结果**: ✅ 成功

#### 离线RL基础模块测试

```bash
$ python -c "
from common.offline.buffer import ReplayBuffer
from common.offline.networks import Actor, Critic, TwinQ
from common.offline.utils import set_seed, compute_mean_std
print('All offline modules OK')
"
```
**结果**: ✅ 成功

#### 数据收集模块测试

```bash
$ python -c "
from data_collection.offline_data_collection.core.environment_factory import EnvironmentFactory
from data_collection.offline_data_collection.core.model_loader import ModelLoader
print('Data collection modules OK')
"
```
**结果**: ✅ 成功

### 6.2 测试总结

| 模块 | 状态 | 说明 |
|------|------|------|
| 在线RL训练脚本 | ✅ 通过 | `train_online_rl.py --help` 正常 |
| 离线RL基础模块 | ✅ 通过 | buffer, networks, utils 全部可导入 |
| 数据收集模块 | ✅ 通过 | 修复循环导入后正常 |
| TD3_BC算法 | ✅ 通过 | 可正常导入 |
| CQL/IQL算法 | ⚠️ 待处理 | 需要安装 pyrallis, d4rl, wandb |

---

## 7. 当前状态与后续工作

### 7.1 当前状态

**✅ 已完成**:
1. 数据收集完成（3个环境，1M transitions each）
2. 代码重构完成（方案F：online/offline物理隔离）
3. 在线RL模块验证通过
4. 离线RL基础模块验证通过
5. 数据收集模块验证通过
6. 循环导入问题已修复
7. gymnasium兼容性已处理

**⚠️ 待处理**:
1. CQL/IQL的pyrallis装饰器问题
2. 创建统一的 `scripts/train_offline_rl.py` 入口
3. 离线RL改用SwanLab日志（可选）

### 7.2 立即可执行

**在线RL训练**:
```bash
cd /data/liyuefeng/offline-slate-rl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gems

python scripts/train_online_rl.py \
    --agent=SAC \
    --belief=GRU \
    --ranker=GeMS \
    --item_embedds=mf \
    --env_name=topics
```

**离线RL训练 (TD3+BC)**:
```bash
# 需要创建 scripts/train_offline_rl.py
# 或直接运行 agents/offline/td3_bc.py
```

### 7.3 后续工作

**优先级1 (高)**:
- 创建 `scripts/train_offline_rl.py` 统一入口
- 解决CQL/IQL的依赖问题
- 启动TD3+BC训练验证

**优先级2 (中)**:
- 将离线RL日志从WandB迁移到SwanLab
- 运行CQL和IQL实验
- 收集性能基准数据

**优先级3 (低)**:
- 添加更多离线RL算法（AWAC, SAC-N等）
- 实现在线评估功能

### 7.4 为Decision Diffuser做准备

本框架为Decision Diffuser开发提供:
1. **数据接口**: 已适配的D4RL格式数据加载
2. **网络结构**: 可复用的Actor/Critic网络
3. **训练框架**: 清晰的在线/离线分离架构
4. **性能基准**: TD3+BC/CQL/IQL的性能作为对比

---

## 8. 重构记录 (2025-12-05)

### 8.1 重构背景

原项目存在以下问题：
- `src/offline_rl/` 和 `src/online_rl/` 目录冗余
- 两个不同的 `ReplayBuffer` 实现混淆
- 导入路径混乱

### 8.2 重构方案 (方案F)

**核心思想**：
- `logger.py` 作为共享文件放在 `common/` 根目录
- 在线RL专用文件放在 `common/online/`
- 离线RL专用文件放在 `common/offline/`

**根本原因**：在线RL使用PyTorch Lightning，离线RL使用纯PyTorch，两者的buffer、训练循环、参数配置方式完全不同，无法共用。

### 8.3 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `src/common/__init__.py` | 重写 |
| `src/common/online/__init__.py` | 新建 |
| `src/common/online/buffer.py` | 新建 |
| `src/common/online/data_module.py` | 新建 |
| `src/common/online/env_wrapper.py` | 新建 |
| `src/common/online/argument_parser.py` | 复制 |
| `src/common/offline/__init__.py` | 新建 |
| `src/common/offline/buffer.py` | 复制 |
| `src/common/offline/networks.py` | 复制 |
| `src/common/offline/utils.py` | 复制 |
| `scripts/train_online_rl.py` | 导入修改 |
| `src/agents/online.py` | 导入修改 |
| `src/agents/offline/td3_bc.py` | 导入修改 |
| `src/agents/offline/cql.py` | 导入修改 |
| `src/agents/offline/iql.py` | 导入修改 |
| `src/training/online_loops.py` | 导入修改 |
| `src/envs/RecSim/simulators.py` | 导入修改 |
| `src/belief_encoders/gru_belief.py` | 导入修改 |
| `src/data_collection/.../environment_factory.py` | 导入修改 |
| `src/data_collection/.../model_loader.py` | 导入修改 |

**总计**: 20个文件涉及修改

### 8.4 删除的目录

- `src/offline_rl/` (整个目录)
- `src/online_rl/` (整个目录)
- `src/common/data_utils.py` (已拆分)
- `src/common/argument_parser.py` (已移动)

### 8.5 修复的问题

1. **循环导入**: 修改 `common/online/__init__.py`，不在包初始化时导入 `EnvWrapper`
2. **gymnasium兼容**: 将 `import gym` 改为 `import gymnasium as gym`

详细记录见: `document/REFACTORING_FEASIBILITY_ANALYSIS_20251204.md`

---

## 附录

### A. 环境配置

- Python: 3.9.23
- PyTorch: 1.10.1+cu113
- NumPy: 1.22.4
- gymnasium: 1.1.1
- CUDA: Available
- Conda环境: gems

### B. 数据集路径

```
data/offline_datasets/
├── diffuse_topdown_expert.npz    # 253MB
├── diffuse_mix_expert.npz        # 261MB
└── diffuse_divpen_expert.npz     # 254MB
```

### C. 参考文献

- TD3+BC: [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860)
- CQL: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- IQL: [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- CORL: https://github.com/tinkoff-ai/CORL
- GeMS: Generative Model for Slate Recommendation

### D. 相关文档

- 重构分析: `document/REFACTORING_FEASIBILITY_ANALYSIS_20251204.md`
- 包含方案F详细设计、执行记录、补充修复、动态验证测试

---

**文档版本**: v2.0
**最后更新**: 2025-12-05
