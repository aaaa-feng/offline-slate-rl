# 代码修复清单

**日期**: 2025-12-01
**状态**: gems_env.py已修复，CQL/IQL待修复

---

## ✅ 已完成的修复

### 1. gems_env.py - 环境包装器

**修复内容**:
- ✅ 正确加载belief_encoder（使用ModelLoader）
- ✅ 正确加载GeMS ranker（使用ModelLoader）
- ✅ 实现`reset()`方法，正确初始化belief state
- ✅ 实现`step()`方法，正确更新belief state
- ✅ 实现`_decode_action()`方法，使用ranker将latent action解码为slate
- ✅ 添加清晰的警告信息，说明在线评估的限制

**关键改进**:
```python
# 加载belief encoder和ranker
self.model_loader = ModelLoader()
self.belief_encoder = self.model_loader.load_belief_encoder(env_name)
self.ranker = self.model_loader.load_ranker(env_name, ranker_type="GeMS")

# reset时初始化belief state
self.belief_state = self.belief_encoder.forward(self.current_obs)

# step时更新belief state
self.belief_state = self.belief_encoder.forward(next_obs, done=done)

# 解码latent action
slate = self.ranker.rank(latent_tensor)
```

---

## 🔴 待修复：CQL算法文件

**文件**: `offline_rl_baselines/algorithms/cql.py`

### 问题1: 冗余的ReplayBuffer定义

**位置**: Line 123-182
**问题**: 文件内部重新定义了ReplayBuffer类，与`common/buffer.py`冲突
**修复**: 删除整个ReplayBuffer类定义（Line 123-182）

### 问题2: 冗余的工具函数

**位置**:
- Line 86-90: `soft_update()` - 已在`common/utils.py`中定义
- Line 91-95: `compute_mean_std()` - 已在`common/utils.py`中定义
- Line 97-99: `normalize_states()` - 可以删除或移到common
- Line 185-196: `set_seed()` - 已在`common/utils.py`中定义

**修复**: 删除这些函数，使用common中的版本

### 问题3: d4rl依赖

**位置**: Line 849
```python
dataset = d4rl.qlearning_dataset(env)
```

**修复**: 改为加载本地.npz文件
```python
dataset = np.load(config.dataset_path)
```

### 问题4: pyrallis依赖

**位置**: Line 842
```python
@pyrallis.wrap()
def train(config: TrainConfig):
```

**修复**:
1. 删除`@pyrallis.wrap()`装饰器
2. 将`TrainConfig`改为dataclass（保持不变）
3. 创建独立的训练脚本`scripts/train_cql.py`使用argparse

### 问题5: 训练函数中的d4rl评估

**位置**: Line 987
```python
{"d4rl_normalized_score": normalized_eval_score}
```

**问题**: 使用了d4rl的归一化评分
**修复**:
- 选项1: 删除d4rl归一化评分
- 选项2: 使用自定义的归一化方法

### 问题6: 默认配置不适用

**位置**: Line 31-85 (TrainConfig)
```python
env: str = "halfcheetah-medium-expert-v2"  # MuJoCo环境
```

**修复**: 改为GeMS环境
```python
env_name: str = "diffuse_topdown"
state_dim: int = 20
action_dim: int = 32
```

---

## 🔴 待修复：IQL算法文件

**文件**: `offline_rl_baselines/algorithms/iql.py`

### 相同的问题

IQL文件存在与CQL相同的问题：
1. ✅ 冗余的ReplayBuffer定义
2. ✅ 冗余的工具函数
3. ✅ d4rl依赖（Line 285）
4. ✅ pyrallis依赖（Line 256）
5. ✅ 默认配置不适用

---

## 📝 建议的修复方案

### 方案A: 完整重构（推荐，但耗时）

1. **清理CQL/IQL文件**:
   - 删除所有冗余代码
   - 只保留核心算法类（ContinuousCQL, IQL）
   - 移除d4rl和pyrallis依赖

2. **创建训练脚本**:
   - `scripts/train_cql.py` - 参考`train_td3_bc.py`
   - `scripts/train_iql.py` - 参考`train_td3_bc.py`

3. **添加训练函数**:
   - 在`algorithms/cql.py`末尾添加`train_cql(config)`函数
   - 在`algorithms/iql.py`末尾添加`train_iql(config)`函数

### 方案B: 最小修改（快速，但不彻底）

1. **只修复关键问题**:
   - 修改Line 849: 改为加载本地数据
   - 修改Line 871-877: 使用GemsReplayBuffer
   - 删除`@pyrallis.wrap()`装饰器

2. **创建简单的训练脚本**:
   - 直接调用修改后的`train()`函数
   - 使用argparse解析参数并构造TrainConfig

### 方案C: 先验证TD3+BC，再决定（最务实）

1. **立即测试TD3+BC**:
   ```bash
   python offline_rl_baselines/scripts/train_td3_bc.py \
       --env_name diffuse_topdown \
       --seed 0 \
       --max_timesteps 10000 \  # 先跑10K steps测试
       --device cuda
   ```

2. **如果TD3+BC工作正常**:
   - 再花时间修复CQL/IQL
   - 因为至少有一个baseline可用

3. **如果TD3+BC有问题**:
   - 先解决TD3+BC的问题
   - 再考虑CQL/IQL

---

## 🎯 关键理解：潜空间训练

### 数据流程

```
数据收集阶段:
RecSim obs → belief_encoder → belief_state (20维)
                                    ↓
                            SAC agent → latent_action (32维)
                                    ↓
                            GeMS ranker → slate (10个物品)
                                    ↓
                            environment → reward

保存到数据集:
- observations: belief_state (20维)
- actions: latent_action (32维)
- rewards, next_observations, terminals
```

```
离线RL训练阶段:
加载数据集 → (belief_state, latent_action, reward)
                    ↓
        在潜空间中训练策略: belief_state → latent_action
                    ↓
        不需要ranker！训练完全在潜空间进行
```

```
在线评估阶段（可选）:
RecSim obs → belief_encoder → belief_state (20维)
                                    ↓
                    训练好的策略 → latent_action (32维)
                                    ↓
                    GeMS ranker → slate (10个物品)
                                    ↓
                    environment → reward
```

### 关键点

1. **训练不需要ranker**:
   - 离线RL算法在潜空间中训练
   - 输入: belief_state (20维)
   - 输出: latent_action (32维)
   - 完全不涉及slate

2. **评估需要ranker**:
   - 如果要在真实环境中评估
   - 需要将latent_action解码为slate
   - 这时才需要ranker

3. **当前状态**:
   - ✅ 数据已收集（包含belief_state和latent_action）
   - ✅ TD3+BC可以在潜空间中训练
   - ✅ gems_env.py已修复，支持在线评估（如果需要）
   - ⚠️ CQL/IQL需要修复才能训练

---

## 📊 测试计划

### 阶段1: 验证TD3+BC（立即执行）

```bash
# 测试数据加载
python -c "
import numpy as np
data = np.load('offline_datasets/diffuse_topdown_expert.npz')
print('Data loaded successfully')
print('Observations:', data['observations'].shape)
print('Actions:', data['actions'].shape)
"

# 测试短时间训练（10K steps，约5-10分钟）
python offline_rl_baselines/scripts/train_td3_bc.py \
    --env_name diffuse_topdown \
    --seed 0 \
    --max_timesteps 10000 \
    --batch_size 256 \
    --device cuda \
    --no_normalize  # 先不归一化，测试基本流程

# 检查日志
tail -f offline_rl_baselines/experiments/logs/td3_bc_*.log
```

### 阶段2: 修复CQL/IQL（如果TD3+BC成功）

根据方案A或方案B进行修复

### 阶段3: 完整训练（如果测试成功）

```bash
# 运行完整的TD3+BC训练（1M steps）
bash offline_rl_baselines/scripts/run_all_baselines.sh td3_bc
```

---

## 💡 建议

1. **优先级**:
   - 🔥 **最高**: 测试TD3+BC是否能正常训练
   - 🔥 **高**: 修复CQL/IQL的关键问题（d4rl依赖）
   - 📝 **中**: 清理冗余代码
   - 📝 **低**: 完善在线评估功能

2. **时间分配**:
   - TD3+BC测试: 10-30分钟
   - CQL/IQL最小修复: 1-2小时
   - CQL/IQL完整重构: 4-6小时

3. **风险控制**:
   - 先确保至少有一个算法（TD3+BC）可用
   - 再逐步添加其他算法
   - 避免同时修改多个文件导致难以调试

---

**下一步行动**: 请决定采用哪个方案，或者先测试TD3+BC
