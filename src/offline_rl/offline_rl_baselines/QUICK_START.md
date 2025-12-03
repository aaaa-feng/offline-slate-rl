# 离线RL Baseline 快速开始指南

## 🎯 当前状态总览

### ✅ 已完成
- ✅ **目录结构**: 清晰的模块化设计
- ✅ **基础设施**: ReplayBuffer, Networks, Utils
- ✅ **TD3+BC**: 完整实现，立即可用
- ✅ **CQL/IQL**: 算法文件已移植，imports已适配
- ✅ **数据收集**: 正在进行中（约3.6小时完成）

### ⏳ 待完善
- ⏳ **CQL训练脚本**: 需要参考TD3+BC实现完整训练函数
- ⏳ **IQL训练脚本**: 需要参考TD3+BC实现完整训练函数

---

## 🚀 立即可执行（数据收集完成后）

### 步骤1：检查数据收集状态

```bash
# 查看进程
ps aux | grep collect_data.py

# 查看进度（当前约4.4%，预计3.6小时完成）
tail -f offline_data_collection/logs/collect_diffuse_topdown_*.log

# 检查数据文件（完成后会生成）
ls -lh offline_datasets/*.npz
```

### 步骤2：数据收集完成后，立即训练TD3+BC

```bash
cd /data/liyuefeng/gems/gems_official/official_code
source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh
conda activate gems

# 训练单个环境（测试）
python offline_rl_baselines/scripts/train_td3_bc.py \
    --env_name diffuse_topdown \
    --seed 0 \
    --max_timesteps 1000000 \
    --batch_size 256 \
    --alpha 2.5 \
    --device cuda

# 或批量运行所有实验（3环境 × 3seeds = 9个实验）
bash offline_rl_baselines/scripts/run_all_baselines.sh td3_bc
```

### 步骤3：监控训练

```bash
# 查看训练日志
ls offline_rl_baselines/experiments/logs/

# 实时查看某个实验
tail -f offline_rl_baselines/experiments/logs/td3_bc_diffuse_topdown_seed0_*.log

# 查看所有训练进程
ps aux | grep train_td3_bc.py
```

---

## 📊 数据收集进度

### 当前状态（2025-11-30 08:44）
- **diffuse_topdown**: 444/10000 episodes (4.4%)
- **diffuse_mix**: 444/10000 episodes (4.4%)
- **diffuse_divpen**: 444/10000 episodes (4.4%)

### 预计完成时间
- **速度**: 1.4-1.5秒/episode
- **剩余时间**: 约3.6小时
- **预计完成**: 今天下午约12:30

### 数据输出
完成后会生成以下文件：
```
offline_datasets/
├── diffuse_topdown_expert.npz  # ~500MB
├── diffuse_mix_expert.npz      # ~500MB
└── diffuse_divpen_expert.npz   # ~500MB
```

每个文件包含：
- `observations`: (N, 20) - Belief states
- `actions`: (N, 32) - Latent actions
- `rewards`: (N,) - 奖励
- `next_observations`: (N, 20) - 下一状态
- `terminals`: (N,) - 终止标志

---

## 🔧 完善CQL/IQL的步骤（可选）

如果您想在数据收集期间完善CQL和IQL，可以参考以下步骤：

### 方法：参考TD3+BC的实现

1. **打开参考文件**
   ```bash
   # 查看TD3+BC的完整实现
   cat offline_rl_baselines/algorithms/td3_bc.py
   ```

2. **在CQL/IQL文件末尾添加训练函数**

   在 `algorithms/cql.py` 或 `algorithms/iql.py` 末尾添加：

   ```python
   def train_cql(config):  # 或 train_iql
       """训练CQL on GeMS dataset"""
       # 1. 加载数据
       dataset = np.load(config.dataset_path)

       # 2. 创建buffer
       buffer = GemsReplayBuffer(...)
       buffer.load_d4rl_dataset({
           'observations': dataset['observations'],
           'actions': dataset['actions'],
           'rewards': dataset['rewards'],
           'next_observations': dataset['next_observations'],
           'terminals': dataset['terminals'],
       })

       # 3. 初始化算法（使用文件中已有的类）
       # ...

       # 4. 训练循环
       for t in range(config.max_timesteps):
           batch = buffer.sample(config.batch_size)
           # 训练一步
           ...
   ```

3. **修改训练脚本调用新函数**

   在 `scripts/train_cql.py` 中：
   ```python
   from offline_rl_baselines.algorithms.cql import train_cql, CQLConfig

   config = CQLConfig(...)
   train_cql(config)
   ```

---

## 📋 实验时间线

### 今天（2025-11-30）
- ⏰ **12:30**: 数据收集完成
- ✅ **12:30-13:00**: 验证数据格式
- 🚀 **13:00**: 启动TD3+BC训练（9个实验）

### 明天（2025-12-01）
- 📊 **上午**: 检查TD3+BC训练进度
- 📈 **下午**: 分析初步结果
- 🔧 **晚上**: 根据需要完善CQL/IQL

### 2-3天后
- ✅ **TD3+BC完成**: 收集所有结果
- 📊 **性能分析**: 对比不同环境和seeds
- 📝 **准备报告**: 总结baseline性能

---

## 💡 关键建议

### 1. 优先级策略
- **高优先级**: TD3+BC（已完成，立即可用）
- **中优先级**: CQL/IQL（如果时间充足）
- **低优先级**: 其他算法（AWAC, SAC-N等）

### 2. 时间分配
- **数据收集**: 自动进行（3.6小时）
- **TD3+BC训练**: 后台运行（每个实验6-12小时）
- **CQL/IQL完善**: 如果需要（2-4小时开发）

### 3. 验证策略
- **先跑TD3+BC**: 验证整个流程可行性
- **分析结果**: 确认baseline性能合理
- **再决定**: 是否需要CQL/IQL

---

## 📞 常见问题

### Q1: 为什么CQL/IQL没有完整的训练脚本？

**A**: 为了快速验证，我们优先完成了TD3+BC（最简单的算法）。CQL和IQL的算法文件已经移植并适配好imports，只需要添加训练函数即可使用。

### Q2: 如何验证代码是否可用？

**A**: 等数据收集完成后，先运行TD3+BC：
```bash
python offline_rl_baselines/scripts/train_td3_bc.py --env_name diffuse_topdown --seed 0
```
如果TD3+BC能正常训练，说明整个框架可用。

### Q3: 数据收集失败怎么办？

**A**: 检查日志：
```bash
tail -100 offline_data_collection/logs/collect_diffuse_topdown_*.log
```
如果有错误，可以重新启动数据收集。

### Q4: 训练时GPU内存不足怎么办？

**A**: 减小batch size：
```bash
python offline_rl_baselines/scripts/train_td3_bc.py --batch_size 128
```

---

## ✅ 检查清单

### 数据收集完成前
- [x] 目录结构创建完成
- [x] 基础设施代码完成
- [x] TD3+BC算法完成
- [x] CQL/IQL算法文件移植
- [x] 训练脚本创建
- [ ] 等待数据收集完成（约3.6小时）

### 数据收集完成后
- [ ] 验证数据文件存在
- [ ] 检查数据格式正确
- [ ] 启动TD3+BC训练
- [ ] 监控训练进度
- [ ] 收集实验结果

### 可选任务
- [ ] 完善CQL训练脚本
- [ ] 完善IQL训练脚本
- [ ] 添加更多算法（AWAC, SAC-N等）

---

## 📚 文档索引

- **README.md**: 完整的系统说明
- **ALGORITHMS_STATUS.md**: 算法迁移状态
- **QUICK_START.md**: 本文档（快速开始指南）

---

**最后更新**: 2025-11-30 08:44
**下一步**: 等待数据收集完成（约3.6小时）
