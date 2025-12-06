# Offline Slate Recommendation with Reinforcement Learning

**项目描述**: 基于 GeMS (Generative Model for Slate recommendation) 的离线强化学习推荐系统研究项目

**主要研究方向**:
- 在线强化学习训练 (SAC + GeMS)
- 离线数据集收集 (Random, Medium, Expert)
- 离线强化学习算法 (CQL, IQL, BCQ 等)
- 推荐系统中的用户行为建模

**项目状态**: 活跃开发中 (2025-12)

---

## 📁 项目目录结构

### 核心代码目录

#### `src/` - 源代码根目录
项目的所有核心实现代码

- **`src/agents/`** - RL 智能体实现
  - `online/` - 在线 RL 算法 (SAC, DQN, SlateQ, REINFORCE, WolpertingerSAC)
  - `offline/` - 离线 RL 算法 (CQL, IQL, BCQ 等)
  - 每个算法包含 actor, critic, 以及训练逻辑

- **`src/belief_encoders/`** - 信念状态编码器
  - `gru_belief.py` - GRU-based 信念编码器
  - 用于 POMDP 环境中的状态表示学习

- **`src/rankers/`** - 推荐排序模型
  - `gems/` - GeMS 生成式排序模型
    - `rankers.py` - GeMS, TopK, kArgmax 等排序器
    - `item_embeddings.py` - 物品嵌入表示 (scratch, MF, ideal)
    - `vae.py` - 变分自编码器实现
  - 负责将 RL action 转换为推荐 slate

- **`src/envs/`** - 推荐环境模拟器
  - `RecSim/` - 基于 RecSim 的推荐环境
    - `simulators.py` - TopicRec 环境实现
    - `user_model.py` - 用户行为模型 (Diffuse, Focused)
    - `click_model.py` - 点击模型 (tdPBM, mixPBM)
  - 模拟用户与推荐系统的交互

- **`src/common/`** - 通用工具和组件
  - `online/` - 在线训练相关
    - `data_module.py` - Replay Buffer 数据模块
    - `env_wrapper.py` - 环境包装器
    - `argument_parser.py` - 命令行参数解析
  - `offline/` - 离线训练相关
  - `logger.py` - SwanLab 日志记录器

- **`src/training/`** - 训练循环实现
  - `online_loops.py` - 在线 RL 训练循环
    - `TrainingEpisodeLoop` - 训练 episode 循环
    - `ValEpisodeLoop` - 验证循环
    - `TestEpisodeLoop` - 测试循环
    - `ResettableFitLoop` - 可重置的 fit 循环
  - `offline_loops.py` - 离线 RL 训练循环

- **`src/data_collection/`** - 数据收集工具
  - 用于收集离线数据集的脚本和工具

- **`src/utils/`** - 工具函数
  - 各种辅助函数和工具

### 配置和脚本目录

#### `config/` - 配置文件
- `paths.py` - 项目路径配置
  - 定义所有数据、模型、日志的标准路径
  - 统一管理文件系统结构

#### `scripts/` - 执行脚本
- **`scripts/train_online_rl.py`** - 在线 RL 训练主脚本 ⭐
  - 支持 SAC, DQN, SlateQ, REINFORCE 等算法
  - 支持 GeMS, TopK, kArgmax 等排序器
  - 集成 SwanLab 云端日志
  - 支持灵活的 checkpoint 策略

- **`scripts/batch_runs/`** - 批量实验脚本
  - `run_medium_collection_training.sh` - Medium 数据收集训练 (50k steps)
  - `run_rl_training_batch.sh` - 批量 RL 训练 (复现实验)
  - 自动分配 GPU,并行训练多个环境

- **`scripts/collect_offline_data.py`** - 离线数据收集脚本
  - 使用训练好的模型收集轨迹数据
  - 支持 Random, Medium, Expert 策略

### 数据和模型目录

#### `data/` - 数据存储
- **`data/datasets/`** - 原始数据集
  - 在线训练数据集
  - 预训练数据集

- **`data/embeddings/`** - 物品嵌入
  - `item_embeddings_diffuse.pt` - Diffuse 用户环境嵌入
  - `item_embeddings_focused.pt` - Focused 用户环境嵌入
  - MF (Matrix Factorization) 预训练嵌入

- **`data/offline_datasets/`** - 离线 RL 数据集
  - `random/` - Random 策略数据 (10k trajectories per env)
  - `medium/` - Medium 策略数据 (50k steps 模型收集) ⭐ 当前收集中
  - `expert/` - Expert 策略数据 (100k steps 模型收集,未来)

#### `checkpoints/` - 模型检查点
- **`checkpoints/gems/`** - GeMS 排序器预训练模型
  - 各环境的 GeMS VAE 模型
  - 用于初始化 RL 训练

- **`checkpoints/online_rl/`** - 在线 RL 训练模型 ⭐
  - 按环境组织: `diffuse_topdown/`, `focused_mix/`, 等
  - 每个环境包含:
    - `*_best.ckpt` - 最佳验证 reward 模型
    - `*_step50000.ckpt` - 50k 步模型 (用于 Medium 数据收集)
    - `*_last.ckpt` - 最终模型

- **`checkpoints/offline_rl/`** - 离线 RL 训练模型
  - 离线算法训练的模型

#### `results/` - 实验结果
- **`results/online_rl/`** - 在线 RL 实验结果
  - 测试结果 `.pt` 文件
  - 性能指标统计

- **`results/offline_rl/`** - 离线 RL 实验结果

### 日志和文档目录

#### `experiments/` - 实验记录
- **`experiments/logs/`** - 训练日志 ⭐
  - `log_58407201/SAC_GeMS/` - 主要实验日志 (seed 58407201)
    - `replication_experiment_20251129/` - 复现实验 (12个实验)
    - `medium_collection_20251206/` - Medium 数据收集训练 (6个实验,进行中)
  - 包含完整的训练输出、验证结果、错误信息

- **`experiments/swanlog/`** - SwanLab 本地日志
  - SwanLab 云端同步的本地副本
  - 包含实验配置、指标、图表

#### `document/` - 项目文档 📚
- **操作指南**:
  - `conversation_2025-12-06_session1.md` - Medium 数据收集操作指南 ⭐
  - `EXPERIMENT_GUIDE.md` - 实验执行指南
  - `DATA_AND_WORKFLOW_EXPLANATION.md` - 数据和工作流说明

- **实验记录**:
  - `FOCUSED_SAC_GEMS_EXPERIMENTS.md` - Focused 环境实验记录
  - `baseline_experiments_params.md` - Baseline 实验参数
  - `RL_TRAINING_PARAMETERS_ANALYSIS.md` - RL 训练参数分析

- **项目分析**:
  - `COMPLETE_PROJECT_ANALYSIS_REPORT.md` - 完整项目分析报告
  - `PROJECT_REVIEW_20251201.md` - 项目回顾 (2025-12-01)
  - `REFACTORING_FEASIBILITY_ANALYSIS_20251204.md` - 重构可行性分析

- **工作记录**:
  - `conversation_2025-11-28*.md` - 会话记录 (5个 sessions)
  - `conversation_2025-11-29_session1.md` - 会话记录
  - `conversation_2025-11-30*.md` - 会话记录 (2个 sessions)
  - `conversation_2025-12-04*.md` - 会话记录 (4个 sessions)
  - `conversation_2025-12-05*.md` - 会话记录 (5个 sessions)
  - `work_summary_2025-12-04.md` - 工作总结
  - `model_management_plan.md` - 模型管理计划
  - `model_migration_summary.md` - 模型迁移总结

#### `backups/` - 备份文件
- 旧版本代码和配置的备份

---

## 🧪 实验概览

### 实验 1: 复现实验 (Replication Experiments)

**目的**: 复现论文结果,验证代码正确性

**时间**: 2025-11-28 ~ 2025-11-29

**实验数量**: 12 个实验 (6 环境 × 2 参数集)

**环境列表**:
1. `diffuse_topdown` - Diffuse 用户 + Top-down 点击模型
2. `diffuse_mix` - Diffuse 用户 + Mix 点击模型
3. `diffuse_divpen` - Diffuse 用户 + Mix 点击模型 + 高多样性惩罚
4. `focused_topdown` - Focused 用户 + Top-down 点击模型
5. `focused_mix` - Focused 用户 + Mix 点击模型
6. `focused_divpen` - Focused 用户 + Mix 点击模型 + 高多样性惩罚

**参数集**:
- **Params1**: `lambda_KL=0.5`, `lambda_click=0.2`
- **Params2**: `lambda_KL=1.0`, `lambda_click=0.5` (论文官方参数)

**日志位置**: `experiments/logs/log_58407201/SAC_GeMS/replication_experiment_20251129/`

**SwanLab 项目**: [GeMS_RL_Training_202512](https://swanlab.cn/@Cliff/GeMS_RL_Training_202512)

### 实验 2: Medium 数据收集训练 (Medium Data Collection)

**目的**: 训练 50k 步模型,用于收集 Medium 质量离线数据集

**时间**: 2025-12-06 (进行中)

**实验数量**: 6 个实验 (6 环境 × 1 参数集)

**训练配置**:
- 训练步数: 100,000 steps
- 保存步数: 50,000 steps (用于数据收集)
- 验证频率: 每 1000 episodes
- 参数: `lambda_KL=1.0`, `lambda_click=0.5` (论文官方参数)

**GPU 分配**: GPU 1, 2, 3 (每个 GPU 2个环境)

**日志位置**: `experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/`

**模型保存**: `checkpoints/online_rl/{env_name}/*_step50000.ckpt`

**下一步**: 使用 50k 步模型收集 10,000 条轨迹/环境

---

## 📊 实验参数详细表格

### 表 1: 算法和模型参数

| 参数类别 | 参数名称 | 值 | 说明 |
|---------|---------|-----|------|
| **RL 算法** | `--agent` | `SAC` | Soft Actor-Critic |
| | `--gamma` | `0.8` | 折扣因子 |
| | `--alpha` | `0.2` | 熵正则化系数 |
| | `--auto_entropy` | `True` | 自动调整熵系数 |
| **Q-Network** | `--q_lr` | `0.001` | Q 网络学习率 |
| | `--hidden_layers_qnet` | `256` | Q 网络隐藏层大小 |
| | `--target_update_frequency` | `1` | 目标网络更新频率 |
| | `--tau` | `0.002` | 软更新系数 |
| **Policy Network** | `--pi_lr` | `0.003` | 策略网络学习率 |
| | `--hidden_layers_pinet` | `256` | 策略网络隐藏层大小 |
| **Belief Encoder** | `--belief` | `GRU` | 信念编码器类型 |
| | `--belief_state_dim` | `20` | 信念状态维度 |
| | `--beliefs` | `actor critic` | 使用信念的组件 |
| **Ranker (GeMS)** | `--ranker` | `GeMS` | 生成式排序模型 |
| | `--latent_dim` | `32` | VAE 潜在空间维度 |
| | `--lambda_KL` | `0.5` / `1.0` | KL 散度损失权重 |
| | `--lambda_click` | `0.2` / `0.5` | 点击预测损失权重 |
| | `--lambda_prior` | `0.0` | 先验损失权重 |
| | `--ranker_embedds` | `scratch` | 排序器嵌入初始化 |
| | `--ranker_sample` | `False` | 是否采样 |
| **Item Embeddings** | `--item_embedds` | `scratch` | 物品嵌入初始化方式 |
| | `--item_embedd_dim` | `20` | 物品嵌入维度 |
| | `--num_items` | `1000` | 物品总数 |

### 表 2: 环境参数

| 参数类别 | 参数名称 | 值 | 说明 |
|---------|---------|-----|------|
| **环境基础** | `--env_name` | `topics` | TopicRec 环境 |
| | `--episode_length` | `100` | Episode 长度 |
| | `--num_topics` | `10` | 主题数量 |
| | `--topic_size` | `2` | 每个主题的物品数 |
| **用户模型** | `--env_embedds` | `item_embeddings_diffuse.pt` / `item_embeddings_focused.pt` | 用户类型 |
| | `--env_omega` | `0.9` | 用户兴趣衰减因子 |
| | `--short_term_boost` | `1.0` | 短期兴趣提升 |
| | `--env_offset` | `0.28` | 兴趣偏移 |
| | `--env_slope` | `100` | 兴趣斜率 |
| **点击模型** | `--click_model` | `tdPBM` / `mixPBM` | 点击模型类型 |
| **用户行为** | `--boredom_threshold` | `5` | 厌倦阈值 |
| | `--recent_items_maxlen` | `10` | 最近物品记忆长度 |
| | `--boredom_moving_window` | `5` | 厌倦滑动窗口 |
| **多样性** | `--diversity_penalty` | `1.0` / `3.0` | 多样性惩罚系数 |
| | `--diversity_threshold` | `4` | 多样性阈值 |

### 表 3: 训练参数

| 参数类别 | 参数名称 | 复现实验 | Medium 收集 | 说明 |
|---------|---------|---------|------------|------|
| **训练步数** | `--max_steps` | `100000` | `100000` | 最大训练步数 |
| | `--random_steps` | `2000` | `2000` | 随机探索步数 |
| **验证** | `--check_val_every_n_epoch` | `1000` | `1000` | 验证频率 (episodes) |
| | `--val_step_length` | `200` | `200` | 验证 episode 长度 |
| | `--test_size` | `500` | `500` | 测试集大小 |
| **Replay Buffer** | `--capacity` | `10000` | `10000` | Buffer 容量 |
| | `--batch_size` | `20` | `20` | 批次大小 |
| **Checkpoint** | `--save_every_n_steps` | `0` | `50000` | 步数间隔保存 |
| **日志** | `--log_every_n_steps` | `1` | `1` | 日志记录频率 |
| | `--progress_bar` | `True` | `False` | 是否显示进度条 |
| **随机种子** | `--seed` | `58407201` | `58407201` | 全局随机种子 |
| | `--ranker_seed` | `58407201` | `58407201` | 排序器随机种子 |

### 表 4: 环境配置对照表

| 环境名称 | 用户类型 | 点击模型 | 多样性惩罚 | 环境嵌入文件 | 说明 |
|---------|---------|---------|-----------|-------------|------|
| `diffuse_topdown` | Diffuse | tdPBM | 1.0 | `item_embeddings_diffuse.pt` | 分散兴趣 + 位置偏差 |
| `diffuse_mix` | Diffuse | mixPBM | 1.0 | `item_embeddings_diffuse.pt` | 分散兴趣 + 混合点击 |
| `diffuse_divpen` | Diffuse | mixPBM | 3.0 | `item_embeddings_diffuse.pt` | 分散兴趣 + 高多样性 |
| `focused_topdown` | Focused | tdPBM | 1.0 | `item_embeddings_focused.pt` | 集中兴趣 + 位置偏差 |
| `focused_mix` | Focused | mixPBM | 1.0 | `item_embeddings_focused.pt` | 集中兴趣 + 混合点击 |
| `focused_divpen` | Focused | mixPBM | 3.0 | `item_embeddings_focused.pt` | 集中兴趣 + 高多样性 |

### 表 5: 参数集对照表

| 参数集 | lambda_KL | lambda_click | 用途 | 说明 |
|-------|-----------|--------------|------|------|
| Params1 | 0.5 | 0.2 | 复现实验 | 探索性参数 |
| Params2 | 1.0 | 0.5 | 复现实验 + Medium 收集 | 论文官方参数 ⭐ |

---

## 🚀 快速开始

### 环境配置

```bash
# 1. 激活 conda 环境
conda activate gems

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证 GPU 可用性
nvidia-smi
```

### 运行单个实验

```bash
# 训练 SAC+GeMS (focused_topdown 环境)
python scripts/train_online_rl.py \
    --agent=SAC \
    --belief=GRU \
    --ranker=GeMS \
    --item_embedds=scratch \
    --env_name=topics \
    --device=cuda \
    --seed=58407201 \
    --ranker_seed=58407201 \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --name="SAC+GeMS" \
    --latent_dim=32 \
    --lambda_KL=1.0 \
    --lambda_click=0.5 \
    --ranker_dataset=focused_topdown \
    --click_model=tdPBM \
    --env_embedds=item_embeddings_focused.pt \
    --diversity_penalty=1.0 \
    --gamma=0.8 \
    --swan_project="GeMS_RL_Training_202512" \
    --swan_mode=cloud
```

### 运行批量实验

```bash
# Medium 数据收集训练 (6个环境并行)
bash scripts/batch_runs/run_medium_collection_training.sh

# 监控训练进度
tail -f experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log
```

### 收集离线数据

```bash
# 使用训练好的模型收集数据
python scripts/collect_offline_data.py \
    --agent=SAC \
    --belief=GRU \
    --ranker=GeMS \
    --model_checkpoint=checkpoints/online_rl/focused_topdown/*_step50000.ckpt \
    --num_trajectories=10000 \
    --output_path=data/offline_datasets/medium/focused_topdown_medium_10000traj.pkl
```

---

## 📈 监控和日志

### SwanLab 云端监控

**项目链接**: [https://swanlab.cn/@Cliff/GeMS_RL_Training_202512](https://swanlab.cn/@Cliff/GeMS_RL_Training_202512)

**监控指标**:
- `train_reward` - 训练 reward
- `val_reward` - 验证 reward
- `train_ep_length` - Episode 长度
- Loss 曲线 (Q-loss, Policy-loss, Alpha-loss)

### 本地日志查看

```bash
# 查看训练进度
grep "Training Step" experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log | tail -20

# 查看验证结果
grep "VALIDATION" experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/*.log | tail -20

# 实时监控
tail -f experiments/logs/log_58407201/SAC_GeMS/medium_collection_20251206/focused_topdown_*.log
```

### 检查模型保存

```bash
# 查看所有 50k 步模型
ls -lh checkpoints/online_rl/*/SAC+GeMS_Medium_*_step50000.ckpt

# 查看特定环境的所有模型
ls -lh checkpoints/online_rl/focused_topdown/
```

---

## 📝 重要文档

### 操作指南
- [Medium 数据收集操作指南](document/conversation_2025-12-06_session1.md) - 详细的数据收集流程
- [实验执行指南](document/EXPERIMENT_GUIDE.md) - 如何运行实验
- [数据和工作流说明](document/DATA_AND_WORKFLOW_EXPLANATION.md) - 数据流程说明

### 实验分析
- [完整项目分析报告](document/COMPLETE_PROJECT_ANALYSIS_REPORT.md) - 项目全面分析
- [RL 训练参数分析](document/RL_TRAINING_PARAMETERS_ANALYSIS.md) - 参数调优分析
- [Focused 环境实验记录](document/FOCUSED_SAC_GEMS_EXPERIMENTS.md) - Focused 环境结果

### 项目管理
- [项目回顾 2025-12-01](document/PROJECT_REVIEW_20251201.md) - 项目进展回顾
- [模型管理计划](document/model_management_plan.md) - 模型组织方案
- [重构可行性分析](document/REFACTORING_FEASIBILITY_ANALYSIS_20251204.md) - 代码重构分析

---

## 🔧 常见问题

### Q1: 验证为什么在 Step 999 而不是 Step 1000?

**A**: 因为 PyTorch Lightning 的自定义训练循环将每个 episode 视为一个 "epoch"。`check_val_every_n_epoch=1000` 表示每 1000 个 episodes 验证一次,验证发生在第 1000 个 episode 结束时,即 Step 999。

### Q2: 如何确认模型在 50k 步保存成功?

**A**:
```bash
# 检查文件是否存在
ls checkpoints/online_rl/*/SAC+GeMS_Medium_*_step50000.ckpt

# 检查文件大小 (应该几百 MB)
ls -lh checkpoints/online_rl/*/SAC+GeMS_Medium_*_step50000.ckpt
```

### Q3: 训练中断如何恢复?

**A**: 目前不支持自动恢复。需要手动检查日志,确定中断步数,然后重新启动训练。

### Q4: 如何修改 GPU 分配?

**A**: 编辑 `scripts/batch_runs/run_medium_collection_training.sh`,修改 `GPU_IDS` 数组:
```bash
GPU_IDS=(1 2 3)  # 修改为你想使用的 GPU ID
```

### Q5: SwanLab 日志上传失败怎么办?

**A**:
1. 检查网络连接
2. 使用 `--swan_mode=local` 仅保存本地日志
3. 使用 `--swan_mode=offline` 离线模式,稍后同步

---

## 📚 参考资料

### 论文
- **GeMS**: "Generative Model for Slate Recommendation" (原始论文)
- **SAC**: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"

### 相关项目
- [RecSim](https://github.com/google-research/recsim) - Google 推荐系统模拟器
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - 深度学习框架

### 工具
- [SwanLab](https://swanlab.cn/) - 实验跟踪和可视化平台

---

## 👥 贡献者

- **Cliff** - 项目负责人
- **Claude Code** - AI 编程助手

---

## 📄 许可证

Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

---

## 📞 联系方式

- **SwanLab 项目**: [GeMS_RL_Training_202512](https://swanlab.cn/@Cliff/GeMS_RL_Training_202512)
- **问题反馈**: 查看项目文档或 SwanLab 实验记录

---

**最后更新**: 2025-12-06
**项目版本**: v1.0
**文档作者**: Claude Code
