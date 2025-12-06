# GeMS 实验完整指南

## 实验运行顺序

完整的实验流程包含以下4个步骤，必须按顺序执行：

### 步骤1: 生成日志数据 (Generate Logged Data)
**脚本**: `RecSim/generate_dataset.py`

为6个不同的环境生成训练数据：
- TopDown-focused
- TopDown-diffuse  
- Mixed-focused
- Mixed-diffuse
- DivPen-focused
- DivPen-diffuse

### 步骤2: 训练矩阵分解 (Matrix Factorization) - 仅用于baseline方法
**脚本**: `GeMS/train_MF.py`

为需要预训练embeddings的baseline方法（TopK, WkNN）生成MF embeddings。

### 步骤3: 预训练GeMS Ranker
**脚本**: `GeMS/pretrain_ranker.py`

在日志数据上预训练GeMS变分自编码器。

### 步骤4: 训练和测试RL Agent
**脚本**: `train_agent.py`

训练SAC+GeMS或baseline方法（SAC+TopK, SlateQ, REINFORCE+SoftMax, SAC+WkNN等）。

---

## 完整参数列表

### 步骤1: 生成日志数据参数

#### 必需参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--n_sess` | int | **必需** | 生成的轨迹数量（会话数） |
| `--env_name` | str | **必需** | 环境类型，目前只支持 "TopicRec" |
| `--path` | str | "data/RecSim/datasets/default" | 生成数据集的保存路径 |

#### 环境参数 (TopicRec)
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--num_items` | int | - | 物品总数 |
| `--boredom_threshold` | int | - | 厌倦阈值（用户对某个主题感到厌倦的阈值） |
| `--recent_items_maxlen` | int | - | 最近点击物品的最大长度 |
| `--boredom_moving_window` | int | - | 厌倦计算的滑动窗口大小 |
| `--short_term_boost` | float | - | 短期奖励提升系数 |
| `--episode_length` | int | - | 每个episode的长度（步数） |
| `--topic_size` | int | - | 每个主题的物品数量 |
| `--num_topics` | int | - | 主题总数 |
| `--env_offset` | float | - | 环境偏移参数（用于计算相关性分数） |
| `--env_slope` | float | - | 环境斜率参数（用于计算相关性分数） |
| `--env_omega` | float | - | 环境omega参数（用于计算相关性分数） |
| `--diversity_threshold` | int | - | 多样性阈值 |
| `--env_embedds` | str | - | 物品embeddings文件路径（如 "item_embeddings_focused.pt"） |
| `--click_model` | str | - | 点击模型类型："tdPBM" 或 "mixPBM" |
| `--diversity_penalty` | float | - | 多样性惩罚系数（仅用于mixPBM） |
| `--seed` | int | 2021 | 随机种子 |

#### 日志策略参数 (EpsGreedyPolicy)
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--epsilon_pol` | float | - | ε-贪婪策略的探索率 |

#### 作者使用的参数值（步骤1）
```bash
# 对于所有6个环境，基本参数相同：
--n_sess=100000
--epsilon_pol=0.5
--env_name="TopicRec"
--num_items=1000
--boredom_threshold=5
--recent_items_maxlen=10
--boredom_moving_window=5
--short_term_boost=1.0
--episode_length=100
--topic_size=2
--num_topics=10
--env_offset=0.28
--env_slope=100
--env_omega=0.9
--diversity_threshold=4
--seed=2754851

# 不同环境的特定参数：
# TopDown-focused:
--env_embedds="item_embeddings_focused.pt"
--click_model="tdPBM"
--path="data/RecSim/datasets/focused_topdown"

# TopDown-diffuse:
--env_embedds="item_embeddings_diffuse.pt"
--click_model="tdPBM"
--path="data/RecSim/datasets/diffuse_topdown"

# Mixed-focused:
--env_embedds="item_embeddings_focused.pt"
--click_model="mixPBM"
--path="data/RecSim/datasets/focused_mix"

# Mixed-diffuse:
--env_embedds="item_embeddings_diffuse.pt"
--click_model="mixPBM"
--path="data/RecSim/datasets/diffuse_mix"

# DivPen-focused:
--env_embedds="item_embeddings_focused.pt"
--click_model="mixPBM"
--diversity_penalty=3.0
--path="data/RecSim/datasets/focused_divpen"

# DivPen-diffuse:
--env_embedds="item_embeddings_diffuse.pt"
--click_model="mixPBM"
--diversity_penalty=3.0
--path="data/RecSim/datasets/diffuse_divpen"
```

---

### 步骤2: 矩阵分解训练参数

#### 必需参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--MF_dataset` | str | - | 数据集文件名（如 "focused_topdown_moving_env.pt"） |

#### 训练参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--train_val_split_MF` | float | 0.1 | 训练/验证集分割比例 |
| `--batch_size_MF` | int | 256 | MF训练的批次大小 |
| `--lr_MF` | float | 0.0001 | 学习率 |
| `--num_neg_sample_MF` | int | 1 | 负采样数量 |
| `--weight_decay_MF` | float | 0.0 | 权重衰减 |
| `--patience_MF` | int | 3 | 早停的patience |

#### 通用参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--data_dir` | str | "data/GeMS" | 数据目录 |
| `--device` | str | "cpu" | 设备（"cpu" 或 "cuda"） |

#### 作者使用的参数值（步骤2）
```bash
# 对每个数据集运行：
--MF_dataset="focused_topdown_moving_env.pt"  # 或其他5个数据集
--train_val_split_MF=0.1
--batch_size_MF=256
--lr_MF=0.0001
--num_neg_sample_MF=1
--weight_decay_MF=0.0
--patience_MF=3
--device="cuda"
```

---

### 步骤3: GeMS预训练参数

#### 必需参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--ranker` | str | **必需** | Ranker类型，必须为 "GeMS" |
| `--dataset` | str | **必需** | 数据集路径（如 "data/RecSim/datasets/focused_topdown_moving_env.pt"） |
| `--item_embedds` | str | **必需** | Item embeddings类型："scratch", "mf_init", "mf_fixed" |

#### GeMS Ranker参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--latent_dim` | int | - | 潜在空间维度（d） |
| `--lambda_click` | float | - | 点击损失权重 |
| `--lambda_KL` | float | - | KL散度损失权重（beta） |
| `--lambda_prior` | float | - | 先验损失权重 |
| `--ranker_lr` | float | - | Ranker学习率 |

#### 训练参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--max_epochs` | int | 300 | 最大训练轮数 |
| `--batch_size` | int | 256 | 批次大小 |
| `--seed` | int | 2021 | 随机种子 |

#### 环境参数（需要与步骤1一致）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--num_items` | int | - | 物品总数（需与生成数据时一致） |
| `--item_embedd_dim` | int | - | Item embedding维度 |

#### SwanLab参数（可选）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--swan_project` | str | None | SwanLab项目名称 |
| `--swan_workspace` | str | None | SwanLab工作空间 |
| `--swan_mode` | str | None | 日志模式：cloud/local/offline/disabled |
| `--swan_tags` | str[] | None | 标签列表 |
| `--swan_description` | str | None | 实验描述 |
| `--swan_logdir` | str | None | 本地日志目录 |
| `--swan_run_id` | str | None | 运行ID |
| `--swan_resume` | str | None | 恢复策略：must/allow/never |

#### 作者使用的参数值（步骤3）
```bash
# 对每个数据集和每个seed运行：
--ranker="GeMS"
--max_epochs=15  # 注意：README中写的是10，但config中是15
--dataset="data/RecSim/datasets/focused_topdown_moving_env.pt"  # 或其他5个数据集
--seed=58407201  # 或其他9个seeds
--item_embedds="scratch"
--lambda_click=0.5
--lambda_KL=1.0  # 注意：README中写的是0.5，但config中是1.0
--lambda_prior=0.0
--latent_dim=32
--device="cuda"
--batch_size=256
--ranker_lr=0.001
```

**使用的seeds**: [58407201, 496912423, 2465781, 300029, 215567, 23437561, 309081907, 548260111, 51941177, 212407167]

---

### 步骤4: RL Agent训练参数

#### 必需参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--agent` | str | **必需** | Agent类型："SAC", "WolpertingerSAC", "SlateQ", "REINFORCE", "REINFORCESlate" |
| `--belief` | str | **必需** | Belief encoder类型："none", "GRU" |
| `--ranker` | str | **必需** | Ranker类型："none", "topk", "kargmax", "GeMS" |
| `--item_embedds` | str | **必需** | Item embeddings类型："none", "scratch", "mf", "ideal" |
| `--env_name` | str | **必需** | 环境名称，如 "topics" |

#### Agent参数（SAC）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--q_lr` | float | - | Q网络学习率 |
| `--pi_lr` | float | - | 策略网络学习率 |
| `--hidden_layers_qnet` | int[] | - | Q网络隐藏层大小（如 [256]） |
| `--hidden_layers_pinet` | int[] | - | 策略网络隐藏层大小（如 [256]） |
| `--target_update_frequency` | int | - | 目标网络更新频率 |
| `--tau` | float | - | 软更新系数 |
| `--gamma` | float | - | 折扣因子 |
| `--auto_entropy` | bool/str | - | 是否自动调整熵系数 |
| `--alpha` | float | - | 熵系数（如果auto_entropy=False） |

#### Ranker参数（GeMS）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--latent_dim` | int | - | 潜在空间维度 |
| `--lambda_click` | float | - | 点击损失权重 |
| `--lambda_KL` | float | - | KL散度损失权重 |
| `--lambda_prior` | float | - | 先验损失权重 |
| `--ranker_embedds` | str | - | Ranker使用的embeddings类型 |
| `--ranker_sample` | bool/str | - | 是否在rank时采样 |
| `--ranker_dataset` | str | - | 用于加载ranker的数据集名称 |
| `--ranker_seed` | int | - | Ranker预训练时使用的seed |

#### Replay Buffer参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--capacity` | int | 1000000 | 缓冲区容量 |
| `--batch_size` | int | 32 | 批次大小 |
| `--random_steps` | int | - | 随机探索步数 |

#### Belief Encoder参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--belief_state_dim` | int | - | Belief状态维度 |
| `--item_embedd_dim` | int | - | Item embedding维度 |

#### 训练参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--max_steps` | int | 1e6 | 最大训练步数 |
| `--check_val_every_n_epoch` | int | 25 | 每N个epoch验证一次 |
| `--val_step_length` | int | - | 验证episode长度 |
| `--test_size` | int | - | 测试episode数量 |
| `--seed` | int | 2021 | 随机种子 |
| `--name` | str | "default" | 实验名称（用于图例） |

#### 环境参数（需与步骤1一致）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--num_items` | int | - | 物品总数 |
| `--boredom_threshold` | int | - | 厌倦阈值 |
| `--recent_items_maxlen` | int | - | 最近物品最大长度 |
| `--boredom_moving_window` | int | - | 厌倦滑动窗口 |
| `--env_omega` | float | - | 环境omega参数 |
| `--short_term_boost` | float | - | 短期奖励提升 |
| `--episode_length` | int | - | Episode长度 |
| `--env_offset` | float | - | 环境偏移 |
| `--env_slope` | float | - | 环境斜率 |
| `--diversity_threshold` | int | - | 多样性阈值 |
| `--topic_size` | int | - | 主题大小 |
| `--num_topics` | int | - | 主题数量 |
| `--diversity_penalty` | float | - | 多样性惩罚（仅mixPBM） |
| `--click_model` | str | - | 点击模型类型 |
| `--env_embedds` | str | - | 物品embeddings文件 |
| `--MF_checkpoint` | str | - | MF checkpoint名称（用于baseline） |

#### SwanLab参数（可选）
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--swan_project` | str | None | SwanLab项目名称 |
| `--swan_workspace` | str | None | SwanLab工作空间 |
| `--swan_mode` | str | None | 日志模式 |
| `--swan_tags` | str[] | None | 标签列表 |
| `--swan_description` | str | None | 实验描述 |
| `--swan_logdir` | str | None | 本地日志目录 |
| `--swan_run_id` | str | None | 运行ID |
| `--swan_resume` | str | None | 恢复策略 |

#### 作者使用的参数值（步骤4 - SAC+GeMS）
```bash
--agent="SAC"
--belief="GRU"
--ranker="GeMS"
--item_embedds="scratch"
--env_name="topics"
--device="cuda"
--max_steps=100000
--check_val_every_n_epoch=1000
--val_step_length=200
--test_size=500
--latent_dim=32
--name="SAC+GeMS"
--lambda_KL=1.0  # 注意：config中是1.0，README中写的是0.5
--lambda_click=0.5
--lambda_prior=0.0
--ranker_embedds="scratch"
--ranker_sample="False"
--capacity=10000
--batch_size=20
--q_lr=0.001
--hidden_layers_qnet=256
--target_update_frequency=1
--tau=0.002
--pi_lr=0.003
--hidden_layers_pinet=256
--gamma=0.8
--auto_entropy="True"
--alpha=0.2
--random_steps=2000
--belief_state_dim=20
--item_embedd_dim=20
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
--diversity_penalty=1.0  # 根据环境不同，可能是1.0或3.0
--ranker_dataset="focused_topdown_moving_env"  # 根据环境不同
--click_model="tdPBM"  # 或 "mixPBM"
--env_embedds="item_embeddings_focused.pt"  # 或 "item_embeddings_diffuse.pt"
--ranker_seed=58407201  # 需与预训练时一致
--seed=58407201  # 或其他9个seeds
```

#### 作者使用的参数值（步骤4 - SAC+TopK baseline）
```bash
--agent="SAC"
--belief="GRU"
--ranker="topk"
--item_embedds="mf"  # 或 "ideal"
--env_name="topics"
--device="cuda"
--seed=58407201
--max_steps=100000
--check_val_every_n_epoch=1000
--val_step_length=200
--test_size=500
--random_steps=2000
--belief_state_dim=20
--item_embedd_dim=20
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
--diversity_penalty=1.0
--MF_checkpoint="focused_topdown_moving_env"  # 根据环境不同
--click_model="tdPBM"
--env_embedds="item_embeddings_focused.pt"
--beliefs=actor critic
--capacity=10000
--batch_size=20
--q_lr=0.001
--hidden_layers_qnet=256
--target_update_frequency=1
--tau=0.002
--pi_lr=0.003
--hidden_layers_pinet=256
--gamma=0.8
--auto_entropy="True"
--alpha=0.2
--name="SAC+topK-mf"  # 或 "SAC+topK-ideal"
```

---

## 重要注意事项

1. **参数一致性**: 步骤1、3、4中的环境参数必须保持一致（num_items, boredom_threshold等）

2. **Seed一致性**: 
   - 步骤3的ranker_seed必须与步骤4的ranker_seed一致
   - 步骤4的seed用于RL训练，可以与ranker_seed不同

3. **数据集命名**: 生成的数据集会自动添加"_moving_env"后缀，所以实际文件名是"focused_topdown_moving_env.pt"

4. **超参数值差异**: 
   - README中lambda_KL=0.5，但config文件中是1.0
   - README中max_epochs=10，但config文件中是15
   - 建议以config文件为准

5. **6个环境**: 需要对所有6个环境分别运行完整流程

6. **10个seeds**: 每个实验需要运行10个不同的seed以获得统计显著性

---

## 完整实验流程示例

### 以TopDown-focused环境为例：

```bash
# 步骤1: 生成数据
python RecSim/generate_dataset.py \
  --n_sess=100000 \
  --epsilon_pol=0.5 \
  --env_name="TopicRec" \
  --num_items=1000 \
  --boredom_threshold=5 \
  --recent_items_maxlen=10 \
  --boredom_moving_window=5 \
  --short_term_boost=1.0 \
  --episode_length=100 \
  --topic_size=2 \
  --num_topics=10 \
  --env_offset=0.28 \
  --env_slope=100 \
  --env_omega=0.9 \
  --diversity_threshold=4 \
  --env_embedds="item_embeddings_focused.pt" \
  --click_model="tdPBM" \
  --path="data/RecSim/datasets/focused_topdown" \
  --seed=2754851

# 步骤2: 训练MF（仅用于baseline）
python GeMS/train_MF.py \
  --MF_dataset="focused_topdown_moving_env.pt" \
  --device="cuda"

# 步骤3: 预训练GeMS（对每个seed）
for seed in 58407201 496912423 2465781 300029 215567 23437561 309081907 548260111 51941177 212407167; do
  python GeMS/pretrain_ranker.py \
    --ranker="GeMS" \
    --max_epochs=15 \
    --dataset="data/RecSim/datasets/focused_topdown_moving_env.pt" \
    --seed=$seed \
    --item_embedds="scratch" \
    --lambda_click=0.5 \
    --lambda_KL=1.0 \
    --lambda_prior=0.0 \
    --latent_dim=32 \
    --device="cuda" \
    --batch_size=256 \
    --ranker_lr=0.001
done

# 步骤4: 训练RL Agent（对每个seed）
for seed in 58407201 496912423 2465781 300029 215567 23437561 309081907 548260111 51941177 212407167; do
  python train_agent.py \
    --agent="SAC" \
    --belief="GRU" \
    --ranker="GeMS" \
    --item_embedds="scratch" \
    --env_name="topics" \
    --device="cuda" \
    --max_steps=100000 \
    --check_val_every_n_epoch=1000 \
    --val_step_length=200 \
    --test_size=500 \
    --latent_dim=32 \
    --name="SAC+GeMS" \
    --lambda_KL=1.0 \
    --lambda_click=0.5 \
    --lambda_prior=0.0 \
    --ranker_embedds="scratch" \
    --ranker_sample="False" \
    --capacity=10000 \
    --batch_size=20 \
    --q_lr=0.001 \
    --hidden_layers_qnet=256 \
    --target_update_frequency=1 \
    --tau=0.002 \
    --pi_lr=0.003 \
    --hidden_layers_pinet=256 \
    --gamma=0.8 \
    --auto_entropy="True" \
    --alpha=0.2 \
    --random_steps=2000 \
    --belief_state_dim=20 \
    --item_embedd_dim=20 \
    --num_items=1000 \
    --boredom_threshold=5 \
    --recent_items_maxlen=10 \
    --boredom_moving_window=5 \
    --env_omega=0.9 \
    --short_term_boost=1.0 \
    --episode_length=100 \
    --env_offset=0.28 \
    --env_slope=100 \
    --diversity_threshold=4 \
    --topic_size=2 \
    --num_topics=10 \
    --diversity_penalty=1.0 \
    --ranker_dataset="focused_topdown_moving_env" \
    --click_model="tdPBM" \
    --env_embedds="item_embeddings_focused.pt" \
    --ranker_seed=$seed \
    --seed=$seed
done
```

---

## 参数总数统计

- **步骤1 (生成数据)**: 约18个参数
- **步骤2 (MF训练)**: 约8个参数  
- **步骤3 (GeMS预训练)**: 约15个参数
- **步骤4 (RL训练)**: 约40+个参数

总计需要配置的参数超过80个，但大部分在配置文件中已定义，实际运行时只需指定关键参数。





