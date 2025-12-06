# Focused环境SAC+GeMS实验复现指南

## 实验概览

针对**focused环境**中的**SAC+GeMS**方法，需要运行以下3个环境：
1. **TopDown-focused** (tdPBM, diversity_penalty=1.0)
2. **Mixed-focused** (mixPBM, diversity_penalty=1.0)
3. **DivPen-focused** (mixPBM, diversity_penalty=3.0)

## 实验统计

| 步骤 | 每个环境 | 3个环境总计 | 说明 |
|------|---------|------------|------|
| 步骤1: 生成数据 | 1次 | **3次** | 每个环境生成一次数据集 |
| 步骤2: 训练MF | 0次 | **0次** | SAC+GeMS不需要MF embeddings |
| 步骤3: 预训练GeMS | 10次 | **30次** | 每个环境10个seeds |
| 步骤4: 训练SAC+GeMS | 10次 | **30次** | 每个环境10个seeds |
| **总计** | 21次 | **63次** | |

**10个seeds**: 58407201, 496912423, 2465781, 300029, 215567, 23437561, 309081907, 548260111, 51941177, 212407167

---

## 实验顺序

### 必须按顺序执行：

1. **步骤1**: 为3个focused环境生成数据（可并行）
2. **步骤3**: 为每个环境的每个seed预训练GeMS（可并行，但需要步骤1完成）
3. **步骤4**: 为每个环境的每个seed训练SAC+GeMS（可并行，但需要步骤3完成）

---

## 完整实验命令

### 步骤1: 生成日志数据（3个环境）

#### 1.1 TopDown-focused
```bash
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
```

#### 1.2 Mixed-focused
```bash
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
  --click_model="mixPBM" \
  --diversity_penalty=1.0 \
  --path="data/RecSim/datasets/focused_mix" \
  --seed=2754851
```

#### 1.3 DivPen-focused
```bash
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
  --click_model="mixPBM" \
  --diversity_penalty=3.0 \
  --path="data/RecSim/datasets/focused_divpen" \
  --seed=2754851
```

---

### 步骤3: 预训练GeMS Ranker（30个实验）

#### 3.1 TopDown-focused (10个seeds)

**Seed 58407201:**
```bash
python GeMS/pretrain_ranker.py \
  --ranker="GeMS" \
  --max_epochs=15 \
  --dataset="data/RecSim/datasets/focused_topdown.pt" \
  --seed=58407201 \
  --item_embedds="scratch" \
  --lambda_click=0.5 \
  --lambda_KL=1.0 \
  --lambda_prior=0.0 \
  --latent_dim=32 \
  --device="cuda" \
  --batch_size=256 \
  --ranker_lr=0.001
```

**其他9个seeds** (496912423, 2465781, 300029, 215567, 23437561, 309081907, 548260111, 51941177, 212407167):
只需将上面的 `--seed=58407201` 替换为对应的seed值即可。

#### 3.2 Mixed-focused (10个seeds)

**Seed 58407201:**
```bash
python GeMS/pretrain_ranker.py \
  --ranker="GeMS" \
  --max_epochs=15 \
  --dataset="data/RecSim/datasets/focused_mix.pt" \
  --seed=58407201 \
  --item_embedds="scratch" \
  --lambda_click=0.5 \
  --lambda_KL=1.0 \
  --lambda_prior=0.0 \
  --latent_dim=32 \
  --device="cuda" \
  --batch_size=256 \
  --ranker_lr=0.001
```

**其他9个seeds**: 同样只需替换 `--seed` 参数。

#### 3.3 DivPen-focused (10个seeds)

**Seed 58407201:**
```bash
python GeMS/pretrain_ranker.py \
  --ranker="GeMS" \
  --max_epochs=15 \
  --dataset="data/RecSim/datasets/focused_divpen.pt" \
  --seed=58407201 \
  --item_embedds="scratch" \
  --lambda_click=0.5 \
  --lambda_KL=1.0 \
  --lambda_prior=0.0 \
  --latent_dim=32 \
  --device="cuda" \
  --batch_size=256 \
  --ranker_lr=0.001
```

**其他9个seeds**: 同样只需替换 `--seed` 参数。

---

### 步骤4: 训练SAC+GeMS Agent（30个实验）

#### 4.1 TopDown-focused (10个seeds)

**Seed 58407201:**
```bash
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
  --ranker_dataset="focused_topdown" \
  --click_model="tdPBM" \
  --env_embedds="item_embeddings_focused.pt" \
  --ranker_seed=58407201 \
  --seed=58407201
```

**其他9个seeds**: 替换 `--ranker_seed` 和 `--seed` 参数为对应的seed值。

#### 4.2 Mixed-focused (10个seeds)

**Seed 58407201:**
```bash
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
  --ranker_dataset="focused_mix" \
  --click_model="mixPBM" \
  --env_embedds="item_embeddings_focused.pt" \
  --ranker_seed=58407201 \
  --seed=58407201
```

**关键差异**: 
- `--ranker_dataset="focused_mix"` (而不是 "focused_topdown")
- `--click_model="mixPBM"` (而不是 "tdPBM")

**其他9个seeds**: 替换 `--ranker_seed` 和 `--seed` 参数。

#### 4.3 DivPen-focused (10个seeds)

**Seed 58407201:**
```bash
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
  --diversity_penalty=3.0 \
  --ranker_dataset="focused_divpen" \
  --click_model="mixPBM" \
  --env_embedds="item_embeddings_focused.pt" \
  --ranker_seed=58407201 \
  --seed=58407201
```

**关键差异**: 
- `--ranker_dataset="focused_divpen"` (而不是 "focused_topdown")
- `--click_model="mixPBM"` (而不是 "tdPBM")
- `--diversity_penalty=3.0` (而不是 1.0)

**其他9个seeds**: 替换 `--ranker_seed` 和 `--seed` 参数。

---

## 参数总结

### 所有环境共用的参数

#### 环境参数（必须与步骤1一致）
- `--num_items=1000`
- `--boredom_threshold=5`
- `--recent_items_maxlen=10`
- `--boredom_moving_window=5`
- `--env_omega=0.9`
- `--short_term_boost=1.0`
- `--episode_length=100`
- `--env_offset=0.28`
- `--env_slope=100`
- `--diversity_threshold=4`
- `--topic_size=2`
- `--num_topics=10`
- `--env_embedds="item_embeddings_focused.pt"` (所有focused环境相同)

#### GeMS预训练参数（步骤3）
- `--ranker="GeMS"`
- `--max_epochs=15` ⚠️ **争议参数**: README写10，config写15
- `--lambda_click=0.5`
- `--lambda_KL=1.0` ⚠️ **争议参数**: README写0.5，config写1.0
- `--lambda_prior=0.0`
- `--latent_dim=32`
- `--batch_size=256`
- `--ranker_lr=0.001`
- `--item_embedds="scratch"`

#### SAC+GeMS训练参数（步骤4）
- `--agent="SAC"`
- `--belief="GRU"`
- `--ranker="GeMS"`
- `--item_embedds="scratch"`
- `--max_steps=100000`
- `--check_val_every_n_epoch=1000`
- `--val_step_length=200`
- `--test_size=500`
- `--latent_dim=32`
- `--lambda_KL=1.0` ⚠️ **争议参数**: README写0.5，config写1.0
- `--lambda_click=0.5` ⚠️ **争议参数**: README写0.2，config写0.5
- `--lambda_prior=0.0`
- `--capacity=10000`
- `--batch_size=20`
- `--q_lr=0.001`
- `--pi_lr=0.003`
- `--hidden_layers_qnet=256`
- `--hidden_layers_pinet=256`
- `--target_update_frequency=1`
- `--tau=0.002`
- `--gamma=0.8`
- `--auto_entropy="True"`
- `--alpha=0.2`
- `--random_steps=2000`
- `--belief_state_dim=20`
- `--item_embedd_dim=20`

### 环境特定参数

| 环境 | ranker_dataset | click_model | diversity_penalty |
|------|----------------|-------------|-------------------|
| TopDown-focused | `focused_topdown` | `tdPBM` | `1.0` |
| Mixed-focused | `focused_mix` | `mixPBM` | `1.0` |
| DivPen-focused | `focused_divpen` | `mixPBM` | `3.0` |

---

## ⚠️ 争议参数说明

### 1. `lambda_KL` (KL散度损失权重)

- **README示例**: 0.5
- **配置文件**: 1.0
- **建议**: 使用 **1.0**（以配置文件为准）

### 2. `lambda_click` (点击损失权重)

- **README示例 (步骤3)**: 0.2
- **配置文件 (步骤3)**: 0.5
- **README示例 (步骤4)**: 0.2
- **配置文件 (步骤4)**: 0.5
- **建议**: 使用 **0.5**（以配置文件为准）

### 3. `max_epochs` (GeMS预训练轮数)

- **README示例**: 10
- **配置文件**: 15
- **建议**: 使用 **15**（以配置文件为准）

### 参数选择建议

**强烈建议使用配置文件中的参数值**，因为：
1. 配置文件是实际实验使用的参数
2. README可能是早期版本的示例
3. 论文结果应该基于配置文件中的参数

---

## 快速运行脚本

### 批量运行所有实验的bash脚本示例

```bash
#!/bin/bash

# 步骤1: 生成数据（3个环境）
echo "Step 1: Generating datasets..."
python RecSim/generate_dataset.py --n_sess=100000 --epsilon_pol=0.5 --env_name="TopicRec" --num_items=1000 --boredom_threshold=5 --recent_items_maxlen=10 --boredom_moving_window=5 --short_term_boost=1.0 --episode_length=100 --topic_size=2 --num_topics=10 --env_offset=0.28 --env_slope=100 --env_omega=0.9 --diversity_threshold=4 --env_embedds="item_embeddings_focused.pt" --click_model="tdPBM" --path="data/RecSim/datasets/focused_topdown" --seed=2754851

python RecSim/generate_dataset.py --n_sess=100000 --epsilon_pol=0.5 --env_name="TopicRec" --num_items=1000 --boredom_threshold=5 --recent_items_maxlen=10 --boredom_moving_window=5 --short_term_boost=1.0 --episode_length=100 --topic_size=2 --num_topics=10 --env_offset=0.28 --env_slope=100 --env_omega=0.9 --diversity_threshold=4 --env_embedds="item_embeddings_focused.pt" --click_model="mixPBM" --diversity_penalty=1.0 --path="data/RecSim/datasets/focused_mix" --seed=2754851

python RecSim/generate_dataset.py --n_sess=100000 --epsilon_pol=0.5 --env_name="TopicRec" --num_items=1000 --boredom_threshold=5 --recent_items_maxlen=10 --boredom_moving_window=5 --short_term_boost=1.0 --episode_length=100 --topic_size=2 --num_topics=10 --env_offset=0.28 --env_slope=100 --env_omega=0.9 --diversity_threshold=4 --env_embedds="item_embeddings_focused.pt" --click_model="mixPBM" --diversity_penalty=3.0 --path="data/RecSim/datasets/focused_divpen" --seed=2754851

# 步骤3: 预训练GeMS（30个实验）
echo "Step 3: Pre-training GeMS rankers..."
SEEDS=(58407201 496912423 2465781 300029 215567 23437561 309081907 548260111 51941177 212407167)
ENVS=("focused_topdown" "focused_mix" "focused_divpen")

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Training GeMS for $env with seed $seed"
        python GeMS/pretrain_ranker.py \
            --ranker="GeMS" \
            --max_epochs=15 \
            --dataset="data/RecSim/datasets/${env}.pt" \
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
done

# 步骤4: 训练SAC+GeMS（30个实验）
echo "Step 4: Training SAC+GeMS agents..."
# 注意：需要根据环境设置不同的click_model和diversity_penalty
# 这里只给出TopDown-focused的示例，其他环境需要相应修改
for seed in "${SEEDS[@]}"; do
    echo "Training SAC+GeMS for focused_topdown with seed $seed"
    python train_agent.py \
        --agent="SAC" --belief="GRU" --ranker="GeMS" --item_embedds="scratch" \
        --env_name="topics" --device="cuda" --max_steps=100000 \
        --check_val_every_n_epoch=1000 --val_step_length=200 --test_size=500 \
        --latent_dim=32 --name="SAC+GeMS" \
        --lambda_KL=1.0 --lambda_click=0.5 --lambda_prior=0.0 \
        --ranker_embedds="scratch" --ranker_sample="False" \
        --capacity=10000 --batch_size=20 --q_lr=0.001 \
        --hidden_layers_qnet=256 --target_update_frequency=1 --tau=0.002 \
        --pi_lr=0.003 --hidden_layers_pinet=256 --gamma=0.8 \
        --auto_entropy="True" --alpha=0.2 --random_steps=2000 \
        --belief_state_dim=20 --item_embedd_dim=20 \
        --num_items=1000 --boredom_threshold=5 --recent_items_maxlen=10 \
        --boredom_moving_window=5 --env_omega=0.9 --short_term_boost=1.0 \
        --episode_length=100 --env_offset=0.28 --env_slope=100 \
        --diversity_threshold=4 --topic_size=2 --num_topics=10 \
        --diversity_penalty=1.0 --ranker_dataset="focused_topdown" \
        --click_model="tdPBM" --env_embedds="item_embeddings_focused.pt" \
        --ranker_seed=$seed --seed=$seed
done
```

---

## 检查清单

在开始实验前，请确认：

- [ ] 已安装所有依赖 (`pip install -r requirements.txt`)
- [ ] 已激活conda gems环境
- [ ] 有足够的GPU资源（建议使用CUDA）
- [ ] 有足够的存储空间（每个数据集约几百MB，每个checkpoint约几十MB）
- [ ] 已确认使用配置文件中的参数值（lambda_KL=1.0, lambda_click=0.5, max_epochs=15）

---

## 预期输出文件

### 步骤1输出
- `data/RecSim/datasets/focused_topdown.pt`
- `data/RecSim/datasets/focused_mix.pt`
- `data/RecSim/datasets/focused_divpen.pt`

### 步骤3输出
- `data/GeMS/checkpoints/GeMS_focused_topdown_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed{SEED}.ckpt` (10个文件)
- `data/GeMS/checkpoints/GeMS_focused_mix_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed{SEED}.ckpt` (10个文件)
- `data/GeMS/checkpoints/GeMS_focused_divpen_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed{SEED}.ckpt` (10个文件)

### 步骤4输出
- `data/checkpoints/focused_topdown/SAC+GeMS_GeMS_focused_topdown_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed{SEED}_agentseed{SEED}_gamma0.8.ckpt` (10个文件)
- `data/results/focused_topdown/SAC+GeMS_GeMS_focused_topdown_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed{SEED}_agentseed{SEED}_gamma0.8.pt` (10个文件)
- 类似地，focused_mix和focused_divpen环境也会生成对应的checkpoint和result文件

---

## 时间估算

假设每个实验的运行时间：
- 步骤1（生成数据）: 约30分钟/环境
- 步骤3（预训练GeMS）: 约1-2小时/实验（取决于GPU）
- 步骤4（训练SAC+GeMS）: 约2-4小时/实验（取决于GPU）

**总时间估算**: 
- 如果串行运行: 约90-180小时（4-7.5天）
- 如果并行运行（有足够GPU）: 可大幅缩短

---

## 注意事项

1. **Seed一致性**: 步骤3的`--seed`必须与步骤4的`--ranker_seed`一致
2. **参数一致性**: 步骤1、3、4中的环境参数必须完全一致
3. **文件路径**: 确保所有路径正确，特别是数据集路径和checkpoint路径
4. **GPU内存**: 确保GPU有足够内存，可能需要调整batch_size
5. **SwanLab记录**: 如果需要记录实验，可以添加SwanLab相关参数





