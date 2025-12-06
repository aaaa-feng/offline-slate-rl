# 模型迁移总结

## ✅ 迁移完成情况

### 已迁移的模型 (2024-12-04)

#### 1. SAC+GeMS (12个模型)
```
/data/liyuefeng/offline-slate-rl/checkpoints/expert/sac_gems/
├── diffuse_divpen/
│   ├── beta0.5_click0.2.ckpt (3.5M)
│   └── beta1.0_click0.5.ckpt (3.5M)
├── diffuse_mix/
│   ├── beta0.5_click0.2.ckpt (3.5M)
│   └── beta1.0_click0.5.ckpt (3.5M)
├── diffuse_topdown/
│   ├── beta0.5_click0.2.ckpt (3.5M)
│   └── beta1.0_click0.5.ckpt (3.5M)
├── focused_divpen/
│   ├── beta0.5_click0.2.ckpt (3.5M)
│   └── beta1.0_click0.5.ckpt (3.5M)
├── focused_mix/
│   ├── beta0.5_click0.2.ckpt (3.5M)
│   └── beta1.0_click0.5.ckpt (3.5M)
└── focused_topdown/
    ├── beta0.5_click0.2.ckpt (3.5M)
    └── beta1.0_click0.5.ckpt (3.5M)
```

#### 2. SAC+WkNN (3个模型)
```
/data/liyuefeng/offline-slate-rl/checkpoints/expert/sac_wknn/
├── focused_divpen/
│   └── model.ckpt (3.9M)
├── focused_mix/
│   └── model.ckpt (3.9M)
└── focused_topdown/
    └── model.ckpt (3.9M)
```

#### 3. SlateQ (3个模型)
```
/data/liyuefeng/offline-slate-rl/checkpoints/expert/slateq/
├── focused_divpen/
│   └── model.ckpt (4.5M)
├── focused_mix/
│   └── model.ckpt (4.5M)
└── focused_topdown/
    └── model.ckpt (4.5M)
```

**总计**: 18个expert级别模型已成功迁移

## 📍 路径关系说明

### 问题1: 训练模型和数据收集模型的路径是否不一样？

**是的，路径不一样！**

#### 训练模型保存路径（旧项目）
```
/data/liyuefeng/gems/gems_official/official_code/data/checkpoints/{env_name}/
└── SAC+GeMS_..._gamma0.8.ckpt
```

#### 数据收集使用的模型路径（新项目）
```
/data/liyuefeng/offline-slate-rl/checkpoints/expert/{agent}/{env_name}/
└── model.ckpt 或 beta*.ckpt
```

#### 数据收集脚本读取路径
```
/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/sac_gems_models/{env_name}/
└── SAC_GeMS_..._gamma0.8.ckpt
```

**注意**: 目前数据收集脚本还在使用旧的路径结构，需要更新！

## 🔄 训练50k步模型后的完整流程

### 场景：训练一个50k步的medium模型并收集数据

#### Step 1: 修改训练代码（在旧项目中）
```bash
cd /data/liyuefeng/gems/gems_official/official_code
```

编辑 `train_agent.py`，在第281行附近添加：
```python
# 原有的最佳模型checkpoint
ckpt = ModelCheckpoint(monitor='val_reward', dirpath=ckpt_dir,
                       filename=ckpt_name, mode='max')

# 新增：50k步的中间checkpoint
ckpt_medium = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=ckpt_name + "_step50000",
    every_n_train_steps=50000,
    save_top_k=-1  # 保存所有checkpoint
)

# 在trainer中添加这个callback
trainer_agent = pl.Trainer(
    logger=exp_logger,
    enable_progress_bar=args.progress_bar,
    callbacks=[RichProgressBar(), ckpt, ckpt_medium],  # 添加ckpt_medium
    ...
)
```

#### Step 2: 运行训练（在旧项目中）
```bash
cd /data/liyuefeng/gems/gems_official/official_code

# 训练diffuse_topdown环境的medium模型
python train_agent.py \
    --agent=SAC \
    --belief=GRU \
    --ranker=GeMS \
    --item_embedds=scratch \
    --env_name=topics \
    --device=cuda \
    --seed=58407201 \
    --ranker_seed=58407201 \
    --max_steps=100000 \
    --ranker_dataset=diffuse_topdown \
    --latent_dim=32 \
    --lambda_KL=1.0 \
    --lambda_click=0.5 \
    --lambda_prior=0.0 \
    --ranker_embedds=scratch \
    --click_model=tdPBM \
    --env_embedds=item_embeddings_diffuse.pt \
    --gamma=0.8 \
    --name=SAC+GeMS \
    --swan_project=GeMS_RL_Training_202512 \
    --run_name=SAC_GeMS_diffuse_topdown_medium_50k
```

训练完成后，会生成两个checkpoint：
```
/data/liyuefeng/gems/gems_official/official_code/data/checkpoints/diffuse_topdown/
├── SAC+GeMS_..._gamma0.8.ckpt              # 最佳模型（可能在任意步数）
└── SAC+GeMS_..._gamma0.8_step50000.ckpt    # 50k步的模型
```

#### Step 3: 迁移模型到新项目
```bash
# 复制50k步的模型到medium目录
cp /data/liyuefeng/gems/gems_official/official_code/data/checkpoints/diffuse_topdown/SAC+GeMS_*_step50000.ckpt \
   /data/liyuefeng/offline-slate-rl/checkpoints/medium/sac_gems/diffuse_topdown/beta1.0_click0.5_step50k.ckpt

# 验证文件已复制
ls -lh /data/liyuefeng/offline-slate-rl/checkpoints/medium/sac_gems/diffuse_topdown/
```

#### Step 4: 更新数据收集脚本的model_loader.py

需要在 `model_loader.py` 中添加加载medium模型的函数：

```python
def load_medium_models(self):
    """加载medium质量的模型 (50k步训练)"""
    models = {}

    # Medium模型目录
    medium_dir = self.project_root / "checkpoints" / "medium" / "sac_gems"

    for env_name in ["diffuse_topdown", "diffuse_mix", "diffuse_divpen",
                     "focused_topdown", "focused_mix", "focused_divpen"]:
        model_path = medium_dir / env_name / "beta1.0_click0.5_step50k.ckpt"

        if model_path.exists():
            print(f"\n加载 {env_name} 环境的SAC+GeMS medium模型...")
            # 加载模型的逻辑（类似load_focused_models）
            agent, ranker, belief = self._load_sac_gems_checkpoint(
                model_path, env_name
            )
            models[env_name] = {
                'agent': agent,
                'ranker': ranker,
                'belief': belief
            }
        else:
            print(f"⚠️  未找到 {env_name} 的medium模型")

    return models
```

#### Step 5: 更新collect_data.py脚本

在 `collect_data.py` 中添加 `--quality` 参数：

```python
parser.add_argument('--quality', type=str, default='expert',
                    choices=['expert', 'medium', 'random'],
                    help='数据质量级别')

# 在加载模型部分
if args.quality == 'expert':
    if args.env_name.startswith('focused'):
        models = model_loader.load_focused_models()
    else:
        models = model_loader.load_diffuse_models()
elif args.quality == 'medium':
    models = model_loader.load_medium_models()
```

#### Step 6: 运行数据收集
```bash
cd /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection

# 收集medium质量数据
python scripts/collect_data.py \
    --env_name diffuse_topdown \
    --quality medium \
    --episodes 10000 \
    --output_dir /data/liyuefeng/offline-slate-rl/datasets/offline_datasets \
    --gpu 5
```

#### Step 7: 数据保存位置
```
/data/liyuefeng/offline-slate-rl/datasets/offline_datasets/
├── diffuse_topdown_expert/          # expert数据（已有）
│   ├── expert_data.pkl
│   └── expert_data_d4rl.npz
└── diffuse_topdown_medium/          # medium数据（新收集）
    ├── medium_data.pkl
    └── medium_data_d4rl.npz
```

## 📊 当前数据收集状态

### 正在进行的数据收集
- **任务**: Focused环境的expert数据收集
- **模型**: SAC+GeMS (beta1.0_click0.5)
- **进度**: ~5% (约455/10000 episodes)
- **GPU**: 5, 6, 7
- **预计完成时间**: 约65分钟

### 使用的模型路径（当前）
```
/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/sac_gems_models/
├── focused_topdown/SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt
├── focused_mix/SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt
└── focused_divpen/SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt
```

**注意**: 这些是从旧项目直接复制过来的，使用的是旧的命名和路径结构。

## 🎯 下一步计划

### 立即任务
1. ✅ 创建新的checkpoints目录结构
2. ✅ 迁移所有expert模型到新结构
3. ⏳ 等待focused expert数据收集完成
4. ⏸️ 更新数据收集脚本以支持新的checkpoints路径结构

### 后续任务（收集medium数据）
5. ⏸️ 修改训练代码支持50k步checkpoint保存
6. ⏸️ 训练6个环境的medium模型（50k步）
7. ⏸️ 更新model_loader.py添加load_medium_models()
8. ⏸️ 收集medium质量数据

## 💡 关键要点

### 路径关系总结
1. **训练时**: 模型保存在旧项目的 `data/checkpoints/{env_name}/`
2. **迁移后**: 模型统一管理在新项目的 `checkpoints/{quality}/{agent}/{env_name}/`
3. **数据收集**: 脚本从 `checkpoints/` 读取模型，收集数据到 `datasets/offline_datasets/`

### 为什么需要迁移？
- **统一管理**: 所有模型集中在一个地方，便于管理
- **按质量分类**: expert/medium/random 清晰分类
- **易于扩展**: 未来添加新模型或新质量级别很容易
- **避免混淆**: 旧项目和新项目的模型分离，不会互相干扰

### 训练50k模型的关键点
1. **修改训练代码**: 添加 `ModelCheckpoint` 在50k步保存
2. **运行完整训练**: 仍然训练100k步，但会在50k步额外保存一个checkpoint
3. **迁移到正确位置**: 复制到 `checkpoints/medium/` 目录
4. **更新数据收集脚本**: 添加加载medium模型的逻辑
5. **收集数据**: 使用 `--quality medium` 参数收集数据
