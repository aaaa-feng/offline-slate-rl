# 模型管理和数据收集路径规划

## 📁 当前路径结构

### 1. 训练模型保存位置（旧项目）
```
/data/liyuefeng/gems/gems_official/official_code/data/checkpoints/
├── diffuse_divpen/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   └── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (3.5M)
├── diffuse_mix/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   └── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (3.5M)
├── diffuse_topdown/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   └── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (3.5M)
├── focused_divpen/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   ├── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (3.5M)
│   └── SAC+WkNN_seed58407201_gamma0.8.ckpt (3.9M)
├── focused_mix/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   ├── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (3.5M)
│   └── SAC+WkNN_seed58407201_gamma0.8.ckpt (3.9M)
├── focused_topdown/
│   ├── SAC+GeMS_..._beta0.5_..._gamma0.8.ckpt (3.5M)
│   ├── SAC+GeMS_..._beta1.0_..._gamma0.8.ckpt (2.6M)
│   ├── SAC+GeMS_..._beta1.0_..._gamma0.8-v1.ckpt (2.6M)
│   ├── SAC+GeMS_..._beta1.0_..._gamma0.8-v2.ckpt (3.5M)
│   └── SAC+WkNN_seed58407201_gamma0.8.ckpt (3.9M)
└── default/
    ├── REINFORCE+SoftMax_seed58407201_gamma0.8.ckpt (3.4M)
    ├── SlateQ_seed58407201_gamma0.8.ckpt (2.2M)
    ├── SlateQ_seed58407201_gamma0.8-v1.ckpt (4.5M)
    └── SlateQ_seed58407201_gamma0.8-v2.ckpt (4.5M)
```

### 2. 数据收集使用的模型位置（新项目）
```
/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/sac_gems_models/
├── diffuse_divpen/
│   └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
├── diffuse_mix/
│   └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
├── diffuse_topdown/
│   └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
├── focused_divpen/
│   └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
├── focused_mix/
│   └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
└── focused_topdown/
    └── SAC_GeMS_..._beta1.0_..._gamma0.8.ckpt (已复制，用于收集expert数据)
```

## 🎯 新的模型管理方案

### 方案设计原则
1. **集中管理**: 所有模型统一存放在新项目的checkpoints目录
2. **按质量分类**: expert (100k步) / medium (50k步) / random
3. **按agent分类**: SAC+GeMS / SAC+WkNN / SlateQ / REINFORCE
4. **易于扩展**: 支持未来添加新的训练步数或agent

### 推荐的新路径结构
```
/data/liyuefeng/offline-slate-rl/checkpoints/
├── expert/                          # Expert级别模型 (100k步训练完成)
│   ├── sac_gems/
│   │   ├── diffuse_divpen/
│   │   │   ├── beta0.5_click0.2.ckpt
│   │   │   └── beta1.0_click0.5.ckpt
│   │   ├── diffuse_mix/
│   │   │   ├── beta0.5_click0.2.ckpt
│   │   │   └── beta1.0_click0.5.ckpt
│   │   ├── diffuse_topdown/
│   │   │   ├── beta0.5_click0.2.ckpt
│   │   │   └── beta1.0_click0.5.ckpt
│   │   ├── focused_divpen/
│   │   │   ├── beta0.5_click0.2.ckpt
│   │   │   └── beta1.0_click0.5.ckpt
│   │   ├── focused_mix/
│   │   │   ├── beta0.5_click0.2.ckpt
│   │   │   └── beta1.0_click0.5.ckpt
│   │   └── focused_topdown/
│   │       ├── beta0.5_click0.2.ckpt
│   │       └── beta1.0_click0.5.ckpt
│   ├── sac_wknn/
│   │   ├── focused_divpen/
│   │   │   └── model.ckpt
│   │   ├── focused_mix/
│   │   │   └── model.ckpt
│   │   └── focused_topdown/
│   │       └── model.ckpt
│   ├── slateq/
│   │   ├── focused_divpen/
│   │   │   └── model.ckpt
│   │   ├── focused_mix/
│   │   │   └── model.ckpt
│   │   └── focused_topdown/
│   │       └── model.ckpt
│   └── reinforce/
│       └── default/
│           └── model.ckpt
│
├── medium/                          # Medium级别模型 (50k步训练)
│   ├── sac_gems/
│   │   ├── diffuse_divpen/
│   │   │   └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   │   ├── diffuse_mix/
│   │   │   └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   │   ├── diffuse_topdown/
│   │   │   └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   │   ├── focused_divpen/
│   │   │   └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   │   ├── focused_mix/
│   │   │   └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   │   └── focused_topdown/
│   │       └── beta1.0_click0.5_step50k.ckpt  (待训练)
│   └── [其他agent的medium模型...]
│
└── random/                          # Random策略模型
    └── [如果需要的话]
```

### 数据收集脚本使用的模型路径
```
/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/
├── expert/                          # 软链接到 checkpoints/expert/
│   ├── sac_gems/
│   ├── sac_wknn/
│   ├── slateq/
│   └── reinforce/
└── medium/                          # 软链接到 checkpoints/medium/
    └── sac_gems/
```

## 🔄 路径关系说明

### 训练模型 → 数据收集的流程

1. **训练阶段** (在旧项目中)
   ```
   训练脚本运行 → 保存checkpoint到:
   /data/liyuefeng/gems/gems_official/official_code/data/checkpoints/{env_name}/
   ```

2. **模型迁移** (整理到新项目)
   ```
   旧checkpoint → 复制到新项目:
   /data/liyuefeng/offline-slate-rl/checkpoints/{quality}/{agent}/{env_name}/
   ```

3. **数据收集准备**
   ```
   创建软链接:
   /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/{quality}/{agent}/
   → 指向 checkpoints/{quality}/{agent}/
   ```

4. **数据收集运行**
   ```
   collect_data.py 读取模型:
   models/{quality}/{agent}/{env_name}/model.ckpt

   收集数据保存到:
   /data/liyuefeng/offline-slate-rl/datasets/offline_datasets/{env_name}_{quality}/
   ```

## 📝 训练50k步模型后的操作流程

### 场景：训练一个50k步的medium模型

1. **修改训练代码** (在旧项目中)
   ```python
   # 在 train_agent.py 中添加中间checkpoint保存
   ckpt_medium = ModelCheckpoint(
       dirpath=ckpt_dir,
       filename=ckpt_name + "_step50000",
       every_n_train_steps=50000,
       save_top_k=-1
   )
   ```

2. **运行训练** (在旧项目中)
   ```bash
   cd /data/liyuefeng/gems/gems_official/official_code
   python train_agent.py --agent=SAC --ranker=GeMS --env_name=topics \
       --ranker_dataset=diffuse_topdown --max_steps=50000 ...
   ```

   训练完成后，模型保存在:
   ```
   /data/liyuefeng/gems/gems_official/official_code/data/checkpoints/diffuse_topdown/
   └── SAC+GeMS_..._step50000.ckpt
   ```

3. **迁移模型到新项目**
   ```bash
   # 复制到新项目的medium目录
   cp /data/liyuefeng/gems/gems_official/official_code/data/checkpoints/diffuse_topdown/SAC+GeMS_..._step50000.ckpt \
      /data/liyuefeng/offline-slate-rl/checkpoints/medium/sac_gems/diffuse_topdown/beta1.0_click0.5_step50k.ckpt
   ```

4. **更新数据收集脚本的model_loader.py**
   ```python
   # 在 model_loader.py 中添加 load_medium_models() 函数
   def load_medium_models(self):
       """加载medium质量的模型 (50k步训练)"""
       models_dir = self.base_dir / "medium" / "sac_gems"
       # ... 加载逻辑
   ```

5. **运行数据收集**
   ```bash
   cd /data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection
   python scripts/collect_data.py \
       --env_name diffuse_topdown \
       --quality medium \
       --episodes 10000 \
       --output_dir /data/liyuefeng/offline-slate-rl/datasets/offline_datasets \
       --gpu 5
   ```

6. **数据保存位置**
   ```
   /data/liyuefeng/offline-slate-rl/datasets/offline_datasets/
   ├── diffuse_topdown_expert/      # expert数据 (已有)
   │   ├── expert_data.pkl
   │   └── expert_data_d4rl.npz
   └── diffuse_topdown_medium/      # medium数据 (新收集)
       ├── medium_data.pkl
       └── medium_data_d4rl.npz
   ```

## ✅ 下一步行动计划

### 立即执行
1. ✅ 创建新的checkpoints目录结构
2. ✅ 迁移所有expert模型到新结构
3. ✅ 更新数据收集脚本以支持新路径
4. ⏳ 等待当前focused expert数据收集完成

### 后续任务
5. ⏸️ 修改训练代码支持50k步checkpoint保存
6. ⏸️ 训练6个环境的medium模型 (50k步)
7. ⏸️ 收集medium质量数据
8. ⏸️ 验证expert和medium数据质量

## 📊 模型性能对比 (用于选择最佳模型)

### SAC+GeMS (Final Episode Reward)
| Environment | beta0.5_click0.2 | beta1.0_click0.5 | 选择 |
|-------------|------------------|------------------|------|
| diffuse_divpen | 272 | 175 | beta0.5 ✓ |
| diffuse_mix | 205 | 258 | beta1.0 ✓ |
| diffuse_topdown | 348 | 240 | beta0.5 ✓ |
| focused_divpen | 212 | 208 | 相近 |
| focused_mix | 237 | 68 | beta0.5 ✓ |
| focused_topdown | 357 | 310 | beta0.5 ✓ |

**注意**: 目前数据收集使用的是beta1.0模型，但从性能来看beta0.5在多数环境中表现更好。
建议后续收集数据时使用性能最好的模型。

### Baseline性能 (Focused环境)
| Agent | focused_topdown | focused_mix | focused_divpen |
|-------|-----------------|-------------|----------------|
| SAC+GeMS (beta1.0) | 310 | 68 | 208 |
| SAC+WkNN | 68 | 48 | 30 |
| SlateQ | 190 | 230 | 41 |

**观察**: SAC+GeMS在大多数环境中表现最好，但SlateQ在focused_mix上表现出色。
