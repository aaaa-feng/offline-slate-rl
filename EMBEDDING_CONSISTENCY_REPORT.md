# Embedding一致性调查报告

**调查日期**：2026-01-27
**调查范围**：在线RL → 离线数据收集 → 离线GeMS训练 → 离线TD3+BC训练
**调查目标**：识别embedding在整个流程中的不一致问题

---

## 执行摘要

本次调查通过靶向溯源方法，追踪了embedding在4个训练阶段的变化路径。调查发现了**1个CRITICAL级别问题**和**3个HIGH级别问题**，这些问题导致了整个流程中存在系统性的embedding不一致。

### 核心发现

🔴 **CRITICAL**：在线训练时Agent使用scratch embedding（随机初始化），而环境使用ground truth embedding（固定值），两者存在根本性不一致。

🟠 **HIGH**：数据收集、离线GeMS训练、TD3+BC训练都保留或加剧了这种不一致，形成了一条"Embedding漂移链"。

### 关键影响

- **状态空间不一致**：Agent的状态表示与环境的真实状态基于不同的embedding空间
- **数据质量问题**：收集的离线数据中包含不一致的状态表示
- **训练效果受限**：离线算法可能因embedding不匹配而性能受限

---

## 1. 调查背景

### 1.1 项目概述

本项目实现了一个完整的从在线RL到离线RL的训练流程：

```
阶段1：在线SAC+GeMS训练
   ↓
阶段2：离线数据收集
   ↓
阶段3：离线GeMS VAE训练
   ↓
阶段4：离线TD3+BC训练
```

### 1.2 调查动机

在项目实施过程中，发现在线训练使用了`--item_embedds=scratch`参数，这意味着Agent的embedding是随机初始化的。而环境的embedding是从固定文件加载的ground truth值。这引发了对整个流程中embedding一致性的担忧。

### 1.3 调查方法

采用**靶向溯源调查**方法，通过4个步骤追踪embedding的变化：

1. ✅ **审计在线训练checkpoint配置**：确认训练时使用的embedding模式
2. ✅ **审计数据收集日志**：确认数据收集时embedding的实际加载情况
3. ✅ **审计离线GeMS训练参数**：确认GeMS训练时embedding的处理方式
4. ✅ **审计TD3+BC代码逻辑**：确认最终训练时embedding的初始化逻辑

---

## 2. 详细调查发现

### 2.1 发现1：在线训练阶段的根本矛盾（CRITICAL）

**严重程度**：🔴 CRITICAL

**证据来源**：
- 文件：`src/data_collection/offline_data_collection/models/medium/model_info.json`
- 关键信息：checkpoint名称包含`scratch`关键字

**问题描述**：

在线训练阶段存在根本性的embedding不一致：

```
环境Embedding：item_embeddings_diffuse.pt（固定的ground truth）
Agent Embedding：--item_embedds=scratch（随机初始化）
```

**具体证据**：

从`model_info.json`中可以看到，所有checkpoint的`original_path`都包含`scratch`关键字：

```
SAC+GeMS_Medium_GeMS_diffuse_topdown_...scratch_seed58407201...
SAC+GeMS_Medium_GeMS_diffuse_mix_...scratch_seed58407201...
SAC+GeMS_Medium_GeMS_diffuse_divpen_...scratch_seed58407201...
```

这证实了在线训练时使用了`--item_embedds=scratch`参数。

**影响分析**：

1. **状态空间定义不一致**：
   - 环境根据ground truth embedding计算用户偏好和点击概率
   - Agent根据random embedding编码belief state
   - 两者的语义空间完全不同

2. **学习效率问题**：
   - Agent需要学习一个从random embedding到环境行为的映射
   - 这个映射本质上是在补偿embedding不匹配
   - 可能导致学习效率降低

3. **策略质量问题**：
   - 学到的策略可能不是最优的
   - 性能可能受限于embedding不匹配

**严重程度判断**：

这是一个CRITICAL级别的问题，因为它影响了整个流程的基础。所有后续阶段都建立在这个不一致的基础上。

---

### 2.2 发现2：数据收集保留了不一致（HIGH）

**严重程度**：🟠 HIGH

**证据来源**：
- 文件：`test/datacollecttest/full_10k_datasets_b3/mix_divpen_10k.log`
- 关键行：第45-46行、第64行

**问题描述**：

数据收集过程保留了在线训练阶段的embedding不一致。

**具体证据**：

从数据收集日志中可以看到：

```
第45行：✅ Ranker embeddings 已从 checkpoint 加载: torch.Size([1000, 20])
第46行：✅ Belief embeddings 已从 checkpoint 加载: torch.Size([1000, 20])
第64行：✅ 找到物品embeddings文件: .../item_embeddings_diffuse.pt
```

这说明：
- Agent的embeddings从checkpoint加载（这是在线训练时scratch随机初始化的结果）
- 环境的embeddings从文件加载（ground truth）

**影响分析**：

1. **数据质量问题**：
   - 收集的数据中，belief state基于scratch embedding
   - 但环境的真实状态基于ground truth embedding
   - 数据的语义一致性受到影响

2. **离线学习挑战**：
   - 离线算法需要从这些不一致的数据中学习
   - 可能影响离线学习的效果

**严重程度判断**：

这是一个HIGH级别的问题，因为它直接影响了离线数据的质量，而数据质量是离线RL成功的关键。

---

### 2.3 发现3：离线GeMS训练进一步修改Embedding（HIGH）

**严重程度**：🟠 HIGH

**证据来源**：
- 文件：`scripts/train_gems_offline.py`
- 关键行：第116行

**问题描述**：

离线GeMS训练使用默认参数`--fixed_embedds=scratch`，这意味着embedding在训练过程中是可训练的，会被修改。

**具体证据**：

从`train_gems_offline.py`的代码中可以看到：

```python
parser.add_argument("--fixed_embedds", type=str, default="scratch",
                   help="Embedding mode (scratch/mf_fixed)")
```

默认值是`"scratch"`，这意味着：
- Embedding不是frozen的
- 在GeMS训练过程中会被梯度更新
- 训练后的embedding与初始embedding不同

**影响分析**：

1. **Embedding进一步偏移**：
   - 初始embedding来自在线训练checkpoint（已经是scratch随机初始化的结果）
   - 训练后embedding被进一步修改
   - 与数据收集时的embedding产生新的偏差

2. **生成模型不一致**：
   - GeMS学到的生成模型基于修改后的embedding
   - 但数据是基于原始embedding收集的
   - 可能影响生成质量

**严重程度判断**：

这是一个HIGH级别的问题，因为它在已有不一致的基础上又引入了新的偏差。

---

### 2.4 发现4：TD3+BC使用GeMS修改后的Embedding（HIGH）

**严重程度**：🟠 HIGH

**证据来源**：
- 文件：`src/agents/offline/td3_bc.py`
- 关键行：第795-798行、第813行、第823行

**问题描述**：

TD3+BC最终使用的是GeMS训练后修改的embedding，而不是ground truth embedding。

**具体证据**：

从`td3_bc.py`的代码中可以看到：

```python
# 第795行：先加载ground truth
temp_embeddings = ItemEmbeddings.from_pretrained(
    config.item_embedds_path,  # item_embeddings_diffuse.pt
    config.device
)

# 第813行：但使用scratch模式加载GeMS
ranker = GeMS.load_from_checkpoint(
    gems_path,
    item_embeddings=temp_embeddings,
    fixed_embedds="scratch",  # 存在逻辑冲突
    ...
)

# 第823行：最终提取GeMS训练后的embedding
gems_embedding_weights = ranker.item_embeddings.weight.data.clone()
```

这说明：
- 虽然初始加载了ground truth embedding（第795行）
- 但最终使用的是GeMS checkpoint中的embedding（第823行）
- 如果GeMS修改了embedding（发现3），这里就会使用修改后的版本

**影响分析**：

1. **与环境不一致**：
   - TD3+BC使用GeMS修改后的embedding
   - 环境仍然使用ground truth embedding
   - 两者不一致

2. **与数据不一致**：
   - 数据收集时使用的是在线训练checkpoint的embedding
   - TD3+BC使用的是GeMS修改后的embedding
   - 两者也不一致

**严重程度判断**：

这是一个HIGH级别的问题，因为它导致最终训练阶段仍然存在embedding不一致。

---

## 3. Embedding漂移链可视化

### 3.1 完整流程图

```
阶段1：在线训练
├─ 环境：item_embeddings_diffuse.pt（固定）
└─ Agent：scratch随机初始化 ❌ 不一致

阶段2：数据收集
├─ 环境：item_embeddings_diffuse.pt（固定）
└─ Agent：从checkpoint加载（scratch结果）❌ 不一致

阶段3：离线GeMS训练
├─ 初始：从checkpoint加载（scratch结果）
└─ 训练后：被修改（fixed_embedds=scratch）❌ 进一步偏移

阶段4：TD3+BC训练
├─ 环境：item_embeddings_diffuse.pt（固定）
└─ Agent：使用GeMS修改后的embedding ❌ 不一致
```

### 3.2 Embedding演变路径

```
Ground Truth (固定) ─────────────────────────────────> 环境始终使用
                                                      (所有阶段)

Random Init (阶段1) ──> Checkpoint (阶段2) ──> GeMS Modified (阶段3) ──> TD3+BC使用
                                                                        (阶段4)
```

### 3.3 关键矛盾总结

1. **环境的embedding从未改变**：始终使用ground truth
2. **Agent的embedding经历了3次变化**：random → checkpoint → GeMS modified
3. **每个阶段的Agent embedding都与环境不一致**

---

## 4. 影响评估

### 4.1 对在线训练的影响（阶段1）

**问题**：Agent学习的状态空间与环境定义的状态空间不一致

**具体表现**：
- 环境根据ground truth embedding计算用户偏好和点击概率
- Agent根据random embedding编码belief state
- 两者的语义空间完全不同

**潜在后果**：
- Agent可能学到错误的策略
- 性能可能受限于embedding不匹配
- 但如果Agent能够适应，可能仍能学到有效策略（通过学习补偿映射）

### 4.2 对数据收集的影响（阶段2）

**问题**：收集的数据中状态表示不一致

**具体表现**：
- 数据中的belief state基于scratch embedding
- 但环境的真实状态基于ground truth embedding
- 数据的语义一致性受到影响

**潜在后果**：
- 离线算法学习时面临不一致的状态表示
- 可能影响离线学习的效果
- 数据质量下降

### 4.3 对离线GeMS训练的影响（阶段3）

**问题**：Embedding在训练中被修改

**具体表现**：
- 初始embedding来自在线训练checkpoint（scratch结果）
- 训练后embedding被修改（fixed_embedds=scratch允许训练）
- 修改后的embedding与数据收集时不同

**潜在后果**：
- GeMS学到的生成模型基于修改后的embedding
- 与数据收集时的embedding空间不一致
- 可能影响生成质量

### 4.4 对TD3+BC训练的影响（阶段4）

**问题**：使用GeMS修改后的embedding，与环境不一致

**具体表现**：
- TD3+BC使用GeMS训练后的embedding
- 环境仍然使用ground truth embedding
- 两者不一致

**潜在后果**：
- 策略学习时状态表示不一致
- 可能影响最终性能
- 评估时的性能可能不准确

---

## 5. 建议和下一步行动

### 5.1 建议1：验证实际影响（优先级：HIGH）

**目标**：确定embedding不一致是否真的影响了性能

**验证方法**：

1. **比较在线训练的性能指标**
   - 检查训练曲线是否收敛
   - 对比使用ideal embedding的基线性能
   - 分析是否存在性能瓶颈

2. **比较离线算法的性能**
   - 检查TD3+BC的最终性能
   - 对比使用一致embedding的版本
   - 评估性能差距

3. **计算embedding的实际差异**
   - 加载ground truth embedding
   - 加载各阶段的checkpoint embedding
   - 计算余弦相似度、欧氏距离等指标
   - 可视化embedding空间的差异

**预期结果**：
- 如果性能差异显著（>10%），需要考虑修复
- 如果性能差异较小（<5%），可以记录为技术债务

### 5.2 建议2：如果需要修复，采用统一Embedding策略（优先级：MEDIUM）

**方案A：全流程使用Ground Truth Embedding**

```
阶段1：--item_embedds=ideal（使用环境的ground truth）
阶段2：自动一致（从checkpoint加载）
阶段3：--fixed_embedds=mf_fixed（冻结embedding）
阶段4：自动一致（使用GeMS的embedding）
```

**优点**：
- 所有阶段embedding完全一致
- 状态空间定义统一
- 理论上最优

**缺点**：
- 需要重新训练所有阶段
- 成本较高（时间和计算资源）

**方案B：全流程使用Scratch Embedding（当前接近此方案）**

```
阶段1：--item_embedds=scratch（当前已是）
阶段2：自动一致（从checkpoint加载）
阶段3：--fixed_embedds=scratch（当前已是）
阶段4：需要修改环境加载逻辑，使用GeMS的embedding
```

**优点**：
- 只需修改阶段4的环境初始化
- 成本较低

**缺点**：
- 环境的embedding不再是ground truth
- 可能影响环境的语义一致性

### 5.3 建议3：记录为技术债务（优先级：LOW）

**如果性能影响不大**：
- 记录当前的embedding不一致问题
- 在文档中说明这是已知限制
- 继续当前实验
- 在未来的项目中避免此问题

---

## 6. 总结

### 6.1 调查完成情况

✅ **已完成4个阶段的靶向溯源调查**
- 阶段1：在线训练checkpoint配置审计
- 阶段2：数据收集日志审计
- 阶段3：离线GeMS训练参数审计
- 阶段4：TD3+BC代码逻辑审计

✅ **已识别4个关键问题点**
- 1个CRITICAL级别问题
- 3个HIGH级别问题

✅ **已分析影响和潜在后果**
- 对每个阶段的影响进行了详细评估
- 识别了embedding漂移链

✅ **已提供修复建议**
- 3个不同优先级的建议
- 2个具体的修复方案

### 6.2 核心结论

**主要问题**：
整个流程存在系统性的embedding不一致问题。在线训练时Agent使用scratch embedding（随机初始化），而环境使用ground truth embedding（固定值），这个根本性矛盾在后续所有阶段都被保留或加剧。

**Embedding漂移链**：
```
Ground Truth (固定) → 环境始终使用
Random Init → Checkpoint → GeMS Modified → TD3+BC使用
```

**关键影响**：
- 状态空间定义不一致
- 数据质量受影响
- 可能限制最终性能

### 6.3 下一步行动

**立即行动**（优先级：HIGH）：
1. 验证实际性能影响
2. 计算embedding的实际差异
3. 评估是否需要修复

**如果影响显著**（优先级：MEDIUM）：
1. 选择修复方案（方案A或方案B）
2. 重新训练受影响的阶段
3. 验证修复效果

**如果影响不大**（优先级：LOW）：
1. 记录为技术债务
2. 在文档中说明已知限制
3. 继续当前实验

---

## 附录

### A. 关键文件清单

**Embedding文件**：
- `data/embeddings/item_embeddings_diffuse.pt`
- `data/embeddings/item_embeddings_focused.pt`

**Checkpoint文件**：
- 在线训练：`models/medium/sac_gems_models/*/SAC_GeMS_*.ckpt`
- 离线GeMS：`checkpoints/gems/*.ckpt`
- TD3+BC：`checkpoints/td3_bc/*.pt`

**代码文件**：
- `src/data_collection/offline_data_collection/core/model_loader.py`
- `scripts/train_gems_offline.py`
- `config/offline_config.py`
- `src/agents/offline/td3_bc.py`

**日志文件**：
- `test/datacollecttest/full_10k_datasets_b3/mix_divpen_10k.log`

### B. 调查方法论

本次调查采用**靶向溯源**方法，通过以下步骤追踪embedding的变化：

1. **配置审计**：检查训练参数和checkpoint命名
2. **日志分析**：确认实际运行时的embedding加载情况
3. **代码审计**：分析embedding初始化和处理逻辑
4. **交叉验证**：通过多个证据源确认发现

这种方法的优点是：
- 快速定位问题
- 证据充分
- 可追溯性强

---

**报告生成日期**：2026-01-27
**报告版本**：v1.0
**调查人员**：Claude Sonnet 4.5

