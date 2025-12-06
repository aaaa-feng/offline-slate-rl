#!/usr/bin/env python3
"""
SAC+GeMS 完整交互测试
展示从模型加载、环境初始化到数据收集的所有细节
用于验证SAC+GeMS模型的正确性和性能
"""
import sys
import os
from pathlib import Path

# 动态获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import numpy as np
from model_loader import ModelLoader
from environment_factory import EnvironmentFactory
from data_formats import SlateDataset, SlateTrajectory, SlateTransition, SlateObservation, SlateAction, SlateInfo
from metrics import SlateMetrics

print("="*80)
print("SAC+GeMS 完整交互测试")
print("="*80)
print(f"测试目的: 验证SAC+GeMS模型加载和性能")
print(f"预期性能: ~250-320分 (训练日志: 317.75分)")
print("="*80)

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    """打印子分隔线"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def test_complete_pipeline():
    """测试完整的数据收集流程"""

    print_section("完整的数据收集流程详细测试")
    print("本测试将展示:")
    print("  1. 环境加载的所有参数")
    print("  2. 模型加载的所有组件")
    print("  3. 每一次交互的完整过程")
    print("  4. 状态如何建立")
    print("  5. 潜空间动作如何得到")
    print("  6. 真实推荐如何得到")
    print("  7. 用户心智向量如何设定")
    print("  8. 数据如何保存")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 环境名称
    env_name = "diffuse_topdown"
    print(f"测试环境: {env_name}")

    # ============================================================================
    # 第1步：加载模型
    # ============================================================================
    print_section("第1步：加载模型 (SAC + GeMS + scratch)")

    model_loader = ModelLoader()
    # 临时修改models_dir为expert级别模型目录
    quality_level = "expert"  # 可选: "expert", "medium", "random"
    expert_models_dir = Path(__file__).resolve().parent.parent / "models" / quality_level / env_name
    model_loader.models_dir = str(expert_models_dir)
    print(f"模型目录: {model_loader.models_dir}")
    print(f"质量级别: {quality_level}")

    print("\n加载模型组件...")
    agent, ranker, belief_encoder = model_loader.load_agent(
        env_name=env_name,
        agent_type="SAC",
        ranker_type="GeMS",
        embedding_type="scratch"
    )

    print_subsection("1.1 Agent (SAC) 配置")
    print(f"类型: {type(agent).__name__}")
    print(f"  state_dim (belief state维度): {agent.state_dim}")
    print(f"  action_dim (latent action维度): {agent.action_dim}")
    print(f"  num_actions (Q网络输出维度): {agent.num_actions}")
    print(f"  gamma (折扣因子): {agent.gamma}")
    print(f"  alpha (熵正则化系数): {agent.alpha}")
    print(f"  device: {agent.my_device}")
    print(f"\n  PolicyNet结构:")
    print(f"    输入: belief_state ({agent.state_dim}维)")
    print(f"    输出: mean + std ({agent.action_dim} * 2 = {agent.action_dim * 2}维)")
    print(f"    激活: Tanh squashing")
    print(f"\n  QNet结构:")
    print(f"    输入: belief_state + latent_action ({agent.state_dim} + {agent.action_dim}维)")
    print(f"    输出: Q值 ({agent.num_actions}维)")

    print_subsection("1.2 Ranker (GeMS) 配置")
    print(f"类型: {type(ranker).__name__}")
    print(f"  latent_dim: {ranker.latent_dim}")
    print(f"  item_embedd_dim: {ranker.item_embedd_dim}")
    print(f"  rec_size (slate大小): {ranker.rec_size}")
    print(f"  num_items (物品总数): {ranker.num_items}")
    print(f"  device: {ranker.device}")
    print(f"\n  工作原理:")
    print(f"    1. 接收latent_action ({ranker.latent_dim}维)")
    print(f"    2. 通过decoder解码为slate embeddings")
    print(f"    3. 计算与所有物品embeddings的相似度")
    print(f"    4. 选择Top-{ranker.rec_size}个物品作为slate")

    print_subsection("1.3 Belief Encoder (GRU) 配置")
    print(f"类型: {type(belief_encoder).__name__}")
    print(f"  hidden_dim (GRU隐藏层维度): {belief_encoder.hidden_dim}")
    print(f"  belief_state_dim (输出维度): {belief_encoder.belief_state_dim}")
    print(f"  item_embedd_dim: {belief_encoder.item_embedd_dim}")
    print(f"  rec_size: {belief_encoder.rec_size}")
    print(f"  beliefs (分支): {belief_encoder.beliefs}")
    print(f"\n  工作原理:")
    print(f"    1. 接收原始观察 (slate + clicks)")
    print(f"    2. 通过GRU编码历史信息")
    print(f"    3. 输出belief_state ({belief_encoder.belief_state_dim}维)")

    # ============================================================================
    # 第2步：创建环境
    # ============================================================================
    print_section("第2步：创建推荐环境 (TopicRec)")

    env_factory = EnvironmentFactory()
    env_config = env_factory.get_env_config(env_name)

    print_subsection("2.1 环境配置参数 (完整)")
    config_categories = {
        "基础配置": ['env_name', 'num_items', 'rec_size', 'episode_length'],
        "用户模型": ['num_topics', 'topic_size', 'env_omega', 'env_alpha'],
        "点击模型": ['click_model', 'env_offset', 'env_slope', 'rel_threshold', 'prop_threshold'],
        "厌倦机制": ['boredom_threshold', 'recent_items_maxlen', 'boredom_moving_window', 'short_term_boost'],
        "多样性": ['diversity_penalty', 'diversity_threshold'],
        "其他": ['click_only_once', 'env_embedds', 'item_embedd_dim', 'sim_seed']
    }

    for category, keys in config_categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in env_config:
                print(f"  {key}: {env_config[key]}")

    print("\n创建环境...")
    import os
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(str(project_root))
    environment = env_factory.create_environment(env_name)

    print_subsection("2.2 环境实例详细信息")
    print(f"环境类型: {type(environment).__name__}")
    print(f"\n物品空间:")
    print(f"  num_items: {environment.num_items}")
    item_dim = env_config.get('item_embedd_dim', 20)
    print(f"  item_embedd_dim: {item_dim}")
    print(f"  rec_size: {environment.rec_size}")
    
    print(f"\n用户模型:")
    print(f"  num_topics: {environment.num_topics}")
    print(f"  topic_size: {environment.topic_size}")
    print(f"  omega (兴趣衰减): {environment.omega}")
    print(f"  alpha (兴趣增强): {environment.alpha}")
    
    print(f"\n点击模型 ({environment.click_model}):")
    # 加载物品embeddings
    project_root = Path(__file__).resolve().parent.parent
    item_embeddings_path = project_root / "data" / "RecSim" / "embeddings" / env_config['env_embedds']
    item_embeddings_path = str(item_embeddings_path)
    item_embeddings = torch.load(item_embeddings_path, map_location=device)
    print(f"\n物品Embeddings:")
    print(f"  路径: {item_embeddings_path}")
    print(f"  形状: {item_embeddings.shape}")
    print(f"  示例 (物品0): {item_embeddings[0].cpu().numpy()}")

    metrics_calculator = SlateMetrics(item_embeddings, env_config['num_items'])

    # ============================================================================
    # 第3步：环境重置
    # ============================================================================
    print_section("第3步：环境重置 - 初始化用户状态")

    obs, info = environment.reset()

    print_subsection("3.1 初始观察 (obs)")
    print("这是环境返回的原始观察，包含:")
    for key, value in obs.items():
        if torch.is_tensor(value):
            print(f"\n  {key}:")
            print(f"    类型: tensor")
            print(f"    形状: {value.shape}")
            print(f"    dtype: {value.dtype}")
            if key == 'slate':
                print(f"    内容 (初始推荐的10个物品): {value.cpu().tolist()}")
            elif key == 'clicks':
                print(f"    内容 (用户点击): {value.cpu().tolist()}")
                print(f"    说明: 环境reset后的初始用户响应")
        else:
            print(f"\n  {key}: {value}")

    print_subsection("3.2 初始信息 (info)")
    print("这是环境的额外信息，包含用户内部状态:")
    for key, value in info.items():
        if torch.is_tensor(value):
            print(f"\n  {key}:")
            print(f"    类型: tensor")
            print(f"    形状: {value.shape}")
            if key == 'user_state':
                print(f"    内容 (用户心智向量): {value.cpu().numpy()}")
                print(f"    说明: {environment.num_topics}个主题 × {environment.topic_size}维 = {value.shape[0]}维")
                print(f"    解释: 表示用户对每个主题的兴趣程度")
            elif key == 'bored':
                print(f"    内容 (厌倦状态): {value.cpu().tolist()}")
                print(f"    说明: 每个主题是否达到厌倦阈值")
            elif key == 'scores':
                print(f"    内容 (相关性分数): {value.cpu().numpy()}")
                print(f"    说明: 每个推荐物品与用户兴趣的匹配度")
        else:
            print(f"\n  {key}: {value}")

    # ============================================================================
    # 第4步：初始化Belief State
    # ============================================================================
    print_section("第4步：初始化Belief State")

    print_subsection("4.1 重置GRU Hidden State")
    for module in belief_encoder.beliefs:
        belief_encoder.hidden[module] = torch.zeros(
            1, 1, belief_encoder.hidden_dim,
            device=belief_encoder.my_device
        )
        print(f"  {module} hidden state: shape={belief_encoder.hidden[module].shape}")

    print_subsection("4.2 第一次Belief编码")
    print("输入: 原始观察 (dict)")
    print("  - slate: 推荐的物品列表")
    print("  - clicks: 用户点击反馈")
    print("\n处理过程:")
    print("  1. 提取slate和clicks")
    print("  2. 获取物品embeddings")
    print("  3. 拼接为输入向量")
    print("  4. 通过GRU编码")
    print("  5. 输出belief_state")

    belief_state = belief_encoder.forward(obs)
    
    print(f"\n输出: Belief State (tensor)")
    print(f"  形状: {belief_state.shape}")
    print(f"  dtype: {belief_state.dtype}")
    print(f"  device: {belief_state.device}")
    print(f"  内容: {belief_state.cpu().numpy()}")
    print(f"  统计: mean={belief_state.mean():.4f}, std={belief_state.std():.4f}, min={belief_state.min():.4f}, max={belief_state.max():.4f}")

    # ============================================================================
    # 第5步：交互循环
    # ============================================================================
    print_section("第5步：交互循环 - 展示前3步的完整过程")

    trajectory = SlateTrajectory()
    episode_slates = []

    done = False
    timestep = 0
    max_display_steps = 3

    agent.eval()
    ranker.eval()
    belief_encoder.eval()

    with torch.no_grad():
        while not done and timestep < 100:
            if timestep < max_display_steps:
                print_subsection(f"时间步 {timestep}")

            # 当前belief state
            current_belief_state = belief_state.clone().detach()

            if timestep < max_display_steps:
                print(f"\n【输入】当前Belief State:")
                print(f"  形状: {current_belief_state.shape}")
                print(f"  内容: {current_belief_state.cpu().numpy()}")
                print(f"  说明: 这是GRU编码的用户历史交互信息")

            # ========================================================================
            # 动作生成
            # ========================================================================
            if timestep < max_display_steps:
                print(f"\n【动作生成】完整流程:")

            # Step 1: SAC生成latent action
            if timestep < max_display_steps:
                print(f"\n  Step 1: SAC PolicyNet生成latent action")
                print(f"    输入: belief_state ({current_belief_state.shape[0]}维)")
                print(f"    处理:")
                print(f"      1. PolicyNet前向传播")
                print(f"      2. 输出mean和std")
                print(f"      3. 使用mean (贪婪策略, sample=False)")
                print(f"      4. Tanh squashing到[-1, 1]")

            latent_action = agent.get_action(current_belief_state, sample=False)

            if timestep < max_display_steps:
                print(f"    输出: latent_action")
                print(f"      形状: {latent_action.shape}")
                print(f"      dtype: {latent_action.dtype}")
                print(f"      内容: {latent_action.cpu().numpy()}")
                print(f"      统计: mean={latent_action.mean():.4f}, std={latent_action.std():.4f}")
                print(f"      值域: [{latent_action.min():.4f}, {latent_action.max():.4f}]")
                print(f"      说明: 这是一个{latent_action.shape[0]}维的连续向量，表示推荐意图")

            # Step 2: GeMS Ranker解码为slate
            if timestep < max_display_steps:
                print(f"\n  Step 2: GeMS Ranker解码为slate")
                print(f"    输入: latent_action ({latent_action.shape[0]}维)")
                print(f"    处理:")
                print(f"      1. 计算latent_action与所有{ranker.num_items}个物品embeddings的相似度")
                print(f"      2. 选择相似度最高的Top-{ranker.rec_size}个物品")
                print(f"      3. 返回物品ID列表")

            slate = ranker.rank(latent_action)

            if timestep < max_display_steps:
                print(f"    输出: slate")
                print(f"      形状: {slate.shape}")
                print(f"      dtype: {slate.dtype}")
                print(f"      内容 (推荐的{ranker.rec_size}个物品ID): {slate.cpu().tolist()}")
                print(f"      说明: 这是最终推荐给用户的物品列表")

            # 保存latent action
            latent_action = latent_action.clone().detach()

            # 创建观察和动作
            observation = SlateObservation(
                belief_state=current_belief_state,
                raw_obs=None
            )

            slate_list = slate.cpu().tolist()
            action = SlateAction(
                discrete_slate=slate_list,
                latent_action=latent_action
            )

            if timestep < max_display_steps:
                print(f"\n  数据保存:")
                print(f"    ✓ discrete_slate: {slate_list}")
                print(f"    ✓ latent_action: shape={latent_action.shape}")

            # ========================================================================
            # 环境交互
            # ========================================================================
            if timestep < max_display_steps:
                print(f"\n【环境交互】用户模拟:")
                print(f"  输入: slate (推荐列表)")

            next_obs_raw, reward, done, next_info = environment.step(slate)

            if timestep < max_display_steps:
                print(f"\n  用户行为模拟过程:")
                print(f"    1. 计算每个物品与用户兴趣的相关性分数")
                print(f"    2. 根据点击模型 ({environment.click_model}) 生成点击概率")
                print(f"    3. 采样生成点击行为")
                print(f"    4. 更新用户心智向量 (兴趣衰减/增强)")
                print(f"    5. 检查厌倦状态")
                print(f"    6. 计算reward")

                print(f"\n  输出:")
                print(f"    reward: {reward}")
                print(f"    done: {done}")
                print(f"    clicks: {next_obs_raw['clicks'].cpu().tolist()}")
                print(f"    点击数: {next_obs_raw['clicks'].sum().item()}")
                
                print(f"\n  用户状态更新:")
                print(f"    新的心智向量: {next_info['user_state'].cpu().numpy()}")
                print(f"    厌倦状态: {next_info['bored'].cpu().tolist()}")
                print(f"    相关性分数: {next_info['scores'].cpu().numpy()}")

            # 保存clicks
            clicks = next_obs_raw.get('clicks', torch.zeros(len(slate)))
            if not torch.is_tensor(clicks):
                clicks = torch.tensor(clicks)

            # 更新belief state
            if timestep < max_display_steps:
                print(f"\n【Belief State更新】")
                print(f"  输入: next_obs_raw (新的观察)")
                print(f"  处理: 通过GRU更新hidden state")

            next_belief_state = belief_encoder.forward(next_obs_raw, done=done)
            if next_belief_state is None:
                next_belief_state = belief_state.clone().detach()
            else:
                next_belief_state = next_belief_state.clone().detach()

            if timestep < max_display_steps:
                print(f"  输出: next_belief_state")
                print(f"    形状: {next_belief_state.shape}")
                print(f"    内容: {next_belief_state.cpu().numpy()}")
                print(f"    说明: 编码了最新的交互历史")

            next_observation = SlateObservation(
                belief_state=next_belief_state,
                raw_obs=next_obs_raw
            )

            # ========================================================================
            # 计算指标
            # ========================================================================
            episode_slates.append(slate_list)
            diversity_score = metrics_calculator.calculate_diversity_score(slate_list)
            coverage_score = metrics_calculator.calculate_coverage_score(slate_list, episode_slates)

            if timestep < max_display_steps:
                print(f"\n【指标计算】")
                print(f"  diversity_score: {diversity_score:.4f}")
                print(f"    说明: 基于物品embeddings的余弦相似度，越高越多样")
                print(f"  coverage_score: {coverage_score:.4f}")
                print(f"    说明: episode内推荐过的唯一物品数 / 总物品数")
                print(f"  click_through_rate: {clicks.sum().item() / len(slate):.4f}")
                print(f"    说明: 点击数 / 推荐数")

            # 创建信息和转移
            info_data = SlateInfo(
                clicks=clicks,
                diversity_score=diversity_score,
                coverage_score=coverage_score,
                episode_return=0.0,
                episode_id=0,
                timestep=timestep
            )

            transition = SlateTransition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                done=done,
                info=info_data
            )

            trajectory.add_transition(transition)

            # 更新状态
            belief_state = next_belief_state
            timestep += 1

            if timestep == max_display_steps:
                print(f"\n{'='*80}")
                print(f"  ... (省略后续 {100 - max_display_steps} 步，继续收集数据) ...")
                print(f"{'='*80}")

    # ============================================================================
    # 第6步：数据保存
    # ============================================================================
    print_section("第6步：数据保存与验证")

    # 更新episode return
    episode_return = trajectory.get_return()
    for transition in trajectory.transitions:
        transition.info.episode_return = episode_return

    print_subsection("6.1 Episode统计")
    print(f"  Episode长度: {len(trajectory.transitions)}")
    print(f"  总回报: {episode_return:.2f}")
    print(f"  平均reward: {episode_return / len(trajectory.transitions):.2f}")

    # 创建数据集
    dataset = SlateDataset(f"{env_name}_test")
    dataset.add_trajectory(trajectory)

    print_subsection("6.2 数据集统计")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 转换为D4RL格式
    print_subsection("6.3 转换为D4RL格式")
    d4rl_data = dataset.to_d4rl_format()

    print("\nD4RL数据格式 (标准离线RL格式):")
    for key, value in d4rl_data.items():
        print(f"\n  {key}:")
        print(f"    shape: {value.shape}")
        print(f"    dtype: {value.dtype}")
        
        if key == 'observations':
            print(f"    说明: belief_state (GRU编码的用户历史)")
            print(f"    第一个样本: {value[0]}")
        elif key == 'actions':
            print(f"    说明: latent_action (SAC输出的连续动作) ✅")
            print(f"    第一个样本: {value[0]}")
            print(f"    统计: mean={value.mean():.4f}, std={value.std():.4f}")
            print(f"    值域: [{value.min():.4f}, {value.max():.4f}]")
        elif key == 'slates':
            print(f"    说明: discrete_slate (GeMS解码的离散推荐)")
            print(f"    第一个样本: {value[0]}")
        elif key == 'rewards':
            print(f"    说明: 用户点击产生的即时奖励")
            print(f"    统计: sum={value.sum():.2f}, mean={value.mean():.4f}")
        elif key == 'clicks':
            print(f"    说明: 用户点击行为 (0/1向量)")
            print(f"    总点击数: {value.sum()}")

    # ============================================================================
    # 第7步：数据验证
    # ============================================================================
    print_section("第7步：数据验证")

    print("验证关键字段:")

    checks = [
        ("observations形状", d4rl_data['observations'].shape == (100, 20), f"期望(100, 20), 实际{d4rl_data['observations'].shape}"),
        ("observations类型", d4rl_data['observations'].dtype in [np.float32, np.float64], f"期望float, 实际{d4rl_data['observations'].dtype}"),
        ("actions形状", d4rl_data['actions'].shape == (100, 32), f"期望(100, 32), 实际{d4rl_data['actions'].shape}"),
        ("actions类型", d4rl_data['actions'].dtype in [np.float32, np.float64], f"期望float, 实际{d4rl_data['actions'].dtype}"),
        ("actions是连续值", d4rl_data['actions'].dtype in [np.float32, np.float64], "✓ 连续latent action"),
        ("slates形状", d4rl_data['slates'].shape == (100, 10), f"期望(100, 10), 实际{d4rl_data['slates'].shape}"),
        ("slates类型", d4rl_data['slates'].dtype in [np.int32, np.int64], f"期望int, 实际{d4rl_data['slates'].dtype}"),
        ("rewards非零", d4rl_data['rewards'].sum() > 0, f"总reward={d4rl_data['rewards'].sum():.2f}"),
    ]

    all_passed = True
    for check_name, result, detail in checks:
        status = "✓" if result else "✗"
        print(f"\n  [{status}] {check_name}")
        print(f"      {detail}")
        if not result:
            all_passed = False

    # ============================================================================
    # 总结
    # ============================================================================
    print_section("测试完成！")

    if all_passed:
        print("\n✅ 所有验证通过！")
    else:
        print("\n⚠️ 部分验证失败，请检查上述错误")

    print("\n完整数据流总结:")
    print("  " + "─" * 76)
    print("  原始观察 (dict: slate + clicks)")
    print("    ↓ [Belief Encoder - GRU]")
    print("  Belief State (20维 tensor)")
    print("    ↓ [SAC PolicyNet - Gaussian]")
    print("  Latent Action (32维连续向量) ✅ 已保存到actions字段")
    print("    ↓ [GeMS Ranker - Decoder + Similarity]")
    print("  Slate (10个物品ID) ✅ 已保存到slates字段")
    print("    ↓ [环境交互 - 用户模拟]")
    print("  Reward + Clicks + Next Observation")
    print("    ↓ [数据格式转换]")
    print("  D4RL格式数据 ✅ 可用于TD3+BC等离线RL算法")
    print("  " + "─" * 76)

    print("\n关键配置总结:")
    print(f"  策略: SAC + GeMS + scratch embeddings")
    print(f"  环境: {env_name}")
    print(f"  sample: False (贪婪策略)")
    print(f"  belief_state_dim: {agent.state_dim}")
    print(f"  latent_action_dim: {agent.action_dim}")
    print(f"  slate_size: {ranker.rec_size}")
    print(f"  num_items: {ranker.num_items}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_complete_pipeline()
