#!/usr/bin/env python3
"""
数据收集前的完整测试脚本
目标：5分钟内验证所有配置是否正确，避免3-4小时后才发现问题

测试内容：
1. 模型加载验证
2. 环境参数验证
3. 性能基准测试（与训练日志对比）
4. 数据格式验证
5. 关键配置检查
"""
import torch
import sys
import os
from pathlib import Path
import numpy as np

# 添加GeMS路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_loader import ModelLoader
from environment_factory import EnvironmentFactory

# 训练日志中的基准性能
TRAINING_BENCHMARKS = {
    'diffuse_topdown': 317.75,
    'diffuse_mix': None,  # 需要从训练日志中获取
    'diffuse_divpen': None  # 需要从训练日志中获取
}

# 训练时的环境参数（从train_agent.py的默认值）
EXPECTED_ENV_PARAMS = {
    'num_items': 1000,
    'rec_size': 10,
    'episode_length': 100,
    'num_topics': 10,
    'topic_size': 2,
    'env_offset': 0.28,
    'env_slope': 100,
    'env_omega': 0.9,
    'env_alpha': 1.0,
    'boredom_threshold': 5,
    'recent_items_maxlen': 10,
    'boredom_moving_window': 5,
    'short_term_boost': 1.0,
    'diversity_penalty': 1.0,  # diffuse_divpen是3.0
    'diversity_threshold': 4,
    'click_model': 'tdPBM',
    'click_only_once': False,
    'sim_seed': 24321357327
}

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_result(test_name, passed, details=""):
    """打印测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"     {details}")

def test_model_loading(env_name):
    """测试1: 模型加载验证"""
    print_section(f"测试1: 模型加载验证 - {env_name}")

    try:
        model_loader = ModelLoader()
        models = model_loader.load_diffuse_models()

        if env_name not in models:
            print_result("模型加载", False, f"未找到 {env_name} 的模型")
            return None, None, None

        agent, ranker, belief_encoder = models[env_name]

        # 检查模型类型
        print_result("Agent类型", agent.__class__.__name__ == "SAC", f"类型: {agent.__class__.__name__}")
        print_result("Ranker类型", ranker.__class__.__name__ == "GeMS", f"类型: {ranker.__class__.__name__}")
        print_result("Belief类型", belief_encoder.__class__.__name__ == "GRUBelief", f"类型: {belief_encoder.__class__.__name__}")

        # 检查关键参数
        print_result("Agent动作维度", agent.action_dim == 32, f"action_dim={agent.action_dim} (期望32)")
        print_result("Ranker潜在维度", ranker.latent_dim == 32, f"latent_dim={ranker.latent_dim} (期望32)")
        print_result("Belief状态维度", belief_encoder.get_state_dim() == 20, f"state_dim={belief_encoder.get_state_dim()} (期望20)")

        # 检查action bounds
        has_bounds = hasattr(agent, 'action_center') and hasattr(agent, 'action_scale')
        print_result("Action bounds设置", has_bounds,
                    f"center={agent.action_center.mean().item():.2f}, scale={agent.action_scale.mean().item():.2f}" if has_bounds else "未设置")

        # 检查模型是否在eval模式
        agent.eval()
        ranker.eval()
        belief_encoder.eval()
        print_result("模型评估模式", True, "已设置为eval模式")

        return agent, ranker, belief_encoder

    except Exception as e:
        print_result("模型加载", False, f"错误: {e}")
        return None, None, None

def test_environment_params(env_name):
    """测试2: 环境参数验证"""
    print_section(f"测试2: 环境参数验证 - {env_name}")

    try:
        env_factory = EnvironmentFactory()
        config = env_factory.get_env_config(env_name)

        all_match = True
        for param, expected_value in EXPECTED_ENV_PARAMS.items():
            if param == 'diversity_penalty' and env_name == 'diffuse_divpen':
                expected_value = 3.0

            actual_value = config.get(param)
            matches = actual_value == expected_value

            if not matches:
                all_match = False
                print_result(f"参数 {param}", matches,
                           f"实际={actual_value}, 期望={expected_value}")

        if all_match:
            print_result("所有环境参数", True, "与训练配置完全一致")

        # 创建环境测试
        original_cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT_ROOT))
            environment = env_factory.create_environment(env_name)
            print_result("环境创建", True, f"成功创建 {env_name} 环境")
            return environment
        finally:
            os.chdir(original_cwd)

    except Exception as e:
        print_result("环境参数验证", False, f"错误: {e}")
        return None

def test_performance_benchmark(env_name, agent, ranker, belief_encoder, environment, num_episodes=100):
    """测试3: 性能基准测试"""
    print_section(f"测试3: 性能基准测试 - {env_name} ({num_episodes} episodes)")

    if agent is None or ranker is None or belief_encoder is None or environment is None:
        print_result("性能测试", False, "模型或环境未正确加载")
        return

    device = agent.my_device
    returns = []

    print("正在测试性能...")
    with torch.inference_mode():
        for ep in range(num_episodes):
            obs, info = environment.reset()

            # 重置belief encoder的hidden状态
            for module in belief_encoder.beliefs:
                belief_encoder.hidden[module] = torch.zeros(1, 1, belief_encoder.hidden_dim, device=belief_encoder.my_device)

            obs = belief_encoder.forward(obs)

            done = False
            ep_return = 0.0

            while not done:
                latent_action = agent.get_action(obs, sample=False)
                slate = ranker.rank(latent_action)

                next_obs_raw, reward, done, next_info = environment.step(slate)
                obs = belief_encoder.forward(next_obs_raw, done=done)

                ep_return += reward

            returns.append(ep_return)

            # 每10个episode打印一次进度
            if (ep + 1) % 10 == 0:
                print(f"  进度: {ep+1}/{num_episodes} episodes, 当前平均: {np.mean(returns):.2f}")

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    print(f"\n性能统计:")
    print(f"  平均回报: {avg_return:.2f} ± {std_return:.2f}")
    print(f"  范围: [{min_return:.2f}, {max_return:.2f}]")

    # 与训练基准对比
    training_benchmark = TRAINING_BENCHMARKS.get(env_name)
    if training_benchmark is not None:
        diff = avg_return - training_benchmark
        diff_pct = (diff / training_benchmark) * 100

        print(f"\n与训练基准对比:")
        print(f"  训练test_reward: {training_benchmark:.2f}")
        print(f"  当前平均回报:   {avg_return:.2f}")
        print(f"  差距:           {diff:.2f} ({diff_pct:+.1f}%)")

        # 判断是否通过（允许10%的性能下降）
        passed = diff_pct >= -10
        print_result("性能基准", passed,
                    f"性能{'符合' if passed else '低于'}预期（允许-10%以内）")
    else:
        print(f"\n⚠️  警告: 未找到 {env_name} 的训练基准，无法对比")

    return avg_return

def test_data_format(env_name, agent, ranker, belief_encoder, environment):
    """测试4: 数据格式验证"""
    print_section(f"测试4: 数据格式验证 - {env_name}")

    if agent is None or ranker is None or belief_encoder is None or environment is None:
        print_result("数据格式测试", False, "模型或环境未正确加载")
        return

    try:
        # 收集一个episode的数据
        obs, info = environment.reset()

        for module in belief_encoder.beliefs:
            belief_encoder.hidden[module] = torch.zeros(1, 1, belief_encoder.hidden_dim, device=belief_encoder.my_device)

        obs = belief_encoder.forward(obs)

        done = False
        latent_actions = []
        slates = []

        with torch.inference_mode():
            while not done:
                latent_action = agent.get_action(obs, sample=False)
                slate = ranker.rank(latent_action)

                # 保存数据
                latent_actions.append(latent_action.cpu().numpy())
                slates.append(slate.cpu().tolist() if torch.is_tensor(slate) else slate)

                next_obs_raw, reward, done, next_info = environment.step(slate)
                obs = belief_encoder.forward(next_obs_raw, done=done)

        # 验证数据格式
        latent_actions = np.array(latent_actions)
        slates = np.array(slates)

        print_result("Latent action维度", latent_actions.shape[1] == 32,
                    f"shape={latent_actions.shape} (期望 (100, 32))")
        print_result("Slate维度", slates.shape[1] == 10,
                    f"shape={slates.shape} (期望 (100, 10))")
        print_result("Episode长度", len(latent_actions) == 100,
                    f"length={len(latent_actions)} (期望100)")

        # 检查latent action的数值范围
        action_mean = np.mean(latent_actions)
        action_std = np.std(latent_actions)
        action_min = np.min(latent_actions)
        action_max = np.max(latent_actions)

        print(f"\nLatent action统计:")
        print(f"  均值: {action_mean:.4f}")
        print(f"  标准差: {action_std:.4f}")
        print(f"  范围: [{action_min:.4f}, {action_max:.4f}]")

        # 检查是否在合理范围内（-3到3之间，因为action_scale=3.0）
        in_range = action_min >= -10 and action_max <= 10
        print_result("Latent action范围", in_range,
                    "数值在合理范围内" if in_range else "数值超出预期范围")

    except Exception as e:
        print_result("数据格式验证", False, f"错误: {e}")

def test_key_configurations():
    """测试5: 关键配置检查"""
    print_section("测试5: 关键配置检查")

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print_result("CUDA可用性", cuda_available,
                f"CUDA {'可用' if cuda_available else '不可用'}")

    if cuda_available:
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前GPU: {torch.cuda.current_device()}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")

    # 检查关键文件是否存在 - 使用统一路径配置
    sys.path.insert(0, str(PROJECT_ROOT / "config"))
    from paths import get_embeddings_path, OFFLINE_DATASETS_DIR

    embeddings_path = get_embeddings_path("item_embeddings_diffuse.pt")
    print_result("物品embeddings文件", embeddings_path.exists(),
                f"路径: {embeddings_path}")

    # 检查输出目录
    output_dir = OFFLINE_DATASETS_DIR
    print_result("输出目录", True,
                f"路径: {output_dir} {'(已存在)' if output_dir.exists() else '(将创建)'}")

def main():
    """主测试流程"""
    print("\n" + "="*80)
    print("  数据收集前完整测试")
    print("  目标: 5分钟内验证所有配置")
    print("="*80)

    # 测试所有环境
    test_envs = ['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen']

    # 测试5: 关键配置检查（只需要执行一次）
    test_key_configurations()

    # 对每个环境进行测试
    for env_name in test_envs:
        print("\n" + "#"*80)
        print(f"#  开始测试环境: {env_name}")
        print("#"*80)

        # 测试1: 模型加载
        agent, ranker, belief_encoder = test_model_loading(env_name)

        # 测试2: 环境参数
        environment = test_environment_params(env_name)

        # 测试3: 性能基准（100 episodes，约2分钟）
        if agent and ranker and belief_encoder and environment:
            avg_return = test_performance_benchmark(env_name, agent, ranker, belief_encoder, environment, num_episodes=100)

            # 测试4: 数据格式
            test_data_format(env_name, agent, ranker, belief_encoder, environment)
        else:
            print(f"\n⚠️  跳过 {env_name} 的性能和数据格式测试（模型或环境加载失败）")

    # 最终总结
    print("\n" + "="*80)
    print("  测试完成！")
    print("="*80)
    print("\n如果所有测试都通过，可以开始数据收集。")
    print("如果有测试失败，请先修复问题再开始数据收集。")
    print("\n开始数据收集命令:")
    print("  bash offline_data_collection/start_collection.sh")
    print("="*80)

if __name__ == "__main__":
    main()
