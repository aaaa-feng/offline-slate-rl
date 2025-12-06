#!/usr/bin/env python3
"""
验证数据收集代码的性能
对比训练测试代码的性能
"""
import torch
import sys
from pathlib import Path

# 添加GeMS路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_loader import ModelLoader
from environment_factory import EnvironmentFactory

def test_performance(env_name='diffuse_topdown', num_episodes=100):
    """测试模型性能"""
    print(f"测试 {env_name} 环境性能...")
    print(f"测试episodes: {num_episodes}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print("加载模型...")
    model_loader = ModelLoader()
    models = model_loader.load_diffuse_models()
    agent, ranker, belief_encoder = models[env_name]

    # 设置为评估模式
    agent.eval()
    ranker.eval()
    belief_encoder.eval()

    # 2. 创建环境
    print("创建环境...")
    env_factory = EnvironmentFactory()
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
        environment = env_factory.create_environment(env_name)
    finally:
        os.chdir(original_cwd)

    # 3. 测试性能（不重置随机种子）
    print("\n测试1: 不重置随机种子")
    returns_no_reset = []

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

            returns_no_reset.append(ep_return)

    print(f"  平均回报: {sum(returns_no_reset)/len(returns_no_reset):.2f}")
    print(f"  标准差: {torch.tensor(returns_no_reset).std().item():.2f}")
    print(f"  最小值: {min(returns_no_reset):.2f}")
    print(f"  最大值: {max(returns_no_reset):.2f}")

    # 4. 测试性能（重置随机种子）
    print("\n测试2: 重置随机种子（模拟训练测试）")
    environment.reset_random_state()  # 重置随机种子
    returns_with_reset = []

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

            returns_with_reset.append(ep_return)

    print(f"  平均回报: {sum(returns_with_reset)/len(returns_with_reset):.2f}")
    print(f"  标准差: {torch.tensor(returns_with_reset).std().item():.2f}")
    print(f"  最小值: {min(returns_with_reset):.2f}")
    print(f"  最大值: {max(returns_with_reset):.2f}")

    # 5. 对比
    print("\n="*60)
    print("对比结果:")
    print(f"  不重置随机种子: {sum(returns_no_reset)/len(returns_no_reset):.2f}")
    print(f"  重置随机种子:   {sum(returns_with_reset)/len(returns_with_reset):.2f}")
    print(f"  差距:           {sum(returns_with_reset)/len(returns_with_reset) - sum(returns_no_reset)/len(returns_no_reset):.2f}")
    print(f"  训练日志test_reward: 317.75")
    print("="*60)

if __name__ == "__main__":
    test_performance('diffuse_topdown', num_episodes=100)
