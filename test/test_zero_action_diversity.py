"""
测试 Zero-Action 生成的 Slate 多样性

目的：验证 Zero-Action 是否总是生成相同的 Slate
如果是，说明环境中存在"无敌热门商品组合"

作者: Claude Code
日期: 2026-01-12
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.offline.eval_env import OfflineEvalEnv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)


def main():
    """主测试流程"""

    print("\n" + "=" * 80)
    print("=== 测试 Zero-Action 生成的 Slate 多样性 ===")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 初始化环境
    # ========================================================================
    print("初始化评估环境...")
    eval_env = OfflineEvalEnv(
        env_name="diffuse_mix",
        dataset_quality="expert",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=58407201,
        verbose=False
    )
    print("✅ 环境初始化完成")
    print()

    # ========================================================================
    # 2. 测试 Zero-Action 的 Slate 生成
    # ========================================================================
    print("=" * 80)
    print("测试 1: 同一个 Zero-Action 生成的 Slate 是否相同？")
    print("=" * 80)

    zero_action = torch.zeros(1, 32).to(eval_env.device)

    slates = []
    print("\n生成 5 次 Slate（使用相同的 Zero-Action）：")
    print("-" * 80)

    for i in range(5):
        with torch.no_grad():
            slate = eval_env.ranker.rank(zero_action).squeeze(0).cpu().numpy()

        slates.append(slate)
        print(f"Slate {i+1}: {slate}")

    # 检查是否完全相同
    first_slate = slates[0]
    all_same = all(np.array_equal(first_slate, s) for s in slates[1:])

    print("-" * 80)
    if all_same:
        print("🔴 结论：Zero-Action 总是生成**完全相同**的 Slate")
        print()
        print("   这说明：")
        print("   1. GeMS Decoder 是确定性的（没有随机采样）")
        print("   2. 存在一个'最优商品组合'")
        print("   3. 环境对这个组合的奖励很高（232分）")
        print()
        print("   影响：")
        print("   - 用户兴趣迁移可能太慢")
        print("   - 或者这些商品太'万能'，适合所有用户")
        print("   - RL 很难找到比这更好的策略")
    else:
        print("🟢 结论：Zero-Action 生成的 Slate 是**变化的**")
        print()
        print("   这说明：")
        print("   - Ranker 内部有随机性")
        print("   - 或者 Action 处理有噪声")

    print("=" * 80)
    print()

    # ========================================================================
    # 3. 测试不同 Episode 中 Zero-Action 的表现
    # ========================================================================
    print("=" * 80)
    print("测试 2: 在不同 Episode 中，Zero-Action 的表现是否稳定？")
    print("=" * 80)
    print()

    class ZeroAgent:
        def act(self, obs, deterministic=True):
            return np.zeros(32)
        def reset_hidden(self):
            pass

    zero_agent = ZeroAgent()

    print("运行 10 个 Episode，记录每个 Episode 的 Reward：")
    print("-" * 80)

    episode_rewards = []
    for i in range(10):
        obs = eval_env.env.reset()
        zero_agent.reset_hidden()

        episode_reward = 0.0
        done = False

        while not done:
            latent_action = zero_agent.act(obs)
            latent_action_tensor = torch.FloatTensor(latent_action).unsqueeze(0).to(eval_env.device)

            with torch.no_grad():
                slate = eval_env.ranker.rank(latent_action_tensor).squeeze(0)

            obs, reward, done, info = eval_env.env.step(slate)

            if isinstance(reward, torch.Tensor):
                reward = reward.item()

            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("-" * 80)
    print(f"平均 Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print()

    if std_reward < 20:
        print("🔴 结论：Zero-Action 的表现**非常稳定**（标准差 < 20）")
        print()
        print("   这说明：")
        print("   - 不同用户对这个 Slate 的反应都很好")
        print("   - 环境的随机性很小")
        print("   - RL 很难通过'个性化'来提升")
    else:
        print("🟢 结论：Zero-Action 的表现**有波动**（标准差 >= 20）")
        print()
        print("   这说明：")
        print("   - 不同用户的偏好有差异")
        print("   - RL 有机会通过'个性化'来提升")

    print("=" * 80)
    print()

    # ========================================================================
    # 4. 最终建议
    # ========================================================================
    print("=" * 80)
    print("=== 最终建议 ===")
    print("=" * 80)
    print()

    if all_same and std_reward < 20:
        print("🔴 问题确认：Expert 数据集 + 当前环境设置不适合展示 RL 的价值")
        print()
        print("建议：")
        print("  1. 立即切换到 Medium 数据集")
        print("  2. 或者调整环境参数（增加 boredom_threshold，增强兴趣迁移）")
        print("  3. 不要在 Expert 上继续浪费时间")
    else:
        print("🟢 环境有一定的随机性和个性化空间")
        print()
        print("建议：")
        print("  1. 可以继续在 Expert 上优化")
        print("  2. 但仍建议测试 Medium 数据集，对比效果")

    print("=" * 80)


if __name__ == "__main__":
    main()
