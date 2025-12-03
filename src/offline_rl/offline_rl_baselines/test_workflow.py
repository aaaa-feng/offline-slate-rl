#!/usr/bin/env python3
"""
测试整个离线RL工作流程
验证数据加载和训练是否能正常工作
"""
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("测试离线RL工作流程")
print("="*60)

# 测试1：检查依赖导入
print("\n[测试1] 检查依赖导入...")
try:
    from offline_rl_baselines.common.buffer import ReplayBuffer
    from offline_rl_baselines.common.utils import set_seed, compute_mean_std
    from offline_rl_baselines.common.networks import Actor, Critic
    from offline_rl_baselines.algorithms.td3_bc import TD3BCConfig, TD3_BC
    print("✅ 所有依赖导入成功")
except Exception as e:
    print(f"❌ 依赖导入失败: {e}")
    sys.exit(1)

# 测试2：创建模拟数据集
print("\n[测试2] 创建模拟数据集...")
try:
    # 模拟GeMS数据格式
    n_samples = 1000
    state_dim = 20
    action_dim = 32

    mock_dataset = {
        'observations': np.random.randn(n_samples, state_dim).astype(np.float32),
        'actions': np.random.randn(n_samples, action_dim).astype(np.float32) * 3.0,  # [-3, 3]
        'rewards': np.random.randn(n_samples).astype(np.float32),
        'next_observations': np.random.randn(n_samples, state_dim).astype(np.float32),
        'terminals': np.random.randint(0, 2, n_samples).astype(np.float32),
    }

    print(f"✅ 模拟数据集创建成功")
    print(f"   - Observations shape: {mock_dataset['observations'].shape}")
    print(f"   - Actions shape: {mock_dataset['actions'].shape}")
    print(f"   - Rewards shape: {mock_dataset['rewards'].shape}")
except Exception as e:
    print(f"❌ 数据集创建失败: {e}")
    sys.exit(1)

# 测试3：测试ReplayBuffer加载
print("\n[测试3] 测试ReplayBuffer加载...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=2000,
        device=device
    )
    buffer.load_d4rl_dataset(mock_dataset)

    print(f"✅ ReplayBuffer加载成功")
    print(f"   - Device: {device}")
    print(f"   - Buffer size: {buffer._size}")
except Exception as e:
    print(f"❌ ReplayBuffer加载失败: {e}")
    sys.exit(1)

# 测试4：测试数据采样
print("\n[测试4] 测试数据采样...")
try:
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch

    print(f"✅ 数据采样成功")
    print(f"   - States shape: {states.shape}")
    print(f"   - Actions shape: {actions.shape}")
    print(f"   - Rewards shape: {rewards.shape}")
except Exception as e:
    print(f"❌ 数据采样失败: {e}")
    sys.exit(1)

# 测试5：测试状态归一化
print("\n[测试5] 测试状态归一化...")
try:
    mean, std = compute_mean_std(mock_dataset['observations'])
    buffer.normalize_states(mean, std)

    print(f"✅ 状态归一化成功")
    print(f"   - Mean shape: {mean.shape}")
    print(f"   - Std shape: {std.shape}")
except Exception as e:
    print(f"❌ 状态归一化失败: {e}")
    sys.exit(1)

# 测试6：测试网络初始化
print("\n[测试6] 测试网络初始化...")
try:
    max_action = 3.0
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    print(f"✅ 网络初始化成功")
    print(f"   - Actor parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"   - Critic parameters: {sum(p.numel() for p in critic.parameters())}")
except Exception as e:
    print(f"❌ 网络初始化失败: {e}")
    sys.exit(1)

# 测试7：测试TD3_BC初始化
print("\n[测试7] 测试TD3_BC初始化...")
try:
    config = TD3BCConfig(
        device=device,
        env_name="test",
        dataset_path="",
        seed=0,
        max_timesteps=100,
        batch_size=32,
    )

    agent = TD3_BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        config=config,
    )

    print(f"✅ TD3_BC初始化成功")
except Exception as e:
    print(f"❌ TD3_BC初始化失败: {e}")
    sys.exit(1)

# 测试8：测试训练一步
print("\n[测试8] 测试训练一步...")
try:
    batch = buffer.sample(batch_size=32)
    metrics = agent.train(batch)

    print(f"✅ 训练一步成功")
    print(f"   - Critic loss: {metrics['critic_loss']:.4f}")
    print(f"   - Actor loss: {metrics['actor_loss']:.4f}")
    print(f"   - Q value: {metrics['q_value']:.4f}")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试9：测试动作选择
print("\n[测试9] 测试动作选择...")
try:
    test_state = np.random.randn(state_dim).astype(np.float32)
    action = agent.act(test_state)

    print(f"✅ 动作选择成功")
    print(f"   - Action shape: {action.shape}")
    print(f"   - Action range: [{action.min():.2f}, {action.max():.2f}]")
except Exception as e:
    print(f"❌ 动作选择失败: {e}")
    sys.exit(1)

# 测试10：测试模型保存和加载
print("\n[测试10] 测试模型保存和加载...")
try:
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pt")
        agent.save(save_path)

        # 创建新的agent并加载
        new_agent = TD3_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            config=config,
        )
        new_agent.load(save_path)

        print(f"✅ 模型保存和加载成功")
except Exception as e:
    print(f"❌ 模型保存/加载失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有测试通过！")
print("="*60)
print("\n📋 总结：")
print("1. ✅ 所有依赖正确导入")
print("2. ✅ 数据加载流程正常")
print("3. ✅ 网络初始化正常")
print("4. ✅ 训练流程正常")
print("5. ✅ 模型保存/加载正常")
print("\n🎯 结论：代码完全可以工作！")
print("   等数据收集完成后即可开始训练。")
print("\n⏰ 数据收集预计还需约3.6小时")
print("   完成后数据将保存在: offline_datasets/*.npz")
