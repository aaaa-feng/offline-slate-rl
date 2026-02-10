import torch
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 路径设置
project_root = Path("/data/liyuefeng/offline-slate-rl")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src/data_collection/offline_data_collection"))

from core.model_loader import ModelLoader
from envs.RecSim.recsim_env import create_environment

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始动态交互测试 (Device: {device})")

    # 1. 加载环境
    env_name = "diffuse_mix"
    # 注意：这里我们使用默认参数，看 Boredom=4 是否能制裁 Zero-Action
    env = create_environment(env_name, seed=42)
    print(f"✅ 环境 {env_name} 加载完成")

    # 2. 加载模型组件
    loader = ModelLoader()
    
    # 2.1 加载 Online VAE (从指定路径)
    # 注意：需要你确认 ModelLoader 是否能加载这个特定的 checkponit，或者我们手动加载
    # 这里为了简便，我们假设 ModelLoader 可以加载 expert 模型（即 Offline VAE + SAC）
    # 然后我们手动替换/对比 VAE
    
    print("📦 加载 SAC + Offline VAE 模型...")
    # 指向你指定的 expert 模型路径
    model_path = "/data/liyuefeng/offline-slate-rl/src/data_collection/offline_data_collection/models/expert/sac_gems_models/diffuse_mix/SAC_GeMS_diffuse_mix_expert_beta1.0_click0.5_div1.0_gamma0.8_dim32_seed58407201.ckpt"
    
    agent, ranker, belief_encoder = loader.load_agent(
        env_name=env_name,
        checkpoint_path=model_path
    )
    
    # 3. 定义测试循环
    def run_episode(policy_type="sac"):
        obs, _ = env.reset()
        if belief_encoder:
            # 重置 belief
            for module in belief_encoder.beliefs:
                belief_encoder.hidden[module] = torch.zeros(1, 1, belief_encoder.hidden_dim, device=device)
            obs = belief_encoder.forward(obs)
            
        total_reward = 0
        rewards = []
        action_norms = []
        
        done = False
        step = 0
        while not done and step < 50: # 测试 50 步
            # 获取 Current Belief
            current_belief = None
            if isinstance(obs, dict) and 'actor' in obs:
                current_belief = obs['actor']
            elif isinstance(obs, torch.Tensor):
                current_belief = obs
            
            # 决策
            if policy_type == "sac":
                # SAC Agent 输出
                z = agent.get_action(current_belief, sample=False)
            elif policy_type == "zero":
                # Zero Action
                z = torch.zeros(1, 32).to(device)
            elif policy_type == "random":
                # Random Action
                z = torch.randn(1, 32).to(device)

            # 记录 z 的模长
            action_norms.append(torch.norm(z).item())

            # 解码 Slate
            slate = ranker.rank(z)
            
            # 环境交互
            next_obs_raw, reward, done, _ = env.step(slate)
            
            # 更新 Belief
            if belief_encoder:
                next_obs = belief_encoder.forward(next_obs_raw, done=done)
            else:
                next_obs = next_obs_raw
                
            obs = next_obs
            total_reward += reward
            rewards.append(reward.item())
            step += 1
            
        return total_reward, rewards, action_norms

    # 4. 开始对比测试
    num_episodes = 5
    results = {
        "sac": {"rewards": [], "norms": []},
        "zero": {"rewards": [], "norms": []},
        # "random": {"rewards": [], "norms": []} 
    }

    print("\n🏁 开始运行 Episode 对比...")
    
    for i in range(num_episodes):
        # 设置相同的 seed 以保证用户一致
        env.seed(100 + i) 
        
        # Test SAC
        r_sac, trace_sac, norm_sac = run_episode("sac")
        results["sac"]["rewards"].append(r_sac)
        results["sac"]["norms"].extend(norm_sac)
        
        # Test Zero (Same user)
        env.seed(100 + i) # Reset same user
        r_zero, trace_zero, norm_zero = run_episode("zero")
        results["zero"]["rewards"].append(r_zero)
        
        print(f"Episode {i+1}: SAC Reward = {r_sac:.2f}, Zero-Action Reward = {r_zero:.2f}")

    # 5. 统计分析
    avg_sac = np.mean(results["sac"]["rewards"])
    avg_zero = np.mean(results["zero"]["rewards"])
    avg_sac_norm = np.mean(results["sac"]["norms"])

    print("\n================ 最终测试报告 ================")
    print(f"SAC Agent 平均回报: {avg_sac:.2f}")
    print(f"Zero-Action 平均回报: {avg_zero:.2f}")
    print(f"SAC 输出动作平均模长: {avg_sac_norm:.4f}")
    
    if avg_sac_norm < 0.1:
        print("⚠️ 警告: SAC Agent 输出了接近 0 的动作，它可能坍缩成了 Zero-Action！")
    else:
        print("✅ SAC Agent 输出了非零动作，它在尝试由于 Zero-Action 不同的策略。")

    if avg_sac > avg_zero:
        print("🎉 SAC 战胜了 Baseline！")
    else:
        print("❄️ SAC 未能战胜 Baseline (万金油策略太强了)。")
        
    print("============================================")

if __name__ == "__main__":
    run_test()