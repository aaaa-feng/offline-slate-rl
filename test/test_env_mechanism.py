#!/usr/bin/env python
"""
测试脚本：分析 Zero-Action 为什么在不同用户上表现稳定

核心问题：
- 每个 episode 会重置用户兴趣向量
- Zero-Action (z=0) 总是推荐相同的 slate
- 但不同用户 + 相同推荐 应该产生不同的点击行为
- 为什么最终 reward 却很稳定 (~232 分)?

假设：
A. 用户兴趣向量虽然随机，但分布集中，导致某些物品对所有用户都有高相关性
B. 物品 embedding 中存在"万金油"物品，与任何用户都匹配
C. VAE 的 z=0 恰好解码出这些万金油物品

测试内容：
1. 分析用户兴趣向量的分布特征
2. 分析物品 embedding 的分布特征
3. 测试 z=0 解码出的 slate
4. 计算该 slate 与不同用户的相关性分布
5. 对比 online VAE vs offline VAE
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "config"))

from paths import get_embeddings_path


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def analyze_user_embedding_distribution(num_samples: int = 1000, seed: int = 58407201):
    """分析用户兴趣向量的分布特征"""
    print_section("1. 用户兴趣向量分布分析")

    device = torch.device("cpu")
    rd_gen = torch.Generator(device=device)
    rd_gen.manual_seed(seed)

    num_topics = 10
    topic_size = 2

    # 收集多个用户的 embedding
    user_embeddings = []
    topic_distributions = []

    for _ in range(num_samples):
        # 复现 simulators.py 中的用户生成逻辑
        user_embedd = torch.abs(torch.clamp(
            0.4 * torch.randn(num_topics, topic_size, device=device, generator=rd_gen),
            -1, 1
        ))

        user_comp_dist = torch.rand(num_topics, device=device, generator=rd_gen).pow(3)
        user_comp_dist /= torch.sum(user_comp_dist)

        user_embedd *= user_comp_dist.unsqueeze(1)

        topic_norm = torch.linalg.norm(user_embedd, dim=1)
        user_embedd = user_embedd.flatten() / torch.sum(topic_norm)

        user_embeddings.append(user_embedd)
        topic_distributions.append(user_comp_dist)

    user_embeddings = torch.stack(user_embeddings)  # (num_samples, 20)
    topic_distributions = torch.stack(topic_distributions)  # (num_samples, 10)

    print(f"\n用户 embedding 形状: {user_embeddings.shape}")
    print(f"用户 embedding 范围: [{user_embeddings.min():.4f}, {user_embeddings.max():.4f}]")
    print(f"用户 embedding 均值: {user_embeddings.mean():.4f}")
    print(f"用户 embedding 标准差: {user_embeddings.std():.4f}")

    # 分析主题分布
    print(f"\n主题分布 (pow(3) 后):")
    print(f"  均值: {topic_distributions.mean(dim=0).numpy()}")
    print(f"  标准差: {topic_distributions.std(dim=0).numpy()}")

    # 计算用户之间的相似度
    user_similarity = torch.matmul(user_embeddings, user_embeddings.t())
    mask = ~torch.eye(num_samples, dtype=torch.bool)
    pairwise_sim = user_similarity[mask]

    print(f"\n用户间相似度:")
    print(f"  均值: {pairwise_sim.mean():.4f}")
    print(f"  标准差: {pairwise_sim.std():.4f}")
    print(f"  最小值: {pairwise_sim.min():.4f}")
    print(f"  最大值: {pairwise_sim.max():.4f}")

    return user_embeddings


def analyze_item_embedding_distribution():
    """分析物品 embedding 的分布特征"""
    print_section("2. 物品 Embedding 分布分析")

    # 加载物品 embedding
    diffuse_path = get_embeddings_path("item_embeddings_diffuse.pt")
    focused_path = get_embeddings_path("item_embeddings_focused.pt")

    print(f"\n加载 diffuse embedding: {diffuse_path}")
    item_embedd_diffuse = torch.load(str(diffuse_path), map_location="cpu")

    print(f"加载 focused embedding: {focused_path}")
    item_embedd_focused = torch.load(str(focused_path), map_location="cpu")

    for name, item_embedd in [("diffuse", item_embedd_diffuse), ("focused", item_embedd_focused)]:
        print(f"\n--- {name} embedding ---")
        print(f"形状: {item_embedd.shape}")
        print(f"范围: [{item_embedd.min():.4f}, {item_embedd.max():.4f}]")
        print(f"均值: {item_embedd.mean():.4f}")
        print(f"标准差: {item_embedd.std():.4f}")

        # 计算每个物品的 L2 范数
        item_norms = torch.linalg.norm(item_embedd, dim=1)
        print(f"物品范数 - 均值: {item_norms.mean():.4f}, 标准差: {item_norms.std():.4f}")
        print(f"物品范数 - 最小: {item_norms.min():.4f}, 最大: {item_norms.max():.4f}")

        # 找出范数最大的物品
        top_norm_items = torch.topk(item_norms, k=20)
        print(f"范数最大的 20 个物品: {top_norm_items.indices.tolist()}")
        print(f"对应范数: {top_norm_items.values.tolist()[:10]}...")

    return item_embedd_diffuse, item_embedd_focused


def load_gems_model(checkpoint_path: str, device: str = "cpu"):
    """加载 GeMS 模型"""
    from rankers.gems.rankers import GeMS
    from rankers.gems.item_embeddings import ItemEmbeddings

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从 checkpoint 中提取超参数
    hparams = checkpoint.get('hyper_parameters', {})

    # 创建 item embeddings
    item_embedd_path = get_embeddings_path("item_embeddings_diffuse.pt")
    item_embedd_data = torch.load(str(item_embedd_path), map_location=device)

    item_embeddings = ItemEmbeddings(
        num_items=hparams.get('num_items', 1000),
        item_embedd_dim=hparams.get('item_embedd_dim', 20),
        device=torch.device(device)
    )
    item_embeddings.embedd.weight.data.copy_(item_embedd_data)

    # 创建 GeMS 模型
    model = GeMS(
        item_embeddings=item_embeddings,
        item_embedd_dim=hparams.get('item_embedd_dim', 20),
        device=torch.device(device),
        rec_size=hparams.get('rec_size', 10),
        latent_dim=hparams.get('latent_dim', 32),
        lambda_click=hparams.get('lambda_click', 0.5),
        lambda_KL=hparams.get('lambda_KL', 1.0),
        lambda_prior=hparams.get('lambda_prior', 0.0),
        ranker_lr=hparams.get('ranker_lr', 3e-3),
        fixed_embedds=hparams.get('fixed_embedds', 'mf_fixed'),
        ranker_sample=hparams.get('ranker_sample', False),
        hidden_layers_infer=hparams.get('hidden_layers_infer', [512, 256]),
        hidden_layers_decoder=hparams.get('hidden_layers_decoder', [256, 512]),
    )

    # 加载权重
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def analyze_zero_action_slate(online_vae_path: str, offline_vae_path: str):
    """分析 z=0 解码出的 slate"""
    print_section("3. Zero-Action Slate 分析")

    device = "cpu"

    # 加载 online VAE
    print(f"\n加载 Online VAE: {online_vae_path}")
    try:
        online_model = load_gems_model(online_vae_path, device)
        z_zero = torch.zeros(online_model.latent_dim, device=device)
        online_slate = online_model.rank(z_zero)
        print(f"Online VAE z=0 解码的 slate: {online_slate.tolist()}")
    except Exception as e:
        print(f"加载 Online VAE 失败: {e}")
        online_slate = None

    # 加载 offline VAE
    print(f"\n加载 Offline VAE: {offline_vae_path}")
    try:
        offline_model = load_gems_model(offline_vae_path, device)
        z_zero = torch.zeros(offline_model.latent_dim, device=device)
        offline_slate = offline_model.rank(z_zero)
        print(f"Offline VAE z=0 解码的 slate: {offline_slate.tolist()}")
    except Exception as e:
        print(f"加载 Offline VAE 失败: {e}")
        offline_slate = None

    return online_slate, offline_slate


def analyze_slate_user_relevance(slate: torch.Tensor, item_embedd: torch.Tensor,
                                  num_users: int = 1000, seed: int = 58407201):
    """分析给定 slate 与不同用户的相关性分布"""
    print_section("4. Slate-用户相关性分析")

    if slate is None:
        print("Slate 为空，跳过分析")
        return

    device = torch.device("cpu")
    rd_gen = torch.Generator(device=device)
    rd_gen.manual_seed(seed)

    num_topics = 10
    topic_size = 2
    offset = 0.15
    slope = 20

    # 获取 slate 中物品的 embedding
    slate_embedd = item_embedd[slate]  # (rec_size, item_embedd_dim)

    # 生成多个用户并计算相关性
    all_relevances = []
    all_click_probs = []

    for _ in range(num_users):
        # 生成用户 embedding
        user_embedd = torch.abs(torch.clamp(
            0.4 * torch.randn(num_topics, topic_size, device=device, generator=rd_gen),
            -1, 1
        ))
        user_comp_dist = torch.rand(num_topics, device=device, generator=rd_gen).pow(3)
        user_comp_dist /= torch.sum(user_comp_dist)
        user_embedd *= user_comp_dist.unsqueeze(1)
        topic_norm = torch.linalg.norm(user_embedd, dim=1)
        user_embedd = user_embedd.flatten() / torch.sum(topic_norm)

        # 计算相关性
        max_score = torch.max(torch.linalg.norm(item_embedd, dim=1))
        score = torch.matmul(slate_embedd, user_embedd)
        norm_score = score / max_score
        relevances = 1 / (1 + torch.exp(-(norm_score - offset) * slope))

        all_relevances.append(relevances)
        all_click_probs.append(relevances.sum().item())  # 简化的期望点击数

    all_relevances = torch.stack(all_relevances)  # (num_users, rec_size)
    all_click_probs = np.array(all_click_probs)

    print(f"\nSlate: {slate.tolist()}")
    print(f"\n每个位置的相关性统计 (across {num_users} users):")
    for i in range(len(slate)):
        rel_i = all_relevances[:, i]
        print(f"  位置 {i} (物品 {slate[i].item()}): "
              f"均值={rel_i.mean():.4f}, 标准差={rel_i.std():.4f}, "
              f"范围=[{rel_i.min():.4f}, {rel_i.max():.4f}]")

    print(f"\n期望点击数统计:")
    print(f"  均值: {all_click_probs.mean():.4f}")
    print(f"  标准差: {all_click_probs.std():.4f}")
    print(f"  范围: [{all_click_probs.min():.4f}, {all_click_probs.max():.4f}]")
    print(f"  变异系数 (CV): {all_click_probs.std() / all_click_probs.mean():.4f}")

    return all_relevances, all_click_probs


def run_episode_simulation(num_episodes: int = 50, seed: int = 58407201):
    """运行完整的 episode 模拟，验证 Zero-Action 的表现"""
    print_section("5. Episode 模拟测试")

    # 这里需要完整的环境和模型，先打印说明
    print("\n此测试需要完整的环境设置，将在后续实现...")
    print("预期测试内容:")
    print("  1. 使用 Online VAE 的 z=0 策略运行 50 episodes")
    print("  2. 使用 Offline VAE 的 z=0 策略运行 50 episodes")
    print("  3. 记录每个 episode 的 reward 分布")
    print("  4. 分析 reward 的稳定性来源")


def main():
    print("\n" + "=" * 60)
    print("  Zero-Action 稳定性分析测试")
    print("=" * 60)

    # 1. 分析用户 embedding 分布
    user_embeddings = analyze_user_embedding_distribution(num_samples=1000)

    # 2. 分析物品 embedding 分布
    item_embedd_diffuse, item_embedd_focused = analyze_item_embedding_distribution()

    # 3. 分析 z=0 解码的 slate
    online_vae_path = str(project_root / "checkpoints/gems/online/GeMS_diffuse_mix_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201.ckpt")
    offline_vae_path = str(project_root / "checkpoints/gems/offline/GeMS_diffuse_mix_expert_latent32_beta1.0_click0.5_seed58407201.ckpt")

    online_slate, offline_slate = analyze_zero_action_slate(online_vae_path, offline_vae_path)

    # 4. 分析 slate 与用户的相关性
    if online_slate is not None:
        print("\n--- Online VAE Slate 分析 ---")
        analyze_slate_user_relevance(online_slate, item_embedd_diffuse)

    if offline_slate is not None:
        print("\n--- Offline VAE Slate 分析 ---")
        analyze_slate_user_relevance(offline_slate, item_embedd_diffuse)

    # 5. Episode 模拟
    run_episode_simulation(num_episodes=50)

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
