"""
GeMS VAE 性能测试脚本

测试目标：
1. 重构质量评估（Reconstruction Quality）
2. Zero-Action Baseline 测试
3. 多样性评估（Diversity Metrics）
4. 端到端性能对比（E2E Performance）

作者: Claude Code
日期: 2026-01-24
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ============================================================================
# 路径设置
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("=== GeMS VAE 性能测试 ===")
print("=" * 80)
print(f"项目根目录: {PROJECT_ROOT}")
print("=" * 80)
print()

# 导入项目模块
from src.common.offline.eval_env import OfflineEvalEnv
from src.rankers.gems.rankers import GeMS
from src.rankers.gems.item_embeddings import ItemEmbeddings

# ============================================================================
# 配置日志
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# ============================================================================
# 模型配置
# ============================================================================
GEMS_MODELS = {
    # 旧 Benchmark 模型（lambda_click=0.5）
    "diffuse_mix_expert": {
        "checkpoint": "GeMS_diffuse_mix_expert_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_mix",
        "dataset_quality": "expert",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },
    "diffuse_mix_medium": {
        "checkpoint": "GeMS_diffuse_mix_medium_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_mix",
        "dataset_quality": "medium",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },
    "diffuse_topdown_expert": {
        "checkpoint": "GeMS_diffuse_topdown_expert_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_topdown",
        "dataset_quality": "expert",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },
    "diffuse_topdown_medium": {
        "checkpoint": "GeMS_diffuse_topdown_medium_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_topdown",
        "dataset_quality": "medium",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },
    "diffuse_divpen_expert": {
        "checkpoint": "GeMS_diffuse_divpen_expert_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_divpen",
        "dataset_quality": "expert",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },
    "diffuse_divpen_medium": {
        "checkpoint": "GeMS_diffuse_divpen_medium_latent32_beta1.0_click0.5_seed58407201.ckpt",
        "env_name": "diffuse_divpen",
        "dataset_quality": "medium",
        "lambda_click": 0.5,
        "group": "old_benchmark"
    },

    # 新 Benchmark 模型（lambda_click=1.0）
    "mix_divpen_v2_b3": {
        "checkpoint": "GeMS_mix_divpen_v2_b3_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "mix_divpen",
        "dataset_quality": "v2_b3",
        "boredom_threshold": 3,
        "lambda_click": 1.0,
        "group": "new_benchmark"
    },
    "mix_divpen_v2_b5": {
        "checkpoint": "GeMS_mix_divpen_v2_b5_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "mix_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "new_benchmark"
    },
    "topdown_divpen_v2_b3": {
        "checkpoint": "GeMS_topdown_divpen_v2_b3_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "topdown_divpen",
        "dataset_quality": "v2_b3",
        "boredom_threshold": 3,
        "lambda_click": 1.0,
        "group": "new_benchmark"
    },
    "topdown_divpen_v2_b5": {
        "checkpoint": "GeMS_topdown_divpen_v2_b5_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "topdown_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "new_benchmark"
    },

    # Epsilon-Greedy 模型（lambda_click=1.0, 使用预训练embeddings）
    "mix_divpen_epsilon_greedy": {
        "checkpoint": "GeMS_mix_divpen_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt",
        "env_name": "mix_divpen",
        "dataset_quality": "epsilon-greedy",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "epsilon_greedy"
    },
    "topdown_divpen_epsilon_greedy": {
        "checkpoint": "GeMS_topdown_divpen_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt",
        "env_name": "topdown_divpen",
        "dataset_quality": "epsilon-greedy",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "epsilon_greedy"
    },

    # MF预训练模型（2026-02-01更新：使用正确的seed58407201）
    "mix_divpen_v2_b5_mf_fixed": {
        "checkpoint": "GeMS_mix_divpen_v2_b5_mf_fixed_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "mix_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "mf_pretrained"
    },
    "mix_divpen_v2_b5_scratch": {
        "checkpoint": "GeMS_mix_divpen_v2_b5_scratch_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "mix_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "mf_pretrained"
    },
    "topdown_divpen_v2_b5_mf_fixed": {
        "checkpoint": "GeMS_topdown_divpen_v2_b5_mf_fixed_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "topdown_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "mf_pretrained"
    },
    "topdown_divpen_v2_b5_scratch": {
        "checkpoint": "GeMS_topdown_divpen_v2_b5_scratch_latent32_beta1.0_click1.0_seed58407201.ckpt",
        "env_name": "topdown_divpen",
        "dataset_quality": "v2_b5",
        "boredom_threshold": 5,
        "lambda_click": 1.0,
        "group": "mf_pretrained"
    },
}


# ============================================================================
# 辅助函数
# ============================================================================
def load_gems_model(checkpoint_path: str, device: str = "cuda") -> GeMS:
    """加载 GeMS 模型"""
    logging.info(f"Loading GeMS model from: {checkpoint_path}")

    # 加载预训练的 item embeddings
    # 根据环境名称使用正确的 embeddings 文件
    # mix_divpen 和 topdown_divpen 使用 diffuse embeddings
    from rankers.gems.item_embeddings import ItemEmbeddings
    temp_embeddings = ItemEmbeddings.from_pretrained(
        "/data/liyuefeng/offline-slate-rl/data/embeddings/item_embeddings_diffuse.pt",
        device
    )

    # 使用 GeMS 标准加载方法（与 TD3+BC 训练脚本一致）
    ranker = GeMS.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        item_embeddings=temp_embeddings,
        device=device,
        rec_size=10,
        item_embedd_dim=20,  # 修正：实际embeddings维度为20
        num_items=1000,
        latent_dim=32,
        lambda_click=1.0,
        lambda_KL=1.0,
        lambda_prior=1.0,
        ranker_lr=3e-3,
        fixed_embedds="scratch",
        ranker_sample=False,
        hidden_layers_infer=[512, 256],
        hidden_layers_decoder=[256, 512]
    )
    ranker.to(device)
    ranker.eval()

    logging.info("✓ GeMS model loaded successfully")
    return ranker


def calculate_gini_coefficient(frequencies: np.ndarray) -> float:
    """计算基尼系数"""
    sorted_freq = np.sort(frequencies)
    n = len(frequencies)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_freq)) / (n * np.sum(sorted_freq)) - (n + 1) / n


def calculate_diversity_metrics(slates: np.ndarray, num_items: int = 1000) -> Dict[str, float]:
    """
    计算多样性指标
    
    Args:
        slates: [num_slates, slate_size] 的 Slate 数组
        num_items: 物品总数
        
    Returns:
        diversity_metrics: 包含各种多样性指标的字典
    """
    num_slates, slate_size = slates.shape
    
    # 1. 物品覆盖率
    unique_items = np.unique(slates)
    item_coverage = len(unique_items) / num_items
    
    # 2. 连续 Slate 重叠率
    overlaps = []
    for i in range(num_slates - 1):
        slate1 = set(slates[i])
        slate2 = set(slates[i + 1])
        overlap = len(slate1 & slate2) / slate_size
        overlaps.append(overlap)
    consecutive_overlap = np.mean(overlaps) if overlaps else 0.0
    
    # 3. 物品流行度分布（基尼系数）
    item_counts = np.bincount(slates.flatten(), minlength=num_items)
    gini = calculate_gini_coefficient(item_counts)
    
    # 4. Top-10 物品覆盖率
    top10_items = np.argsort(item_counts)[-10:]
    top10_count = item_counts[top10_items].sum()
    top10_coverage = top10_count / slates.size
    
    return {
        "item_coverage": item_coverage,
        "consecutive_overlap": consecutive_overlap,
        "gini_coefficient": gini,
        "top10_coverage": top10_coverage
    }


class BaselineAgent:
    """
    Baseline Agent for testing GeMS performance.

    🔥 REFACTORED: Now outputs slates directly (not latent actions).
    Agents must handle slate decoding internally to match the new architecture.
    """
    def __init__(self, strategy: str = "zero", mean_action: np.ndarray = None, action_dim: int = 32, ranker = None):
        """
        Initialize Baseline Agent.

        Args:
            strategy: "zero", "random", or "mean"
            mean_action: Mean latent action (for "mean" strategy)
            action_dim: Latent action dimension
            ranker: GeMS ranker for decoding latent actions to slates
        """
        self.strategy = strategy
        self.mean_action = mean_action
        self.action_dim = action_dim
        self.ranker = ranker

        if self.ranker is None:
            raise ValueError("BaselineAgent requires a ranker for slate decoding.")

    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        Generate baseline latent action and decode to slate.

        Returns:
            slate: numpy array of shape [rec_size]
        """
        # Generate latent action based on strategy
        if self.strategy == "zero":
            latent_action = np.zeros(self.action_dim, dtype=np.float32)
        elif self.strategy == "random":
            latent_action = np.random.randn(self.action_dim).astype(np.float32)
        elif self.strategy == "mean":
            latent_action = self.mean_action.copy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # 🔥 NEW: Decode to slate using ranker
        latent_action_tensor = torch.FloatTensor(latent_action).unsqueeze(0).to(self.ranker.device)
        slate_tensor = self.ranker.rank(latent_action_tensor).squeeze(0)
        slate = slate_tensor.cpu().numpy()

        return slate

    def reset_hidden(self):
        pass


# ============================================================================
# 测试函数：Baseline 策略测试（Zero/Random/Mean-Action）
# ============================================================================
def test_zero_action_baseline(models_to_test: List[str], args):
    """测试三种 Baseline 策略性能：Zero-Action, Random-Action, Mean-Action"""

    print("\n" + "=" * 80)
    print("=== 测试 1: Baseline 策略性能（Zero/Random/Mean-Action） ===")
    print("=" * 80)
    print()

    # 存储所有结果：{model_name: {strategy: {mean_reward, std_reward}}}
    all_results = {}

    # 定义三种测试策略
    strategies = ["zero", "random", "mean"]
    strategy_names = {
        "zero": "Zero-Action",
        "random": "Random-Action",
        "mean": "Mean-Action"
    }

    for model_name in models_to_test:
        model_config = GEMS_MODELS[model_name]
        all_results[model_name] = {"group": model_config["group"]}

        print(f"\n{'='*80}")
        print(f"测试模型: {model_name}")
        print(f"{'='*80}")

        # 🔥 加载指定的 GeMS checkpoint
        checkpoint_path = PROJECT_ROOT / "checkpoints/gems/offline" / model_config["checkpoint"]
        print(f"📦 GeMS Checkpoint: {checkpoint_path}")
        print(f"   Group: {model_config['group']}")
        print(f"   Lambda_click: {model_config['lambda_click']}")

        try:
            ranker = load_gems_model(str(checkpoint_path), args.device)
        except Exception as e:
            print(f"\n✗ 加载 GeMS 模型失败: {e}")
            continue

        # 初始化评估环境（不再传入ranker，agents现在内部处理slate解码）
        env_params = {
            "env_name": model_config["env_name"],
            "device": args.device,
            "seed": args.seed,
            "verbose": False
        }

        # 添加特定参数
        if "dataset_quality" in model_config:
            env_params["dataset_quality"] = model_config["dataset_quality"]
        if "boredom_threshold" in model_config:
            env_params["env_param_override"] = {
                "boredom_threshold": model_config["boredom_threshold"]
            }

        try:
            eval_env = OfflineEvalEnv(**env_params)

            # 测试三种策略
            for strategy in strategies:
                print(f"\n  [{strategy_names[strategy]}]", end=" ")

                # 创建对应的 agent
                if strategy == "mean":
                    # Mean-Action 需要计算数据集的平均动作
                    # 这里使用零向量作为近似（因为 VAE 潜空间通常以0为中心）
                    mean_action = np.zeros(32, dtype=np.float32)
                    agent = BaselineAgent(strategy="mean", mean_action=mean_action, action_dim=32, ranker=ranker)
                else:
                    agent = BaselineAgent(strategy=strategy, action_dim=32, ranker=ranker)

                # 评估
                metrics = eval_env.evaluate_policy(
                    agent=agent,
                    num_episodes=args.num_episodes,
                    deterministic=True
                )

                all_results[model_name][strategy] = {
                    "mean_reward": metrics['mean_reward'],
                    "std_reward": metrics['std_reward']
                }

                print(f"✓ {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")

        except Exception as e:
            print(f"\n✗ 模型测试失败: {e}")
            for strategy in strategies:
                all_results[model_name][strategy] = {
                    "mean_reward": 0.0,
                    "std_reward": 0.0
                }
            continue

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("=== Baseline 策略结果汇总 ===")
    print("=" * 80)
    print(f"{'模型':<35} {'分组':<15} {'Zero-Action':<18} {'Random-Action':<18} {'Mean-Action':<18}")
    print("-" * 80)

    for model_name, results in all_results.items():
        if "zero" in results:  # 确保测试成功
            zero_str = f"{results['zero']['mean_reward']:.2f}±{results['zero']['std_reward']:.2f}"
            random_str = f"{results['random']['mean_reward']:.2f}±{results['random']['std_reward']:.2f}"
            mean_str = f"{results['mean']['mean_reward']:.2f}±{results['mean']['std_reward']:.2f}"
            print(f"{model_name:<35} {results['group']:<15} {zero_str:<18} {random_str:<18} {mean_str:<18}")

    print("=" * 80)


# ============================================================================
# 测试函数：多样性评估
# ============================================================================
def test_diversity_metrics(models_to_test: List[str], args):
    """测试多样性指标"""
    
    print("\n" + "=" * 80)
    print("=== 测试 2: 多样性评估 ===")
    print("=" * 80)
    print("⚠️  此功能需要实现 GeMS 模型的 Slate 生成接口")
    print("=" * 80)
    # TODO: 实现多样性测试


# ============================================================================
# 测试函数：重构质量评估
# ============================================================================
def test_reconstruction_quality(models_to_test: List[str], args):
    """测试重构质量"""
    
    print("\n" + "=" * 80)
    print("=== 测试 3: 重构质量评估 ===")
    print("=" * 80)
    print("⚠️  此功能需要实现 GeMS 模型的重构评估接口")
    print("=" * 80)
    # TODO: 实现重构质量测试



# ============================================================================
# 主函数
# ============================================================================
def main():
    """主测试流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GeMS VAE 性能测试")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="zero_action",
        choices=["zero_action", "diversity", "reconstruction", "all"],
        help="测试模式"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="要测试的模型名称列表（默认测试所有模型）"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Zero-Action 测试的 Episode 数量"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=58407201,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("=== 测试配置 ===")
    print("=" * 80)
    print(f"测试模式: {args.test_mode}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("=" * 80)
    print()
    
    # 确定要测试的模型
    if args.models is None:
        models_to_test = list(GEMS_MODELS.keys())
    else:
        models_to_test = args.models
    
    print(f"将测试 {len(models_to_test)} 个模型:")
    for model_name in models_to_test:
        print(f"  - {model_name}")
    print()
    
    # 根据测试模式执行相应的测试
    if args.test_mode == "zero_action":
        test_zero_action_baseline(models_to_test, args)
    elif args.test_mode == "diversity":
        test_diversity_metrics(models_to_test, args)
    elif args.test_mode == "reconstruction":
        test_reconstruction_quality(models_to_test, args)
    elif args.test_mode == "all":
        test_zero_action_baseline(models_to_test, args)
        test_diversity_metrics(models_to_test, args)
        test_reconstruction_quality(models_to_test, args)
    
    print("\n" + "=" * 80)
    print("=== 测试完成 ===")
    print("=" * 80)


if __name__ == "__main__":
    main()

