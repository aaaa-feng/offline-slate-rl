"""
TD3+BC for GeMS datasets (Enhanced Version with SwanLab)
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2106.06860.pdf

Enhancements:
- SwanLab logging support
- Simplified checkpoint/log structure
- Comprehensive metrics monitoring
- Reward normalization by default
"""
import copy
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入路径配置
sys.path.insert(0, str(PROJECT_ROOT.parent))
from config.offline import paths
from config.offline.config import TD3BCConfig, auto_generate_paths, auto_generate_swanlab_config

from common.offline.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.offline.utils import set_seed, compute_mean_std, soft_update
from common.offline.networks import Actor, Critic
from common.offline.eval_env import OfflineEvalEnv
from common.offline.checkpoint_utils import resolve_gems_checkpoint
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.item_embeddings import ItemEmbeddings

# SwanLab Logger import (离线RL专用版本)
try:
    from common.offline.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    # 如果 swanlab 包不存在,使用 dummy logger
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available, using dummy logger")

    class SwanlabLogger:
        """Dummy logger when swanlab is not available"""
        def __init__(self, *args, **kwargs):
            pass

        def log_metrics(self, metrics, step=None):
            pass

        def log_hyperparams(self, params):
            pass

        @property
        def experiment(self):
            class DummyExperiment:
                def finish(self):
                    pass
            return DummyExperiment()

TensorBatch = List[torch.Tensor]


def compute_action_normalization_params(
    dataset: Dict[str, np.ndarray],
    ranker,
    device: str,
    batch_size: int = 1000,
    use_fake_clicks: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🔥 补丁1：从 slates + clicks (或 fake_clicks) 动态推断 latent_action 并计算归一化参数

    新数据格式移除了预编码的 'actions' 字段，训练时需要：
    1. 使用 ranker.run_inference(slates, clicks) 推断 latent_action (返回 latent_mu, log_latent_var)
    2. 计算归一化参数：action_center, action_scale

    Args:
        dataset: 包含 'slates' 和 'clicks' 的字典
        ranker: GeMS ranker 模型
        device: 设备
        batch_size: 批处理大小
        use_fake_clicks: 是否用全零 clicks 推断动作（避免点击噪声）

    Returns:
        action_center: (action_max + action_min) / 2
        action_scale: (action_max - action_min) / 2 + 1e-6
    """
    if use_fake_clicks:
        print("🔥 [补丁1] 从 slates + fake_clicks 推断 latent_action 并计算归一化参数...")
    else:
        print("🔥 [补丁1] 从 slates + clicks 推断 latent_action 并计算归一化参数...")

    slates = torch.tensor(dataset['slates'], dtype=torch.long, device=device)
    clicks = None
    if not use_fake_clicks:
        clicks = torch.tensor(dataset['clicks'], dtype=torch.float32, device=device)

    num_samples = slates.shape[0]
    all_latent_actions = []

    ranker.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_slates = slates[i:i+batch_size]
            if use_fake_clicks:
                batch_clicks = torch.zeros_like(batch_slates, dtype=torch.float32)
            else:
                batch_clicks = clicks[i:i+batch_size]

            # 推断 latent_action (使用 run_inference 获取 mu 和 log_var)
            latent_mu, log_latent_var = ranker.run_inference(batch_slates, batch_clicks)
            all_latent_actions.append(latent_mu.cpu())

    # 合并所有 latent_action
    all_latent_actions = torch.cat(all_latent_actions, dim=0).to(device)

    # 计算归一化参数（与 buffer.py 一致）
    action_min = all_latent_actions.min(dim=0)[0]
    action_max = all_latent_actions.max(dim=0)[0]

    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6

    print(f"  ✅ 推断完成: {num_samples} samples")
    print(f"  Action range: [{action_min.min().item():.4f}, {action_max.max().item():.4f}]")
    print(f"  Action center shape: {action_center.shape}")
    print(f"  Action scale shape: {action_scale.shape}")

    return action_center, action_scale


class TD3_BC:
    """TD3+BC algorithm with Dual-Stream End-to-End GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config: TD3BCConfig,
        ranker_params: Dict,  # 🔥 GeMS-aligned: 接收 Ranker 参数
        ranker=None,  # 🔥 [FIX] 添加 ranker 参数（用于实时推断）
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0  # 归一化后固定为 1.0

        # ========================================================================
        # 🔥 关键：从 Ranker 参数中提取组件（复刻 BC 逻辑）
        # ========================================================================

        # 🔥 [FIX] 保存 ranker 引用（用于训练时实时推断动作）
        self.ranker = ranker

        # 验证：如果未提供 ranker，发出警告
        if self.ranker is None:
            logging.warning(
                "⚠️  TD3_BC initialized without ranker. "
                "Agent will not be able to perform inference (act())."
            )
            logging.warning("    This is acceptable for training-only scenarios.")

        # 1. Action Bounds（直接使用 Ranker 的）
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)

        # 2. Item Embeddings（使用 GeMS 训练后的）
        self.item_embeddings = ranker_params['item_embeddings']

        # 3. 初始化 GRU belief encoder
        gru_mode = "Shared GRU" if config.use_shared_gru else "Dual-Stream GRU"
        input_dim = config.rec_size * (config.item_embedd_dim + 1)

        self.belief = GRUBelief(
            item_embeddings=self.item_embeddings,  # 🔥 传入 GeMS 的 Embeddings
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=self.device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=["actor", "critic"],  # DUAL-STREAM ARCHITECTURE
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim  # 🔥 显式传入
        )

        # 4. 🔥 关键：双重保险 - 再次冻结 Embeddings
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()

        # ⚠️ 关键：冻结GeMS模型
        for param in self.ranker.parameters():
            param.requires_grad = False
        self.ranker.eval()

        # Initialize Actor network
        self.actor = Actor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=self.max_action,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Initialize Critic networks
        self.critic_1 = Critic(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # CRITICAL: 共享GRU优化器配置 (GRU只在Actor优化器中)
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["shared"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.actor_lr)

        self.critic_optimizer = torch.optim.Adam([
            {'params': self.critic_1.parameters()},
            {'params': self.critic_2.parameters()}
        ], lr=config.critic_lr)

        self.total_it = 0
        gru_mode = "Shared GRU" if config.use_shared_gru else "Dual-Stream GRU"
        logging.info(f"TD3_BC initialized: {gru_mode}, action_dim={action_dim}, hidden_dim={config.belief_hidden_dim}")

    def train(self, batch) -> Dict[str, float]:
        """
        训练一步 (双流端到端训练: Actor GRU + Critic GRU)

        Args:
            batch: TrajectoryBatch with obs, actions, rewards, dones
        """
        self.total_it += 1

        # Step 1: GRU 前向传播（根据配置选择单流或双流）
        if self.config.use_shared_gru:
            s, ns = self.belief.forward_batch_shared(batch)
        else:
            states, next_states = self.belief.forward_batch(batch)
            s = states["actor"]
            ns = next_states["actor"]

        # 🔬 表征坍缩探针：每 100 步检测一次 GRU 表征空间的健康度
        representation_rank = 0.0
        representation_singular_max = 0.0
        representation_singular_min = 0.0
        representation_singular_ratio = 0.0
        if self.total_it % 100 == 0:
            with torch.no_grad():
                # SVD 分解：s 的形状是 [batch_size, hidden_dim]
                try:
                    U, Sigma, Vt = torch.linalg.svd(s, full_matrices=False)

                    # 有效秩（Effective Rank）：衡量表征空间的维度
                    # 公式：(Σ σ_i)^2 / Σ (σ_i^2)
                    representation_rank = (Sigma.sum() ** 2) / (Sigma ** 2).sum()
                    representation_rank = representation_rank.item()

                    # 最大和最小奇异值
                    representation_singular_max = Sigma[0].item()
                    representation_singular_min = Sigma[-1].item()

                    # 奇异值比率（条件数）：衡量表征空间的病态程度
                    representation_singular_ratio = representation_singular_max / (representation_singular_min + 1e-8)
                except Exception as e:
                    # SVD 可能失败（比如矩阵退化），记录为 0
                    pass

        # Step 2: 🔥 [FIX] 实时推断 Latent Actions（因为 Buffer 里没有 action 了）
        # 从 batch.obs 中提取 slates 和 clicks
        flat_slates = torch.cat(batch.obs["slate"], dim=0)  # [sum_seq_lens, rec_size]
        flat_clicks = torch.cat(batch.obs["clicks"], dim=0)  # [sum_seq_lens, rec_size]

        with torch.no_grad():
            # 使用零填充 clicks 推断动作，避免点击噪声导致监督不一致
            fake_clicks = torch.zeros_like(flat_slates, dtype=torch.float32)
            true_actions, _ = self.ranker.run_inference(flat_slates, fake_clicks)

            # 归一化（使用初始化时计算的参数）
            true_actions = (true_actions - self.action_center) / self.action_scale

        rewards = torch.cat(batch.rewards, dim=0) if batch.rewards else None
        dones = torch.cat(batch.dones, dim=0) if batch.dones else None

        # Step 3: Critic Update (TD3 Loss)
        with torch.no_grad():
            # 使用共享 GRU 的 next_state 生成 next_action
            noise = (torch.randn_like(true_actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(ns) + noise).clamp(
                -self.max_action, self.max_action
            )

            # 🔬 记录 policy noise 统计
            policy_noise_mean = noise.mean().item()
            policy_noise_std = noise.std().item()

            # 使用共享 GRU 的 next_state 计算 target Q
            target_q1 = self.critic_1_target.q1(ns, next_action)
            target_q2 = self.critic_2_target.q1(ns, next_action)
            target_q = torch.min(target_q1, target_q2)

            if rewards is not None and dones is not None:
                # 计算 Bellman 备份
                target_q = rewards + (1 - dones) * self.config.gamma * target_q
            else:
                # 如果没有 reward/done，使用简化版本
                target_q = target_q * self.config.gamma

        # 使用共享 GRU 的 current_state 计算 current Q
        # ⚠️ 必须使用 s.detach() 避免幽灵梯度
        current_q1 = self.critic_1.q1(s.detach(), true_actions)
        current_q2 = self.critic_2.q1(s.detach(), true_actions)

        # 🔬 计算 TD Error 和 Bellman 备份质量指标
        td_error = torch.abs(current_q1 - target_q).mean().item()

        # 🔬 Reward 和 Done 统计
        reward_mean = rewards.mean().item() if rewards is not None else 0.0
        reward_std = rewards.std().item() if rewards is not None else 0.0
        done_rate = dones.mean().item() if dones is not None else 0.0

        # 🔬 Bootstrap ratio (γ * next_Q 占 target_Q 的比例)
        if rewards is not None and dones is not None:
            next_q_contribution = (1 - dones) * self.config.gamma * torch.min(target_q1, target_q2)
            bootstrap_ratio = (next_q_contribution / (target_q + 1e-8)).mean().item()
        else:
            bootstrap_ratio = 1.0

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize Critic + Critic GRU
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 🔧 HOT-FIX: Critic Gradient Clipping (防止梯度爆炸)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            10.0  # 从 float('inf') 改为 10.0
        )
        self.critic_optimizer.step()

        # 🔬 计算 Critic 权重范数
        critic_weight_norm = sum(p.norm().item() for p in self.critic_1.parameters()) + \
                            sum(p.norm().item() for p in self.critic_2.parameters())

        # Step 4: Actor Update (TD3+BC Loss) - Delayed
        actor_loss = None
        bc_loss = None
        action_l2_distance = 0.0
        action_cosine_sim = 0.0
        action_infer_l2 = 0.0
        action_infer_cos = 0.0
        slate_accuracy = 0.0
        lambda_value = 0.0
        q_term = 0.0
        bc_weight_ratio = 0.0
        actor_action_mean = 0.0
        actor_action_std = 0.0
        actor_action_max = 0.0
        actor_action_min = 0.0
        dataset_action_mean = 0.0
        dataset_action_std = 0.0
        actor_weight_norm = 0.0
        actor_target_diff = 0.0
        gru_grad_norm = 0.0
        gru_hidden_mean = 0.0
        gru_hidden_std = 0.0
        gru_hidden_max = 0.0
        gru_hidden_min = 0.0
        grad_conflict = 0.0  # 🔬 梯度冲突指标

        if self.total_it % self.config.policy_freq == 0:
            # 使用共享 GRU 的 state 生成 action
            pi = self.actor(s)

            # CRITICAL: 使用 detached state 计算 Q 值
            # 这样梯度不会流回 Critic
            q = self.critic_1.q1(s.detach(), pi)
            lmbda = self.config.alpha / q.abs().mean().detach()

            # 🔬 记录 lambda 和 Q term
            lambda_value = lmbda.item()
            q_term = -lmbda * q.mean()

            # ✅ 新BC Loss：与Critic一致的动作监督（fake_clicks 推断）
            pi_denormalized = pi * self.action_scale + self.action_center
            true_slate = torch.cat(batch.obs["slate"], dim=0)

            # BC Loss：使用与Critic一致的动作监督
            bc_loss = F.mse_loss(pi, true_actions)

            # 🔬 梯度冲突分析探针：计算 RL 和 BC 梯度的余弦相似度
            # 计算 RL 梯度
            loss_rl = -lmbda * q.mean()
            self.actor_optimizer.zero_grad()
            loss_rl.backward(retain_graph=True)
            grad_rl = torch.cat([p.grad.flatten().clone() for p in self.actor.parameters() if p.grad is not None])

            # 计算 BC 梯度
            self.actor_optimizer.zero_grad()
            bc_loss.backward(retain_graph=True)
            grad_bc = torch.cat([p.grad.flatten().clone() for p in self.actor.parameters() if p.grad is not None])

            # 计算余弦相似度（-1 表示完全对抗，+1 表示完全一致）
            with torch.no_grad():
                if grad_rl.norm() > 1e-8 and grad_bc.norm() > 1e-8:
                    grad_conflict = F.cosine_similarity(grad_rl.unsqueeze(0), grad_bc.unsqueeze(0)).item()
                else:
                    grad_conflict = 0.0

            # 🔬 Actor 行为分析指标
            with torch.no_grad():
                actor_action_mean = pi.mean().item()
                actor_action_std = pi.std().item()
                actor_action_max = pi.max().item()
                actor_action_min = pi.min().item()
                dataset_action_mean = true_actions.mean().item()
                dataset_action_std = true_actions.std().item()

            # 🔬 监控指标
            with torch.no_grad():
                # L2距离 (欧氏距离)
                action_l2_distance = torch.norm(pi - true_actions, dim=1).mean().item()
                # 余弦相似度
                action_cosine_sim = F.cosine_similarity(pi, true_actions, dim=1).mean().item()
                # 监督一致性：真实 clicks vs fake_clicks 的动作差异
                true_actions_click, _ = self.ranker.run_inference(flat_slates, flat_clicks)
                true_actions_click = (true_actions_click - self.action_center) / self.action_scale
                action_infer_l2 = torch.norm(true_actions_click - true_actions, dim=1).mean().item()
                action_infer_cos = F.cosine_similarity(true_actions_click, true_actions, dim=1).mean().item()
                # Slate准确率 (核心指标 - 仅用于监控，不参与梯度)
                policy_slate_logits = self.ranker.decode_to_slate_logits(pi_denormalized)
                predicted_slate = policy_slate_logits.argmax(dim=2)
                slate_accuracy = (predicted_slate == true_slate).float().mean().item()

                # 🔬 GRU Hidden State 统计
                gru_hidden_mean = s.mean().item()
                gru_hidden_std = s.std().item()
                gru_hidden_max = s.max().item()
                gru_hidden_min = s.min().item()

            # TD3+BC loss
            actor_loss = -lmbda * q.mean() + bc_loss

            # 🔬 BC weight ratio (BC loss 占总 loss 的比例)
            bc_weight_ratio = (bc_loss / (actor_loss.abs() + 1e-8)).item()

            # Optimize Actor + Actor GRU
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 🔧 HOT-FIX: Actor Gradient Clipping (防止梯度爆炸)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)  # 从 float('inf') 改为 100.0

            # 🔬 GRU Gradient Norm (单独记录)
            gru_params = list(self.belief.gru["shared"].parameters())
            gru_grad_norm = sum(p.grad.norm().item() for p in gru_params if p.grad is not None)

            self.actor_optimizer.step()

            # 🔬 计算 Actor 权重范数
            actor_weight_norm = sum(p.norm().item() for p in self.actor.parameters())

            # 🔬 计算 Actor-Target 差异（更新前）
            actor_target_diff = sum(
                (p1 - p2).norm().item()
                for p1, p2 in zip(self.actor.parameters(), self.actor_target.parameters())
            )

            # Update target networks
            soft_update(self.actor_target, self.actor, self.config.tau)
            soft_update(self.critic_1_target, self.critic_1, self.config.tau)
            soft_update(self.critic_2_target, self.critic_2, self.config.tau)

        # 🔬 计算 Critic-Target 差异（在 soft_update 之后，每步都计算）
        critic_target_diff = sum(
            (p1 - p2).norm().item()
            for p1, p2 in zip(self.critic_1.parameters(), self.critic_1_target.parameters())
        ) + sum(
            (p1 - p2).norm().item()
            for p1, p2 in zip(self.critic_2.parameters(), self.critic_2_target.parameters())
        )

        # 返回完整的监控指标
        return {
            # === Loss 指标 ===
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
            "bc_loss": bc_loss.item() if bc_loss is not None else 0.0,

            # === Q 值指标 ===
            "q_value": current_q1.mean().item(),
            "q1_value": current_q1.mean().item(),
            "q2_value": current_q2.mean().item(),
            "q_max": max(current_q1.max().item(), current_q2.max().item()),
            "q_min": min(current_q1.min().item(), current_q2.min().item()),
            "q_std": current_q1.std().item(),
            "target_q": target_q.mean().item(),
            "target_q_max": target_q.max().item(),
            "target_q_min": target_q.min().item(),

            # === 梯度指标 ===
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item() if actor_loss is not None else 0.0,
            "gru_grad_norm": gru_grad_norm,

            # === OOD 监控指标 ===
            "action_l2_distance": action_l2_distance,
            "action_cosine_sim": action_cosine_sim,
            "slate_accuracy": slate_accuracy,

            # === 监督信号一致性监控 ===
            "action_infer_l2": action_infer_l2,
            "action_infer_cos": action_infer_cos,

            # === BC 正则化效果指标 ===
            "lambda_value": lambda_value,
            "q_term": q_term.item() if isinstance(q_term, torch.Tensor) else q_term,
            "bc_weight_ratio": bc_weight_ratio,

            # === Actor 行为分析指标 ===
            "actor_action_mean": actor_action_mean,
            "actor_action_std": actor_action_std,
            "actor_action_max": actor_action_max,
            "actor_action_min": actor_action_min,
            "dataset_action_mean": dataset_action_mean,
            "dataset_action_std": dataset_action_std,

            # === 训练稳定性指标 ===
            "td_error": td_error,
            "actor_weight_norm": actor_weight_norm,
            "critic_weight_norm": critic_weight_norm,
            "actor_target_diff": actor_target_diff,
            "critic_target_diff": critic_target_diff,

            # === Bellman 备份质量指标 ===
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "done_rate": done_rate,
            "bootstrap_ratio": bootstrap_ratio,

            # === Policy Noise 指标 ===
            "policy_noise_mean": policy_noise_mean,
            "policy_noise_std": policy_noise_std,

            # === GRU Hidden State 指标 ===
            "gru_hidden_mean": gru_hidden_mean,
            "gru_hidden_std": gru_hidden_std,
            "gru_hidden_max": gru_hidden_max,
            "gru_hidden_min": gru_hidden_min,

            # === 🔬 表征坍缩探针指标 ===
            "representation_rank": representation_rank,
            "representation_singular_max": representation_singular_max,
            "representation_singular_min": representation_singular_min,
            "representation_singular_ratio": representation_singular_ratio,

            # === 🔬 梯度冲突探针指标 ===
            "grad_conflict": grad_conflict,
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        选择动作并解码为slate (端到端推理)

        Args:
            obs: Dict with 'slate' and 'clicks' (torch.Tensor or numpy arrays)
            deterministic: 是否确定性选择 (TD3+BC 总是确定性的)

        Returns:
            slate: 推荐slate (shape: [rec_size]), numpy array
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # 只使用 Actor GRU 编码 (推理时不需要 Critic)
        belief_state = self.belief.forward(obs_tensor, done=False)["actor"]

        # Actor 预测
        raw_action = self.actor(belief_state)  # [1, action_dim]

        # 反归一化
        latent_action = raw_action * self.action_scale + self.action_center

        # 🔥 NEW: 使用 GeMS ranker 解码 latent action 为 slate
        if self.ranker is None:
            raise RuntimeError(
                "TD3_BC.act() requires a ranker for slate decoding. "
                "Please provide ranker during initialization."
            )

        # 确保设备一致性
        latent_action = latent_action.to(self.device)

        # 添加 batch 维度 (ranker 期望 [batch_size, latent_dim])
        latent_action_batched = latent_action.unsqueeze(0)  # [1, latent_dim]

        # 解码为 slate
        slate_tensor = self.ranker.rank(latent_action_batched)  # [1, rec_size]

        # 移除 batch 维度并转换为 numpy
        slate_output = slate_tensor.squeeze(0).cpu().numpy()  # [rec_size]

        return slate_output

    def reset_hidden(self):
        """
        重置双流 GRU 隐藏状态 (在每个 episode 开始时调用)
        使用 dummy obs + done=True 来优雅地重置
        """
        dummy_obs = {
            "slate": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
            "clicks": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device)
        }
        self.belief.forward(dummy_obs, done=True)

    def save(self, filepath: str):
        """保存模型（包含所有必要信息，支持独立加载）"""
        torch.save({
            'belief_state_dict': self.belief.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'ranker_state_dict': self.ranker.state_dict(),  # 🔥 保存完整的GeMS模型
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'embeddings_meta': {
                'num_items': self.item_embeddings.num_items,
                'embedd_dim': self.item_embeddings.embedd_dim,
            },
            'action_dim': self.action_dim,
            'total_it': self.total_it,
            'config': self.config,
        }, filepath)
        logging.info(f"✅ Model saved to {filepath} (with embeddings_meta)")

    def load(self, filepath: str):
        """加载模型 (包含双流 GRU + Actor + Critics + 归一化参数)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.belief.load_state_dict(checkpoint['belief_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])

        # 🔥 加载完整的GeMS模型（向后兼容）
        if 'ranker_state_dict' in checkpoint:
            self.ranker.load_state_dict(checkpoint['ranker_state_dict'])
            logging.info("✅ Loaded ranker from checkpoint")
        else:
            logging.warning("⚠️  Old checkpoint format: ranker not found in checkpoint")

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """从 Checkpoint 独立加载，无需 GeMS"""
        logging.info("=" * 80)
        logging.info("=== Loading TD3_BC from Checkpoint (Standalone) ===")
        logging.info(f"Checkpoint: {checkpoint_path}")
        logging.info("=" * 80)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 1. 恢复 Embeddings
        embeddings_meta = checkpoint['embeddings_meta']
        belief_state = checkpoint['belief_state_dict']
        embedding_weights = belief_state['item_embeddings.actor.embedd.weight']

        agent_embeddings = ItemEmbeddings(
            num_items=embeddings_meta['num_items'],
            item_embedd_dim=embeddings_meta['embedd_dim'],
            device=device,
            weights=embedding_weights
        )
        logging.info(f"✅ Embeddings restored: {embeddings_meta['num_items']} items")

        # 2. 构建 ranker_params
        ranker_params = {
            'item_embeddings': agent_embeddings,
            'action_center': checkpoint['action_center'],
            'action_scale': checkpoint['action_scale'],
            'num_items': embeddings_meta['num_items'],
            'item_embedd_dim': embeddings_meta['embedd_dim']
        }

        # 3. 创建 Agent
        agent = cls(
            action_dim=checkpoint['action_dim'],
            config=checkpoint['config'],
            ranker_params=ranker_params,
            ranker=None  # 🔥 [FIX] 从 checkpoint 加载时不需要 ranker（仅推理）
        )

        # 4. 加载权重
        agent.belief.load_state_dict(belief_state)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.total_it = checkpoint['total_it']

        logging.info(f"✅ TD3_BC loaded from {checkpoint_path} (standalone)")
        logging.info("=" * 80)
        return agent


def train_td3_bc(config: TD3BCConfig):
    """
    Train TD3+BC on GeMS dataset with SwanLab logging

    Args:
        config: Training configuration
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d")

    # 自动生成路径配置
    config = auto_generate_paths(config, timestamp)

    # 自动生成 SwanLab 配置
    config = auto_generate_swanlab_config(config)

    # 创建目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 生成日志文件名
    log_filename = f"{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_seed{config.seed}_{config.run_id}.log"
    log_filepath = os.path.join(config.log_dir, log_filename)

    # 清除已有的handlers并重新配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置logging (输出到文件和stdout)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ],
        force=True
    )

    # Set seed
    set_seed(config.seed)
    logging.info(f"Global seed set to {config.seed}")

    # 🔥 输出完整的运行命令
    logging.info("=" * 80)
    logging.info("=== Training Command ===")
    logging.info("=" * 80)
    command_str = " ".join(sys.argv)
    logging.info(f"Command: {command_str}")
    logging.info("=" * 80)
    logging.info("")

    # Initialize SwanLab
    swan_logger = None
    if config.use_swanlab:
        if not SWANLAB_AVAILABLE:
            logging.warning("SwanLab not available, disabling SwanLab logging")
            config.use_swanlab = False
        else:
            try:
                swan_logger = SwanlabLogger(
                    project=config.swan_project,
                    experiment_name=config.run_name,
                    workspace=config.swan_workspace,
                    description=config.swan_description,
                    tags=config.swan_tags,
                    config=config.__dict__,
                    mode=config.swan_mode,
                    logdir=config.swan_logdir,
                )
                logging.info(f"SwanLab initialized: project={config.swan_project}, run={config.run_name}")
            except Exception as e:
                logging.warning(f"SwanLab initialization failed: {e}")
                config.use_swanlab = False

    # 打印配置信息
    logging.info("=" * 80)
    logging.info("=== Configuration ===")
    logging.info("=" * 80)
    logging.info(f"Environment: {config.env_name} ({config.dataset_quality})")
    logging.info(f"Algorithm: TD3+BC (alpha={config.alpha}, gamma={config.gamma})")
    logging.info(f"Training: max_steps={config.max_timesteps}, batch={config.batch_size}, lr={config.learning_rate}")
    logging.info(f"GRU Mode: {'Shared GRU' if config.use_shared_gru else 'Dual-Stream GRU'}")
    logging.info(f"Device: {config.device}")
    logging.info(f"Seed: {config.seed}")
    logging.info(f"Dataset: {config.dataset_path}")
    logging.info(f"Log file: {log_filepath}")
    logging.info("=" * 80)
    logging.info("")

    # ========================================================================
    # 🔥 关键：加载 GeMS 并提取组件（复刻 BC 逻辑）
    # ========================================================================
    from rankers.gems.rankers import GeMS

    logging.info("=" * 80)
    logging.info("=== Loading GeMS Ranker ===")
    logging.info("=" * 80)

    # 1. 使用统一配置模块解析 GeMS Checkpoint 路径和参数
    gems_path, lambda_click = resolve_gems_checkpoint(
        env_name=config.env_name,
        dataset_quality=config.dataset_quality,
        gems_embedding_mode=config.gems_embedding_mode
    )

    logging.info(f"Checkpoint: {gems_path}")
    logging.info(f"Embedding mode: {config.gems_embedding_mode}")
    logging.info(f"Lambda_click: {lambda_click}")
    logging.info(f"Latent dim: 32")

    # 2. 加载 GeMS Ranker
    temp_embeddings = ItemEmbeddings.from_pretrained(
        config.item_embedds_path,
        config.device
    )

    ranker = GeMS.load_from_checkpoint(
        gems_path,
        map_location=config.device,
        item_embeddings=temp_embeddings,
        device=config.device,
        rec_size=config.rec_size,
        item_embedd_dim=config.item_embedd_dim,
        num_items=config.num_items,
        latent_dim=32,
        lambda_click=lambda_click,
        lambda_KL=1.0,
        lambda_prior=1.0,
        ranker_lr=3e-3,
        fixed_embedds="scratch",
        ranker_sample=False,
        hidden_layers_infer=[512, 256],
        hidden_layers_decoder=[256, 512]
    )
    ranker.eval()
    ranker.freeze()
    ranker = ranker.to(config.device)

    logging.info("✓ GeMS loaded and frozen")
    logging.info("=" * 80)
    logging.info("")

    # 3. 提取 GeMS 训练后的 Embeddings
    logging.info("=" * 80)
    logging.info("=== Extracting Embeddings ===")
    logging.info("=" * 80)

    gems_embedding_weights = ranker.item_embeddings.weight.data.clone()

    agent_embeddings = ItemEmbeddings(
        num_items=ranker.item_embeddings.num_embeddings,
        item_embedd_dim=ranker.item_embeddings.embedding_dim,
        device=config.device,
        weights=gems_embedding_weights
    )

    # 4. 提前冻结
    for param in agent_embeddings.parameters():
        param.requires_grad = False

    logging.info(f"Source: GeMS trained embeddings")
    logging.info(f"Num items: {ranker.item_embeddings.num_embeddings}")
    logging.info(f"Embedding dim: {ranker.item_embeddings.embedding_dim}")
    logging.info("✓ Embeddings extracted and frozen")
    logging.info("=" * 80)
    logging.info("")

    # 5. 准备 Ranker 参数包
    ranker_params = {
        'item_embeddings': agent_embeddings,
        'action_center': ranker.action_center,
        'action_scale': ranker.action_scale,
        'num_items': ranker.num_items,
        'item_embedd_dim': ranker.item_embedd_dim
    }

    # ========================================================================
    # 加载数据集
    # ========================================================================
    logging.info("=" * 80)
    logging.info("=== Loading Dataset ===")
    logging.info("=" * 80)

    dataset = np.load(config.dataset_path)
    num_transitions = len(dataset['slates'])

    logging.info(f"Path: {config.dataset_path}")
    logging.info(f"Transitions: {num_transitions}")

    # 统计 episodes 数量
    if 'episode_ids' in dataset:
        num_episodes = len(np.unique(dataset['episode_ids']))
        logging.info(f"Episodes: {num_episodes}")

    logging.info(f"Has rewards: {'Yes' if 'rewards' in dataset else 'No'}")
    logging.info(f"Has terminals: {'Yes' if 'terminals' in dataset else 'No'}")

    # 🔥 [FIX] 阻断点3：移除 IN-MEMORY ACTION RELABELING 逻辑
    # 新数据格式不包含预编码的 actions 字段，训练时实时推断

    # 🔥 [补丁1] 计算动作归一化参数（从 slates + fake_clicks 推断）
    logging.info("=" * 80)
    logging.info("=== Computing Action Normalization ===")
    logging.info("=" * 80)
    logging.info(f"Method: Infer from slates + fake_clicks")
    logging.info(f"Samples: {num_transitions}")

    action_center, action_scale = compute_action_normalization_params(
        dataset={'slates': dataset['slates'], 'clicks': dataset['clicks']},
        ranker=ranker,
        device=config.device,
        batch_size=1000,
        use_fake_clicks=True
    )

    # Get dimensions (从 action_center 推断 action_dim)
    action_dim = action_center.shape[0]

    logging.info(f"Action dim: {action_dim}")
    logging.info(f"Action range: [{action_center.min().item() - action_scale.max().item():.4f}, {action_center.max().item() + action_scale.max().item():.4f}]")
    logging.info("✓ Normalization parameters computed")
    logging.info("=" * 80)
    logging.info("")

    # Create trajectory replay buffer
    replay_buffer = TrajectoryReplayBuffer(device=config.device)

    # 🔥 [FIX] 加载新格式数据（不包含 actions 字段）
    dataset_dict = {
        'episode_ids': dataset['episode_ids'],
        'slates': dataset['slates'],
        'clicks': dataset['clicks'],
        'next_slates': dataset['next_slates'],  # 🔥 新增
        'next_clicks': dataset['next_clicks'],  # 🔥 新增
    }

    # 可选字段
    if 'rewards' in dataset:
        dataset_dict['rewards'] = dataset['rewards'] / 100.0
    if 'terminals' in dataset:
        dataset_dict['terminals'] = dataset['terminals']

    replay_buffer.load_d4rl_dataset(dataset_dict)

    logging.info("✓ Dataset loaded into replay buffer")
    logging.info("=" * 80)
    logging.info("")

    # 🔥 [补丁1] 更新 ranker_params 中的 action bounds（使用预热步骤计算的参数）
    ranker_params['action_center'] = action_center
    ranker_params['action_scale'] = action_scale

    # Initialize TD3+BC agent (with Dual-Stream E2E GRU)
    logging.info("=" * 80)
    logging.info("=== Initializing TD3+BC Agent ===")
    logging.info("=" * 80)
    logging.info(f"Action dim: {action_dim}")
    logging.info(f"Belief hidden dim: {config.belief_hidden_dim}")
    logging.info(f"Actor hidden dim: {config.hidden_dim}")
    logging.info(f"Critic hidden dim: {config.hidden_dim}")
    logging.info(f"GRU mode: {'Shared GRU' if config.use_shared_gru else 'Dual-Stream GRU'}")

    agent = TD3_BC(
        action_dim=action_dim,
        config=config,
        ranker_params=ranker_params,  # 🔥 传入 Ranker 参数
        ranker=ranker,  # 🔥 [FIX] 传入 ranker（用于实时推断）
    )

    logging.info("✓ Agent initialized")
    logging.info("=" * 80)
    logging.info("")

    # Initialize evaluation environment
    logging.info("=" * 80)
    logging.info("=== Creating Evaluation Environment ===")
    logging.info("=" * 80)

    try:
        eval_env = OfflineEvalEnv(
            env_name=config.env_name,
            dataset_quality=config.dataset_quality,
            device=config.device,
            seed=config.seed,
            verbose=False
            # 🔥 不再需要ranker参数：Agent现在直接输出slate
        )
        logging.info(f"Environment: {config.env_name}")
        logging.info(f"Click model: {eval_env.env_config['click_model']}")
        logging.info(f"Diversity penalty: {eval_env.env_config['diversity_penalty']}")
        logging.info(f"Episode length: {eval_env.env_config['episode_length']}")
        if 'boredom_threshold' in eval_env.env_config:
            logging.info(f"Boredom threshold: {eval_env.env_config['boredom_threshold']}")
        logging.info("✓ Eval environment ready")
    except Exception as e:
        logging.warning(f"Failed to initialize eval env: {e}")
        eval_env = None
        logging.info("✗ Eval environment creation failed")

    logging.info("=" * 80)
    logging.info("")

    # Training loop
    logging.info("")
    logging.info("=" * 80)
    logging.info("=== Training Started ===")
    logging.info("=" * 80)
    logging.info("")

    for t in range(int(config.max_timesteps)):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)

        # Train
        train_metrics = agent.train(batch)

        # Logging
        if (t + 1) % 1000 == 0:
            # 构建统一的 SwanLab 指标字典（带命名空间前缀）
            swanlab_metrics = {
                # === Loss 指标 ===
                "train/actor_loss": train_metrics['actor_loss'],
                "train/critic_loss": train_metrics['critic_loss'],
                "train/bc_loss": train_metrics['bc_loss'],

                # === Q 值指标 ===
                "train/q_value_mean": train_metrics['q_value'],
                "train/q1_value": train_metrics['q1_value'],
                "train/q2_value": train_metrics['q2_value'],
                "train/q_value_std": train_metrics['q_std'],
                "train/q_max": train_metrics['q_max'],
                "train/q_min": train_metrics['q_min'],
                "train/target_q_mean": train_metrics['target_q'],
                "train/target_q_max": train_metrics['target_q_max'],
                "train/target_q_min": train_metrics['target_q_min'],

                # === 梯度指标 ===
                "train/actor_grad_norm": train_metrics['actor_grad_norm'],
                "train/critic_grad_norm": train_metrics['critic_grad_norm'],
                "train/gru_grad_norm": train_metrics['gru_grad_norm'],

                # === OOD 监控指标 ===
                "train/action_l2_distance": train_metrics['action_l2_distance'],
                "train/action_cosine_sim": train_metrics['action_cosine_sim'],
                "train/slate_accuracy": train_metrics['slate_accuracy'],

                # === 监督一致性监控 ===
                "train/action_infer_l2": train_metrics['action_infer_l2'],
                "train/action_infer_cos": train_metrics['action_infer_cos'],

                # === BC 正则化效果指标 ===
                "train/lambda_value": train_metrics['lambda_value'],
                "train/q_term": train_metrics['q_term'],
                "train/bc_weight_ratio": train_metrics['bc_weight_ratio'],

                # === Actor 行为分析指标 ===
                "train/actor_action_mean": train_metrics['actor_action_mean'],
                "train/actor_action_std": train_metrics['actor_action_std'],
                "train/actor_action_max": train_metrics['actor_action_max'],
                "train/actor_action_min": train_metrics['actor_action_min'],
                "train/dataset_action_mean": train_metrics['dataset_action_mean'],
                "train/dataset_action_std": train_metrics['dataset_action_std'],

                # === 训练稳定性指标 ===
                "train/td_error": train_metrics['td_error'],
                "train/actor_weight_norm": train_metrics['actor_weight_norm'],
                "train/critic_weight_norm": train_metrics['critic_weight_norm'],
                "train/actor_target_diff": train_metrics['actor_target_diff'],
                "train/critic_target_diff": train_metrics['critic_target_diff'],

                # === Bellman 备份质量指标 ===
                "train/reward_mean": train_metrics['reward_mean'],
                "train/reward_std": train_metrics['reward_std'],
                "train/done_rate": train_metrics['done_rate'],
                "train/bootstrap_ratio": train_metrics['bootstrap_ratio'],

                # === Policy Noise 指标 ===
                "train/policy_noise_mean": train_metrics['policy_noise_mean'],
                "train/policy_noise_std": train_metrics['policy_noise_std'],

                # === GRU Hidden State 指标 ===
                "train/gru_hidden_mean": train_metrics['gru_hidden_mean'],
                "train/gru_hidden_std": train_metrics['gru_hidden_std'],
                "train/gru_hidden_max": train_metrics['gru_hidden_max'],
                "train/gru_hidden_min": train_metrics['gru_hidden_min'],

                # === 🔬 表征坍缩探针指标 ===
                "train/representation_rank": train_metrics['representation_rank'],
                "train/representation_singular_max": train_metrics['representation_singular_max'],
                "train/representation_singular_min": train_metrics['representation_singular_min'],
                "train/representation_singular_ratio": train_metrics['representation_singular_ratio'],

                # === 🔬 梯度冲突探针指标 ===
                "train/grad_conflict": train_metrics['grad_conflict'],
            }

            # 详细的多行日志（显示所有关键指标）
            logging.info(f"=" * 80)
            logging.info(f"Step {t+1}")
            logging.info(f"-" * 80)

            # Loss 指标
            logging.info(
                f"Loss: critic={train_metrics['critic_loss']:.3f}, "
                f"actor={train_metrics['actor_loss']:.3f}, "
                f"bc={train_metrics['bc_loss']:.3f}, "
                f"td_err={train_metrics['td_error']:.3f}"
            )

            # Q 值指标
            logging.info(
                f"Q-values: mean={train_metrics['q_value']:.2f}, "
                f"q1={train_metrics['q1_value']:.2f}, "
                f"q2={train_metrics['q2_value']:.2f}, "
                f"std={train_metrics['q_std']:.2f}, "
                f"max={train_metrics['q_max']:.2f}, "
                f"min={train_metrics['q_min']:.2f}"
            )

            logging.info(
                f"Target Q: mean={train_metrics['target_q']:.2f}, "
                f"max={train_metrics['target_q_max']:.2f}, "
                f"min={train_metrics['target_q_min']:.2f}"
            )

            # BC 正则化指标
            logging.info(
                f"BC Reg: λ={train_metrics['lambda_value']:.4f}, "
                f"q_term={train_metrics['q_term']:.3f}, "
                f"bc_ratio={train_metrics['bc_weight_ratio']:.3f}"
            )

            # Actor 行为分析
            logging.info(
                f"Actor Action: mean={train_metrics['actor_action_mean']:.3f}, "
                f"std={train_metrics['actor_action_std']:.3f}, "
                f"max={train_metrics['actor_action_max']:.3f}, "
                f"min={train_metrics['actor_action_min']:.3f}"
            )

            logging.info(
                f"Dataset Action: mean={train_metrics['dataset_action_mean']:.3f}, "
                f"std={train_metrics['dataset_action_std']:.3f}"
            )

            # OOD 监控指标
            logging.info(
                f"OOD: cos_sim={train_metrics['action_cosine_sim']:.3f}, "
                f"l2_dist={train_metrics['action_l2_distance']:.3f}, "
                f"slate_acc={train_metrics['slate_accuracy']*100:.1f}%"
            )

            # 监督一致性
            logging.info(
                f"Supervision: infer_cos={train_metrics['action_infer_cos']:.3f}, "
                f"infer_l2={train_metrics['action_infer_l2']:.3f}"
            )

            # 梯度指标
            logging.info(
                f"Gradients: actor={train_metrics['actor_grad_norm']:.2f}, "
                f"critic={train_metrics['critic_grad_norm']:.2f}, "
                f"gru={train_metrics['gru_grad_norm']:.2f}"
            )

            # 权重范数
            logging.info(
                f"Weight Norms: actor={train_metrics['actor_weight_norm']:.1f}, "
                f"critic={train_metrics['critic_weight_norm']:.1f}"
            )

            # Target 网络差异
            logging.info(
                f"Target Diff: actor={train_metrics['actor_target_diff']:.3f}, "
                f"critic={train_metrics['critic_target_diff']:.3f}"
            )

            # Bellman 备份质量
            logging.info(
                f"Bellman: reward_mean={train_metrics['reward_mean']:.3f}, "
                f"reward_std={train_metrics['reward_std']:.3f}, "
                f"done_rate={train_metrics['done_rate']:.3f}, "
                f"bootstrap={train_metrics['bootstrap_ratio']:.3f}"
            )

            # Policy Noise
            logging.info(
                f"Policy Noise: mean={train_metrics['policy_noise_mean']:.4f}, "
                f"std={train_metrics['policy_noise_std']:.4f}"
            )

            # GRU Hidden State
            logging.info(
                f"GRU Hidden: mean={train_metrics['gru_hidden_mean']:.3f}, "
                f"std={train_metrics['gru_hidden_std']:.3f}, "
                f"max={train_metrics['gru_hidden_max']:.3f}, "
                f"min={train_metrics['gru_hidden_min']:.3f}"
            )

            # 🔬 表征坍缩探针（每100步更新一次）
            if train_metrics['representation_rank'] > 0:
                logging.info(
                    f"🔬 Representation: rank={train_metrics['representation_rank']:.2f}, "
                    f"σ_max={train_metrics['representation_singular_max']:.2f}, "
                    f"σ_min={train_metrics['representation_singular_min']:.4f}, "
                    f"ratio={train_metrics['representation_singular_ratio']:.1f}"
                )

            # 🔬 梯度冲突探针
            if train_metrics['grad_conflict'] != 0:
                conflict_status = '对抗' if train_metrics['grad_conflict'] < -0.5 else '一致' if train_metrics['grad_conflict'] > 0.5 else '中性'
                logging.info(
                    f"🔬 Gradient Conflict: cos_sim={train_metrics['grad_conflict']:.3f} ({conflict_status})"
                )

            logging.info(f"=" * 80)

            if config.use_swanlab and swan_logger:
                swan_logger.log_metrics(swanlab_metrics, step=t+1)

        # Evaluation
        if eval_env is not None and (t + 1) % config.eval_freq == 0:
            eval_metrics = eval_env.evaluate_policy(
                agent=agent,
                num_episodes=10,
                deterministic=True
            )

            logging.info(f"Eval @ Step {t+1}: reward={eval_metrics['mean_reward']:.2f}±{eval_metrics['std_reward']:.2f}, len={eval_metrics['mean_episode_length']:.1f}")

            if swan_logger:
                swan_logger.log_metrics({
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/std_reward': eval_metrics['std_reward'],
                    'eval/mean_episode_length': eval_metrics['mean_episode_length'],
                }, step=t+1)

        # Save checkpoint
        if (t + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"td3bc_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_step{t+1}.pt"
            )
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(
        config.checkpoint_dir,
        f"td3bc_{config.env_name}_{config.dataset_quality}_alpha{config.alpha}_lr{config.actor_lr}_seed{config.seed}_{config.run_id}_final.pt"
    )
    agent.save(final_path)

    # Final evaluation
    if eval_env is not None:
        final_eval_metrics = eval_env.evaluate_policy(
            agent=agent,
            num_episodes=100,
            deterministic=True
        )

        logging.info(f"Final eval: reward={final_eval_metrics['mean_reward']:.2f}±{final_eval_metrics['std_reward']:.2f}, len={final_eval_metrics['mean_episode_length']:.1f}")

        if swan_logger:
            swan_logger.log_metrics({
                'final_eval/mean_reward': final_eval_metrics['mean_reward'],
                'final_eval/std_reward': final_eval_metrics['std_reward'],
                'final_eval/mean_episode_length': final_eval_metrics['mean_episode_length'],
            }, step=config.max_timesteps)

    logging.info("Training completed")

    if config.use_swanlab and swan_logger:
        swan_logger.experiment.finish()

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TD3+BC (TD3 with Behavior Cloning) on offline datasets")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment",
                        help="实验名称")
    parser.add_argument("--env_name", type=str, default="diffuse_mix",
                        help="环境名称")
    parser.add_argument("--dataset_quality", type=str, default="expert",
                        help="数据集质量 (旧benchmark: random/medium/expert, 新benchmark: v2_b3/v2_b5)")
    parser.add_argument("--seed", type=int, default=58407201,
                        help="随机种子")
    parser.add_argument("--run_id", type=str, default="",
                        help="唯一运行标识符 (格式: MMDD_HHMM, 如果为空则自动生成)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    # 数据集配置
    parser.add_argument("--dataset_path", type=str, default="",
                        help="数据集路径 (如果为空则自动生成)")

    # 训练配置
    parser.add_argument("--max_timesteps", type=int, default=int(1e6),
                        help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="批次大小")
    parser.add_argument("--eval_freq", type=int, default=int(5e3),
                        help="评估频率 (训练步数)")
    parser.add_argument("--save_freq", type=int, default=int(5e4),
                        help="保存频率 (训练步数)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="日志记录频率")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")

    # TD3+BC特定参数
    parser.add_argument("--alpha", type=float, default=2.5,
                        help="BC正则化系数")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="软更新系数")
    parser.add_argument("--policy_noise", type=float, default=0.2,
                        help="策略噪声")
    parser.add_argument("--noise_clip", type=float, default=0.5,
                        help="噪声裁剪")
    parser.add_argument("--policy_freq", type=int, default=2,
                        help="策略更新频率")

    # GRU架构配置
    parser.add_argument("--use_shared_gru", action="store_true", default=True,
                        help="使用共享GRU（推荐）")
    parser.add_argument("--no_shared_gru", action="store_false", dest="use_shared_gru",
                        help="使用双GRU（原始方案）")

    # SwanLab配置
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                        help="是否使用SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab",
                        help="禁用SwanLab")

    # GeMS配置（2026-01-30新增）
    parser.add_argument("--gems_embedding_mode", type=str, default="mf_fixed",
                        choices=["default", "mf_fixed", "mf_scratch", "epsilon-greedy"],
                        help="GeMS embedding模式: default(旧checkpoint), mf_fixed(MF冻结), mf_scratch(MF可训练)")

    args = parser.parse_args()

    config = TD3BCConfig(
        experiment_name=args.experiment_name,
        env_name=args.env_name,
        dataset_quality=args.dataset_quality,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        dataset_path=args.dataset_path,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        use_shared_gru=args.use_shared_gru,
        use_swanlab=args.use_swanlab,
        gems_embedding_mode=args.gems_embedding_mode,
    )

    train_td3_bc(config)
