"""
离线RL评估环境封装

此模块封装完整的评估流程,确保评估环境参数与数据收集时一致。
包括:
- 环境创建
- Ranker (GeMS VAE) 加载
- 完整推理流程: Agent (内置GRU) → Ranker → Slate → Environment

注: 端到端模式,Agent自带GRU Belief Encoder
"""

import sys
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from config.offline import paths
from config.offline.env_params import get_env_config
from common.offline.checkpoint_utils import resolve_gems_checkpoint, extract_boredom_threshold

# 导入在线RL组件
from common.online.data_module import BufferDataModule
from common.online.env_wrapper import EnvWrapper
from rankers.gems.rankers import GeMS
from rankers.gems.item_embeddings import ItemEmbeddings


class OfflineEvalEnv:
    """
    离线RL评估环境

    封装完整的评估流程,包括环境、Ranker和Belief Encoder的加载。
    确保评估环境参数与数据收集时完全一致。
    """

    def __init__(
        self,
        env_name: str,
        dataset_quality: str = "medium",
        device: str = "cuda",
        seed: int = 58407201,
        verbose: bool = True,
        env_param_override: Optional[Dict[str, Any]] = None,
        ranker = None,  # 🔥 可选：从 Agent 传入的 ranker（包含完整的 GeMS 模型）
        dataset_path: Optional[str] = None,  # 🔥 NEW: 用于加载数据集频率字典
    ):
        """
        初始化离线评估环境

        Args:
            env_name: 环境名称 (如 diffuse_mix)
            dataset_quality: 数据集质量 (random/medium/expert)
            device: 设备
            seed: 随机种子
            verbose: 是否打印详细信息
            env_param_override: 环境参数覆盖字典 (用于测试不同环境配置)
            ranker: 可选的ranker实例（如果提供，则不会单独加载GeMS）
        """
        self.env_name = env_name
        self.dataset_quality = dataset_quality
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.env_param_override = env_param_override

        # 加载环境配置（传递 dataset_quality 以加载正确的元数据文件）
        self.env_config = get_env_config(env_name, dataset_quality)

        # 🔥 对于新 benchmark，从 dataset_quality 中自动提取 boredom threshold
        if self.env_name in ['mix_divpen', 'topdown_divpen']:
            boredom = extract_boredom_threshold(self.dataset_quality, self.env_name)
            if boredom is not None:
                # 初始化 env_param_override（如果用户没有提供）
                if self.env_param_override is None:
                    self.env_param_override = {}
                # 只在用户没有显式设置时才覆盖
                if 'boredom_threshold' not in self.env_param_override:
                    self.env_param_override['boredom_threshold'] = boredom

        if self.verbose:
            logging.info(f"Initializing OfflineEvalEnv for {env_name}")
            logging.info(f"  Click model: {self.env_config['click_model']}")
            logging.info(f"  Diversity penalty: {self.env_config['diversity_penalty']}")
            logging.info(f"  Ranker dataset: {self.env_config['ranker_dataset']}")

        # 🔥 DEPRECATED: Ranker参数已废弃，agents现在内部处理slate解码
        if ranker is not None:
            logging.warning(
                "⚠️  OfflineEvalEnv no longer requires ranker parameter. "
                "Agents now handle slate decoding internally."
            )
            logging.warning("    This parameter will be removed in future versions.")

        # 初始化组件
        self.env = None
        self.ranker = None  # 🔥 不再使用ranker（agents现在输出slate）
        self.item_embeddings = None
        self.ranker_checkpoint_path = None  # 用于日志输出

        # 创建环境
        self._create_environment()

        # 加载 Item Embeddings
        self._load_item_embeddings()

        # 🔥 DEPRECATED: 不再加载 ranker（agents 现在输出 slate）
        # 保留此逻辑仅用于向后兼容，但实际不再使用
        self.ranker_checkpoint_path = "deprecated"

        # 🔥 NEW: 加载数据集频率字典 (如果提供路径)
        self.item_freq_counter = None
        self.combo_freq_counter = None
        self.item_total = 0
        
        if dataset_path:
            self._load_dataset_frequency(dataset_path)

        # 强制打印参数摘要 (无视 verbose 设置，确保可观测性)
        logging.info(f"[eval_env.py] Eval env: {self.env_name}/{self.dataset_quality}, click_model={self.env_config['click_model']}, ep_len={self.env_config['episode_length']}")

    def _load_dataset_frequency(self, dataset_path: str):
        """
        加载数据集频率字典 (一次性成本)
        
        1. 统计每个 item 的出现频率
        2. 统计每个组合的出现频率 (Top-1000)
        """
        logging.info(f"Loading dataset frequency from: {dataset_path}")
        
        dataset = np.load(dataset_path)
        n_transitions = len(dataset['slates'])
        
        # ========== 1. Item 频率统计 ==========
        all_items = dataset['slates'].flatten()
        self.item_freq_counter = Counter(all_items)
        self.item_total = sum(self.item_freq_counter.values())
        logging.info(f"  Item freq: {len(self.item_freq_counter)} unique items, {self.item_total} total exposures")
        
        # ========== 2. Combo 频率统计 (Top-1000) ==========
        combo_counter = Counter()
        for slate in dataset['slates'][:min(100000, n_transitions)]:
            combo_key = tuple(slate.tolist())
            combo_counter[combo_key] += 1
        
        self.combo_freq_counter = dict(combo_counter.most_common(1000))
        logging.info(f"  Combo freq: {len(self.combo_freq_counter)} top combos cached")

    def _create_environment(self):
        """创建环境 (使用与数据收集一致的参数)"""
        if self.verbose:
            logging.info("Creating environment...")

        # 🔥 应用环境参数覆盖 (用于测试不同配置)
        if self.env_param_override:
            if self.verbose:
                logging.info("⚠️  Applying environment parameter overrides:")
            for key, value in self.env_param_override.items():
                if key in self.env_config:
                    old_value = self.env_config[key]
                    self.env_config[key] = value
                    if self.verbose:
                        logging.info(f"  {key}: {old_value} → {value}")
                else:
                    logging.warning(f"  Unknown parameter: {key}")

        # 创建空的buffer (评估时不需要)
        buffer = BufferDataModule(
            offline_data=[],
            batch_size=1,
            capacity=100,
            device=self.device
        )

        # 创建环境包装器
        self.env = EnvWrapper(
            buffer=buffer,
            env_name=self.env_config["env_name"],
            click_model=self.env_config["click_model"],
            diversity_penalty=self.env_config["diversity_penalty"],
            num_items=self.env_config["num_items"],
            episode_length=self.env_config["episode_length"],
            boredom_threshold=self.env_config["boredom_threshold"],
            recent_items_maxlen=self.env_config["recent_items_maxlen"],
            boredom_moving_window=self.env_config["boredom_moving_window"],
            env_omega=self.env_config["env_omega"],
            short_term_boost=self.env_config["short_term_boost"],
            env_offset=self.env_config["env_offset"],
            env_slope=self.env_config["env_slope"],
            diversity_threshold=self.env_config["diversity_threshold"],
            topic_size=self.env_config["topic_size"],
            num_topics=self.env_config["num_topics"],
            # RecSim基类必需参数
            rec_size=10,  # slate大小
            dataset_name=self.env_config["dataset_name"],
            sim_seed=self.seed,
            filename="",  # 评估时不需要保存文件
            # TopicRec额外必需参数
            env_alpha=1.0,
            env_propensities=[],
            env_embedds=self.env_config["item_embeddings"],  # 使用配置中的embeddings文件
            click_only_once=False,
            rel_threshold=None,
            prop_threshold=None,
            device=self.device,
            seed=self.seed
        )

        if self.verbose:
            logging.info(f"  Environment created: {self.env_config['env_name']}")

    def _load_item_embeddings(self):
        """加载Item Embeddings"""
        if self.verbose:
            logging.info("Loading item embeddings...")

        embeddings_path = paths.get_embeddings_path(
            self.env_config["item_embeddings"]
        )

        self.item_embeddings = ItemEmbeddings.from_scratch(
            self.env_config["num_items"],
            self.env_config["item_embedd_dim"],
            device=self.device
        )

        # 加载预训练的embeddings
        if embeddings_path.exists():
            loaded_data = torch.load(embeddings_path, map_location=self.device)
            # 检查加载的是Tensor还是state_dict
            if isinstance(loaded_data, torch.Tensor):
                # 如果是Tensor,直接赋值给weight
                self.item_embeddings.embedd.weight.data = loaded_data
            else:
                # 如果是state_dict,使用load_state_dict
                self.item_embeddings.load_state_dict(loaded_data)

            if self.verbose:
                logging.info(f"  Loaded embeddings from: {embeddings_path}")
        else:
            logging.warning(f"  Embeddings file not found: {embeddings_path}")

    def _load_ranker(self):
        """
        [DEPRECATED] Agents现在内部处理slate解码

        此方法保留用于向后兼容，但不再执行任何操作。
        """
        logging.warning(
            "⚠️  _load_ranker() is deprecated. "
            "Slate decoding is now handled by agents."
        )
        return None

    def evaluate_policy(
        self,
        agent,
        num_episodes: int = 100,
        deterministic: bool = True,
        noise_sigma: float = 0.0,  # 🔥 NEW: 噪声强度（0.0 表示无噪声）
        return_hamming_stats: bool = False  # 🔥 NEW: 是否返回汉明距离统计
    ) -> Dict[str, float]:
        """
        评估策略性能 (端到端模式)

        Args:
            agent: 离线 RL agent (BC/TD3+BC/CQL/IQL) - 必须是端到端架构
            num_episodes: 评估轮数
            deterministic: 是否使用确定性策略
            noise_sigma: 噪声强度（用于 Ranker 蝴蝶效应测试）
            return_hamming_stats: 是否返回汉明距离统计

        Returns:
            评估指标字典
        """
        if self.verbose:
            logging.info(f"Evaluating policy for {num_episodes} episodes (E2E mode)...")
            logging.info(f"Using Ranker: {self.ranker_checkpoint_path}")
            if noise_sigma > 0:
                logging.info(f"🔬 Perturbation Test Mode: σ={noise_sigma}")

        episode_rewards = []
        episode_lengths = []
        hamming_distances = []  # 🔥 记录汉明距离
        early_termination_count = 0  # 🔥 记录提前终止

        # =======================================================
        # 🔬 推荐位坍缩探针 (Policy Autopsy) 统计
        # =======================================================
        slate_unique_counts = []       # 每步 slate 的 unique item 数
        slate_full_repeat_count = 0    # 全部位置都相同 (unique==1) 的 slate 计数
        global_item_counter = Counter()  # 全评估过程 item 曝光频次计数
        total_slate_steps = 0          # 累计 slate 步数
        
        # 🔥 NEW: Slate 热门探针统计
        slate_list_for_probe = []  # 记录所有 slates 用于频率分析
        
        # 🔥 NEW: 归因探针统计 (Attribution Probe)
        expected_clicks_total = 0.0    # 期望点击总数
        actual_clicks_total = 0        # 实际点击总数
        bored_topic_count_total = 0    # Bored topic 累计
        bored_topic_count_steps = 0    # 统计 bored 的步数
        diversity_trigger_steps = 0    # 触发 diversity penalty 的步数
        repeat_exposure_ratio_total = 0.0  # 重复曝光比例累计
        repeat_exposure_steps = 0      # 统计重复曝光的步数

        # Penalty 自动探测
        detected_penalty_key = "none"
        penalty_total = 0.0
        penalty_steps = 0
        _penalty_hints = ["penalty", "divpen", "diversity", "reward_penalty"]  # 子串匹配关键词

        for episode in range(num_episodes):
            # 重置环境
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            # 重置 Agent 的 GRU hidden state
            if hasattr(agent, 'reset_hidden'):
                agent.reset_hidden()

            while not done:
                # =======================================================
                # 🔥 修复的逻辑：如果启用加噪，必须用 noisy_slate 与环境交互！
                # =======================================================
                if noise_sigma > 0 and hasattr(agent, 'actor') and hasattr(agent, 'ranker'):
                    with torch.no_grad():
                        # 1. 构造输入 tensor
                        slate_tensor = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
                        clicks_tensor = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)
                        obs_tensor = {"slate": slate_tensor, "clicks": clicks_tensor}

                        # 2. 获取 belief state
                        belief_state = agent.belief.forward(obs_tensor, done=False)["actor"]

                        # 3. Actor 预测（clean action）
                        raw_action, _ = agent.actor(belief_state, deterministic=True, need_log_prob=False)
                        clean_z = raw_action * agent.action_scale + agent.action_center

                        ranker_device = next(agent.ranker.parameters()).device
                        clean_z_device = clean_z.to(ranker_device)

                        # 4. 生成 noisy action
                        noisy_z = clean_z_device + torch.randn_like(clean_z_device) * noise_sigma

                        # 5. 分别解码出 clean 和 noisy 的 slate
                        clean_slate = agent.ranker.rank(clean_z_device.unsqueeze(0)).squeeze(0)
                        noisy_slate = agent.ranker.rank(noisy_z.unsqueeze(0)).squeeze(0)

                        # 6. 计算汉明距离并记录 (只计算，不影响环境)
                        hamming_dist = (clean_slate != noisy_slate).float().mean().item()
                        hamming_distances.append(hamming_dist)

                        # 7. 🔥 致命修复：必须使用 noisy_slate 与环境进行真实交互！
                        slate = noisy_slate.cpu().numpy()
                else:
                    # 正常模式 (无噪声)：直接使用 agent.act()
                    slate = agent.act(obs, deterministic=deterministic)

                # 转换为 tensor (如果 agent 返回 numpy array)
                if isinstance(slate, np.ndarray):
                    slate = torch.from_numpy(slate).long().to(self.device)

                # =======================================================
                # 🔬 Slate 坍缩探针统计 (在 env.step 前记录 slate 状态)
                # =======================================================
                # 统一转成一维 int list 再统计
                if isinstance(slate, torch.Tensor):
                    slate_list = slate.cpu().numpy().flatten().tolist()
                else:
                    slate_list = np.asarray(slate).flatten().tolist()

                # 统计 unique item 数
                unique_count = len(set(slate_list))
                slate_unique_counts.append(unique_count)

                # 统计全部位置都相同 (unique==1) 的 slate
                if unique_count == 1:
                    slate_full_repeat_count += 1

                # 累计 item 曝光频次
                for item_id in slate_list:
                    global_item_counter[item_id] += 1

                total_slate_steps += 1
                
                # 🔥 NEW: 记录 slate 用于后续频率分析
                slate_list_for_probe.append(slate_list)

                # 环境执行
                obs, reward, done, info = self.env.step(slate)

                # =======================================================
                # 🔬 归因探针累计 (Attribution Probe Accumulation)
                # =======================================================
                if isinstance(info, dict):
                    # 1. Expected/Actual Clicks
                    if "expected_clicks" in info:
                        expected_clicks_total += info["expected_clicks"]
                    if "actual_clicks" in info:
                        actual_clicks_total += info["actual_clicks"]
                    
                    # 2. Boredom 指标
                    if "bored_topic_count" in info:
                        bored_topic_count_total += info["bored_topic_count"]
                        bored_topic_count_steps += 1
                    
                    # 3. Diversity Penalty 指标
                    if "diversity_penalty_triggered" in info:
                        if info["diversity_penalty_triggered"]:
                            diversity_trigger_steps += 1
                    
                    # 4. Repeat Exposure 指标
                    if "repeat_exposure_ratio" in info:
                        repeat_exposure_ratio_total += info["repeat_exposure_ratio"]
                        repeat_exposure_steps += 1

                # =======================================================
                # 🔬 Penalty 字段自动探测 (从 info 中探测，子串匹配更稳健)
                # =======================================================
                if detected_penalty_key == "none" and isinstance(info, dict):
                    # 首次匹配：遍历 info.items()，子串匹配关键词
                    for key, val in info.items():
                        key_lower = key.lower()
                        if any(hint in key_lower for hint in _penalty_hints):
                            try:
                                float(val)
                                detected_penalty_key = key
                                break
                            except (TypeError, ValueError):
                                continue

                # 后续每步累计已锁定的 penalty key
                if detected_penalty_key != "none" and isinstance(info, dict):
                    if detected_penalty_key in info:
                        try:
                            penalty_val = float(info[detected_penalty_key])
                            penalty_total += penalty_val
                            penalty_steps += 1
                        except (TypeError, ValueError):
                            pass

                # 转换 reward 为 Python 标量
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()

                episode_reward += reward
                episode_length += 1

            # 记录提前终止 (Boredom trigger)
            if episode_length < self.env_config.get("episode_length", 50):
                early_termination_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        if episode_rewards:
            rewards = np.asarray(episode_rewards, dtype=np.float64)
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            median_reward = float(np.median(rewards))
            min_reward = float(np.min(rewards))
            max_reward = float(np.max(rewards))

            # IQM: mean of middle 50% rewards (25%~75%)
            if rewards.size >= 4:
                sorted_rewards = np.sort(rewards)
                low = int(sorted_rewards.size * 0.25)
                high = int(sorted_rewards.size * 0.75)
                if high <= low:
                    iqm_reward = mean_reward
                else:
                    iqm_reward = float(np.mean(sorted_rewards[low:high]))
            else:
                iqm_reward = mean_reward
        else:
            mean_reward = 0.0
            std_reward = 0.0
            median_reward = 0.0
            iqm_reward = 0.0
            min_reward = 0.0
            max_reward = 0.0

        # =======================================================
        # 🔬 全局集中度指标计算 (HHI, Gini)
        # =======================================================
        def _compute_hhi(counts: Counter) -> float:
            """计算 HHI 集中度指标：HHI = sum(p_i^2)"""
            total = sum(counts.values())
            if total == 0:
                return 0.0
            hhi = sum((count / total) ** 2 for count in counts.values())
            return float(hhi)

        def _compute_gini(counts: Counter) -> float:
            """计算 Gini 系数 (基于频次分布)"""
            if len(counts) == 0:
                return 0.0
            values = sorted(counts.values())
            n = len(values)
            total = sum(values)
            if total == 0:
                return 0.0
            # Gini = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
            # 使用简化公式
            cumsum = np.cumsum(values)
            gini = (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * total) / (n * total)
            return float(gini)

        # 计算 Slate 统计指标
        slate_unique_mean = float(np.mean(slate_unique_counts)) if slate_unique_counts else 0.0
        slate_full_repeat_rate = slate_full_repeat_count / total_slate_steps if total_slate_steps > 0 else 0.0
        global_unique_items = len(global_item_counter)
        global_hhi = _compute_hhi(global_item_counter)
        global_gini = _compute_gini(global_item_counter)

        # 计算 TopK exposure share
        if total_slate_steps > 0 and global_item_counter:
            total_exposure = sum(global_item_counter.values())
            sorted_items = sorted(global_item_counter.items(), key=lambda x: x[1], reverse=True)
            top1_count = sorted_items[0][1] if len(sorted_items) > 0 else 0
            top5_count = sum(count for _, count in sorted_items[:5])
            top1_exposure_share = top1_count / total_exposure
            top5_exposure_share = top5_count / total_exposure
            # 生成 top5_items_str: "118:13.7%,46:11.3%,345:7.4%,..."
            top5_items_str = ",".join(
                f"{item_id}:{count/total_exposure*100:.1f}%"
                for item_id, count in sorted_items[:5]
            )
        else:
            top1_exposure_share = 0.0
            top5_exposure_share = 0.0
            top5_items_str = ""

        # Penalty per step
        penalty_per_step = penalty_total / penalty_steps if penalty_steps > 0 else 0.0

        # 🔥 NEW: Slate 热门探针指标 (使用 dataset 频率字典)
        if self.item_freq_counter and slate_list_for_probe:
            # ========== Item 频率分位数 ==========
            item_percentiles = []
            for slate in slate_list_for_probe:
                for item_id in slate:
                    freq = self.item_freq_counter.get(item_id, 0)
                    percentile = freq / self.item_total * 100
                    item_percentiles.append(percentile)
            
            item_freq_percentile_mean = float(np.mean(item_percentiles)) if item_percentiles else 0.0
            item_freq_percentile_median = float(np.median(item_percentiles)) if item_percentiles else 0.0
            
            # ========== Combo 命中率 ==========
            combo_hits = 0
            combo_counter = Counter()
            
            for slate in slate_list_for_probe:
                combo_key = tuple(slate)
                combo_counter[combo_key] += 1
                
                if combo_key in self.combo_freq_counter:
                    combo_hits += 1
            
            combo_soft_hit_rate = combo_hits / len(slate_list_for_probe) if slate_list_for_probe else 0.0
            
            # Top1 combo 重复率
            combo_top1_repeat_share = 0.0
            if combo_counter and slate_list_for_probe:
                top1_combo, top1_count = combo_counter.most_common(1)[0]
                combo_top1_repeat_share = top1_count / len(slate_list_for_probe)
        else:
            item_freq_percentile_mean = 0.0
            item_freq_percentile_median = 0.0
            combo_soft_hit_rate = 0.0
            combo_top1_repeat_share = 0.0

        # 🔥 NEW: 归因探针指标计算 (Attribution Probe Metrics)
        expected_clicks_mean_per_step = expected_clicks_total / total_slate_steps if total_slate_steps > 0 else 0.0
        actual_clicks_mean_per_step = actual_clicks_total / total_slate_steps if total_slate_steps > 0 else 0.0
        click_efficiency_ratio = actual_clicks_mean_per_step / max(expected_clicks_mean_per_step, 1e-8)
        bored_topic_count_mean = bored_topic_count_total / bored_topic_count_steps if bored_topic_count_steps > 0 else 0.0
        diversity_penalty_trigger_rate = diversity_trigger_steps / total_slate_steps if total_slate_steps > 0 else 0.0
        repeat_exposure_ratio = repeat_exposure_ratio_total / repeat_exposure_steps if repeat_exposure_steps > 0 else 0.0

        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "median_reward": median_reward,
            "iqm_reward": iqm_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            # 🔬 推荐位坍缩探针指标
            "slate_unique_mean": slate_unique_mean,
            "slate_full_repeat_rate": slate_full_repeat_rate,
            "global_unique_items": global_unique_items,
            "global_hhi": global_hhi,
            "global_gini": global_gini,
            "top1_exposure_share": top1_exposure_share,
            "top5_exposure_share": top5_exposure_share,
            "penalty_per_step": penalty_per_step,
            "penalty_detected_key": detected_penalty_key,
            # 🔥 NEW: Slate 热门探针指标
            "item_freq_percentile_mean": item_freq_percentile_mean,
            "item_freq_percentile_median": item_freq_percentile_median,
            "combo_soft_hit_rate": combo_soft_hit_rate,
            "combo_top1_repeat_share": combo_top1_repeat_share,
            # 🔥 NEW: 归因探针指标 (Attribution Probe)
            "expected_clicks_mean": expected_clicks_mean_per_step,
            "actual_clicks_mean": actual_clicks_mean_per_step,
            "click_efficiency_ratio": click_efficiency_ratio,
            "bored_topic_count_mean": bored_topic_count_mean,
            "diversity_penalty_trigger_rate": diversity_penalty_trigger_rate,
            "repeat_exposure_ratio": repeat_exposure_ratio,
        }

        # 🔥 NEW: 添加提前终止率（无论是否加噪都计算）
        metrics["early_termination_rate"] = early_termination_count / num_episodes if num_episodes > 0 else 0.0

        # 🔥 NEW: 如果启用加噪测试，添加汉明距离统计
        if return_hamming_stats and noise_sigma > 0 and hamming_distances:
            metrics["hamming_distance_mean"] = float(np.mean(hamming_distances))
            metrics["hamming_distance_std"] = float(np.std(hamming_distances))

        # 🔬 推荐位坍缩探针日志 - 始终输出（无视 verbose 设置）
        logging.info(f"  Slate Probe: unique_mean={slate_unique_mean:.2f}, full_repeat_rate={slate_full_repeat_rate:.4f}, global_unique={global_unique_items}, hhi={global_hhi:.4f}, gini={global_gini:.4f}")
        logging.info(f"  Exposure Top5: {top5_items_str} | top1_share={top1_exposure_share:.2%}, top5_share={top5_exposure_share:.2%}")
        logging.info(f"  Penalty Probe: key={detected_penalty_key}, per_step={penalty_per_step:.4f}")
        if self.item_freq_counter:
            logging.info(f"  Slate Hot Probe: item_freq_pct_mean={item_freq_percentile_mean:.2f}%, combo_hit_rate={combo_soft_hit_rate:.2%}, combo_top1_repeat={combo_top1_repeat_share:.2%}")
        
        # 🔥 NEW: 归因探针日志 - 始终输出
        logging.info(f"  Attribution Probe: expected_clicks={expected_clicks_mean_per_step:.3f}, actual_clicks={actual_clicks_mean_per_step:.3f}, efficiency={click_efficiency_ratio:.2%}, bored_mean={bored_topic_count_mean:.2f}, div_trigger={diversity_penalty_trigger_rate:.2%}, repeat_ratio={repeat_exposure_ratio:.2%}")
        
        if self.verbose:
            logging.info(f"Evaluation completed ({num_episodes} episodes):")
            logging.info(f"  Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
            logging.info(f"  Median reward: {metrics['median_reward']:.4f}")
            logging.info(f"  IQM reward: {metrics['iqm_reward']:.4f}")
            # 🔥 NEW: 如果有加噪测试结果，也打印出来
            if noise_sigma > 0 and hamming_distances:
                logging.info(f"  Hamming Distance: {metrics.get('hamming_distance_mean', 0):.4f} ± {metrics.get('hamming_distance_std', 0):.4f}")
                logging.info(f"  Early Termination Rate: {metrics['early_termination_rate']:.1%}")

        return metrics




