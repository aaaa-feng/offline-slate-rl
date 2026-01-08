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
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from config import paths
from config.env_params import get_env_config

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
        verbose: bool = True
    ):
        """
        初始化离线评估环境

        Args:
            env_name: 环境名称 (如 diffuse_mix)
            dataset_quality: 数据集质量 (random/medium/expert)
            device: 设备
            seed: 随机种子
            verbose: 是否打印详细信息
        """
        self.env_name = env_name
        self.dataset_quality = dataset_quality
        self.device = device
        self.seed = seed
        self.verbose = verbose

        # 加载环境配置
        self.env_config = get_env_config(env_name)

        if self.verbose:
            logging.info(f"Initializing OfflineEvalEnv for {env_name}")
            logging.info(f"  Click model: {self.env_config['click_model']}")
            logging.info(f"  Diversity penalty: {self.env_config['diversity_penalty']}")
            logging.info(f"  Ranker dataset: {self.env_config['ranker_dataset']}")

        # 初始化组件
        self.env = None
        self.ranker = None
        self.item_embeddings = None
        self.ranker_checkpoint_path = None  # 用于日志输出

        # 创建环境
        self._create_environment()

        # 加载Item Embeddings
        self._load_item_embeddings()

        # 加载Ranker
        self._load_ranker()

        # 强制打印参数摘要 (无视 verbose 设置,确保可观测性)
        logging.info("")
        logging.info("=" * 70)
        logging.info("Offline Evaluation Environment Setup")
        logging.info("=" * 70)
        logging.info(f"Env Name:            {self.env_name}")
        logging.info(f"Dataset Quality:     {self.dataset_quality}")
        logging.info(f"Click Model:         {self.env_config['click_model']}")
        logging.info(f"Diversity Penalty:   {self.env_config['diversity_penalty']}")
        logging.info(f"Episode Length:      {self.env_config['episode_length']}")
        logging.info(f"Boredom Threshold:   {self.env_config['boredom_threshold']}")
        logging.info(f"Ranker Source:       {self.ranker_checkpoint_path}")
        logging.info("=" * 70)
        logging.info("✅ OfflineEvalEnv initialization completed (GeMS-only E2E mode)")
        logging.info("")

    def _create_environment(self):
        """创建环境 (使用与数据收集一致的参数)"""
        if self.verbose:
            logging.info("Creating environment...")

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
        """加载Ranker (GeMS VAE) - 直接从纯GeMS checkpoint加载"""
        if self.verbose:
            logging.info("Loading Ranker (GeMS)...")

        # 构建GeMS checkpoint路径
        project_root = PROJECT_ROOT.parent
        gems_ckpt_dir = project_root / "checkpoints" / "gems" / "offline"

        # 根据环境名称和数据集质量构建checkpoint文件名
        # 格式: GeMS_{env_name}_{quality}_latent32_beta1.0_click0.5_seed58407201.ckpt
        dataset_name = self.env_config["dataset_name"]
        gems_ckpt_name = f"GeMS_{dataset_name}_{self.dataset_quality}_latent32_beta1.0_click0.5_seed58407201.ckpt"
        gems_ckpt_path = gems_ckpt_dir / gems_ckpt_name

        if not gems_ckpt_path.exists():
            raise FileNotFoundError(f"GeMS checkpoint not found: {gems_ckpt_path}")

        # 保存路径用于日志输出
        self.ranker_checkpoint_path = str(gems_ckpt_path)

        if self.verbose:
            logging.info(f"  Loading Ranker from: {gems_ckpt_path}")

        # 使用GeMS.load_from_checkpoint标准方法加载
        self.ranker = GeMS.load_from_checkpoint(
            str(gems_ckpt_path),
            map_location=self.device,
            item_embeddings=self.item_embeddings,
            item_embedd_dim=self.env_config["item_embedd_dim"],
            device=self.device,
            rec_size=10,
            latent_dim=self.env_config["latent_dim"],
            lambda_click=0.5,
            lambda_KL=1.0,
            lambda_prior=1.0,
            ranker_lr=3e-3,
            fixed_embedds="scratch",
            ranker_sample=False,
            hidden_layers_infer=[512, 256],
            hidden_layers_decoder=[256, 512]
        )

        self.ranker.eval()
        self.ranker.freeze()

        # 显式强制设备同步 (不依赖 load_from_checkpoint 的自动管理)
        self.ranker = self.ranker.to(self.device)

        if self.verbose:
            logging.info(f"  ✅ GeMS Ranker loaded successfully")

    def evaluate_policy(
        self,
        agent,
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        评估策略性能 (端到端模式)

        Args:
            agent: 离线RL agent (BC/TD3+BC/CQL/IQL) - 必须是端到端架构
            num_episodes: 评估轮数
            deterministic: 是否使用确定性策略

        Returns:
            评估指标字典
        """
        if self.verbose:
            logging.info(f"Evaluating policy for {num_episodes} episodes (E2E mode)...")
            logging.info(f"Using Ranker: {self.ranker_checkpoint_path}")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            # 重置环境
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            # 重置Agent的GRU hidden state
            if hasattr(agent, 'reset_hidden'):
                agent.reset_hidden()

            while not done:
                # 端到端模式: Agent直接处理原始obs并输出latent_action
                latent_action = agent.act(obs, deterministic=deterministic)

                # 将latent_action转为tensor给Ranker (添加batch维度)
                latent_action_tensor = torch.FloatTensor(latent_action).unsqueeze(0).to(self.device)

                # 防御性检查: 确保 tensor 确实在正确的设备上
                if latent_action_tensor.device != self.ranker.device:
                    latent_action_tensor = latent_action_tensor.to(self.ranker.device)

                # Ranker解码为slate
                with torch.no_grad():
                    slate = self.ranker.rank(latent_action_tensor).squeeze(0)

                # 环境执行
                obs, reward, done, info = self.env.step(slate)

                # 转换reward为Python标量
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        metrics = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        }

        if self.verbose:
            logging.info(f"Evaluation completed:")
            logging.info(f"  Mean reward: {metrics['mean_reward']:.4f}")
            logging.info(f"  Std reward: {metrics['std_reward']:.4f}")

        return metrics





