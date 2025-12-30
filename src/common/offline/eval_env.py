"""
离线RL评估环境封装

此模块封装完整的评估流程,确保评估环境参数与数据收集时一致。
包括:
- 环境创建
- Ranker (GeMS VAE) 加载
- Belief Encoder (GRU) 加载
- 完整推理流程: Agent → Ranker → Slate → Environment
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
from envs.RecSim.simulators import TopicRec
from belief_encoders.gru_belief import BeliefEncoder, GRUBelief
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
        device: str = "cuda",
        seed: int = 58407201,
        verbose: bool = True
    ):
        """
        初始化离线评估环境

        Args:
            env_name: 环境名称 (如 diffuse_mix)
            device: 设备
            seed: 随机种子
            verbose: 是否打印详细信息
        """
        self.env_name = env_name
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
        self.belief_encoder = None
        self.item_embeddings = None

        # 创建环境
        self._create_environment()

        # 加载Item Embeddings
        self._load_item_embeddings()

        # 加载Ranker
        self._load_ranker()

        # 加载Belief Encoder
        self._load_belief_encoder()

        if self.verbose:
            logging.info("OfflineEvalEnv initialization completed")

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
        """加载Ranker (GeMS VAE)"""
        if self.verbose:
            logging.info("Loading Ranker (GeMS)...")

        # 构建checkpoint路径 (PROJECT_ROOT已经是src目录,需要回到项目根目录)
        project_root = PROJECT_ROOT.parent
        data_collection_models_dir = project_root / "src" / "data_collection" / "offline_data_collection" / "models"
        sac_gems_ckpt_path = data_collection_models_dir / self.env_config["sac_gems_checkpoint"]

        if not sac_gems_ckpt_path.exists():
            raise FileNotFoundError(f"SAC+GeMS checkpoint not found: {sac_gems_ckpt_path}")

        # 加载checkpoint
        checkpoint = torch.load(sac_gems_ckpt_path, map_location=self.device)
        state_dict = checkpoint['state_dict']

        # 创建GeMS Ranker (使用与checkpoint匹配的hidden layers配置)
        self.ranker = GeMS(
            item_embeddings=self.item_embeddings,
            latent_dim=self.env_config["latent_dim"],
            belief_state_dim=self.env_config["belief_state_dim"],
            item_embedd_dim=self.env_config["item_embedd_dim"],
            rec_size=10,  # slate大小
            hidden_layers_infer=[512, 256],  # 从checkpoint推断出的配置
            hidden_layers_decoder=[256, 512],  # 从checkpoint推断出的配置
            lambda_click=0.5,
            lambda_KL=1.0,
            lambda_prior=0.0,
            ranker_lr=0.001,
            fixed_embedds=False,
            ranker_sample=False,
            device=self.device
        )

        # 提取ranker相关的权重
        ranker_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ranker.'):
                new_key = key.replace('ranker.', '')
                ranker_state_dict[new_key] = value

        # 加载权重
        self.ranker.load_state_dict(ranker_state_dict)
        self.ranker.eval()

        # 确保所有参数都在正确的设备上
        self.ranker = self.ranker.to(self.device)

        if self.verbose:
            logging.info(f"  Ranker loaded from: {sac_gems_ckpt_path}")

    def _load_belief_encoder(self):
        """加载Belief Encoder (GRU)"""
        if self.verbose:
            logging.info("Loading Belief Encoder (GRU)...")

        # 构建checkpoint路径 (PROJECT_ROOT已经是src目录,需要回到项目根目录)
        project_root = PROJECT_ROOT.parent
        data_collection_models_dir = project_root / "src" / "data_collection" / "offline_data_collection" / "models"
        sac_gems_ckpt_path = data_collection_models_dir / self.env_config["sac_gems_checkpoint"]

        # 加载checkpoint
        checkpoint = torch.load(sac_gems_ckpt_path, map_location=self.device)
        state_dict = checkpoint['state_dict']

        # 创建Belief Encoder
        self.belief_encoder = GRUBelief(
            belief_state_dim=self.env_config["belief_state_dim"],
            hidden_dim=self.env_config["belief_state_dim"],
            input_dim=10 * (self.env_config["item_embedd_dim"] + 1),  # rec_size * (item_embedd_dim + 1)
            item_embeddings=self.item_embeddings,
            item_embedd_dim=self.env_config["item_embedd_dim"],
            rec_size=10,
            ranker=self.ranker,
            belief_lr=0.001,
            hidden_layers_reduction=[],
            beliefs=["actor"],  # 只需要actor的GRU
            device=self.device
        )

        # 提取Belief Encoder的权重 (只加载actor相关的权重,过滤掉critic)
        belief_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('belief.'):
                # 只移除 'belief.' 前缀,保留完整的嵌套结构
                new_key = key.replace('belief.', '', 1)
                # 过滤掉critic相关的权重 (我们只需要actor)
                if 'critic' not in new_key:
                    belief_state_dict[new_key] = value

        # 加载权重
        self.belief_encoder.load_state_dict(belief_state_dict)
        self.belief_encoder.eval()

        # 确保所有参数都在正确的设备上
        self.belief_encoder = self.belief_encoder.to(self.device)

        if self.verbose:
            logging.info(f"  Belief Encoder loaded from: {sac_gems_ckpt_path}")

    def evaluate_policy(
        self,
        agent,
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        评估策略性能

        Args:
            agent: 离线RL agent (BC/TD3+BC/CQL/IQL)
            num_episodes: 评估轮数
            deterministic: 是否使用确定性策略

        Returns:
            评估指标字典
        """
        if self.verbose:
            logging.info(f"Evaluating policy for {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            # 重置环境
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            # 重置GRU hidden state
            if hasattr(self.belief_encoder, 'reset_hidden'):
                self.belief_encoder.reset_hidden()

            while not done:
                # 1. 将原始obs转换为belief_state (使用GRU)
                with torch.no_grad():
                    belief_state = self.belief_encoder.forward(obs, done=False)

                # 2. Agent输出latent_action (32维)
                # belief_state是tensor,需要转为numpy给agent
                belief_state_np = belief_state.cpu().numpy()
                latent_action = agent.act(belief_state_np, deterministic=deterministic)

                # 3. 将latent_action转为tensor给Ranker (添加batch维度)
                latent_action_tensor = torch.FloatTensor(latent_action).unsqueeze(0).to(self.device)

                # 4. Ranker解码为slate
                with torch.no_grad():
                    slate = self.ranker.rank(latent_action_tensor).squeeze(0)

                # 5. 环境执行
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





