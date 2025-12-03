"""
GeMS环境的Gym包装器
用于离线RL算法的在线评估
注意：这个环境主要用于评估，训练时直接使用离线数据
"""
import gym
import numpy as np
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加GeMS路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # 指向official_code
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GemsGymEnv(gym.Env):
    """
    GeMS环境包装成标准Gym接口

    注意：由于GeMS使用latent action (32维)，评估时需要GeMS的ranker来解码
    如果没有ranker，将使用简化的评估方式
    """

    metadata = {'render.modes': []}

    def __init__(self, env_name: str, use_ranker: bool = False):
        """
        Args:
            env_name: 环境名称 (diffuse_topdown, diffuse_mix, diffuse_divpen)
            use_ranker: 是否使用GeMS ranker进行解码（需要加载模型）
        """
        super().__init__()

        self.env_name = env_name
        self.use_ranker = use_ranker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 临时切换目录以加载GeMS模块
        cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            from offline_data_collection.environment_factory import EnvironmentFactory
            from offline_data_collection.model_loader import ModelLoader

            # 创建环境
            self.env_factory = EnvironmentFactory()
            self.env = self.env_factory.create_environment(env_name)

            # 初始化ModelLoader
            self.model_loader = ModelLoader()

            # 加载belief encoder（评估时必需）
            try:
                self.belief_encoder = self.model_loader.load_belief_encoder(env_name)
                self.belief_encoder.eval()
                print(f"✅ Belief encoder loaded for {env_name}")
            except Exception as e:
                print(f"❌ Failed to load belief encoder: {e}")
                print(f"⚠️  Online evaluation will NOT work without belief encoder!")
                self.belief_encoder = None

            # 如果需要ranker，尝试加载
            if use_ranker:
                try:
                    # 加载GeMS ranker（用于将latent action解码为slate）
                    self.ranker = self.model_loader.load_ranker(
                        env_name=env_name,
                        ranker_type="GeMS",  # 使用GeMS ranker
                        embedding_type="ideal"
                    )
                    self.ranker.eval()
                    print(f"✅ GeMS ranker loaded for {env_name}")
                except Exception as e:
                    print(f"❌ Failed to load ranker: {e}")
                    print(f"⚠️  Will use random slate for evaluation (NOT meaningful!)")
                    self.ranker = None
            else:
                self.ranker = None
                print(f"⚠️  Ranker not loaded. Online evaluation will use random slates.")

        finally:
            os.chdir(cwd)

        # 定义空间
        # Observation: belief state (20维)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )

        # Action: latent action (32维)
        self.action_space = gym.spaces.Box(
            low=-3.0,  # 根据SAC的action bounds
            high=3.0,
            shape=(32,),
            dtype=np.float32
        )

        # 内部状态
        self.current_obs = None
        self.step_count = 0
        self.episode_return = 0.0
        self.belief_state = None

    def reset(self) -> np.ndarray:
        """重置环境"""
        obs_tuple = self.env.reset()
        # RecSim环境返回(obs, info)
        if isinstance(obs_tuple, tuple):
            self.current_obs, _ = obs_tuple
        else:
            self.current_obs = obs_tuple

        self.step_count = 0
        self.episode_return = 0.0

        # 重置belief encoder的hidden state
        if self.belief_encoder is not None:
            # 重置GRU的hidden状态
            for module in self.belief_encoder.beliefs:
                self.belief_encoder.hidden[module] = torch.zeros(
                    1, 1, self.belief_encoder.hidden_dim,
                    device=self.belief_encoder.my_device
                )
            # 第一次调用belief encoder，将原始obs转换为belief state
            self.belief_state = self.belief_encoder.forward(self.current_obs)
        else:
            # 如果没有belief encoder，返回零向量
            self.belief_state = torch.zeros(20, device=self.device)

        # 返回belief state (numpy格式)
        return self.belief_state.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        Args:
            action: latent action (32维)

        Returns:
            observation, reward, done, info
        """
        # 将latent action解码成slate
        slate = self._decode_action(action)

        # 在环境中执行
        next_obs, reward, done, info = self.env.step(slate)

        # 更新belief state
        if self.belief_encoder is not None:
            # 使用belief encoder更新belief state
            next_belief_tensor = self.belief_encoder.forward(next_obs, done=done)
            if next_belief_tensor is None:
                # 如果done=True，belief encoder可能返回None
                next_belief_tensor = self.belief_state.clone()
            self.belief_state = next_belief_tensor
        else:
            # 如果没有belief encoder，使用零向量
            self.belief_state = torch.zeros(20, device=self.device)

        # 更新状态
        self.current_obs = next_obs
        self.step_count += 1
        self.episode_return += reward

        # 转换为numpy
        next_belief = self.belief_state.cpu().detach().numpy()

        # 添加额外信息
        info['episode_return'] = self.episode_return
        info['step_count'] = self.step_count

        return next_belief, reward, done, info

    def _extract_belief_state(self, obs: Any) -> np.ndarray:
        """
        从observation提取belief state

        注意：此方法已废弃，belief state的更新在reset()和step()中直接处理
        保留此方法仅为兼容性
        """
        if self.belief_state is not None:
            return self.belief_state.cpu().detach().numpy()
        else:
            return np.zeros(20, dtype=np.float32)

    def _decode_action(self, latent_action: np.ndarray) -> torch.Tensor:
        """
        将latent action解码成slate

        Args:
            latent_action: 32维连续latent action (numpy array)

        Returns:
            slate: 10个物品ID的tensor
        """
        if self.ranker is not None:
            # 使用GeMS ranker解码
            try:
                with torch.no_grad():
                    # 转换为tensor
                    latent_tensor = torch.FloatTensor(latent_action).to(self.device)
                    # 调用ranker的rank方法（与collect_data.py一致）
                    slate = self.ranker.rank(latent_tensor)
                    return slate
            except Exception as e:
                print(f"❌ Error decoding action with ranker: {e}")
                print(f"⚠️  Falling back to random slate")

        # Fallback: 随机选择slate
        # 警告：这会导致评估结果完全无意义！
        num_items = 1000  # GeMS环境固定1000个物品
        slate_size = 10
        slate_list = list(np.random.choice(num_items, size=slate_size, replace=False))
        slate = torch.tensor(slate_list, device=self.device)
        return slate

    def seed(self, seed: int = None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def render(self, mode='human'):
        """渲染（不实现）"""
        pass

    def close(self):
        """关闭环境"""
        pass


def wrap_env(
    env: gym.Env,
    state_mean: np.ndarray = 0.0,
    state_std: np.ndarray = 1.0,
) -> gym.Env:
    """
    包装环境以进行状态归一化

    Args:
        env: 原始环境
        state_mean: 状态均值
        state_std: 状态标准差

    Returns:
        包装后的环境
    """
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env
