"""
Replay Buffer for offline RL
不依赖d4rl，直接加载GeMS数据集
"""
import torch
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

class ReplayBuffer:
    """Replay buffer for offline RL training"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """
        加载D4RL格式的数据集（支持新旧两种格式）

        新格式（V4重构）：
            - slates: (N, rec_size) 原始推荐slate
            - clicks: (N, rec_size) 用户点击
            - next_slates: (N, rec_size) 下一个slate
            - next_clicks: (N, rec_size) 下一个点击
            - rewards, terminals

        旧格式（向后兼容）：
            - observations: (N, state_dim) 预编码的belief state
            - actions: (N, action_dim) 预编码的latent action
            - next_observations: (N, state_dim)
            - rewards, terminals

        ⚠️ 语义变化：
            新格式下，self._states 存储的是 slates（离散ID），而非预编码状态
            新格式下，self._actions 存储的是 clicks（0/1），而非latent action
            使用方算法需要在训练时实时编码/推断
        """
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        # 🔥 检测数据格式
        if 'slates' in data:
            # ========== 新格式（V4重构）==========
            print("🔥 检测到新格式数据集（slates + clicks）")

            n_transitions = data["slates"].shape[0]
            if n_transitions > self._buffer_size:
                raise ValueError(
                    f"Replay buffer is smaller than the dataset! "
                    f"Buffer size: {self._buffer_size}, Dataset size: {n_transitions}"
                )

            # 存储原始 slates 和 clicks（不再是预编码的状态/动作）
            self._states[:n_transitions] = self._to_tensor(data["slates"])
            self._actions[:n_transitions] = self._to_tensor(data["clicks"])
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
            self._next_states[:n_transitions] = self._to_tensor(data["next_slates"])
            # Note: next_clicks 可以选择性存储，这里暂不存储（如需要可扩展）
            self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

            print(f"✅ 新格式数据集加载成功: {n_transitions} transitions")
            print(f"   States = slates (shape: {data['slates'].shape})")
            print(f"   Actions = clicks (shape: {data['clicks'].shape})")

        else:
            # ========== 旧格式（向后兼容）==========
            print("⚠️  检测到旧格式数据集（observations + actions）")

            n_transitions = data["observations"].shape[0]
            if n_transitions > self._buffer_size:
                raise ValueError(
                    f"Replay buffer is smaller than the dataset! "
                    f"Buffer size: {self._buffer_size}, Dataset size: {n_transitions}"
                )

            self._states[:n_transitions] = self._to_tensor(data["observations"])
            self._actions[:n_transitions] = self._to_tensor(data["actions"])
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
            self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
            self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

            print(f"✅ 旧格式数据集加载成功: {n_transitions} transitions")

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        采样一个batch的数据

        Returns:
            [states, actions, rewards, next_states, dones]
        """
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def normalize_states(self, mean: np.ndarray, std: np.ndarray):
        """
        对状态进行归一化

        Args:
            mean: 状态均值
            std: 状态标准差
        """
        mean = self._to_tensor(mean)
        std = self._to_tensor(std)
        self._states = (self._states - mean) / std
        self._next_states = (self._next_states - mean) / std
        print(f"States normalized with mean shape: {mean.shape}, std shape: {std.shape}")

    def normalize_rewards(self, mean: float = None, std: float = None):
        """
        对奖励进行归一化

        Args:
            mean: 奖励均值（如果为None，则自动计算）
            std: 奖励标准差（如果为None，则自动计算）
        """
        rewards = self._rewards[:self._size]
        if mean is None:
            mean = rewards.mean().item()
        if std is None:
            std = rewards.std().item()
            std = max(std, 1e-6)  # 防止除零

        self._rewards = (self._rewards - mean) / std
        print(f"Rewards normalized: mean={mean:.4f}, std={std:.4f}")
        return mean, std

    def scale_rewards(self, scale: float = 1.0):
        """
        缩放奖励

        Args:
            scale: 缩放因子
        """
        self._rewards = self._rewards * scale
        print(f"Rewards scaled by {scale}")

    def normalize_actions(self, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对动作进行 Min-Max 归一化到 [-1, 1] 范围
        使用与在线 SAC 训练一致的归一化方式

        公式 (与 online.py 第 478-480 行一致):
            action_min = min(actions)
            action_scale = (max(actions) - min(actions)) / 2
            action_center = action_min + action_scale
            action_normalized = (action - action_center) / action_scale

        Returns:
            action_center: 归一化中心点 (用于反归一化)
            action_scale: 缩放比例 (用于反归一化)
        """
        actions = self._actions[:self._size]

        # 计算每个维度的 min 和 max
        action_min = actions.min(dim=0)[0]
        action_max = actions.max(dim=0)[0]

        # 计算 center 和 scale (与在线训练一致)
        action_scale = (action_max - action_min) / 2 + eps
        action_center = action_min + action_scale

        # 执行归一化
        self._actions[:self._size] = (self._actions[:self._size] - action_center) / action_scale

        print(f"Actions normalized to [-1, 1]")
        print(f"  Original range: [{action_min.min().item():.4f}, {action_max.max().item():.4f}]")
        print(f"  Action center shape: {action_center.shape}")
        print(f"  Action scale shape: {action_scale.shape}")

        return action_center, action_scale


@dataclass
class TrajectoryBatch:
    """
    Trajectory batch for RNN-based agents

    🔥 V4重构：添加 next_obs，actions 改为可选

    Attributes:
        obs: Dict with keys 'slate' and 'clicks', each containing List[Tensor]
             - slate: List of [seq_len, rec_size] tensors
             - clicks: List of [seq_len, rec_size] tensors
        next_obs: Dict with keys 'slate' and 'clicks' for next observations
        actions: List of [seq_len, action_dim] tensors (optional, 新格式不再使用)
        rewards: List of [seq_len, 1] tensors (optional, for value-based methods)
        dones: List of [seq_len, 1] tensors (optional, for value-based methods)
    """
    obs: Dict[str, List[torch.Tensor]]
    next_obs: Dict[str, List[torch.Tensor]] = None  # 🔥 [FIX] 添加
    actions: List[torch.Tensor] = None  # 🔥 [FIX] 改为可选
    rewards: List[torch.Tensor] = None
    dones: List[torch.Tensor] = None


class TrajectoryReplayBuffer:
    """
    Trajectory-based replay buffer for RNN agents
    Stores complete episodes and samples trajectories for end-to-end training
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize trajectory replay buffer

        Args:
            device: Device to store tensors on
        """
        self._device = device
        self._trajectories = []  # List[Dict] with keys: 'slate', 'clicks', 'action', 'reward', 'done'
        self._num_episodes = 0
        self._num_transitions = 0
        self._action_center = None
        self._action_scale = None

    def normalize_actions(self, data: Dict[str, np.ndarray], eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对动作进行 Min-Max 归一化到 [-1, 1] 范围
        使用正确的公式: action_center = (action_max + action_min) / 2

        Args:
            data: 包含 'actions' 的字典
            eps: 防止除零的小常数

        Returns:
            action_center: 归一化中心点 (用于反归一化)
            action_scale: 缩放比例 (用于反归一化)
        """
        all_actions = torch.tensor(data["actions"], dtype=torch.float32, device=self._device)

        # 计算每个维度的 min 和 max
        action_min = all_actions.min(dim=0)[0]
        action_max = all_actions.max(dim=0)[0]

        # 使用正确的公式计算 center 和 scale
        action_center = (action_max + action_min) / 2
        action_scale = (action_max - action_min) / 2 + eps

        # 归一化所有动作
        normalized_actions = (all_actions - action_center) / action_scale

        print(f"Actions normalized to [-1, 1]")
        print(f"  Original range: [{action_min.min().item():.4f}, {action_max.max().item():.4f}]")
        print(f"  Normalized range: [{normalized_actions.min().item():.4f}, {normalized_actions.max().item():.4f}]")
        print(f"  Action center shape: {action_center.shape}")
        print(f"  Action scale shape: {action_scale.shape}")

        return action_center, action_scale, normalized_actions

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """
        加载D4RL格式的数据集，按episode_ids分割成trajectories

        🔥 V4重构：不再依赖预编码的 actions 字段
        - 只加载原始数据：slates, clicks, next_slates, next_clicks
        - 动作归一化由外部（TD3初始化时）计算

        Args:
            data: 包含以下字段的字典:
                - episode_ids: (N,) episode标识
                - slates: (N, rec_size) 推荐slate
                - clicks: (N, rec_size) 用户点击
                - next_slates: (N, rec_size) 下一个slate
                - next_clicks: (N, rec_size) 下一个点击
                - rewards: (N,) 奖励（可选）
                - terminals: (N,) 终止标志（可选）
        """
        if self._trajectories:
            raise ValueError("Trying to load data into non-empty replay buffer")

        # 🔥 [FIX] 移除动作归一化逻辑（改由外部计算）
        # self._action_center 和 self._action_scale 将由外部传入

        # 转换为tensor
        episode_ids = data["episode_ids"]
        slates = torch.tensor(data["slates"], dtype=torch.long, device=self._device)
        clicks = torch.tensor(data["clicks"], dtype=torch.float32, device=self._device)

        # 🔥 [FIX] 读取 next_slates 和 next_clicks
        next_slates = torch.tensor(data["next_slates"], dtype=torch.long, device=self._device)
        next_clicks = torch.tensor(data["next_clicks"], dtype=torch.float32, device=self._device)

        # 可选字段
        if "rewards" in data:
            rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self._device)
        else:
            rewards = None

        if "terminals" in data:
            dones = torch.tensor(data["terminals"], dtype=torch.float32, device=self._device)
        else:
            dones = None

        # 按episode_ids分割成trajectories
        print("Splitting into trajectories...")
        unique_episode_ids = np.unique(episode_ids)
        self._num_episodes = len(unique_episode_ids)

        for ep_id in unique_episode_ids:
            # 找到属于当前episode的所有transition
            mask = episode_ids == ep_id
            indices = np.where(mask)[0]

            # 提取当前episode的数据
            ep_slates = slates[indices]
            ep_clicks = clicks[indices]
            # 🔥 [FIX] 提取 next_slates 和 next_clicks
            ep_next_slates = next_slates[indices]
            ep_next_clicks = next_clicks[indices]

            trajectory = {
                "slate": ep_slates,
                "clicks": ep_clicks,
                "next_slate": ep_next_slates,  # 🔥 [FIX] 添加
                "next_clicks": ep_next_clicks,  # 🔥 [FIX] 添加
                # "action": ep_actions,  # 🔥 [FIX] 删除，数据集中没有 action 了
            }

            if rewards is not None:
                trajectory["reward"] = rewards[indices]
            if dones is not None:
                trajectory["done"] = dones[indices]

            self._trajectories.append(trajectory)
            self._num_transitions += len(indices)

        print(f"Dataset loaded: {self._num_episodes} episodes, {self._num_transitions} transitions")
        print(f"Average episode length: {self._num_transitions / self._num_episodes:.1f}")

    def sample(self, batch_size: int) -> TrajectoryBatch:
        """
        采样一个batch的trajectories

        🔥 V4重构：返回 next_slate 和 next_clicks，不再返回 actions

        Args:
            batch_size: 要采样的episode数量

        Returns:
            TrajectoryBatch对象，包含:
                - obs: Dict with 'slate' and 'clicks' as List[Tensor]
                - next_obs: Dict with 'next_slate' and 'next_clicks' as List[Tensor]
                - rewards: List[Tensor] (if available)
                - dones: List[Tensor] (if available)
        """
        # 随机采样episode indices
        indices = np.random.randint(0, self._num_episodes, size=batch_size)

        # 收集数据
        slates_list = []
        clicks_list = []
        next_slates_list = []  # 🔥 [FIX] 添加
        next_clicks_list = []  # 🔥 [FIX] 添加
        rewards_list = []
        dones_list = []

        for idx in indices:
            traj = self._trajectories[idx]
            slates_list.append(traj["slate"])
            clicks_list.append(traj["clicks"])

            # 🔥 [FIX] 收集 next 数据
            next_slates_list.append(traj["next_slate"])
            next_clicks_list.append(traj["next_clicks"])

            # actions_list.append(traj["action"])  # 🔥 [FIX] 删除

            if "reward" in traj:
                rewards_list.append(traj["reward"].unsqueeze(-1))  # [seq_len, 1]
            if "done" in traj:
                dones_list.append(traj["done"].unsqueeze(-1))  # [seq_len, 1]

        # 构造batch
        obs = {
            "slate": slates_list,
            "clicks": clicks_list,
        }

        # 🔥 [FIX] 构造 next_obs
        next_obs = {
            "slate": next_slates_list,
            "clicks": next_clicks_list,
        }

        batch = TrajectoryBatch(
            obs=obs,
            next_obs=next_obs,  # 🔥 [FIX] 添加 next_obs
            actions=None,  # 🔥 [FIX] 不再返回 actions
            rewards=rewards_list if rewards_list else None,
            dones=dones_list if dones_list else None,
        )

        return batch

    def get_action_normalization_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作归一化参数（用于Agent初始化）

        Returns:
            action_center: 归一化中心点
            action_scale: 缩放比例
        """
        if self._action_center is None or self._action_scale is None:
            raise ValueError("Action normalization parameters not available. Call load_d4rl_dataset first.")
        return self._action_center, self._action_scale

    def __len__(self) -> int:
        """返回episode数量"""
        return self._num_episodes
