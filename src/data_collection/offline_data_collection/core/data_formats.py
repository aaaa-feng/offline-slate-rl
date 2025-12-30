#!/usr/bin/env python3
"""
离线数据收集的数据格式定义
支持D4RL标准格式和slate推荐特有格式
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

@dataclass
class SlateObservation:
    """Slate推荐的观察数据结构"""
    user_obs: Optional[torch.Tensor] = None      # 用户特征
    item_obs: Optional[torch.Tensor] = None      # 物品特征  
    belief_state: Optional[torch.Tensor] = None  # GRU编码的信念状态
    raw_obs: Optional[Dict] = None               # 原始观察数据

@dataclass
class SlateAction:
    """Slate推荐的动作数据结构"""
    discrete_slate: List[int]                    # 离散slate (物品ID列表)
    latent_action: Optional[torch.Tensor] = None # GeMS的latent action (如果有)
    slate_embedding: Optional[torch.Tensor] = None # Slate embedding (如果有)

@dataclass
class SlateInfo:
    """Slate推荐的额外信息"""
    clicks: torch.Tensor                         # 用户点击 [0,1,0,1,...]
    diversity_score: float                       # 多样性分数
    coverage_score: float                        # 覆盖率分数
    episode_return: float                        # 累积奖励
    episode_id: int                              # 轨迹ID
    timestep: int                                # 时间步
    item_relevances: Optional[torch.Tensor] = None  # Oracle信息：物品相关性 (num_items,)

@dataclass
class SlateTransition:
    """单个转移的完整数据"""
    observation: SlateObservation
    action: SlateAction
    reward: float
    next_observation: SlateObservation
    done: bool
    info: SlateInfo

class SlateTrajectory:
    """完整轨迹数据"""
    def __init__(self):
        self.transitions: List[SlateTransition] = []
        self.episode_id: int = 0
        self.episode_return: float = 0.0
        self.episode_length: int = 0
    
    def add_transition(self, transition: SlateTransition):
        """添加转移"""
        self.transitions.append(transition)
        self.episode_length += 1
        self.episode_return += transition.reward
    
    def get_length(self) -> int:
        return len(self.transitions)
    
    def get_return(self) -> float:
        return sum(t.reward for t in self.transitions)

class SlateDataset:
    """Slate推荐数据集"""
    def __init__(self, name: str = "slate_dataset"):
        self.name = name
        self.trajectories: List[SlateTrajectory] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_trajectory(self, trajectory: SlateTrajectory):
        """添加轨迹"""
        trajectory.episode_id = len(self.trajectories)
        self.trajectories.append(trajectory)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.trajectories:
            return {}
        
        episode_lengths = [traj.get_length() for traj in self.trajectories]
        episode_returns = [traj.get_return() for traj in self.trajectories]
        total_transitions = sum(episode_lengths)
        
        return {
            'num_episodes': len(self.trajectories),
            'total_transitions': total_transitions,
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'avg_episode_return': np.mean(episode_returns),
            'std_episode_return': np.std(episode_returns),
            'min_episode_return': np.min(episode_returns),
            'max_episode_return': np.max(episode_returns),
        }
    
    def to_d4rl_format(self) -> Dict[str, np.ndarray]:
        """转换为D4RL标准格式"""
        if not self.trajectories:
            return {}
        
        # 收集所有转移
        all_transitions = []
        for traj in self.trajectories:
            all_transitions.extend(traj.transitions)
        
        if not all_transitions:
            return {}
        
        # 提取数据
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        timeouts = []
        
        # Slate推荐特有字段
        slates = []
        clicks = []
        diversity_scores = []
        coverage_scores = []
        episode_ids = []
        timesteps = []

        # V2新增字段（向后兼容）
        raw_observations = []
        raw_next_observations = []
        user_states = []
        user_bored = []
        item_relevances = []  # V3新增：Oracle信息

        for i, transition in enumerate(all_transitions):
            # 标准D4RL字段
            obs = transition.observation
            next_obs = transition.next_observation
            
            # 使用belief_state作为主要观察
            if obs.belief_state is not None:
                observations.append(obs.belief_state.cpu().numpy())
            else:
                # 如果没有belief_state，使用零向量占位
                observations.append(np.zeros(32))  # 假设belief_state_dim=32
            
            if next_obs.belief_state is not None:
                next_observations.append(next_obs.belief_state.cpu().numpy())
            else:
                next_observations.append(np.zeros(32))
            
            # 动作优先使用latent_action（连续动作），如果没有则使用slate（离散动作）
            if transition.action.latent_action is not None:
                # 使用连续latent action（用于TD3+BC等离线RL算法）
                actions.append(transition.action.latent_action.cpu().numpy())
            else:
                # 降级方案：使用离散slate（用于random数据或没有ranker的情况）
                slate_array = np.array(transition.action.discrete_slate)
                actions.append(slate_array)
            
            rewards.append(transition.reward)
            terminals.append(transition.done)
            timeouts.append(False)  # 假设没有timeout
            
            # Slate推荐特有字段
            slates.append(transition.action.discrete_slate)
            clicks.append(transition.info.clicks.cpu().numpy())
            diversity_scores.append(transition.info.diversity_score)
            coverage_scores.append(transition.info.coverage_score)
            episode_ids.append(transition.info.episode_id)
            timesteps.append(transition.info.timestep)

            # V2新增：提取raw_obs（向后兼容，静默处理None）
            if obs.raw_obs is not None:
                raw_observations.append(obs.raw_obs)
                # 提取用户状态
                if 'user' in obs.raw_obs and 'user_state' in obs.raw_obs['user']:
                    user_states.append(obs.raw_obs['user']['user_state'].cpu().numpy())
                    user_bored.append(obs.raw_obs['user']['bored'].cpu().numpy())
                else:
                    # 如果raw_obs存在但结构不完整，填充默认值
                    user_states.append(np.zeros(10))
                    user_bored.append(np.zeros(10, dtype=bool))
            else:
                # 旧数据或默认模式：填充None和默认值
                raw_observations.append(None)
                user_states.append(np.zeros(10))
                user_bored.append(np.zeros(10, dtype=bool))

            if next_obs.raw_obs is not None:
                raw_next_observations.append(next_obs.raw_obs)
            else:
                raw_next_observations.append(None)

            # V3新增：提取Oracle信息（向后兼容）
            if transition.info.item_relevances is not None:
                item_relevances.append(transition.info.item_relevances.cpu().numpy())
            else:
                # 兼容旧数据：填充零向量
                item_relevances.append(np.zeros(1000))  # 默认1000个物品
        
        # 转换为numpy数组
        d4rl_data = {
            # 标准D4RL字段
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_observations),
            'terminals': np.array(terminals),
            'timeouts': np.array(timeouts),

            # Slate推荐特有字段
            'slates': np.array(slates),
            'clicks': np.array(clicks),
            'diversity_scores': np.array(diversity_scores),
            'coverage_scores': np.array(coverage_scores),
            'episode_ids': np.array(episode_ids),
            'timesteps': np.array(timesteps),

            # V2新增字段（向后兼容）
            'raw_observations': np.array(raw_observations, dtype=object),
            'raw_next_observations': np.array(raw_next_observations, dtype=object),
            'user_states': np.array(user_states),
            'user_bored': np.array(user_bored),

            # V3新增字段（Oracle信息）
            'item_relevances': np.array(item_relevances),
        }
        
        return d4rl_data
    
    def save(self, filepath: str, format: str = 'pickle'):
        """保存数据集"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'd4rl':
            d4rl_data = self.to_d4rl_format()
            np.savez_compressed(filepath, **d4rl_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str, format: str = 'pickle'):
        """加载数据集"""
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'd4rl':
            data = np.load(filepath)
            # 这里可以实现从D4RL格式重建SlateDataset的逻辑
            # 暂时返回原始数据
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")

def create_empty_observation(belief_state_dim: int = 32) -> SlateObservation:
    """创建空的观察"""
    return SlateObservation(
        belief_state=torch.zeros(belief_state_dim)
    )

def create_empty_action(slate_size: int = 10) -> SlateAction:
    """创建空的动作"""
    return SlateAction(
        discrete_slate=[0] * slate_size
    )

def create_empty_info(slate_size: int = 10) -> SlateInfo:
    """创建空的信息"""
    return SlateInfo(
        clicks=torch.zeros(slate_size),
        diversity_score=0.0,
        coverage_score=0.0,
        episode_return=0.0,
        episode_id=0,
        timestep=0
    )

if __name__ == "__main__":
    # 测试数据格式
    print("测试Slate数据格式...")
    
    # 创建测试数据
    dataset = SlateDataset("test_dataset")
    
    # 创建测试轨迹
    trajectory = SlateTrajectory()
    
    for t in range(5):
        obs = SlateObservation(belief_state=torch.randn(32))
        action = SlateAction(discrete_slate=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        next_obs = SlateObservation(belief_state=torch.randn(32))
        info = SlateInfo(
            clicks=torch.randint(0, 2, (10,)),
            diversity_score=0.8,
            coverage_score=0.1,
            episode_return=10.0,
            episode_id=0,
            timestep=t
        )
        
        transition = SlateTransition(
            observation=obs,
            action=action,
            reward=2.0,
            next_observation=next_obs,
            done=(t == 4),
            info=info
        )
        
        trajectory.add_transition(transition)
    
    dataset.add_trajectory(trajectory)
    
    # 打印统计信息
    stats = dataset.get_stats()
    print("数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试D4RL格式转换
    d4rl_data = dataset.to_d4rl_format()
    print(f"\nD4RL格式数据形状:")
    for key, value in d4rl_data.items():
        print(f"  {key}: {value.shape}")
    
    print("✅ 数据格式测试完成!")
