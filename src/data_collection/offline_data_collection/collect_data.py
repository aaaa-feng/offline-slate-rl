#!/usr/bin/env python3
"""
主数据收集脚本
使用训练好的模型收集离线强化学习数据
"""
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import argparse
from datetime import datetime
from collections import Counter

# 添加父目录到路径以便导入core模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_formats import SlateDataset, SlateTrajectory, SlateTransition, SlateObservation, SlateAction, SlateInfo
from core.model_loader import ModelLoader
from core.environment_factory import EnvironmentFactory
from core.metrics import SlateMetrics, create_item_popularity_dict

# 导入新增的工具函数
from utils.merge_datasets import merge_datasets
from utils.analyze_quality import analyze_dataset_quality as analyze_quality_from_file

class OfflineDataCollector:
    """离线数据收集器"""
    
    def __init__(self, output_dir: str = None,
                 epsilon_greedy: float = 0.0,
                 epsilon_noise_scale: float = 1.0,
                 file_prefix: str = ""):
        # 动态设置默认输出目录
        if output_dir is None:
            # 使用统一路径配置
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / "config"))
            from paths import OFFLINE_DATASETS_DIR
            output_dir = str(OFFLINE_DATASETS_DIR)
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔥 新增：ε-greedy 噪声注入参数
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_noise_scale = epsilon_noise_scale
        self.file_prefix = file_prefix

        # 初始化组件
        self.model_loader = ModelLoader()
        self.env_factory = EnvironmentFactory()
        
        # 数据收集配置
        self.collection_config = {
            'expert': {
                'episodes': 10000,
                'description': 'Expert trajectories from best performing models'
            },
            'medium': {
                'episodes': 10000, 
                'description': 'Medium quality trajectories from decent models'
            },
            'random': {
                'episodes': 5000,
                'description': 'Random trajectories for baseline'
            }
        }
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def collect_trajectories_from_model(self, env_name: str, agent, ranker, belief_encoder,
                                      environment, num_episodes: int,
                                      quality_level: str = "expert",
                                      save_raw_obs: bool = False) -> SlateDataset:
        """
        使用指定模型收集轨迹数据

        Args:
            env_name: 环境名称
            agent: RL智能体
            ranker: 排序器
            belief_encoder: 信念编码器
            environment: 环境实例
            num_episodes: 收集的episode数量
            quality_level: 数据质量级别
            save_raw_obs: 是否保存原始obs (默认False，向后兼容)

        Returns:
            dataset: 收集的数据集
        """
        print(f"开始收集 {env_name} 环境的 {quality_level} 数据...")
        print(f"目标episodes: {num_episodes}")
        
        # 创建数据集
        dataset = SlateDataset(f"{env_name}_{quality_level}")
        dataset.metadata = {
            'env_name': env_name,
            'quality_level': quality_level,
            'agent_type': type(agent).__name__,
            'ranker_type': type(ranker).__name__ if ranker else 'None',
            'collection_time': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # 初始化指标计算器
        env_config = self.env_factory.get_env_config(env_name)
        
        # 加载物品embeddings用于指标计算
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root / "config"))
        from paths import get_embeddings_path
        item_embeddings_path = str(get_embeddings_path(env_config['env_embedds']))
        if os.path.exists(item_embeddings_path):
            item_embeddings = torch.load(item_embeddings_path, map_location=self.device)
        else:
            # 使用随机embeddings
            item_embeddings = torch.randn(env_config['num_items'], env_config['item_embedd_dim'])
        
        metrics_calculator = SlateMetrics(item_embeddings, env_config['num_items'])
        
        # 设置模型为评估模式 (如果模型不为None)
        if agent is not None:
            agent.eval()
        if ranker is not None:
            ranker.eval()
        if belief_encoder is not None:
            belief_encoder.eval()
        
        # 收集数据
        successful_episodes = 0
        failed_episodes = 0
        
        with torch.no_grad():
            for episode_idx in tqdm(range(num_episodes), desc=f"收集{quality_level}数据"):
                try:
                    trajectory = self._collect_single_episode(
                        environment, agent, ranker, belief_encoder,
                        metrics_calculator, episode_idx, quality_level, save_raw_obs
                    )
                    
                    if trajectory and len(trajectory.transitions) > 0:
                        dataset.add_trajectory(trajectory)
                        successful_episodes += 1
                    else:
                        failed_episodes += 1
                        
                except Exception as e:
                    print(f"Episode {episode_idx} 收集失败: {e}")
                    failed_episodes += 1
                    continue
                
                # 每1000个episode打印一次进度
                if (episode_idx + 1) % 1000 == 0:
                    stats = dataset.get_stats()
                    print(f"已完成 {episode_idx + 1}/{num_episodes} episodes")
                    print(f"  成功: {successful_episodes}, 失败: {failed_episodes}")
                    print(f"  平均episode长度: {stats.get('avg_episode_length', 0):.2f}")
                    print(f"  平均episode回报: {stats.get('avg_episode_return', 0):.2f}")
        
        print(f"数据收集完成!")
        print(f"  成功episodes: {successful_episodes}")
        print(f"  失败episodes: {failed_episodes}")
        print(f"  成功率: {successful_episodes/(successful_episodes+failed_episodes)*100:.2f}%")
        
        return dataset
    
    def _collect_single_episode(self, environment, agent, ranker, belief_encoder,
                               metrics_calculator, episode_id: int, quality_level: str = "expert",
                               save_raw_obs: bool = False) -> Optional[SlateTrajectory]:
        """
        收集单个episode的数据

        Returns:
            trajectory: 轨迹数据，如果失败返回None
        """
        try:
            # 重置环境
            obs, info = environment.reset()

            # V2新增：保存原始obs（开关控制）
            import copy
            raw_obs_before_encoding = copy.deepcopy(obs) if save_raw_obs else None

            # 初始化轨迹
            trajectory = SlateTrajectory()

            # 初始化信念状态 (如果有belief_encoder)
            # 关键修复：与训练代码一致，第一次调用belief.forward(obs)将原始obs转换为belief state
            if belief_encoder is not None:
                # 手动重置GRU的hidden状态
                for module in belief_encoder.beliefs:
                    belief_encoder.hidden[module] = torch.zeros(1, 1, belief_encoder.hidden_dim, device=belief_encoder.my_device)
                # 第一次调用：将原始obs转换为belief state
                obs = belief_encoder.forward(obs)

            episode_slates = []  # 用于计算覆盖率
            done = False
            timestep = 0

            # 🔥 修复：从环境获取最大步数，而不是硬编码 100
            # RecSim 环境中使用 self.H 存储 episode_length
            max_steps = getattr(environment, 'H', 100)

            while not done and timestep < max_steps:
                # 🔥 修复：处理 Actor-Critic 分离的 Belief State
                current_belief_state = None

                # 🔥 修复：Random 策略不需要 belief_state，跳过提取
                if quality_level == "random":
                    # Random 策略直接使用原始 obs，不需要提取 belief_state
                    current_belief_state = None
                elif isinstance(obs, torch.Tensor):
                    # obs 是单一的 tensor
                    current_belief_state = obs.clone().detach()
                elif isinstance(obs, dict):
                    # obs 是字典，可能是 Actor-Critic 分离的 Belief State
                    if 'actor' in obs:
                        # 提取 actor 部分给 Agent 使用
                        current_belief_state = obs['actor'].clone().detach()
                    elif 'critic' in obs:
                        # 只有 critic（不常见）
                        current_belief_state = obs['critic'].clone().detach()
                    else:
                        # 无法识别的字典结构
                        raise ValueError(f"无法识别的 obs 字典结构，keys: {obs.keys()}")
                else:
                    # 其他类型
                    raise ValueError(f"obs 类型不正确: {type(obs)}")

                # 创建观察（V2：开关控制raw_obs保存）
                observation = SlateObservation(
                    belief_state=current_belief_state,
                    raw_obs=raw_obs_before_encoding  # 根据save_raw_obs开关决定是否保存
                )

                # 选择动作 (如果模型为None，使用随机策略)
                latent_action = None  # 初始化latent_action

                if agent is None or ranker is None or quality_level == "random":
                    # 随机动作
                    slate = environment.get_random_action()
                    latent_action = None  # 随机策略没有latent action
                else:
                    # 使用训练好的模型
                    if ranker:
                        # 关键修复：保存latent_action
                        latent_action = agent.get_action(current_belief_state, sample=False)

                        # 🔥 新增：ε-greedy 噪声注入
                        if self.epsilon_greedy > 0 and np.random.rand() < self.epsilon_greedy:
                            # 以 epsilon 概率添加高斯噪声到 latent action
                            noise = torch.randn_like(latent_action) * self.epsilon_noise_scale
                            latent_action = latent_action + noise
                            # 截断到合理范围，防止数值爆炸
                            latent_action = torch.clamp(latent_action, -5.0, 5.0)

                        slate = ranker.rank(latent_action)
                        # 关键：clone + detach避免梯度问题
                        latent_action = latent_action.clone().detach()
                    else:
                        slate = agent.get_action(current_belief_state, sample=False)
                        latent_action = None  # 没有ranker时，action就是slate

                # 确保slate是tensor格式 (环境需要tensor)
                if isinstance(slate, list):
                    slate = torch.tensor(slate, device=self.device)
                elif isinstance(slate, np.ndarray):
                    slate = torch.tensor(slate, device=self.device)
                elif torch.is_tensor(slate):
                    slate = slate.to(self.device)

                # 创建动作 (SlateAction需要列表格式)
                slate_list = slate.cpu().tolist() if torch.is_tensor(slate) else slate
                # 关键修复：保存latent_action
                action = SlateAction(
                    discrete_slate=slate_list,
                    latent_action=latent_action  # 保存latent action
                )

                # 环境步进
                next_obs_raw, reward, done, next_info = environment.step(slate)

                # V2新增：保存next_raw_obs（开关控制）
                next_raw_obs_copy = copy.deepcopy(next_obs_raw) if save_raw_obs else None

                # 保存原始观察中的clicks（在转换为belief state之前）
                clicks = next_obs_raw.get('clicks', torch.zeros(len(slate)))
                if not torch.is_tensor(clicks):
                    clicks = torch.tensor(clicks)

                # V3新增：获取Oracle信息（开关控制）
                item_relevances = None
                if save_raw_obs:
                    try:
                        # 从底层模拟器获取真实相关性（上帝视角）
                        item_relevances = environment.get_relevances()
                        if item_relevances is not None and torch.is_tensor(item_relevances):
                            item_relevances = item_relevances.clone().detach()
                    except (AttributeError, Exception) as e:
                        # 如果环境不支持get_relevances，静默跳过
                        item_relevances = None

                # 关键修复：与训练代码一致，调用belief.forward(next_obs, done)更新belief state
                if belief_encoder is not None:
                    next_obs = belief_encoder.forward(next_obs_raw, done=done)

                    # 🔥 修复：处理 next_obs 的 Actor-Critic 分离 Belief State
                    if next_obs is None:
                        # 当 done=True 时，belief_encoder 可能返回 None
                        # 此时沿用当前的 obs
                        if isinstance(obs, dict) and 'actor' in obs:
                            next_obs = obs['actor'].clone().detach()
                            next_belief_state = obs['actor'].clone().detach()
                        elif isinstance(obs, torch.Tensor):
                            next_obs = obs.clone().detach()
                            next_belief_state = obs.clone().detach()
                        else:
                            # 保底处理
                            next_obs = obs
                            next_belief_state = obs
                    else:
                        # next_obs 不是 None，需要提取 belief state
                        if isinstance(next_obs, dict) and 'actor' in next_obs:
                            # Actor-Critic 分离：提取 actor 部分
                            next_belief_state = next_obs['actor'].clone().detach()
                        elif isinstance(next_obs, torch.Tensor):
                            # 单一 tensor
                            next_belief_state = next_obs.clone().detach()
                        elif isinstance(next_obs, dict) and 'critic' in next_obs:
                            # 只有 critic（不常见）
                            next_belief_state = next_obs['critic'].clone().detach()
                        else:
                            # 无法识别的结构
                            raise ValueError(f"无法识别的 next_obs 结构，type: {type(next_obs)}, keys: {next_obs.keys() if isinstance(next_obs, dict) else 'N/A'}")
                else:
                    # 🔥 修复：Random 策略没有 belief_encoder，next_belief_state 应该为 None
                    next_obs = next_obs_raw
                    next_belief_state = None

                next_observation = SlateObservation(
                    belief_state=next_belief_state,
                    raw_obs=next_raw_obs_copy  # V2：根据开关决定是否保存
                )
                
                episode_slates.append(slate_list)
                diversity_score = metrics_calculator.calculate_diversity_score(slate_list)
                coverage_score = metrics_calculator.calculate_coverage_score(slate_list, episode_slates)
                
                # 创建信息
                info_data = SlateInfo(
                    clicks=clicks,
                    diversity_score=diversity_score,
                    coverage_score=coverage_score,
                    episode_return=0.0,  # 将在轨迹完成后更新
                    episode_id=episode_id,
                    timestep=timestep,
                    item_relevances=item_relevances  # V3：Oracle信息
                )
                
                # 创建转移
                transition = SlateTransition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation,
                    done=done,
                    info=info_data
                )
                
                trajectory.add_transition(transition)

                # 更新状态
                obs = next_obs
                raw_obs_before_encoding = next_raw_obs_copy  # V2：更新raw_obs
                timestep += 1
            
            # 更新所有转移的episode_return
            episode_return = trajectory.get_return()
            for transition in trajectory.transitions:
                transition.info.episode_return = episode_return
            
            return trajectory
            
        except Exception as e:
            print(f"收集episode {episode_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_belief_state(self, obs: Dict, belief_encoder) -> torch.Tensor:
        """
        获取belief state

        Args:
            obs: 环境观察
            belief_encoder: 信念编码器

        Returns:
            belief_state: 信念状态向量
        """
        try:
            if belief_encoder is not None:
                # 使用GRUBelief的forward方法
                belief_state = belief_encoder.forward(obs, done=False)
                # 关键修复：clone + detach 避免inference mode冲突
                if belief_state is not None:
                    belief_state = belief_state.clone().detach().to(self.device)
                    return belief_state
                else:
                    return torch.zeros(32, device=self.device)
            else:
                # 如果没有belief encoder，返回随机向量
                return torch.randn(32, device=self.device)
        except Exception as e:
            print(f"获取belief state时出错: {e}")
            return torch.zeros(32, device=self.device)
    
    def collect_all_diffuse_data(self, quality_level: str = 'expert', save_raw_obs: bool = False):
        """
        收集所有diffuse环境的数据

        Args:
            quality_level: 数据质量级别 (expert/medium/random)
            save_raw_obs: 是否保存原始obs (默认False，向后兼容)
        """
        print(f"开始收集所有diffuse环境的 {quality_level} 数据...")

        # 加载训练好的模型
        print(f"加载 {quality_level} 级别的模型...")
        models = self.model_loader.load_diffuse_models(quality_level=quality_level)
        
        # 创建环境 (需要切换到GeMS根目录，因为TopicRec使用相对路径)
        print("创建环境...")
        original_cwd = os.getcwd()
        project_root = Path(__file__).resolve().parent.parent
        try:
            os.chdir(str(project_root))
            environments = self.env_factory.create_all_diffuse_environments()
        finally:
            os.chdir(original_cwd)
        
        # 为每个环境收集数据
        for env_name in ['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen']:
            if env_name not in models or env_name not in environments:
                print(f"⚠️ 跳过 {env_name}: 模型或环境缺失")
                continue
            
            print(f"\n{'='*60}")
            print(f"收集 {env_name} 环境的数据")
            print(f"{'='*60}")
            
            agent, ranker, belief_encoder = models[env_name]
            environment = environments[env_name]

            # 收集指定质量级别的数据
            dataset = self.collect_trajectories_from_model(
                env_name, agent, ranker, belief_encoder, environment,
                self.collection_config[quality_level]['episodes'], quality_level, save_raw_obs
            )

            # 保存数据
            data_path = os.path.join(self.output_dir, env_name, f'{self.file_prefix}{quality_level}_data.pkl')
            dataset.save(data_path, format='pickle')

            # 保存D4RL格式
            d4rl_path = os.path.join(self.output_dir, env_name, f'{self.file_prefix}{quality_level}_data_d4rl.npz')
            dataset.save(d4rl_path, format='d4rl')

            print(f"✅ {env_name} {quality_level}数据已保存:")
            print(f"  Pickle格式: {data_path}")
            print(f"  D4RL格式: {d4rl_path}")

            # 打印统计信息
            stats = dataset.get_stats()
            print(f"  数据集统计:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
        
        print(f"\n🎉 所有数据收集完成!")
        print(f"数据保存在: {self.output_dir}")

def analyze_dataset_quality(dataset, env_name):
    """
    快速分析数据集是否符合 Offline RL 训练要求

    Args:
        dataset: SlateDataset 对象
        env_name: 环境名称
    """
    print(f"\n{'='*20} 数据集快速体检报告 ({env_name}) {'='*20}")

    # 1. 提取所有 transitions
    all_rewards = []
    all_slates = []
    episode_returns = []
    episode_lengths = []

    for trajectory in dataset.trajectories:
        # 提取 rewards
        episode_rewards = []
        episode_slates = []

        for transition in trajectory.transitions:
            # Reward
            episode_rewards.append(transition.reward)

            # Slate (从 action 中提取)
            if hasattr(transition.action, 'discrete_slate'):
                slate = transition.action.discrete_slate
                episode_slates.append(slate)

        all_rewards.extend(episode_rewards)
        all_slates.extend(episode_slates)
        episode_returns.append(sum(episode_rewards))
        episode_lengths.append(len(episode_rewards))

    all_rewards = np.array(all_rewards)

    # --- 指标 1: Reward 区分度 ---
    mean_rew = np.mean(all_rewards)
    std_rew = np.std(all_rewards)
    neg_rate = np.mean(all_rewards < 0) * 100
    zero_rate = np.mean(all_rewards == 0) * 100
    pos_rate = np.mean(all_rewards > 0) * 100

    print(f"\n[1. Reward 分布] -> 决定 RL 能否学到优劣")
    print(f"  - 均值 (Mean): {mean_rew:.4f}")
    print(f"  - 标准差 (Std): {std_rew:.4f} \t{'✅' if std_rew > 5.0 else '⚠️'} 目标: > 5.0, 越大区分度越高")
    print(f"  - 负分比例: {neg_rate:.2f}% \t{'✅' if neg_rate > 5.0 else '❌'} 目标: > 5%, 必须有惩罚")
    print(f"  - 零分比例: {zero_rate:.2f}%")
    print(f"  - 正分比例: {pos_rate:.2f}%")

    # --- 指标 2: 序列多样性 (Consecutive Overlap) ---
    overlaps = []
    if len(all_slates) > 1:
        # 计算连续 slate 的重叠率
        for i in range(min(len(all_slates) - 1, 10000)):  # 限制计算量
            s1 = set(all_slates[i])
            s2 = set(all_slates[i + 1])
            if len(s1) > 0:
                overlap = len(s1 & s2) / len(s1)  # 重叠率
                overlaps.append(overlap)

    avg_overlap = np.mean(overlaps) * 100 if overlaps else 0
    print(f"\n[2. 策略僵化度] -> 决定是否只是复读机")
    print(f"  - 连续 Slate 重叠率: {avg_overlap:.2f}% \t{'✅' if avg_overlap < 50.0 else '⚠️'} 目标: < 50%, 太高说明策略不改错")

    # --- 指标 3: 物品集中度 (Concentration) ---
    if all_slates:
        # 展平所有推荐的物品
        flat_items = []
        for slate in all_slates:
            if isinstance(slate, (list, tuple)):
                flat_items.extend(slate)
            elif isinstance(slate, np.ndarray):
                flat_items.extend(slate.flatten().tolist())

        item_counts = Counter(flat_items)
        total_recs = len(flat_items)
        sorted_counts = item_counts.most_common()

        # 计算前 10% 物品占据的推荐量
        top_10_percent_num = max(1, int(len(item_counts) * 0.1))
        top_10_items = sorted_counts[:top_10_percent_num]
        top_10_coverage = sum([c for i, c in top_10_items]) / total_recs * 100

        print(f"\n[3. 物品覆盖度] -> 决定是否存在马太效应")
        print(f"  - 唯一物品数: {len(item_counts)}")
        print(f"  - Top-10% 物品覆盖率: {top_10_coverage:.2f}% \t{'✅' if top_10_coverage < 60.0 else '⚠️'} 目标: < 60%, 越低越好")
    else:
        top_10_coverage = 0

    # --- 综合判定 ---
    print(f"\n{'='*20} 综合评价 {'='*20}")
    is_good = True

    if std_rew < 1.0:
        print("❌ Reward 几乎没有波动，RL 很难学习！")
        is_good = False

    if neg_rate < 1.0:
        print("❌ 几乎没有负反馈，Critic 容易高估！建议增加 penalty 或 noise。")
        is_good = False

    if avg_overlap > 80.0:
        print("❌ 策略极其僵化，一直推重复内容！建议减小 boredom_threshold。")
        is_good = False

    if all_slates and top_10_coverage > 80.0:
        print("❌ 物品高度集中，存在严重马太效应！建议增加 epsilon_greedy 或使用混合策略。")
        is_good = False

    if is_good:
        print("✅ 数据集初步合格！具备训练 Offline RL 的潜力。")
    else:
        print("⚠️ 数据集存在风险，请调整环境参数或收集策略。")

    print("="*50)

def collect_mixed_strategy_data(collector: OfflineDataCollector, args):
    """
    一键收集混合策略数据（简化版）

    流程：
    1. 构建策略配置列表
    2. 循环收集各策略数据
    3. 自动合并数据集
    4. 输出质量分析
    """
    # 1. 构建策略配置列表（直接从参数读取，epsilon可调）
    strategies = [
        {
            'quality': 'expert',
            'epsilon': args.expert_pure_epsilon,
            'episodes': args.expert_pure_eps,
            'name': f'expert_eps{args.expert_pure_epsilon:.1f}'
        },
        {
            'quality': 'expert',
            'epsilon': args.expert_noisy_epsilon,
            'episodes': args.expert_noisy_eps,
            'name': f'expert_eps{args.expert_noisy_epsilon:.1f}'
        },
        {
            'quality': 'medium',
            'epsilon': args.medium_noisy_epsilon,
            'episodes': args.medium_noisy_eps,
            'name': f'medium_eps{args.medium_noisy_epsilon:.1f}'
        },
        {
            'quality': 'random',
            'epsilon': args.random_epsilon,
            'episodes': args.random_eps,
            'name': f'random_eps{args.random_epsilon:.1f}'
        }
    ]

    # 过滤掉 episodes=0 的策略
    strategies = [s for s in strategies if s['episodes'] > 0]

    total_episodes = sum(s['episodes'] for s in strategies)

    print("=" * 80)
    print("混合策略数据收集配置")
    print("=" * 80)
    print(f"环境: {args.env_name}")
    print(f"总Episodes: {total_episodes}")
    print(f"环境参数: boredom={args.boredom_threshold}, penalty={args.diversity_penalty}, length={args.episode_length}")
    print("\n策略配置:")
    for i, strategy in enumerate(strategies, 1):
        ratio = strategy['episodes'] / total_episodes * 100
        print(f"  {i}. {strategy['name']}: {strategy['episodes']} episodes ({ratio:.1f}%)")
    print("=" * 80)

    # 2. 循环收集各策略数据
    subset_paths = []
    for i, strategy in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] 开始收集策略: {strategy['name']}")
        print("-" * 80)

        # 设置当前策略参数
        args.quality = strategy['quality']
        args.epsilon_greedy = strategy['epsilon']
        args.episodes = strategy['episodes']

        # 收集数据
        subset_path = collect_single_strategy_data(collector, args, strategy['name'])
        subset_paths.append(subset_path)

        print(f"✅ 策略 {strategy['name']} 收集完成: {subset_path}")

    # 3. 自动合并数据集
    if args.auto_merge:
        print("\n" + "=" * 80)
        print("开始合并数据集...")
        print("=" * 80)
        merged_path = merge_datasets(subset_paths, args.output_name, args.env_name)
        print(f"✅ 合并完成: {merged_path}")

        # 4. 输出质量分析
        if args.analyze_quality:
            print("\n" + "=" * 80)
            print("数据质量分析")
            print("=" * 80)
            analyze_quality_from_file(merged_path)

        # 5. 可选：删除临时子数据集
        if not args.keep_subsets:
            print("\n清理临时子数据集...")
            for path in subset_paths:
                if os.path.exists(path):
                    os.remove(path)
                oracle_path = path.replace('_data_d4rl.npz', '_oracle.npz')
                if os.path.exists(oracle_path):
                    os.remove(oracle_path)
            print("✅ 临时文件已清理")

    print("\n" + "=" * 80)
    print("🎉 混合策略数据收集完成！")
    print("=" * 80)

def collect_single_strategy_data(collector: OfflineDataCollector, args, strategy_name: str) -> str:
    """
    收集单个策略的数据

    Args:
        collector: 数据收集器实例
        args: 命令行参数
        strategy_name: 策略名称（用于文件命名）

    Returns:
        数据集文件路径
    """
    # 加载模型（如果需要）
    if args.quality != 'random':
        print(f"  加载 {args.quality} 级别模型...")
        try:
            agent, ranker, belief_encoder = collector.model_loader.load_model(
                env_name=args.env_name,
                quality=args.quality
            )
        except Exception as e:
            print(f"  ❌ 加载模型失败: {e}")
            raise
    else:
        agent, ranker, belief_encoder = None, None, None

    # 创建环境
    print(f"  创建环境...")
    original_cwd = os.getcwd()
    project_root = Path(__file__).resolve().parent.parent

    # 构建环境参数
    env_kwargs = {}
    if args.boredom_threshold is not None:
        env_kwargs['boredom_threshold'] = args.boredom_threshold
    if args.diversity_penalty is not None:
        env_kwargs['diversity_penalty'] = args.diversity_penalty
    if args.episode_length is not None:
        env_kwargs['episode_length'] = args.episode_length

    try:
        os.chdir(str(project_root))
        environment = collector.env_factory.create_environment(args.env_name, **env_kwargs)
    finally:
        os.chdir(original_cwd)

    if environment is None:
        raise ValueError(f"无法创建环境: {args.env_name}")

    # 收集数据
    print(f"  收集 {args.episodes} episodes...")
    dataset = collector.collect_trajectories_from_model(
        args.env_name, agent, ranker, belief_encoder, environment,
        args.episodes, args.quality, args.save_raw_obs
    )

    # 保存数据集
    output_dir = os.path.join(collector.output_dir, args.env_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(output_dir, f"{strategy_name}_data_d4rl.npz")
    dataset.save(output_path, format='d4rl')

    return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='离线数据收集')
    # 动态设置默认输出目录
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    default_output_dir = str(project_root / "data" / "datasets" / "offline")
    parser.add_argument('--output_dir', type=str,
                       default=default_output_dir,
                       help='输出目录')
    parser.add_argument('--env_name', type=str,
                       choices=['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen',
                               'focused_topdown', 'focused_mix', 'focused_divpen', 'all'],
                       default='all',
                       help='环境名称')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='每个质量级别的episodes数量')
    parser.add_argument('--quality', type=str,
                       choices=['expert', 'medium', 'random'],
                       default='expert',
                       help='数据质量级别 (expert/medium/random)')
    parser.add_argument('--save_raw_obs', action='store_true',
                       help='保存原始环境观察(V2格式)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='指定使用的GPU编号')

    # 🔥 新增参数：ε-greedy 噪声注入
    parser.add_argument('--epsilon_greedy', type=float, default=0.0,
                       help='以 epsilon 的概率注入噪声 (0.0-1.0, 默认0.0)')
    parser.add_argument('--epsilon_noise_scale', type=float, default=1.0,
                       help='高斯噪声的标准差 (默认1.0)')

    # 🔥 新增参数：环境参数覆盖
    parser.add_argument('--boredom_threshold', type=int, default=None,
                       help='覆盖 boredom_threshold (越小越容易厌倦)')
    parser.add_argument('--diversity_penalty', type=float, default=None,
                       help='覆盖 diversity_penalty (越大惩罚越重)')
    parser.add_argument('--episode_length', type=int, default=None,
                       help='覆盖 episode_length')

    # 🔥 新增参数：文件前缀（防止覆盖旧数据）
    parser.add_argument('--file_prefix', type=str, default="",
                       help='输出文件的前缀 (e.g. "hard_")')

    # 🆕 新增参数：一键混合策略收集（简化版）
    parser.add_argument('--mix_mode', action='store_true', default=False,
                       help='启用混合策略收集模式')
    parser.add_argument('--total_episodes', type=int, default=10000,
                       help='混合策略模式下的总episode数')

    # 各策略的episode数量（直接指定数量，更清晰）
    parser.add_argument('--expert_pure_eps', type=int, default=1000,
                       help='Pure Expert策略的episodes数')
    parser.add_argument('--expert_pure_epsilon', type=float, default=0.0,
                       help='Pure Expert策略的epsilon值（默认0.0）')

    parser.add_argument('--expert_noisy_eps', type=int, default=4000,
                       help='Noisy Expert策略的episodes数')
    parser.add_argument('--expert_noisy_epsilon', type=float, default=0.3,
                       help='Noisy Expert策略的epsilon值（默认0.3）')

    parser.add_argument('--medium_noisy_eps', type=int, default=3000,
                       help='Noisy Medium策略的episodes数')
    parser.add_argument('--medium_noisy_epsilon', type=float, default=0.3,
                       help='Noisy Medium策略的epsilon值（默认0.3）')

    parser.add_argument('--random_eps', type=int, default=2000,
                       help='Random策略的episodes数')
    parser.add_argument('--random_epsilon', type=float, default=0.0,
                       help='Random策略的epsilon值（默认0.0，因为已经是随机策略）')

    parser.add_argument('--output_name', type=str, default='mixed_data',
                       help='混合策略模式下的输出数据集名称')
    parser.add_argument('--auto_merge', action='store_true', default=True,
                       help='是否自动合并子数据集（默认True）')
    parser.add_argument('--keep_subsets', action='store_true', default=False,
                       help='是否保留子数据集（默认False）')
    parser.add_argument('--analyze_quality', action='store_true', default=True,
                       help='是否输出数据质量分析（默认True）')

    args = parser.parse_args()

    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"设置使用GPU: {args.gpu}")

    # 🔥 构建环境参数覆盖字典
    env_kwargs = {}
    if args.boredom_threshold is not None:
        env_kwargs['boredom_threshold'] = args.boredom_threshold
    if args.diversity_penalty is not None:
        env_kwargs['diversity_penalty'] = args.diversity_penalty
    if args.episode_length is not None:
        env_kwargs['episode_length'] = args.episode_length

    # 打印环境参数覆盖信息
    if env_kwargs:
        print(f"⚠️  环境参数覆盖: {env_kwargs}")

    # 🔥 创建数据收集器（传入新参数）
    collector = OfflineDataCollector(
        args.output_dir,
        epsilon_greedy=args.epsilon_greedy,
        epsilon_noise_scale=args.epsilon_noise_scale,
        file_prefix=args.file_prefix
    )

    # 打印噪声注入信息
    if args.epsilon_greedy > 0:
        print(f"⚠️  ε-greedy 噪声注入: epsilon={args.epsilon_greedy}, scale={args.epsilon_noise_scale}")

    # 🆕 检查是否使用混合策略收集模式
    if args.mix_mode:
        # 使用一键混合策略收集
        collect_mixed_strategy_data(collector, args)
        return

    # 更新配置
    for quality in collector.collection_config:
        collector.collection_config[quality]['episodes'] = args.episodes

    if args.env_name == 'all':
        # 收集所有环境的数据
        collector.collect_all_diffuse_data(quality_level=args.quality, save_raw_obs=args.save_raw_obs)
    else:
        # 收集单个环境的数据
        print(f"收集 {args.env_name} 环境的 {args.quality} 数据...")

        # 根据环境名称判断是diffuse还是focused
        is_focused = args.env_name.startswith('focused')

        # 🔥 优化：只加载需要的单个环境的模型，而不是加载所有模型
        # 🔥 修复：Random 策略不需要加载模型
        if args.quality == "random":
            print(f"使用 Random 策略，跳过模型加载...")
            agent, ranker, belief_encoder = None, None, None
        else:
            print(f"加载 {args.env_name} 环境的 {args.quality} 级别模型...")
            try:
                agent, ranker, belief_encoder = collector.model_loader.load_model(
                    env_name=args.env_name,
                    quality=args.quality
                )
            except Exception as e:
                print(f"❌ 错误: 加载 {args.env_name} 的模型失败: {e}")
                return

        # 创建环境
        print("创建环境...")
        original_cwd = os.getcwd()
        project_root = Path(__file__).resolve().parent.parent
        try:
            os.chdir(str(project_root))
            # 🔥 传递环境参数覆盖
            environment = collector.env_factory.create_environment(args.env_name, **env_kwargs)
        finally:
            os.chdir(original_cwd)

        if environment is None:
            print(f"❌ 错误: 未找到 {args.env_name} 的环境")
            return

        print(f"\n{'='*60}")
        print(f"收集 {args.env_name} 环境的数据")
        print(f"{'='*60}")

        # 🔥 agent, ranker, belief_encoder 已经在上面加载好了
        # environment 也已经在上面创建好了

        # 收集指定质量级别的数据
        dataset = collector.collect_trajectories_from_model(
            args.env_name, agent, ranker, belief_encoder, environment,
            args.episodes, args.quality, args.save_raw_obs
        )

        # 保存数据
        output_dir = os.path.join(collector.output_dir, args.env_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 🔥 使用 file_prefix 生成文件名
        data_path = os.path.join(output_dir, f'{collector.file_prefix}{args.quality}_data.pkl')
        dataset.save(data_path, format='pickle')

        # 保存D4RL格式
        d4rl_path = os.path.join(output_dir, f'{collector.file_prefix}{args.quality}_data_d4rl.npz')
        dataset.save(d4rl_path, format='d4rl')

        print(f"✅ {args.env_name} {args.quality}数据已保存:")
        print(f"  Pickle格式: {data_path}")
        print(f"  D4RL格式: {d4rl_path}")

        # 打印统计信息
        stats = dataset.get_stats()
        print(f"  数据集统计:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

        # 🔥 新增：快速体检数据集质量
        analyze_dataset_quality(dataset, args.env_name)

        print(f"\n🎉 数据收集完成!")
        print(f"数据保存在: {output_dir}")

if __name__ == "__main__":
    main()
