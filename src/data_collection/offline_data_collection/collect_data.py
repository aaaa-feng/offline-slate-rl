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

# 添加父目录到路径以便导入core模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_formats import SlateDataset, SlateTrajectory, SlateTransition, SlateObservation, SlateAction, SlateInfo
from core.model_loader import ModelLoader
from core.environment_factory import EnvironmentFactory
from core.metrics import SlateMetrics, create_item_popularity_dict

class OfflineDataCollector:
    """离线数据收集器"""
    
    def __init__(self, output_dir: str = None):
        # 动态设置默认输出目录
        if output_dir is None:
            # 使用统一路径配置
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / "config"))
            from paths import OFFLINE_DATASETS_DIR
            output_dir = str(OFFLINE_DATASETS_DIR)
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

            while not done and timestep < 100:  # 最大100步
                # 关键修复：obs已经是belief state，直接使用
                current_belief_state = obs.clone().detach() if belief_encoder is not None else obs

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
                    # 当done=True时，belief_encoder可能返回None，使用当前obs作为next_belief_state
                    if next_obs is None:
                        next_belief_state = obs.clone().detach()
                    else:
                        next_belief_state = next_obs.clone().detach()
                else:
                    next_obs = next_obs_raw
                    next_belief_state = next_obs

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
            data_path = os.path.join(self.output_dir, env_name, f'{quality_level}_data.pkl')
            dataset.save(data_path, format='pickle')

            # 保存D4RL格式
            d4rl_path = os.path.join(self.output_dir, env_name, f'{quality_level}_data_d4rl.npz')
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

    args = parser.parse_args()

    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"设置使用GPU: {args.gpu}")
    
    # 创建数据收集器
    collector = OfflineDataCollector(args.output_dir)
    
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

        # 加载训练好的模型
        print(f"加载 {args.quality} 级别的模型...")
        if is_focused:
            models = collector.model_loader.load_focused_models(quality_level=args.quality)
        else:
            models = collector.model_loader.load_diffuse_models(quality_level=args.quality)

        if args.env_name not in models:
            print(f"❌ 错误: 未找到 {args.env_name} 的模型")
            return

        # 创建环境
        print("创建环境...")
        original_cwd = os.getcwd()
        project_root = Path(__file__).resolve().parent.parent
        try:
            os.chdir(str(project_root))
            environment = collector.env_factory.create_environment(args.env_name)
        finally:
            os.chdir(original_cwd)

        if environment is None:
            print(f"❌ 错误: 未找到 {args.env_name} 的环境")
            return

        print(f"\n{'='*60}")
        print(f"收集 {args.env_name} 环境的数据")
        print(f"{'='*60}")

        agent, ranker, belief_encoder = models[args.env_name]
        # environment已经在上面创建好了

        # 收集指定质量级别的数据
        dataset = collector.collect_trajectories_from_model(
            args.env_name, agent, ranker, belief_encoder, environment,
            args.episodes, args.quality, args.save_raw_obs
        )

        # 保存数据
        output_dir = os.path.join(collector.output_dir, args.env_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        data_path = os.path.join(output_dir, f'{args.quality}_data.pkl')
        dataset.save(data_path, format='pickle')

        # 保存D4RL格式
        d4rl_path = os.path.join(output_dir, f'{args.quality}_data_d4rl.npz')
        dataset.save(d4rl_path, format='d4rl')

        print(f"✅ {args.env_name} {args.quality}数据已保存:")
        print(f"  Pickle格式: {data_path}")
        print(f"  D4RL格式: {d4rl_path}")

        # 打印统计信息
        stats = dataset.get_stats()
        print(f"  数据集统计:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

        print(f"\n🎉 数据收集完成!")
        print(f"数据保存在: {output_dir}")

if __name__ == "__main__":
    main()
