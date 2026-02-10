#!/usr/bin/env python3
"""
环境工厂
用于创建和配置推荐环境
"""
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径 - 从core/向上到项目根目录，然后进入src/
OFFLINE_DATA_COLLECTION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = OFFLINE_DATA_COLLECTION_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envs.RecSim.simulators import TopicRec
from common.online.env_wrapper import EnvWrapper
from common.online.data_module import BufferDataModule

class EnvironmentFactory:
    """环境工厂类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 基础配置（与训练代码默认参数一致）
        base_config = {
            'env_name': 'topics',
            'num_items': 1000,
            'rec_size': 10,
            'episode_length': 100,
            'num_topics': 10,
            'topic_size': 2,
            'env_offset': 0.28,  # 修复：训练代码默认值
            'env_slope': 100,     # 修复：训练代码默认值
            'env_omega': 0.9,
            'env_alpha': 1.0,
            'env_propensities': None,
            'boredom_threshold': 5,  # 修复：训练代码默认值
            'recent_items_maxlen': 10,
            'boredom_moving_window': 5,
            'short_term_boost': 1.0,  # 修复：训练代码默认值
            'diversity_penalty': 1.0,
            'diversity_threshold': 4,  # 修复：训练代码默认值
            'click_model': 'tdPBM',
            'click_only_once': False,
            'rel_threshold': None,  # 修复：训练代码默认值
            'prop_threshold': None,  # 修复：训练代码默认值
            'env_embedds': 'item_embeddings_diffuse.pt',
            'item_embedd_dim': 20,
            'sim_seed': 24321357327,
            'filename': None
        }
        
        # Focused环境基础配置（仅更换embeddings）
        focused_base_config = base_config.copy()
        focused_base_config['env_embedds'] = 'item_embeddings_focused.pt'

        # 环境配置映射
        # 注意：divpen环境使用diversity_penalty=3.0，其他环境使用1.0
        # 🔥 修复：明确指定每个环境的 click_model
        self.env_configs = {
            'diffuse_topdown': {
                **base_config,
                'dataset_name': 'diffuse_topdown',
                'click_model': 'tdPBM'  # TopDown 使用 tdPBM
            },
            'diffuse_mix': {
                **base_config,
                'dataset_name': 'diffuse_mix',
                'click_model': 'mixPBM'  # 🔥 修复：Mix 应该使用 mixPBM
            },
            'diffuse_divpen': {
                **base_config,
                'dataset_name': 'diffuse_divpen',
                'click_model': 'mixPBM',  # 🔥 修复：Divpen 应该使用 mixPBM
                'diversity_penalty': 3.0
            },
            'focused_topdown': {
                **focused_base_config,
                'dataset_name': 'focused_topdown',
                'click_model': 'tdPBM',  # TopDown 使用 tdPBM
                'diversity_penalty': 1.0
            },
            'focused_mix': {
                **focused_base_config,
                'dataset_name': 'focused_mix',
                'click_model': 'mixPBM',  # Mix 使用 mixPBM
                'diversity_penalty': 1.0
            },
            'focused_divpen': {
                **focused_base_config,
                'dataset_name': 'focused_divpen',
                'click_model': 'mixPBM',  # Divpen 使用 mixPBM
                'diversity_penalty': 3.0
            }
        }
    
    def create_environment(self, env_name: str, **kwargs) -> TopicRec:
        """
        创建推荐环境
        
        Args:
            env_name: 环境名称 (diffuse_topdown, diffuse_mix, diffuse_divpen)
            **kwargs: 额外的环境参数
            
        Returns:
            environment: TopicRec环境实例
        """
        if env_name not in self.env_configs:
            raise ValueError(f"不支持的环境: {env_name}. 支持的环境: {list(self.env_configs.keys())}")
        
        # 获取基础配置
        config = self.env_configs[env_name].copy()
        
        # 更新配置
        config.update(kwargs)
        config['device'] = self.device
        
        # TopicRec会自己处理embeddings加载，我们只需要确保文件存在
        embeddings_path = PROJECT_ROOT / "data" / "embeddings" / config['env_embedds']
        embeddings_path = str(embeddings_path)
        if os.path.exists(embeddings_path):
            print(f"✅ 找到物品embeddings文件: {embeddings_path}")
        else:
            print(f"⚠️ 未找到物品embeddings文件: {embeddings_path}")
            # 如果文件不存在，设置为None让TopicRec生成随机embeddings
            config['env_embedds'] = None
        
        # 创建环境
        try:
            environment = TopicRec(**config)
            print(f"✅ 成功创建环境: {env_name}")
            return environment
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            raise
    
    def create_env_wrapper(self, env_name: str, buffer_size: int = 10000, **kwargs) -> EnvWrapper:
        """
        创建环境包装器
        
        Args:
            env_name: 环境名称
            buffer_size: 缓冲区大小
            **kwargs: 额外参数
            
        Returns:
            env_wrapper: EnvWrapper实例
        """
        # 创建基础环境
        base_env = self.create_environment(env_name, **kwargs)
        
        # 创建缓冲区
        buffer = BufferDataModule(
            offline_data=[],
            capacity=buffer_size,
            batch_size=32,
            device=self.device
        )
        
        # 创建环境包装器
        config = self.env_configs[env_name].copy()
        config.update(kwargs)
        
        env_wrapper = EnvWrapper(
            buffer=buffer,
            **config
        )
        
        return env_wrapper
    
    def get_env_config(self, env_name: str) -> Dict[str, Any]:
        """
        获取环境配置
        
        Args:
            env_name: 环境名称
            
        Returns:
            config: 环境配置字典
        """
        if env_name not in self.env_configs:
            raise ValueError(f"不支持的环境: {env_name}")
        
        return self.env_configs[env_name].copy()
    
    def list_available_environments(self) -> list:
        """
        列出可用的环境
        
        Returns:
            env_names: 环境名称列表
        """
        return list(self.env_configs.keys())
    
    def create_all_diffuse_environments(self) -> Dict[str, TopicRec]:
        """
        创建所有diffuse环境
        
        Returns:
            environments: {env_name: environment}
        """
        environments = {}
        
        for env_name in self.env_configs.keys():
            if env_name.startswith('diffuse'):
                try:
                    env = self.create_environment(env_name)
                    environments[env_name] = env
                    print(f"✅ {env_name} 环境创建成功")
                except Exception as e:
                    print(f"❌ {env_name} 环境创建失败: {e}")
        
        return environments
    
    def validate_environment(self, env_name: str) -> bool:
        """
        验证环境是否可以正常工作
        
        Args:
            env_name: 环境名称
            
        Returns:
            is_valid: 是否有效
        """
        try:
            # 创建环境
            env = self.create_environment(env_name)
            
            # 测试重置
            obs, info = env.reset()
            
            # 测试随机动作
            random_action = env.get_random_action()
            
            # 测试环境步进
            next_obs, reward, done, next_info = env.step(random_action)
            
            print(f"✅ {env_name} 环境验证成功")
            print(f"  观察形状: {obs}")
            print(f"  动作形状: {random_action}")
            print(f"  奖励: {reward}")
            
            return True
            
        except Exception as e:
            print(f"❌ {env_name} 环境验证失败: {e}")
            return False

if __name__ == "__main__":
    # 测试环境工厂
    print("测试环境工厂...")
    
    factory = EnvironmentFactory()
    
    # 列出可用环境
    available_envs = factory.list_available_environments()
    print(f"可用环境: {available_envs}")
    
    # 测试创建单个环境
    try:
        env = factory.create_environment('diffuse_topdown')
        print("✅ 单个环境创建测试成功")
        print(f"  环境类型: {type(env).__name__}")
        print(f"  物品数量: {env.num_items}")
        print(f"  推荐大小: {env.rec_size}")
    except Exception as e:
        print(f"❌ 单个环境创建测试失败: {e}")
    
    # 测试环境验证
    for env_name in available_envs:
        print(f"\n验证环境: {env_name}")
        is_valid = factory.validate_environment(env_name)
        if not is_valid:
            print(f"⚠️ {env_name} 环境验证失败")
    
    # 测试创建所有环境
    try:
        all_envs = factory.create_all_diffuse_environments()
        print(f"\n✅ 成功创建 {len(all_envs)} 个环境")
        for env_name in all_envs.keys():
            print(f"  - {env_name}")
    except Exception as e:
        print(f"❌ 批量环境创建失败: {e}")
    
    print("\n✅ 环境工厂测试完成!")
