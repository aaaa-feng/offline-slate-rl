#!/usr/bin/env python3
"""
模型加载器
用于加载训练好的GeMS模型进行数据收集
"""
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 添加项目路径 - 从core/向上4级到项目根目录，然后进入src/
# core/ -> offline_data_collection/ -> data_collection/ -> src/ -> offline-slate-rl/
OFFLINE_DATA_COLLECTION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = OFFLINE_DATA_COLLECTION_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.online import SAC, SlateQ, REINFORCE, WolpertingerSAC
from belief_encoders.gru_belief import GRUBelief
from rankers.gems.rankers import GeMS, TopKRanker, kHeadArgmaxRanker
from rankers.gems.item_embeddings import ItemEmbeddings, MFEmbeddings
from common.online.argument_parser import MyParser

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, models_dir: str = None):
        # 动态设置默认模型目录
        if models_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            models_dir = str(project_root / "offline_data_collection" / "best_models_for_data_collection")
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境配置
        self.env_configs = {
            'diffuse_topdown': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_diffuse.pt'
            },
            'diffuse_mix': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_diffuse.pt'
            },
            'diffuse_divpen': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_diffuse.pt'
            },
            'focused_topdown': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_focused.pt'
            },
            'focused_mix': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_focused.pt'
            },
            'focused_divpen': {
                'num_items': 1000,
                'rec_size': 10,
                'num_topics': 10,
                'topic_size': 2,
                'item_embedd_dim': 20,
                'belief_state_dim': 20,
                'env_embedds': 'item_embeddings_focused.pt'
            }
        }
    
    def load_item_embeddings(self, env_name: str, embedding_type: str = "ideal") -> ItemEmbeddings:
        """
        加载物品embeddings
        
        Args:
            env_name: 环境名称
            embedding_type: embedding类型 (ideal, scratch, mf)
            
        Returns:
            item_embeddings: ItemEmbeddings对象
        """
        config = self.env_configs[env_name]
        
        if embedding_type == "ideal":
            # 加载预训练的ideal embeddings - 使用统一路径配置
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / "config"))
            from paths import get_embeddings_path
            embeddings_path = str(get_embeddings_path(config['env_embedds']))
            if os.path.exists(embeddings_path):
                embeddings_tensor = torch.load(embeddings_path, map_location=self.device)
                item_embeddings = ItemEmbeddings(
                    num_items=config['num_items'],
                    item_embedd_dim=config['item_embedd_dim'],
                    device=self.device
                )
                item_embeddings.embedd.weight.data = embeddings_tensor
                print(f"✅ 成功加载ideal embeddings: {embeddings_path}")
            else:
                print(f"⚠️ 未找到ideal embeddings文件 {embeddings_path}, 使用随机初始化")
                item_embeddings = ItemEmbeddings(
                    num_items=config['num_items'],
                    item_embedd_dim=config['item_embedd_dim'],
                    device=self.device
                )
        
        elif embedding_type == "scratch":
            # 随机初始化
            item_embeddings = ItemEmbeddings(
                num_items=config['num_items'],
                item_embedd_dim=config['item_embedd_dim'],
                device=self.device
            )
        
        elif embedding_type == "mf":
            # 加载MF embeddings - 使用统一路径配置
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / "config"))
            from paths import get_mf_embeddings_path
            mf_path = str(get_mf_embeddings_path(f"{env_name}_moving_env"))
            if os.path.exists(mf_path):
                item_embeddings = MFEmbeddings(
                    num_items=config['num_items'],
                    item_embedd_dim=config['item_embedd_dim'],
                    device=self.device
                )
                # 加载MF权重
                mf_checkpoint = torch.load(mf_path, map_location=self.device)
                item_embeddings.load_state_dict(mf_checkpoint)
            else:
                print(f"警告: 未找到MF embeddings文件 {mf_path}, 使用随机初始化")
                item_embeddings = ItemEmbeddings(
                    num_items=config['num_items'],
                    item_embedd_dim=config['item_embedd_dim'],
                    device=self.device
                )
        
        else:
            raise ValueError(f"不支持的embedding类型: {embedding_type}")
        
        return item_embeddings
    
    def load_belief_encoder(self, env_name: str) -> GRUBelief:
        """
        加载信念编码器
        
        Args:
            env_name: 环境名称
            
        Returns:
            belief_encoder: GRUBelief对象
        """
        config = self.env_configs[env_name]
        
        # 创建item embeddings (用于belief encoder)
        item_embeddings = self.load_item_embeddings(env_name, "scratch")
        
        # 计算input_dim: rec_size * (item_embedd_dim + 1)
        input_dim = config['rec_size'] * (config['item_embedd_dim'] + 1)
        
        belief_encoder = GRUBelief(
            hidden_dim=config['belief_state_dim'],
            input_dim=input_dim,
            item_embeddings=item_embeddings,
            belief_state_dim=config['belief_state_dim'],
            item_embedd_dim=config['item_embedd_dim'],
            rec_size=config['rec_size'],
            ranker=True,  # 假设使用ranker
            device=self.device,
            belief_lr=0.001,
            hidden_layers_reduction=[256],
            beliefs=['actor', 'critic']
        )
        
        return belief_encoder
    
    def load_ranker(self, env_name: str, ranker_type: str = "TopK", embedding_type: str = "ideal") -> Any:
        """
        加载ranker模型
        
        Args:
            env_name: 环境名称
            ranker_type: ranker类型 (TopK, GeMS)
            embedding_type: embedding类型
            
        Returns:
            ranker: Ranker对象
        """
        config = self.env_configs[env_name]
        
        if ranker_type == "TopK":
            # 加载item embeddings
            item_embeddings = self.load_item_embeddings(env_name, embedding_type)
            
            ranker = TopKRanker(
                item_embeddings=item_embeddings,
                item_embedd_dim=config['item_embedd_dim'],
                rec_size=config['rec_size'],
                device=self.device
            )
            
        elif ranker_type == "GeMS":
            # 加载GeMS ranker
            item_embeddings = self.load_item_embeddings(env_name, embedding_type)

            ranker = GeMS(
                item_embeddings=item_embeddings,
                item_embedd_dim=config['item_embedd_dim'],
                rec_size=config['rec_size'],
                num_items=config['num_items'],
                latent_dim=32,  # latent维度
                hidden_layers_infer=[512, 256],  # 从checkpoint推断：512 -> 256 -> 64
                hidden_layers_decoder=[256, 512],  # 从checkpoint推断：256 -> 512
                device=self.device,
                lambda_click=0.5,
                lambda_KL=0.5,
                lambda_prior=0.0,
                ranker_lr=0.001,
                fixed_embedds=False,  # 不固定embeddings
                ranker_sample=False   # 不采样，使用argmax
            )
            
            # 尝试加载预训练的GeMS权重
            project_root = Path(__file__).resolve().parent.parent
            gems_checkpoint_path = project_root / "data" / "GeMS" / "checkpoints" / f"GeMS_{env_name}_latentdim32_beta1.0_lambdaclick0.5_lambdaprior0.0_scratch_seed58407201.ckpt"
            gems_checkpoint_path = str(gems_checkpoint_path)
            if os.path.exists(gems_checkpoint_path):
                try:
                    checkpoint = torch.load(gems_checkpoint_path, map_location=self.device)
                    ranker.load_state_dict(checkpoint['state_dict'])
                    print(f"✅ 成功加载GeMS checkpoint: {gems_checkpoint_path}")
                except Exception as e:
                    print(f"⚠️ 加载GeMS checkpoint失败: {e}")
            else:
                print(f"⚠️ 未找到GeMS checkpoint: {gems_checkpoint_path}")
        
        else:
            raise ValueError(f"不支持的ranker类型: {ranker_type}")
        
        return ranker
    
    def load_model(self, env_name: str, agent_type: str = "SAC", ranker_type: str = "TopK", 
                   embedding_type: str = "ideal") -> Tuple[Any, Any, Any]:
        """
        加载完整的模型 (load_agent的别名方法)
        
        Args:
            env_name: 环境名称
            agent_type: agent类型 (SAC, SlateQ, REINFORCE, WolpertingerSAC)
            ranker_type: ranker类型
            embedding_type: embedding类型
            
        Returns:
            (agent, ranker, belief_encoder): 模型组件
        """
        return self.load_agent(env_name, agent_type, ranker_type, embedding_type)
    
    def load_agent(self, env_name: str, agent_type: str = "SAC", ranker_type: str = "TopK", 
                   embedding_type: str = "ideal") -> Tuple[Any, Any, Any]:
        """
        加载完整的agent模型
        
        Args:
            env_name: 环境名称
            agent_type: agent类型 (SAC, SlateQ, REINFORCE, WolpertingerSAC)
            ranker_type: ranker类型
            embedding_type: embedding类型
            
        Returns:
            (agent, ranker, belief_encoder): 模型组件
        """
        config = self.env_configs[env_name]
        
        # 加载组件 (确保与训练时完全一致)
        # Belief Encoder: 使用scratch embeddings (与训练时一致)
        input_dim = config['rec_size'] * (config['item_embedd_dim'] + 1)  # 10 * (20 + 1) = 210
        belief_item_embeds = self.load_item_embeddings(env_name, "scratch")
        
        belief_encoder = GRUBelief(
            hidden_dim=config['belief_state_dim'],      # 20
            input_dim=input_dim,                        # 210
            item_embeddings=belief_item_embeds,
            belief_state_dim=config['belief_state_dim'],# 20
            item_embedd_dim=config['item_embedd_dim'],  # 20
            rec_size=config['rec_size'],                # 10
            ranker=True,  # 与训练时一致
            device=self.device,
            belief_lr=0.001,
            hidden_layers_reduction=[256],
            beliefs=['actor', 'critic']  # 必须与训练时一致
        )
        
        # Ranker: TopK+ideal
        ranker = self.load_ranker(env_name, ranker_type, embedding_type)
        
        # 创建agent (连续动作SAC)
        if agent_type == "SAC":
            # 根据ranker类型确定action_dim
            if ranker_type == "GeMS":
                action_dim = 32  # GeMS的latent_dim
            else:
                action_dim = config['item_embedd_dim']  # TopK使用item_embedd_dim

            agent = SAC(
                belief=belief_encoder,
                ranker=ranker,
                state_dim=config['belief_state_dim'],  # 20
                action_dim=action_dim,   # 32 for GeMS, 20 for TopK
                num_actions=1,  # 关键修复: 连续SAC模式，Q网络输出维度=1
                device=self.device,
                random_steps=1000,
                verbose=False,
                q_lr=0.001,
                pi_lr=0.003,
                gamma=0.8,
                tau=0.002,
                alpha=0.2,
                l2_reg=0.0,
                auto_entropy=True,
                alpha_lr=0.001,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                gradient_steps=1,
                hidden_layers_qnet=[256],
                hidden_layers_pinet=[256],
                target_update_frequency=1
            )
        
        elif agent_type == "SlateQ":
            agent = SlateQ(
                belief=belief_encoder,
                ranker=ranker,
                state_dim=config['belief_state_dim'],
                action_dim=config['num_items'],
                num_actions=config['num_items'],
                device=self.device,
                q_lr=0.001,
                gamma=0.8,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                hidden_layers_qnet=[256]
            )
        
        else:
            raise ValueError(f"不支持的agent类型: {agent_type}")
        # 🏥 统一加载 checkpoint
        checkpoint_dir = self.models_dir  # best_models_for_data_collection目录
        checkpoint_loaded = False

        for checkpoint_file in os.listdir(checkpoint_dir):
            # 找到匹配当前环境的 checkpoint 文件
            if checkpoint_file.endswith('.ckpt') and env_name in checkpoint_file:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    full_state_dict = checkpoint['state_dict']

                    print(f"🏥 开始【统一加载】: {checkpoint_path}")

                    # 1. 过滤掉训练状态相关的键（如优化器状态等），只保留模型权重
                    model_keys_only = {
                        k: v for k, v in full_state_dict.items()
                        if not k.startswith(('q_optimizer.', 'pi_optimizer.', 'alpha_optimizer.', 'global_step', 'epoch'))
                    }

                    # 2. 将所有权重统一加载到 Agent 实例中 (SAC 实例包含 Ranker/Belief 子模块)
                    # strict=False 用于忽略训练无关的键（如优化器状态）
                    load_result = agent.load_state_dict(model_keys_only, strict=False)

                    # 检查是否还有核心组件缺失 (如果模型结构与 checkpoint 不符，可能会缺失)
                    core_missing = [k for k in load_result.missing_keys
                                    if k.startswith(('ranker.', 'belief.', 'QNet', 'PolicyNet'))]

                    if core_missing:
                         print(f"    🚨 警告: Agent 内部核心组件缺失 {len(core_missing)} 个键! 请检查模型结构。")
                         print(f"    🚨 缺失键名示例: {core_missing[:5]}...")
                    else:
                         print(f"  ✅ Agent 核心权重加载成功 (包含 Ranker/Belief).")

                    # 3. 【关键修复】同步权重到外部独立实例
                    # 推理时可能使用外部的 ranker/belief_encoder 实例，必须同步权重

                    # 从 Agent 内部的子模块中提取正确的权重，加载到外部独立创建的实例中
                    if ranker is not None:
                        external_ranker_state = agent.ranker.state_dict()
                        ranker.load_state_dict(external_ranker_state, strict=True)

                    external_belief_state = agent.belief.state_dict()
                    belief_encoder.load_state_dict(external_belief_state, strict=True)

                    print(f"  ✅ 外部 Ranker/Belief 实例权重已成功同步.")

                    # 4. 【关键修复】设置action bounds
                    if 'action_center' in full_state_dict and 'action_scale' in full_state_dict:
                        agent.action_center = full_state_dict['action_center'].to(self.device)
                        agent.action_scale = full_state_dict['action_scale'].to(self.device)
                        print(f"  ✅ Action bounds已从checkpoint加载: center shape={agent.action_center.shape}, scale shape={agent.action_scale.shape}")

                    elif ranker_type == "GeMS":
                        # 【核心修复】动态计算精确的 Action Bounds，而不是使用固定的 3.0
                        # 训练时代码是这样做的：ranker.get_action_bounds(dataset_path)

                        # 使用统一路径配置获取数据集路径
                        sys.path.insert(0, str(PROJECT_ROOT / "config"))
                        from paths import get_online_dataset_path
                        dataset_path = get_online_dataset_path(env_name)
                        
                        if os.path.exists(dataset_path):
                            print(f"  📊 正在从数据集计算精确 Action Bounds: {dataset_path}")
                            # 这会返回精确的 (32,) 向量
                            center, scale = ranker.get_action_bounds(str(dataset_path), batch_size=10)
                            
                            agent.action_center = center.to(self.device)
                            agent.action_scale = scale.to(self.device)
                            
                            print(f"  ✅ 精确 Bounds 已应用!")
                            print(f"     Scale Mean: {scale.mean().item():.4f} (应接近 3.18)")
                            print(f"     Scale Std:  {scale.std().item():.4f}")
                        else:
                            print(f"  ⚠️ 未找到数据集 {dataset_path}，回退到默认值 3.0 (性能可能受损)")
                            agent.action_center = torch.zeros(action_dim, device=self.device)
                            agent.action_scale = 3.0 * torch.ones(action_dim, device=self.device)
                    else:
                        # 其他情况使用默认值
                        agent.action_center = torch.zeros(action_dim, device=self.device)
                        agent.action_scale = torch.ones(action_dim, device=self.device)
                        print(f"  ⚠️ 使用默认action bounds: dim={action_dim}")

                    checkpoint_loaded = True
                    break

                except Exception as e:
                    print(f"⚠️ 统一加载失败: {e}")
                    import traceback
                    traceback.print_exc()

        if not checkpoint_loaded:
            print(f"⚠️ 未找到checkpoint，使用随机初始化的模型")
            # 初始化默认的action bounds
            if ranker_type == "GeMS":
                agent.action_center = torch.zeros(action_dim, device=self.device)
                agent.action_scale = 3.0 * torch.ones(action_dim, device=self.device)
            else:
                agent.action_center = torch.zeros(action_dim, device=self.device)
                agent.action_scale = torch.ones(action_dim, device=self.device)
        
        # 设置为评估模式并移到GPU (这部分保持不变)
        agent.eval()
        agent = agent.to(self.device)
        if ranker is not None:
            ranker.eval()
            ranker = ranker.to(self.device)
        belief_encoder.eval()
        belief_encoder = belief_encoder.to(self.device)
        
        return agent, ranker, belief_encoder
        # # 🏥 手术式加载checkpoint
        # checkpoint_dir = self.models_dir  # best_models_for_data_collection目录
        # checkpoint_loaded = False
        # for checkpoint_file in os.listdir(checkpoint_dir):
        #     if checkpoint_file.endswith('.ckpt') and env_name in checkpoint_file:
        #         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        #         try:
        #             checkpoint = torch.load(checkpoint_path, map_location=self.device)
        #             full_state_dict = checkpoint['state_dict']
                    
        #             print(f"🏥 开始手术式加载: {checkpoint_path}")
        #             print(f"  总键数: {len(full_state_dict)}")
                    
        #             # 1. 提取并加载Agent权重 (PolicyNet, QNet等)
        #             agent_keys = {k: v for k, v in full_state_dict.items() 
        #                          if not k.startswith('belief.') and not k.startswith('ranker.')}
        #             agent_load_result = agent.load_state_dict(agent_keys, strict=False)
        #             print(f"  ✅ Agent加载: {len(agent_keys)}个键")
        #             if agent_load_result.missing_keys:
        #                 print(f"    缺失: {len(agent_load_result.missing_keys)}个")
        #                 print(f"    缺失键名: {agent_load_result.missing_keys}") # <-- 加上这行
                    
        #             # 2. 提取并加载Belief权重 (GRU, embeddings等)
        #             belief_keys = {k.replace('belief.', ''): v for k, v in full_state_dict.items() 
        #                           if k.startswith('belief.')}
        #             belief_load_result = belief_encoder.load_state_dict(belief_keys, strict=False)
        #             print(f"  ✅ Belief加载: {len(belief_keys)}个键")
        #             if belief_load_result.missing_keys:
        #                 print(f"    缺失: {len(belief_load_result.missing_keys)}个")
        #                 print(f"    缺失键名: {belief_load_result.missing_keys}")
                    
        #             # 3. 验证关键组件是否成功加载
        #             agent_success = len(agent_load_result.missing_keys) == 0
        #             belief_success = len(belief_load_result.missing_keys) <= 2  # 允许少量缺失
                    
        #             if agent_success and belief_success:
        #                 print(f"🎉 手术式加载成功!")
        #             else:
        #                 print(f"⚠️ 部分加载失败 - Agent: {agent_success}, Belief: {belief_success}")
                    
        #             checkpoint_loaded = True
        #             break
                    
        #         except Exception as e:
        #             print(f"⚠️ 手术式加载失败: {e}")
        #             import traceback
        #             traceback.print_exc()
        
        # if not checkpoint_loaded:
        #     print(f"⚠️ 未找到checkpoint，使用随机初始化的模型")
        
        # # 设置为评估模式并移到GPU
        # agent.eval()
        # agent = agent.to(self.device)
        # if ranker is not None:
        #     ranker.eval()
        #     ranker = ranker.to(self.device)
        # belief_encoder.eval()
        # belief_encoder = belief_encoder.to(self.device)
        
        # return agent, ranker, belief_encoder

    
    def load_diffuse_models(self, quality_level: str = "expert") -> Dict[str, Tuple[Any, Any, Any]]:
        """
        加载所有diffuse环境的SAC+GeMS模型

        Args:
            quality_level: 模型质量级别 ("expert", "medium", "random")
                - expert: 10w步训练的高质量模型
                - medium: 5w步训练的中等质量模型
                - random: 随机策略模型

        Returns:
            models: {env_name: (agent, ranker, belief_encoder)}
        """
        models = {}

        diffuse_envs = ['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen']

        # 根据质量级别选择模型目录
        models_base_dir = Path(__file__).resolve().parent.parent / "models" / quality_level

        for env_name in diffuse_envs:
            print(f"\n加载 {env_name} 环境的 {quality_level} 级别模型...")
            try:
                # 临时修改models_dir为指定质量级别的模型目录
                original_models_dir = self.models_dir
                self.models_dir = str(models_base_dir / env_name)

                # 加载SAC+GeMS模型
                agent, ranker, belief_encoder = self.load_agent(
                    env_name=env_name,
                    agent_type="SAC",
                    ranker_type="GeMS",  # 使用GeMS ranker
                    embedding_type="scratch"  # GeMS使用scratch embeddings
                )

                # 恢复原始models_dir
                self.models_dir = original_models_dir

                models[env_name] = (agent, ranker, belief_encoder)
                print(f"✅ {env_name} SAC+GeMS模型加载成功")
                print(f"   - Agent动作维度: {agent.action_dim}")
                print(f"   - Ranker类型: {type(ranker).__name__}")
                print(f"   - Ranker latent_dim: {ranker.latent_dim if hasattr(ranker, 'latent_dim') else 'N/A'}")
            except Exception as e:
                print(f"❌ {env_name} SAC+GeMS模型加载失败: {e}")
                import traceback
                traceback.print_exc()

        return models

    def load_focused_models(self, quality_level: str = "expert") -> Dict[str, Tuple[Any, Any, Any]]:
        """
        加载所有focused环境的SAC+GeMS模型

        Args:
            quality_level: 模型质量级别 ("expert", "medium", "random")
                - expert: 10w步训练的高质量模型
                - medium: 5w步训练的中等质量模型
                - random: 随机策略模型

        Returns:
            models: {env_name: (agent, ranker, belief_encoder)}
        """
        models = {}

        focused_envs = ['focused_topdown', 'focused_mix', 'focused_divpen']

        # 根据质量级别选择模型目录
        models_base_dir = Path(__file__).resolve().parent.parent / "models" / quality_level

        for env_name in focused_envs:
            print(f"\n加载 {env_name} 环境的 {quality_level} 级别模型...")
            try:
                # 临时修改models_dir为指定质量级别的模型目录
                original_models_dir = self.models_dir
                self.models_dir = str(models_base_dir / env_name)

                # 加载SAC+GeMS模型
                agent, ranker, belief_encoder = self.load_agent(
                    env_name=env_name,
                    agent_type="SAC",
                    ranker_type="GeMS",  # 使用GeMS ranker
                    embedding_type="scratch"  # GeMS使用scratch embeddings
                )

                # 恢复原始models_dir
                self.models_dir = original_models_dir

                models[env_name] = (agent, ranker, belief_encoder)
                print(f"✅ {env_name} SAC+GeMS模型加载成功")
                print(f"   - Agent动作维度: {agent.action_dim}")
                print(f"   - Ranker类型: {type(ranker).__name__}")
                print(f"   - Ranker latent_dim: {ranker.latent_dim if hasattr(ranker, 'latent_dim') else 'N/A'}")
            except Exception as e:
                print(f"❌ {env_name} SAC+GeMS模型加载失败: {e}")
                import traceback
                traceback.print_exc()

        return models

    def load_diffuse_models_topk(self) -> Dict[str, Tuple[Any, Any, Any]]:
        """
        加载所有diffuse环境的TopK模型（旧方法，仅用于对比）

        Returns:
            models: {env_name: (agent, ranker, belief_encoder)}
        """
        models = {}

        diffuse_envs = ['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen']

        for env_name in diffuse_envs:
            print(f"\n加载 {env_name} 环境的TopK模型...")
            try:
                agent, ranker, belief_encoder = self.load_agent(
                    env_name=env_name,
                    agent_type="SAC",
                    ranker_type="TopK",
                    embedding_type="ideal"
                )
                models[env_name] = (agent, ranker, belief_encoder)
                print(f"✅ {env_name} TopK模型加载成功")
            except Exception as e:
                print(f"❌ {env_name} TopK模型加载失败: {e}")

        return models

if __name__ == "__main__":
    # 测试模型加载
    print("测试模型加载器...")
    
    loader = ModelLoader()
    
    # 测试加载单个环境的模型
    try:
        agent, ranker, belief_encoder = loader.load_agent(
            env_name="diffuse_topdown",
            agent_type="SAC",
            ranker_type="TopK",
            embedding_type="ideal"
        )
        print("✅ 单个模型加载测试成功")
        print(f"  Agent类型: {type(agent).__name__}")
        print(f"  Ranker类型: {type(ranker).__name__}")
        print(f"  Belief Encoder类型: {type(belief_encoder).__name__}")
    except Exception as e:
        print(f"❌ 单个模型加载测试失败: {e}")
    
    # 测试加载所有diffuse模型
    try:
        models = loader.load_diffuse_models()
        print(f"\n✅ 成功加载 {len(models)} 个环境的模型")
        for env_name in models.keys():
            print(f"  - {env_name}")
    except Exception as e:
        print(f"❌ 批量模型加载失败: {e}")
    
    print("\n✅ 模型加载器测试完成!")
