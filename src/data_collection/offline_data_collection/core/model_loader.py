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
    
    def load_ranker(self, env_name: str) -> Any:
        """
        创建 GeMS ranker 结构（用于 SAC+GeMS 模型加载）

        注意：
            - 此方法只创建 ranker 结构，不加载权重
            - 权重将从 SAC+GeMS checkpoint 中加载
            - 不再支持单独加载 GeMS checkpoint
            - 不再支持 TopK ranker

        Args:
            env_name: 环境名称

        Returns:
            ranker: GeMS ranker 对象（未加载权重）
        """
        config = self.env_configs[env_name]

        # 创建 scratch embeddings（权重将从 checkpoint 加载）
        item_embeddings = self.load_item_embeddings(env_name, "scratch")

        ranker = GeMS(
            item_embeddings=item_embeddings,
            item_embedd_dim=config['item_embedd_dim'],
            rec_size=config['rec_size'],
            num_items=config['num_items'],
            latent_dim=32,
            hidden_layers_infer=[512, 256],
            hidden_layers_decoder=[256, 512],
            device=self.device,
            lambda_click=0.5,  # 占位值，将从 checkpoint 加载
            lambda_KL=0.5,
            lambda_prior=0.0,
            ranker_lr=0.001,
            fixed_embedds=False,
            ranker_sample=False
        )

        return ranker

    def load_model(self, env_name: str,
                   checkpoint_path: Optional[str] = None,
                   quality: str = "expert",
                   beta: float = 1.0,
                   lambda_click: float = 0.5) -> Tuple[Any, Any, Any]:
        """
        加载完整的模型（load_agent 的别名方法）

        参数说明请参考 load_agent() 方法
        """
        return self.load_agent(env_name, checkpoint_path, quality, beta, lambda_click)
    
    def load_agent(self, env_name: str,
                   checkpoint_path: Optional[str] = None,
                   quality: str = "expert",
                   beta: float = 1.0,
                   lambda_click: float = 0.5) -> Tuple[Any, Any, Any]:
        """
        加载完整的 SAC+GeMS 模型（用于离线数据收集）

        Args:
            env_name: 环境名称 (diffuse_topdown, diffuse_mix, diffuse_divpen,
                               focused_topdown, focused_mix, focused_divpen)
            checkpoint_path: 可选，指定 checkpoint 文件的完整路径
                            如果不指定，将从 model_info.json 自动查找匹配的模型
            quality: 模型质量级别 (expert, medium, random)
                    仅在 checkpoint_path=None 时使用
            beta: GeMS beta 参数，仅在 checkpoint_path=None 时使用
            lambda_click: GeMS lambda_click 参数，仅在 checkpoint_path=None 时使用

        Returns:
            (agent, ranker, belief_encoder): 模型组件

        注意：
            - 只支持加载 SAC+GeMS 模型（agent_type=SAC, ranker_type=GeMS）
            - embeddings 和所有权重都从 checkpoint 加载，不支持随机初始化
            - 如果指定 checkpoint_path，将忽略 quality/beta/lambda_click 参数

        使用示例：
            # 自动加载（从 model_info.json）
            agent, ranker, belief = loader.load_agent("diffuse_topdown")

            # 指定参数自动加载
            agent, ranker, belief = loader.load_agent(
                "focused_topdown", quality="expert", beta=0.5, lambda_click=0.2
            )

            # 指定完整路径（用于测试）
            agent, ranker, belief = loader.load_agent(
                "diffuse_topdown",
                checkpoint_path="/path/to/SAC+GeMS_xxx.ckpt"
            )
        """
        config = self.env_configs[env_name]

        # ============================================================================
        # 第1步：确定 checkpoint 路径
        # ============================================================================

        # 如果没有指定 checkpoint_path，从 model_info.json 自动查找
        if checkpoint_path is None:
            checkpoint_path = self._get_checkpoint_from_config(
                env_name, quality, beta, lambda_click
            )
            print(f"📋 从配置文件自动选择模型:")
            print(f"   环境: {env_name}")
            print(f"   质量: {quality}")
            print(f"   参数: beta={beta}, lambda_click={lambda_click}")
            print(f"   文件: {os.path.basename(checkpoint_path)}")
        else:
            print(f"📦 使用指定的模型: {os.path.basename(checkpoint_path)}")

        # 验证文件存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint 文件不存在: {checkpoint_path}\n"
                f"请检查路径是否正确，或检查 model_info.json 配置"
            )

        print(f"✅ Checkpoint 文件存在，开始加载...")

        # ============================================================================
        # 第2步：创建模型组件结构
        # ============================================================================

        print(f"\n🔧 创建模型组件结构...")

        # 2.1 创建 Belief Encoder（使用 scratch embeddings，权重将从 checkpoint 加载）
        input_dim = config['rec_size'] * (config['item_embedd_dim'] + 1)  # 10 * (20 + 1) = 210
        belief_item_embeds = self.load_item_embeddings(env_name, "scratch")

        belief_encoder = GRUBelief(
            hidden_dim=config['belief_state_dim'],      # 20
            input_dim=input_dim,                        # 210
            item_embeddings=belief_item_embeds,
            belief_state_dim=config['belief_state_dim'],# 20
            item_embedd_dim=config['item_embedd_dim'],  # 20
            rec_size=config['rec_size'],                # 10
            ranker=True,
            device=self.device,
            belief_lr=0.001,
            hidden_layers_reduction=[256],
            beliefs=['actor', 'critic']
        )
        print(f"  ✅ Belief Encoder 结构已创建")

        # 2.2 创建 Ranker（使用 scratch embeddings，权重将从 checkpoint 加载）
        ranker = self.load_ranker(env_name)
        print(f"  ✅ GeMS Ranker 结构已创建")

        # 2.3 创建 SAC Agent
        action_dim = 32  # GeMS latent_dim
        agent = SAC(
            belief=belief_encoder,
            ranker=ranker,
            state_dim=config['belief_state_dim'],  # 20
            action_dim=action_dim,   # 32 for GeMS
            num_actions=1,  # 连续SAC模式
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
        print(f"  ✅ SAC Agent 结构已创建")

        # ============================================================================
        # 第3步：加载 checkpoint 并验证
        # ============================================================================

        print(f"\n📦 加载 checkpoint...")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['state_dict']
            print(f"  ✅ Checkpoint 加载成功，包含 {len(state_dict)} 个键")
        except Exception as e:
            raise RuntimeError(f"加载 checkpoint 失败: {e}")

        # 验证必要的组件是否存在
        print(f"\n🔍 验证 checkpoint 完整性...")
        required_components = {
            'ranker.item_embeddings.weight': 'Ranker embeddings',
            'belief.ranker.item_embeddings.weight': 'Belief embeddings',
            'PolicyNet.0.weight': 'Policy network',
            'QNet.0.weight': 'Q network'
        }

        missing_components = []
        for key, name in required_components.items():
            if key not in state_dict:
                missing_components.append(f"{name} ({key})")

        if missing_components:
            raise ValueError(
                f"❌ Checkpoint 不完整，缺少以下组件:\n" +
                "\n".join(f"     - {c}" for c in missing_components) +
                f"\n\n这可能不是一个有效的 SAC+GeMS checkpoint 文件。"
            )

        print(f"  ✅ Checkpoint 完整性验证通过")

        # 过滤训练状态相关的键
        model_keys_only = {
            k: v for k, v in state_dict.items()
            if not k.startswith(('q_optimizer.', 'pi_optimizer.', 'alpha_optimizer.',
                                'global_step', 'epoch'))
        }
        print(f"  ℹ️  过滤后保留 {len(model_keys_only)} 个模型权重键")

        # ============================================================================
        # 第4步：加载权重到 Agent
        # ============================================================================

        print(f"\n🔄 加载权重到模型...")

        try:
            load_result = agent.load_state_dict(model_keys_only, strict=False)
        except Exception as e:
            raise RuntimeError(f"加载权重失败: {e}")

        # 检查核心组件是否成功加载
        core_missing = [k for k in load_result.missing_keys
                        if k.startswith(('ranker.', 'belief.', 'QNet', 'PolicyNet'))]

        if core_missing:
            raise RuntimeError(
                f"❌ 核心组件加载失败，缺失 {len(core_missing)} 个键:\n" +
                "\n".join(f"     - {k}" for k in core_missing[:10]) +
                ("\n     ..." if len(core_missing) > 10 else "")
            )

        print(f"  ✅ Agent 权重加载成功")
        print(f"  ✅ Ranker 权重加载成功（包含 embeddings）")
        print(f"  ✅ Belief 权重加载成功（包含 embeddings）")

        # 验证 embeddings 是否成功加载（方案A的关键验证）
        if 'ranker.item_embeddings.weight' in state_dict:
            ranker_embed_shape = state_dict['ranker.item_embeddings.weight'].shape
            print(f"  ✅ Ranker embeddings 已从 checkpoint 加载: {ranker_embed_shape}")
        else:
            raise RuntimeError("❌ Checkpoint 中没有 ranker embeddings，无法加载")

        if 'belief.ranker.item_embeddings.weight' in state_dict:
            belief_embed_shape = state_dict['belief.ranker.item_embeddings.weight'].shape
            print(f"  ✅ Belief embeddings 已从 checkpoint 加载: {belief_embed_shape}")
        else:
            raise RuntimeError("❌ Checkpoint 中没有 belief embeddings，无法加载")

        # ============================================================================
        # 第5步：加载 action_bounds
        # ============================================================================

        print(f"\n🎯 加载 action bounds...")

        if 'action_center' in state_dict and 'action_scale' in state_dict:
            agent.action_center = state_dict['action_center'].to(self.device)
            agent.action_scale = state_dict['action_scale'].to(self.device)
            print(f"  ✅ Action bounds 已从 checkpoint 加载")
            print(f"     Center shape: {agent.action_center.shape}")
            print(f"     Scale shape: {agent.action_scale.shape}")

        else:
            # 如果 checkpoint 中没有，从数据集动态计算
            print(f"  ⚠️  Checkpoint 中没有 action_bounds，尝试从数据集计算...")

            # 使用统一路径配置获取数据集路径
            sys.path.insert(0, str(PROJECT_ROOT / "config"))
            from paths import get_online_dataset_path
            dataset_path = get_online_dataset_path(env_name)

            if os.path.exists(dataset_path):
                print(f"  📊 正在从数据集计算精确 Action Bounds: {dataset_path}")
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

        # ============================================================================
        # 第6步：设置评估模式并移到设备
        # ============================================================================

        print(f"\n🚀 最终设置...")

        agent.eval()
        agent = agent.to(self.device)
        ranker.eval()
        ranker = ranker.to(self.device)
        belief_encoder.eval()
        belief_encoder = belief_encoder.to(self.device)

        print(f"  ✅ 所有组件已设置为评估模式")
        print(f"  ✅ 所有组件已移到设备: {self.device}")

        print(f"\n{'='*80}")
        print(f"✅ SAC+GeMS 模型加载完成!")
        print(f"{'='*80}\n")

        return agent, ranker, belief_encoder

    def _get_checkpoint_from_config(self, env_name: str, quality: str,
                                    beta: float, lambda_click: float) -> str:
        """
        从 model_info.json 获取 checkpoint 路径（V2简化版本）

        Args:
            env_name: 环境名称
            quality: 模型质量级别 (expert, medium)
            beta: GeMS beta 参数（V2中已在JSON配置，此参数仅用于验证）
            lambda_click: GeMS lambda_click 参数（V2中已在JSON配置，此参数仅用于验证）

        Returns:
            checkpoint_path: checkpoint 文件的完整路径
        """
        import json

        # 构建 model_info.json 路径
        models_base_dir = Path(__file__).resolve().parent.parent / "models"
        config_path = models_base_dir / quality / "model_info.json"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 读取V2扁平化配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 直接读取模型信息（无需参数匹配）
        if env_name not in config['models']:
            available_envs = list(config['models'].keys())
            raise ValueError(f"环境 '{env_name}' 不在配置中。可用环境: {available_envs}")

        model_info = config['models'][env_name]

        # 构建完整路径（使用相对路径）
        checkpoint_path = models_base_dir / quality / model_info['filename']

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        print(f"📦 使用指定的模型: {Path(model_info['filename']).name}")
        return str(checkpoint_path)

    def load_diffuse_models(self, quality_level: str = "expert") -> Dict[str, Tuple[Any, Any, Any]]:
        """
        加载所有diffuse环境的SAC+GeMS模型（V2重构版本）

        Args:
            quality_level: 模型质量级别 ("expert", "medium")
                - expert: 100k步训练的高质量模型
                - medium: 50k步训练的中等质量模型

        Returns:
            models: {env_name: (agent, ranker, belief_encoder)}
        """
        models = {}
        diffuse_envs = ['diffuse_topdown', 'diffuse_mix', 'diffuse_divpen']

        # 根据质量级别选择模型目录
        models_base_dir = Path(__file__).resolve().parent.parent / "models" / quality_level

        # 读取model_info.json（V2扁平化结构）
        model_info_path = models_base_dir / "model_info.json"
        import json
        with open(model_info_path, 'r') as f:
            model_config = json.load(f)

        for env_name in diffuse_envs:
            print(f"\n加载 {env_name} 环境的 {quality_level} 级别模型...")
            try:
                # 直接读取模型信息（V2扁平化结构）
                model_info = model_config['models'].get(env_name)
                if not model_info:
                    print(f"⚠️ 未找到 {env_name} 的模型配置")
                    continue

                # 提取参数（直接访问，无需层级判断）
                params = model_info['parameters']
                env_config = model_info['env_config']

                beta = params['beta']
                lambda_click = params['lambda_click']

                print(f"  使用参数: beta={beta}, lambda_click={lambda_click}")

                # 构建checkpoint路径（相对路径）
                checkpoint_path = models_base_dir / model_info['filename']

                # 加载SAC+GeMS模型
                agent, ranker, belief_encoder = self.load_agent(
                    env_name=env_name,
                    checkpoint_path=str(checkpoint_path),
                    quality=quality_level,
                    beta=beta,
                    lambda_click=lambda_click
                )

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
                self.models_dir = str(models_base_dir / "sac_gems_models" / env_name)

                # 加载SAC+GeMS模型
                agent, ranker, belief_encoder = self.load_agent(
                    env_name=env_name,
                    quality=quality_level
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
