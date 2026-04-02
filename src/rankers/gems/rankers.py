"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

from abc import abstractmethod

import torch
import pytorch_lightning as pl

from typing import List, Tuple, Dict, Union
from torch.nn import Sequential, Embedding, Linear, Softmax, CrossEntropyLoss, BCEWithLogitsLoss, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .argument_parser import MyParser
from .item_embeddings import ItemEmbeddings
from .data_utils import Trajectory


class Ranker(pl.LightningModule):
    '''
        Abstract Ranker class.
    '''
    def __init__(self, item_embeddings : ItemEmbeddings, item_embedd_dim : int, device : torch.device,
                    rec_size : int, **kwargs) -> None:
        super().__init__()

        self.my_device = device
        self.rec_size = rec_size
        self.item_embedd_dim = item_embedd_dim
        self.item_embeddings = item_embeddings
        self.num_items = self.item_embeddings.num_items

        action_min = torch.min(self.item_embeddings.embedd.weight.data, dim = 0).values      #item_embedd_dim
        self.action_scale = (torch.max(self.item_embeddings.embedd.weight.data, dim = 0).values - action_min) / 2 #item_embedd_dim
        self.action_center = action_min + self.action_scale

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        arguments = [action.option_strings[0] for action in parser._actions]
        if '--num_items' not in arguments:
            parser.add_argument('--num_items', type=int, default = 1000)
        if '--item_embedd_dim' not in arguments:
            parser.add_argument('--item_embedd_dim', type=int, default = 20)
        if '--rec_size' not in arguments:
            parser.add_argument('--rec_size', type=int, default = 10)
        return parser

class TopKRanker(Ranker):
    '''
        Retrieves the k items closest to the latent action.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.modules = []

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[Ranker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the space of item embeddings.
        '''
        with torch.inference_mode():
            # Handle batch dimension: action shape [batch_size, item_embedd_dim] or [item_embedd_dim]
            if action.dim() == 1:
                action_vec = action.unsqueeze(1)  # [item_embedd_dim, 1]
            else:
                action_vec = action.squeeze(0).unsqueeze(1)  # [batch_size, item_embedd_dim] -> [item_embedd_dim, 1]

            similarity = torch.matmul(self.item_embeddings.get_weights(), action_vec).squeeze(1)  # [num_items]
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.topk(similarity, k = self.rec_size, sorted = True)[1]
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.topk(similarity[unique[counts == 1]], k = self.rec_size, sorted = True)[1]]

    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Inverse mapping: slate → action (approximate for TopKRanker).

            For TopKRanker, the inverse mapping is not unique (many actions can produce
            the same Top-K slate). We use simple average of item embeddings as a
            representative action, avoiding position weighting to prevent training
            target drift.

            Args:
                slates: [batch_size, rec_size] or [batch_size, traj_len, rec_size]
                clicks: Not used, kept for interface compatibility

            Returns:
                actions: [batch_size, item_embedd_dim]
                log_var: [batch_size, item_embedd_dim] (set to -10.0 for deterministic)
        '''
        # Handle batch of trajectories
        if len(slates.shape) == 3:
            slates = slates.flatten(end_dim=1)

        batch_size = slates.shape[0]

        # Get embeddings for all items in slates: [batch_size, rec_size, item_embedd_dim]
        slate_embeddings = self.item_embeddings.embedd(slates)

        # Simple average (no position weighting to avoid training target drift)
        actions = slate_embeddings.mean(dim=1)  # [batch_size, item_embedd_dim]

        # Log variance set to -10.0 (deterministic, exp(-10) ≈ 0)
        log_var = torch.full_like(actions, -10.0)

        return actions, log_var

class kHeadArgmaxRanker(TopKRanker):
    '''
        Retrieves the closest item for each slot of the slate
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.action_center = self.action_center.repeat(self.rec_size)
        self.action_scale = self.action_scale.repeat(self.rec_size)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[TopKRanker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim * self.rec_size, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim * self.rec_size, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be of size item_embedd_dim * rec_size.
        '''
        with torch.inference_mode():
            similarity = torch.matmul(self.item_embeddings.get_weights(), action.reshape(self.item_embedd_dim, self.rec_size))
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.argmax(similarity, dim = 0)
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.argmax(similarity[unique[counts == 1], :], dim = 0)]

    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Inverse mapping: slate → action (exact reconstruction for kHeadArgmaxRanker).

            For kHeadArgmaxRanker, the inverse mapping is exact because each position
            independently selects an item. The action format must match rank()'s expectation:
            action.reshape(item_embedd_dim, rec_size) where each column is a position's embedding.

            Args:
                slates: [batch_size, rec_size] or [batch_size, traj_len, rec_size]
                clicks: Not used, kept for interface compatibility

            Returns:
                actions: [batch_size, item_embedd_dim * rec_size]
                log_var: [batch_size, item_embedd_dim * rec_size] (set to -10.0 for deterministic)
        '''
        # Handle batch of trajectories
        if len(slates.shape) == 3:
            slates = slates.flatten(end_dim=1)

        batch_size = slates.shape[0]

        # Get embeddings for all items: [batch_size, rec_size, item_embedd_dim]
        slate_embeddings = self.item_embeddings.embedd(slates)

        # Transpose to [batch_size, item_embedd_dim, rec_size] then flatten
        # This ensures that after reshape(item_embedd_dim, rec_size), each column is a position's embedding
        actions = slate_embeddings.transpose(1, 2).flatten(start_dim=1)  # [batch_size, item_embedd_dim * rec_size]

        # Log variance set to -10.0 (deterministic, exp(-10) ≈ 0)
        log_var = torch.full_like(actions, -10.0)

        return actions, log_var

class AbstractGeMS(Ranker):
    '''
        Abstract parent for the GeMS family of model classes.
    '''

    def __init__(self, latent_dim : int, lambda_click : float, lambda_KL : float, lambda_prior : float,
                    ranker_lr : float, fixed_embedds : bool, ranker_sample : bool, **kwargs) -> None:
        super().__init__(**kwargs)

        self.modules = []

        self.latent_dim = latent_dim
        self.lambda_click = lambda_click
        self.lambda_KL = lambda_KL
        self.lambda_prior = lambda_prior
        self.lr = ranker_lr
        self.sample = ranker_sample

        # Item embeddings
        item_pre_embeddings = self.item_embeddings.get_weights() # Pre-trained/random item embeddings from PT Lightning
        self.item_embeddings = Embedding(self.num_items, self.item_embedd_dim)
        self.item_embeddings.weight.data.copy_(item_pre_embeddings)
        if fixed_embedds in ["mf_fixed"]: # Use frozen item embeddings
            self.item_embeddings.weight.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[Ranker.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--lambda_click', type=float, default=1.0)
        parser.add_argument('--lambda_KL', type=float, default=1.0)
        parser.add_argument('--lambda_prior', type=float, default=1.0)
        parser.add_argument('--latent_dim', type=int, default=8)
        parser.add_argument('--ranker_lr', type=float, default=3e-3)

        #### For ranker selection in RL4REC
        parser.add_argument('--ranker_dataset', type=str, default=None)
        parser.add_argument('--ranker_embedds', type=str, default=None)
        parser.add_argument('--ranker_seed', type=int, default=None)
        parser.add_argument('--ranker_sample', type=parser.str2bool, default=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.latent_dim, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.latent_dim, device = self.device) - 0.5)

    def get_action_bounds(self, data_path : str, batch_size : int = 10) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Returns the action bounds for continuous control inside GeMS's latent space.
        '''
        data = torch.load(data_path)

        action_min = 1e6 * torch.ones(self.latent_dim)
        action_max = -1e6 * torch.ones(self.latent_dim)
        for i in range(1000 // batch_size):
            # batch = {"slate" : torch.cat([traj["slate"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0),
            #             "clicks" : torch.cat([traj["clicks"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0)}
            ### 1 - Pass through embeddings
            slates = torch.stack([traj["slate"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0)           # batch_size, traj_len, rec_size
            clicks = torch.stack([traj["clicks"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0).float()  # batch_size, traj_len, rec_size

            ### 2 - Pass through inference model
            with torch.inference_mode():
                latent_mu, log_latent_var = self.run_inference(slates, clicks)

            latent_sigma = torch.exp(log_latent_var / 2)
            latent_min = torch.min(latent_mu - latent_sigma, dim = 0).values
            latent_max = torch.max(latent_mu + latent_sigma, dim = 0).values

            action_min = torch.minimum(action_min, latent_min)
            action_max = torch.maximum(action_max, latent_max)

        self.action_scale = (action_max - action_min).to(self.my_device) / 2
        self.action_center = action_min.to(self.my_device) + self.action_scale
        return self.action_center, self.action_scale

    @abstractmethod
    def rank(self, action) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the latent space of the VAE.
        '''
        pass

    @abstractmethod
    def run_inference(self, slates, clicks) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def run_decoder(self, latent_sample) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def run_prior(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        '''
            Pre-training of the ranker.
        '''
        ### 1 - Extract slates and clicks from batch
        slates = torch.stack(batch.obs["slate"]).to(self.device)            # batch_size, rec_size
        clicks = torch.stack(batch.obs["clicks"]).float().to(self.device)            # batch_size, rec_size

        ### 2 - Pass through inference model
        latent_mu, log_latent_var = self.run_inference(slates, clicks)

        ### 3 - Reparameterization trick
        latent_var = log_latent_var.exp()
        latent_sample = latent_mu + torch.randn_like(latent_var) * latent_var    # batch_size, latent_dim

        ### 4 - Pass through decoder model
        item_logits, click_logits = self.run_decoder(latent_sample)

        ### 5 - Pass through prior model
        prior_mu, log_prior_var = self.run_prior()           # batch_size * seq_len, latent_dim * 2
        prior_var = log_prior_var.exp()

        ### 6 - Compute the losses
        slate_loss = CrossEntropyLoss(reduction = 'mean')(item_logits, slates.flatten())   # Softmax is in the CrossEntropyLoss
        click_loss = BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks.flatten(end_dim = -2))
        mean_term = ((latent_mu - prior_mu) ** 2) / prior_var
        KLLoss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()
        prior_reg = torch.sum(prior_mu.pow(2) + log_prior_var.pow(2))

        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KLLoss + self.lambda_prior * prior_reg

        ### 7 - Compute accuracy metrics
        # Slate reconstruction accuracy (Top-1 and Top-5)
        slate_preds = item_logits.argmax(dim=-1)
        slate_top1_acc = (slate_preds == slates.flatten()).float().mean()

        # Top-5 accuracy
        _, slate_top5_preds = item_logits.topk(5, dim=-1)
        slate_top5_acc = (slate_top5_preds == slates.flatten().unsqueeze(-1)).any(dim=-1).float().mean()

        # Click prediction accuracy
        click_preds = (torch.sigmoid(click_logits) > 0.5).float()
        click_acc = (click_preds == clicks.flatten(end_dim=-2)).float().mean()

        ### 8 - Latent space statistics
        latent_mu_norm = latent_mu.norm(dim=-1).mean()
        latent_var_mean = latent_var.mean()

        ### 9 - Log all metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_slateloss", slate_loss, prog_bar=True)
        self.log("train_clickloss", click_loss, prog_bar=True)
        self.log("train_KLloss", KLLoss, prog_bar=True)
        self.log("train_prior_reg", prior_reg)

        # Accuracy metrics
        self.log("train_slate_top1_acc", slate_top1_acc, prog_bar=True)
        self.log("train_slate_top5_acc", slate_top5_acc)
        self.log("train_click_acc", click_acc, prog_bar=True)

        # Latent space metrics
        self.log("train_latent_mu_norm", latent_mu_norm)
        self.log("train_latent_var_mean", latent_var_mean)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        '''
            Validation step during pre-training of the ranker.
        '''
        ### 1 - Pass through embeddings
        slates = torch.stack(batch.obs["slate"]).to(self.device)            # batch_size, rec_size
        clicks = torch.stack(batch.obs["clicks"]).float().to(self.device)            # batch_size, rec_size

        ### 2 - Pass through inference model
        latent_mu, log_latent_var = self.run_inference(slates, clicks)

        ### 3 - Reparameterization trick
        latent_var = log_latent_var.exp()
        # latent_sample = latent_mu + torch.randn_like(latent_var) * latent_var    # batch_size, latent_dim

        ### 4 - Pass through decoder model
        item_logits, click_logits = self.run_decoder(latent_mu)

        ### 5 - Pass through prior model
        prior_mu, log_prior_var = self.run_prior()           # batch_size * seq_len, latent_dim * 2
        prior_var = log_prior_var.exp()

        ### 6 - Compute the losses
        slate_loss = CrossEntropyLoss(reduction = 'mean')(item_logits, slates.flatten())   # Softmax is in the CrossEntropyLoss
        click_loss = BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks.flatten(end_dim = -2))
        mean_term = ((latent_mu - prior_mu) ** 2) / prior_var
        KLLoss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()
        prior_reg = torch.sum(prior_mu.pow(2) + log_prior_var.pow(2))

        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KLLoss # + self.lambda_prior * prior_reg

        ### 7 - Compute accuracy metrics
        # Slate reconstruction accuracy (Top-1 and Top-5)
        slate_preds = item_logits.argmax(dim=-1)
        slate_top1_acc = (slate_preds == slates.flatten()).float().mean()

        # Top-5 accuracy
        _, slate_top5_preds = item_logits.topk(5, dim=-1)
        slate_top5_acc = (slate_top5_preds == slates.flatten().unsqueeze(-1)).any(dim=-1).float().mean()

        # Click prediction accuracy
        click_preds = (torch.sigmoid(click_logits) > 0.5).float()
        click_acc = (click_preds == clicks.flatten(end_dim=-2)).float().mean()

        ### 8 - Latent space statistics
        latent_mu_norm = latent_mu.norm(dim=-1).mean()
        latent_var_mean = latent_var.mean()

        ### 9 - Log all metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_slateloss", slate_loss, prog_bar=True)
        self.log("val_clickloss", click_loss, prog_bar=True)
        self.log("val_KLloss", KLLoss, prog_bar=True)
        self.log("val_prior_reg", prior_reg)

        # Accuracy metrics
        self.log("val_slate_top1_acc", slate_top1_acc, prog_bar=True)
        self.log("val_slate_top5_acc", slate_top5_acc)
        self.log("val_click_acc", click_acc, prog_bar=True)

        # Latent space metrics
        self.log("val_latent_mu_norm", latent_mu_norm)
        self.log("val_latent_var_mean", latent_var_mean)

        return loss

    # def encode(self, obs : Dict) -> torch.FloatTensor:
    #     with torch.inference_mode():
    #         ### 1 - Pass through embeddings
    #         slates = obs["slate"]                     # traj_len, rec_size
    #         clicks = obs["clicks"].float()            # traj_len, rec_size

    #         ### 2 - Pass through inference model
    #         latent_mu, log_latent_var = self.run_inference(slates, clicks)

    #     return latent_mu, log_latent_var

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]
        # return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2),
        #         'monitor': 'val_loss',
        #         }

class GeMS(AbstractGeMS):
    '''
        Slate-VAE.
    '''
    def __init__(self, hidden_layers_infer : List[int], hidden_layers_decoder : List[int], **kwargs) -> None:
        super().__init__(**kwargs)

        # Inference
        layers = []
        input_size = self.rec_size * (self.item_embedd_dim + 1)
        out_size = hidden_layers_infer[:]
        out_size.append(self.latent_dim * 2)    # mu and log_sigma
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.inference = Sequential(*layers)

        # Decoder
        layers = []
        input_size = self.latent_dim
        out_size = hidden_layers_decoder[:]
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            layers.append(ReLU())
        self.decoder = Sequential(*layers)
        self.slate_decoder = Linear(out_size[-1], self.rec_size * self.item_embedd_dim)
        self.click_decoder = Linear(out_size[-1], self.rec_size)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[AbstractGeMS.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--hidden_layers_infer', type=int, nargs='+', default=[512, 256])
        parser.add_argument('--hidden_layers_decoder', type=int, nargs='+', default=[256, 512])
        return parser

    def rank(self, action, clicked = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the latent space of the VAE.
        '''
        with torch.inference_mode():
            item_logits = self.slate_decoder(self.decoder(action)).reshape(self.rec_size, self.item_embedd_dim) \
                          @ self.item_embeddings.weight.t()
        if clicked is None:
            if self.sample:
                dist = torch.distributions.categorical.Categorical(logits = item_logits)
                return dist.sample()
            else:
                return torch.argmax(item_logits, dim = 1)   # rec_size
        else:   # Only with sample = False
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.argmax(item_logits[:, unique[counts == 1]], dim = 1)]

    def run_inference(self, slates, clicks) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if len(slates.shape) == 3 : # Batch of trajectories
            slates = slates.flatten(end_dim = 1)
            clicks = clicks.flatten(end_dim = 1)
        embedds = self.item_embeddings(slates).flatten(start_dim = 1)  # batch_size, rec_size * item_embedd_dim
        latent_params = self.inference(torch.cat([embedds, clicks], dim = 1))   # batch_size, latent_dim * 2
        return latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]

    def run_decoder(self, latent_sample) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = latent_sample.size()[0]
        reconstruction = self.decoder(latent_sample) # batch_size, hidden_layer_size
        item_logits = self.slate_decoder(reconstruction).reshape(batch_size * self.rec_size, self.item_embedd_dim) \
                      @ self.item_embeddings.weight.t().detach() # No backprop to item embeddings for that branch
        # batch_size * rec_size, num_items
        click_logits = self.click_decoder(reconstruction)   # batch_size, rec_size

        return item_logits, click_logits

    def decode_to_slate_logits(self, latent_sample):
        """将latent action解码为slate logits"""
        batch_size = latent_sample.size(0)
        reconstruction = self.decoder(latent_sample)
        slate_embeddings = self.slate_decoder(reconstruction).reshape(
            batch_size * self.rec_size, self.item_embedd_dim)
        item_logits = slate_embeddings @ self.item_embeddings.weight.t()
        return item_logits.reshape(batch_size, self.rec_size, self.num_items)

    def run_prior(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return torch.zeros(self.latent_dim, device = self.device), torch.zeros(self.latent_dim, device = self.device)


# ============================================================================
# Wolpertinger-style Rankers (借鉴自 rl_wolpertinger 项目)
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F


class WolpertingerActor(nn.Module):
    """
    Wolpertinger Actor 网络：生成 proto-action
    
    输入：state (GRU belief state)
    输出：proto-action (item embedding 空间中的连续向量)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(WolpertingerActor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim))
            layers.append(ReLU())
            prev_dim = dim
        layers.append(Linear(prev_dim, action_dim))
        
        self.network = Sequential(*layers)
    
    def forward(self, state):
        """输出 proto-action"""
        return self.network(state)


class WolpertingerRanker(Ranker):
    """
    Wolpertinger-style ranker: Actor → proto-action → kNN → Top-K
    
    核心思想：
    1. Actor 网络生成 proto-action（原型动作）
    2. 在 item embeddings 中进行 kNN 搜索
    3. 从 k 个候选中选择 Top-rec_size
    
    与 TopKRanker 的区别：
    - TopKRanker: 直接相似度排序
    - WolpertingerRanker: 先 kNN 筛选，再选择
    
    Action space: [item_embedd_dim] (20-dim)
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        k: int = 50,  # kNN 候选数量
        actor_hidden_dims: List[int] = None,
        state_dim: int = 20,  # GRU belief state 维度
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.k = min(k, self.num_items)  # 确保 k 不超过 item 总数
        
        # 创建 Actor 网络（注意：在联合训练中，这个 Actor 不会被使用）
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        
        self.actor = WolpertingerActor(
            state_dim=state_dim,
            action_dim=item_embedd_dim,
            hidden_dims=actor_hidden_dims
        )
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1  # 20
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        将 proto-action 解码为 slate
        
        Args:
            action: [batch_size, item_embedd_dim] - proto-action
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        # 处理批次维度
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            proto_action = action[i]  # [item_embedd_dim]
            
            # kNN 搜索：计算与所有 items 的欧氏距离
            distances = torch.linalg.norm(
                self.item_embeddings.get_weights() - proto_action.unsqueeze(0),
                dim=1
            )
            
            # 选择距离最小的 k 个 items
            topk_indices = torch.argsort(distances)[:self.k]
            
            # 从 k 个候选中选择 Top-rec_size
            # 简化版：直接选择最近的 rec_size 个
            slate = topk_indices[:self.rec_size]
            slates.append(slate)
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → proto-action
        
        使用 slate 中 items 的平均 embedding 作为 proto-action
        （与 TopKRanker 相同的策略）
        """
        slate_embeddings = self.item_embeddings(slates)  # [batch, rec_size, embedd_dim]
        proto_action = slate_embeddings.mean(dim=1)  # [batch, embedd_dim]
        log_var = torch.full_like(proto_action, -10.0)  # 确定性（低方差）
        return proto_action, log_var


class WolpertingerActorSlate(nn.Module):
    """
    Wolpertinger Slate Actor 网络：生成 proto-slate
    
    输入：state (GRU belief state)
    输出：proto-slate (rec_size 个 item embeddings)
    """
    def __init__(self, state_dim: int, action_dim: int, rec_size: int, hidden_dims: List[int]):
        super(WolpertingerActorSlate, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim))
            layers.append(ReLU())
            prev_dim = dim
        layers.append(Linear(prev_dim, action_dim * rec_size))
        
        self.network = Sequential(*layers)
    
    def forward(self, state):
        """输出 proto-slate"""
        return self.network(state)


class WolpertingerSlateRanker(Ranker):
    """
    Wolpertinger Slate ranker: Actor → proto-slate → 多位置 kNN → slate
    
    核心思想：
    1. Actor 网络生成 proto-slate（每个位置一个 proto-item）
    2. 对每个位置独立进行 kNN 搜索
    3. 选择每个位置最近的 item
    
    与 kHeadArgmaxRanker 的区别：
    - kHeadArgmaxRanker: 每个位置独立 argmax
    - WolpertingerSlateRanker: 每个位置独立 kNN
    
    Action space: [rec_size * item_embedd_dim] (200-dim)
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        k: int = 50,
        actor_hidden_dims: List[int] = None,
        state_dim: int = 20,
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.k = min(k, self.num_items)
        
        # 扩展 action_center/scale 以支持每个位置
        self.action_center = self.action_center.repeat(rec_size)
        self.action_scale = self.action_scale.repeat(rec_size)
        
        # 创建 Actor 网络
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        
        self.actor = WolpertingerActorSlate(
            state_dim=state_dim,
            action_dim=item_embedd_dim,
            rec_size=rec_size,
            hidden_dims=actor_hidden_dims
        )
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim * self.rec_size, 1  # 200
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        将 proto-slate 解码为 slate
        
        Args:
            action: [batch_size, rec_size * item_embedd_dim] - proto-slate
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            # Reshape 为 [rec_size, item_embedd_dim]
            proto_slate = action[i].reshape(self.rec_size, self.item_embedd_dim)
            
            slate = []
            for pos in range(self.rec_size):
                proto_item = proto_slate[pos]
                
                # 对每个位置做 kNN
                distances = torch.linalg.norm(
                    self.item_embeddings.get_weights() - proto_item.unsqueeze(0),
                    dim=1
                )
                topk_indices = torch.argsort(distances)[:self.k]
                
                # 选择最近的 item
                slate.append(topk_indices[0])
            
            slates.append(torch.tensor(slate, device=self.device))
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → proto-slate
        
        精确重构：每个位置的 embedding 按顺序排列
        （与 kHeadArgmaxRanker 相同的策略）
        """
        slate_embeddings = self.item_embeddings(slates)  # [batch, rec_size, embedd_dim]
        # Transpose 后 flatten: [batch, embedd_dim, rec_size] → [batch, embedd_dim * rec_size]
        proto_slate = slate_embeddings.transpose(1, 2).flatten(start_dim=1)
        log_var = torch.full_like(proto_slate, -10.0)
        return proto_slate, log_var


class GreedySlateRanker(Ranker):
    """
    Greedy slate generator: 迭代贪心选择，考虑累积效应
    
    核心思想：
    1. 迭代选择 items，每次选择使边际收益最大的 item
    2. 考虑已选 items 对后续选择的影响（累积分子/分母）
    3. 使用 mask 机制防止重复选择
    
    与 TopKRanker 的区别：
    - TopKRanker: 一次性选择 Top-K
    - GreedySlateRanker: 迭代选择，考虑累积效应
    
    Action space: [item_embedd_dim] (20-dim)
    
    参考：rl_slate_wolpertinger 项目的 GreedySlateGenerator
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        s_no_click: float = -1.0,  # 无点击的基准分数
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.s_no_click = s_no_click
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1  # 20
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        使用贪心算法生成 slate
        
        Args:
            action: [batch_size, item_embedd_dim] - 用于计算 item scores
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            # 计算所有 items 的评分（相似度）
            action_vec = action[i].unsqueeze(1)  # [item_embedd_dim, 1]
            scores = torch.matmul(
                self.item_embeddings.get_weights(),
                action_vec
            ).squeeze(1)  # [num_items]
            
            # 假设 Q-values 为 1（简化版）
            # 在实际应用中，可以从 Critic 网络获取 Q-values
            qvals = torch.ones_like(scores)
            
            # 贪心选择
            numerator = torch.tensor(0.0, device=self.device)
            denominator = torch.tensor(self.s_no_click, device=self.device)
            mask = torch.ones_like(qvals, dtype=torch.bool)
            
            slate = []
            for _ in range(self.rec_size):
                # 计算每个候选的边际收益
                # marginal_value = (累积奖励 + 新item奖励) / (累积评分 + 新item评分)
                marginal_value = (numerator + scores * qvals) / (denominator + scores)
                
                # 排除已选的 items
                marginal_value[~mask] = float('-inf')
                
                # 选择最大边际收益的 item
                k = torch.argmax(marginal_value)
                
                slate.append(k.item())
                mask[k] = False
                
                # 更新累积值
                numerator = numerator + scores[k] * qvals[k]
                denominator = denominator + scores[k]
            
            slates.append(torch.tensor(slate, device=self.device))
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → action
        
        使用 slate 中 items 的平均 embedding
        """
        slate_embeddings = self.item_embeddings(slates)
        action = slate_embeddings.mean(dim=1)
        log_var = torch.full_like(action, -10.0)
        return action, log_var
