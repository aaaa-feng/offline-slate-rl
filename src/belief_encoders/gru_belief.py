"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

import torch
import pytorch_lightning as pl 

from typing import List, Dict
import copy
from torch.nn import Embedding, Sequential, Linear, ReLU, Tanh, GRU, ModuleDict

from common.online.argument_parser import MyParser
from rankers.gems.item_embeddings import ItemEmbeddings
from rankers.gems.rankers import Ranker

####
# Belief encoders : GRU

class BeliefEncoder(pl.LightningModule):
    '''
        The belief encoder is responsible for projecting the sequence of past observations into a latent space.
    '''
    def __init__(self, item_embeddings : ItemEmbeddings, belief_state_dim : int, item_embedd_dim : int, rec_size : int, ranker : bool,
                    device : torch.device, belief_lr : float, hidden_layers_reduction : List[int], beliefs : List[str], **kwargs) -> None:
        super().__init__()

        self.my_device = device
        self.rec_size = rec_size
        self.item_embedd_dim = item_embedd_dim
        self.belief_state_dim = belief_state_dim
        self.ranker = ranker

        self.beliefs = beliefs

        self.item_embeddings = ModuleDict({})
        self.reduction = ModuleDict({})
        for module in beliefs:
            self.item_embeddings[module] = copy.deepcopy(item_embeddings)
            # Dimensionality reduction
            layers = []
            input_size = (item_embedd_dim + 1) * rec_size
            out_size = hidden_layers_reduction[:]
            out_size.append(self.belief_state_dim)
            for i, layer_size in enumerate(out_size):
                layers.append(Linear(input_size, layer_size))
                input_size = layer_size
                if i != len(out_size) - 1:
                    layers.append(ReLU())
            layers.append(Tanh())   # We constrain the latent space between -1 and 1
            self.reduction[module] = Sequential(*layers)

        self.belief_lr= belief_lr

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--belief_state_dim', type=int, default=8)
        parser.add_argument('--item_embedd_dim', type=int, default=8)
        parser.add_argument('--belief_lr', type=float, default=0.0005)
        parser.add_argument('--hidden_layers_reduction', type=int, nargs='+', default=[256, 64])
        parser.add_argument('--beliefs', type=str, nargs='+', default=["actor", "critic"])
        return parser

    def get_state_dim(self) -> int:
        return self.belief_state_dim

    def forward(self, obs : Dict[str, torch.LongTensor], done : bool = False) -> torch.FloatTensor:
        '''
            Encoding of observations, which are passed one by one (i.e. inference).
            If there is an explicit actor, we don't need to pass through critic belief as it is not going to be queried for action selection.
        '''
        if not done:
            # Then we can compute the new belief
            return torch.empty(0, device = self.device)   # belief_state_dim must be equal to 0 for MAB
    
    def forward_batch(self, batch) -> torch.FloatTensor:
        '''
            Encoding of a batch of trajectories (i.e. training).
        '''
        ### For this simple belief encoder, we flatten out the trajectories
        states, next_states = {}, {}
        clicks = torch.cat(batch.obs["clicks"], dim = 0) # (sum_seq_lengths, rec_size)

        for module in self.beliefs:
            states[module] = torch.empty(len(clicks), 0, device = self.device) # (sum_seq_lengths, 0)
            next_states[module] = torch.cat([states[module][1:].detach(), torch.empty(1, 0, device = self.device)], dim = 0)

        return states, next_states
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.belief_lr)
        return [optimizer]

class GRUBelief(BeliefEncoder):
    '''
        Passes all embeddings into a GRU (or two GRUs in actor critic) to retain memory from past observations.
    '''
    def __init__(self, hidden_dim : int, input_dim : int = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if hidden_dim is None:
            self.hidden_dim = self.belief_state_dim
            self.input_dim = self.rec_size * (self.item_embedd_dim + 1)
        else:
            self.hidden_dim = hidden_dim
            self.input_dim = input_dim

        self.reduction = None

        # Auto-calculate input_dim if not provided
        if self.input_dim is None:
            self.input_dim = self.rec_size * (self.item_embedd_dim + 1)

        self.gru = ModuleDict({})
        self.hidden = {}
        for module in self.beliefs:
            self.gru[module] = GRU(self.input_dim, self.hidden_dim, num_layers = 1, batch_first = True).to(self.my_device)
            self.hidden[module] = torch.zeros(1, 1, self.hidden_dim, device = self.my_device)

        # ✅ 新增：共享GRU（用于TD3+BC改进版）
        self.gru["shared"] = GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True).to(self.my_device)
        self.hidden["shared"] = torch.zeros(1, 1, self.hidden_dim, device=self.my_device)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[BeliefEncoder.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=None)
        return parser

    def forward(self, obs : Dict[str, torch.LongTensor], done : bool = False) -> torch.FloatTensor:
        '''
            Encoding of observations, which are passed one by one (i.e. inference).
            No need to use PL's truncated bptt as we never use this method for training.
        '''
        with torch.inference_mode():
            if not done:
                # Then we can compute the new belief
                states = {}
                if "actor" in self.beliefs:
                    item_embeddings = self.item_embeddings["actor"](obs["slate"])    # (rec_size, item_embedd_dim)
                    obs_embedd = torch.cat([item_embeddings, obs["clicks"].float().unsqueeze(1)], dim = 1).flatten() # (rec_size * (item_embedd_dim + 1))
                    # 确保 hidden state 与输入在同一设备
                    if self.hidden["actor"].device != obs_embedd.device:
                        self.hidden["actor"] = self.hidden["actor"].to(obs_embedd.device)
                    out, self.hidden["actor"] = self.gru["actor"](obs_embedd.unsqueeze(0).unsqueeze(1), self.hidden["actor"])
                    states["actor"] = out.squeeze()
                if "critic" in self.beliefs:
                    item_embeddings = self.item_embeddings["critic"](obs["slate"])    # (rec_size, item_embedd_dim)
                    obs_embedd = torch.cat([item_embeddings, obs["clicks"].float().unsqueeze(1)], dim = 1).flatten() # (rec_size * (item_embedd_dim + 1))
                    # 确保 hidden state 与输入在同一设备
                    if self.hidden["critic"].device != obs_embedd.device:
                        self.hidden["critic"] = self.hidden["critic"].to(obs_embedd.device)
                    out, self.hidden["critic"] = self.gru["critic"](obs_embedd.unsqueeze(0).unsqueeze(1), self.hidden["critic"])
                    states["critic"] = out.squeeze()
                if "belief" in self.beliefs:
                    item_embeddings = self.item_embeddings["belief"](obs["slate"])
                    obs_embedd = torch.cat([item_embeddings, obs["clicks"].float().unsqueeze(1)], dim = 1).flatten()
                    if self.hidden["belief"].device != obs_embedd.device:
                        self.hidden["belief"] = self.hidden["belief"].to(obs_embedd.device)
                    out, self.hidden["belief"] = self.gru["belief"](obs_embedd.unsqueeze(0).unsqueeze(1), self.hidden["belief"])
                    states["belief"] = out.squeeze()
                return states
            else:
                for module in self.beliefs:
                    self.hidden[module] = torch.zeros(1, 1, self.hidden_dim, device = self.my_device)
    
    def forward_batch(self, batch) -> torch.FloatTensor:
        '''
            Encoding of a batch of trajectories (i.e. training).
            Right now return ful episodes but we'll have to see how to make training more efficient.
        '''
        lens = [len(clicks) for clicks in batch.obs["clicks"]]
        batch_size = len(batch.obs["clicks"])

        item_embeddings, obs_embedd, states, next_states = {}, {}, {}, {}
        for module in self.beliefs:
            ### Pass through embeddings
            item_embeddings[module] = [self.item_embeddings[module](rl) for rl in batch.obs["slate"]]  # list( tensor(rec_size, item_embedd_dim), len = batch_size)
            obs_embedd[module] = [torch.cat([embedd, clicks.float().unsqueeze(2)], dim = 2).flatten(start_dim = 1) 
                            for embedd, clicks in zip(item_embeddings[module], batch.obs["clicks"])] # list( tensor(seq_len, rec_size * (item_embedd_dim + 1)), len = batch_size)

            obs_embedd[module] = torch.nn.utils.rnn.pad_sequence(obs_embedd[module], batch_first = True) # (batch_size, max_seq_len, rec_size * (item_embedd_dim + 1))
            obs_embedd[module] = torch.nn.utils.rnn.pack_padded_sequence(obs_embedd[module], lens, batch_first = True, enforce_sorted = False)
            
            ### Project into latent space
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device = self.my_device)

            states[module], _ = self.gru[module](obs_embedd[module], hidden) 
            states[module], _ = torch.nn.utils.rnn.pad_packed_sequence(states[module], batch_first = True)   # (batch_size, max_seq_len, belief_state_dim)
            states[module] = torch.cat([states[module][i, :lens[i], :] for i in range(batch_size)], dim = 0) # (sum_seq_lens, belief_state_dim)
            next_states[module] = torch.cat([states[module][1:].detach(), torch.zeros(1, self.hidden_dim, device = self.my_device)], dim = 0) # (sum_seq_lens, belief_state_dim)

        return states, next_states

    def forward_batch_shared(self, batch):
        """用于TD3+BC改进版的单流共享GRU前向传播"""
        lens = [len(clicks) for clicks in batch.obs["clicks"]]
        batch_size = len(batch.obs["clicks"])

        # 使用actor的item_embeddings
        item_embeddings = [self.item_embeddings["actor"](rl)
                           for rl in batch.obs["slate"]]

        obs_embedd = [torch.cat([embedd, clicks.float().unsqueeze(2)], dim=2).flatten(start_dim=1)
                      for embedd, clicks in zip(item_embeddings, batch.obs["clicks"])]

        obs_embedd = torch.nn.utils.rnn.pad_sequence(obs_embedd, batch_first=True)
        obs_embedd = torch.nn.utils.rnn.pack_padded_sequence(
            obs_embedd, lens, batch_first=True, enforce_sorted=False)

        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=self.my_device)
        states, _ = self.gru["shared"](obs_embedd, hidden)

        states, _ = torch.nn.utils.rnn.pad_packed_sequence(states, batch_first=True)
        states = torch.cat([states[i, :lens[i], :] for i in range(batch_size)], dim=0)

        # ⚠️ 关键：正确处理Episode边界
        next_states_list = []
        start_idx = 0
        for length in lens:
            episode_states = states[start_idx:start_idx+length]
            episode_next_states = torch.cat([
                episode_states[1:].detach(),
                torch.zeros(1, self.hidden_dim, device=self.my_device)
            ], dim=0)
            next_states_list.append(episode_next_states)
            start_idx += length

        next_states = torch.cat(next_states_list, dim=0)
        return states, next_states
