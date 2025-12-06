"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

在线RL的数据模块
基于PyTorch Lightning的DataModule
"""

import torch
import pytorch_lightning as pl
from typing import List

from .buffer import ReplayBuffer, Trajectory


class BufferDataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int) -> None:
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.buffer.sample(self.batch_size)


class BufferDataModule(pl.LightningDataModule):
    '''
        DataModule that serves batches to the agent.
    '''
    def __init__(self, batch_size: int, capacity: int, offline_data: List[Trajectory] = [], **kwargs) -> None:
        super().__init__()

        self.buffer = ReplayBuffer(offline_data, capacity)
        self.buffer_dataset = BufferDataset(self.buffer, batch_size)
        self.num_workers = 0

    def collate_fn(self, batch):
        if batch == [-1]:
            # Special case of num_steps < batch_size
            return 0
        batch = Trajectory(*zip(*batch[0]))
        if batch.next_obs[0] is None:   ## POMDP
            batch.obs = {key: [obs[key] for obs in batch.obs] for key in batch.obs[0].keys()}
            batch.next_obs = None
            batch.action = torch.cat(batch.action, dim=0)
            batch.reward = torch.cat(batch.reward, dim=0)
            batch.done = torch.cat(batch.done, dim=0)
            if batch.info[0] is not None:
                batch.info = torch.cat(batch.info, dim=0)
        else:                           ## MDP
            batch.obs = torch.stack(batch.obs)
            batch.next_obs = torch.stack(batch.next_obs)
            batch.action = torch.stack(batch.action)
            batch.reward = torch.stack(batch.reward, dim=0).squeeze()
            batch.done = torch.stack(batch.done, dim=0).squeeze()
        return batch

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn=self.collate_fn,
                                                num_workers=self.num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn=self.collate_fn,
                                                num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn=self.collate_fn,
                                                num_workers=self.num_workers, shuffle=False)
