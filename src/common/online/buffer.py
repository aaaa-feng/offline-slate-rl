"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

在线RL的经验回放缓冲区
支持动态添加经验，用于与环境交互
"""

from collections import deque
from typing import List
import random

from recordclass import recordclass

Trajectory = recordclass("Trajectory", ("obs", "action", "reward", "next_obs", "done", "info"))


class ReplayBuffer():
    '''
        This ReplayBuffer class supports both tuples of experience and full trajectories,
        and it allows to never discard environment transitions for Offline Dyna.
    '''
    def __init__(self, offline_data: List[Trajectory], capacity: int) -> None:

        self.buffer_env = deque(offline_data, maxlen=capacity)
        self.buffer_model = deque([], maxlen=capacity)

    def push(self, buffer_type: str, *args) -> None:
        """Save a trajectory or tuple of experience"""
        if buffer_type == "env":
            self.buffer_env.append(Trajectory(*args))
        elif buffer_type == "model":
            self.buffer_model.append(Trajectory(*args))
        else:
            raise ValueError("Buffer type must be either 'env' or 'model'.")

    def sample(self, batch_size: int, from_data: bool = False) -> List[Trajectory]:
        if from_data:
            return random.sample(self.buffer_env, batch_size)
        else:
            if len(self.buffer_env + self.buffer_model) < batch_size:
                return -1
            return random.sample(self.buffer_env + self.buffer_model, batch_size)

    def __len__(self) -> int:
        return len(self.buffer_env) + len(self.buffer_model)
