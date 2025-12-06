# -*- coding: utf-8 -*-
"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Online RL utilities module

Note: Use explicit imports to avoid circular import issues:
    from common.online.buffer import ReplayBuffer, Trajectory
    from common.online.data_module import BufferDataset, BufferDataModule
    from common.online.env_wrapper import EnvWrapper, get_file_name
    from common.online.argument_parser import MyParser, MainParser
"""

# Only export names that don't cause circular imports
from .buffer import ReplayBuffer, Trajectory
from .data_module import BufferDataset, BufferDataModule
from .argument_parser import MyParser, MainParser

# EnvWrapper is NOT imported here to avoid circular import with envs.RecSim.simulators
# Use: from common.online.env_wrapper import EnvWrapper, get_file_name
