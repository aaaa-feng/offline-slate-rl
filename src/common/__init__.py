# -*- coding: utf-8 -*-
"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Common utilities module
- logger.py: Shared SwanLab logger
- online/: Online RL utilities
- offline/: Offline RL utilities

Usage:
    from common.online.buffer import ReplayBuffer
    from common.offline.buffer import ReplayBuffer
    from common.logger import SwanlabLogger
"""

# Lazy imports - submodules are imported on demand
# This avoids dependency issues at package import time
