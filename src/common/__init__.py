# -*- coding: utf-8 -*-
"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Common utilities module
- online/: Online RL utilities (includes logger with Lightning support)
- offline/: Offline RL utilities (includes lightweight logger)

Usage:
    # Online RL
    from common.online.buffer import ReplayBuffer
    from common.online.logger import SwanlabLogger

    # Offline RL
    from common.offline.buffer import ReplayBuffer
    from common.offline.logger import SwanlabLogger
"""

# Lazy imports - submodules are imported on demand
# This avoids dependency issues at package import time
