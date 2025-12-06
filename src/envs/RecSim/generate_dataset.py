"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

import torch
import sys
from argparse import ArgumentParser

from simulators import TopicRec
from logging_policies import EpsGreedyPolicy

parser = ArgumentParser()
parser.add_argument('--n_sess', type=int, required = True, help='Number of trajectories to generate.')
#parser.add_argument('--oracle_fraction', type=float, required = True, help='Proportion of oracle items.')
parser.add_argument('--path', type=str, default = "default", help='Path to generated dataset (use "default" for auto-naming).')
parser.add_argument('--env_name', type=str, required = True, choices=["TopicRec"], help='Type of simulator environment.')
parser.add_argument('--seed', type=int, default="2021", help='Random seed.')

def get_elem(l, ch):
    for i,el in enumerate(l):
        if el.startswith(ch):
            return el
env_name = get_elem(sys.argv, "--env_name=").split("=")[1]
if env_name == "TopicRec":
    env_class = TopicRec
    label = "topic"
else:
    raise NotImplementedError("This type of simulator environment has not been implemented yet.")

parser = env_class.add_model_specific_args(parser)
parser = EpsGreedyPolicy.add_model_specific_args(parser)
args = parser.parse_args()
arg_dict = vars(args)

# Seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 使用统一路径配置
if args.path == "default":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "config"))
    from paths import ONLINE_DATASETS_DIR

    filename = label + "_" + args.click_model + "_random" + str(args.epsilon_pol) + "_" + str(args.n_sess // 1000) + "K"
    arg_dict["path"] = str(ONLINE_DATASETS_DIR / filename)
else:
    # 用户指定了完整路径
    arg_dict["path"] = args.path

env = env_class(**arg_dict)
env.set_policy(EpsGreedyPolicy, arg_dict)
env.generate_dataset(args.n_sess, arg_dict["path"])
