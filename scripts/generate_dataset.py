"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Dataset Generation Script - Adapted for offline-slate-rl project structure
"""

import torch
import sys
from pathlib import Path
from argparse import ArgumentParser

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "envs" / "RecSim"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

from envs.RecSim.simulators import TopicRec
from envs.RecSim.logging_policies import EpsGreedyPolicy
from paths import ONLINE_DATASETS_DIR

parser = ArgumentParser()
parser.add_argument('--n_sess', type=int, required=True, help='Number of trajectories to generate.')
parser.add_argument('--path', type=str, default="default", help='Path to generated dataset (use "default" for auto-naming).')
parser.add_argument('--env_name', type=str, required=True, choices=["TopicRec"], help='Type of simulator environment.')
parser.add_argument('--seed', type=int, default=2021, help='Random seed.')

def get_elem(l, ch):
    for i, el in enumerate(l):
        if el.startswith(ch):
            return el
    return None

env_name_arg = get_elem(sys.argv, "--env_name=")
if env_name_arg is None:
    print("Usage: python generate_dataset.py --env_name=TopicRec --n_sess=100000 ...")
    sys.exit(1)

env_name = env_name_arg.split("=")[1]
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

# Use project paths for output
if args.path == "default":
    filename = label + "_" + args.click_model + "_random" + str(args.epsilon_pol) + "_" + str(args.n_sess // 1000) + "K"
    output_dir = str(ONLINE_DATASETS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    arg_dict["path"] = output_dir + "/" + filename
else:
    # Make sure output directory exists
    output_path = Path(args.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("=== Dataset Generation ===")
print("=" * 80)
print(f"Environment: {env_name}")
print(f"Sessions: {args.n_sess}")
print(f"Output: {arg_dict['path']}")
print("=" * 80)

env = env_class(**arg_dict)
env.set_policy(EpsGreedyPolicy, arg_dict)
env.generate_dataset(args.n_sess, arg_dict["path"])

print(f"### Dataset generation complete. Saved to: {arg_dict['path']}")
