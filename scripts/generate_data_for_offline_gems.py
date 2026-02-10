"""
数据收集脚本 - 为GeMS训练收集epsilon-greedy数据
支持 mix_divpen 和 topdown_divpen 环境
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

# Parse arguments
parser = ArgumentParser(description='Generate epsilon-greedy data for GeMS training')
parser.add_argument('--env_name', type=str, required=True,
                   choices=['mix_divpen', 'topdown_divpen'],
                   help='Environment name')
parser.add_argument('--n_sess', type=int, required=True,
                   help='Number of sessions to generate')
parser.add_argument('--seed', type=int, default=58407201,
                   help='Random seed')

# Add TopicRec-specific arguments
parser = TopicRec.add_model_specific_args(parser)
parser = EpsGreedyPolicy.add_model_specific_args(parser)

args = parser.parse_args()
arg_dict = vars(args)

# Seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Setup output path
output_dir = PROJECT_ROOT / "data" / "datasets" / "offline" / f"gems_data_{args.env_name}"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{args.env_name}_epsilon-greedy.pt"
arg_dict["path"] = str(output_path)

print("=" * 80)
print("=== GeMS Data Collection (Epsilon-Greedy) ===")
print("=" * 80)
print(f"Environment: {args.env_name}")
print(f"Click Model: {arg_dict.get('click_model', 'N/A')}")
print(f"Diversity Penalty: {arg_dict.get('diversity_penalty', 'N/A')}")
print(f"Epsilon: {arg_dict.get('epsilon_pol', 'N/A')}")
print(f"Sessions: {args.n_sess}")
print(f"Seed: {args.seed}")
print(f"Output: {output_path}")
print("=" * 80)

# Create environment
print("\nInitializing environment...")
env = TopicRec(**arg_dict)

# Set epsilon-greedy policy
print("Setting epsilon-greedy policy...")
env.set_policy(EpsGreedyPolicy, arg_dict)

# Generate dataset
print(f"\nGenerating {args.n_sess} sessions...")
env.generate_dataset(args.n_sess, arg_dict["path"])

print(f"\n{'=' * 80}")
print(f"✅ Dataset generation complete!")
print(f"📁 Saved to: {output_path}")
print(f"{'=' * 80}")
