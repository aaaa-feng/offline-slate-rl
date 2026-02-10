"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Matrix Factorization Training Script - Adapted for offline-slate-rl project structure
"""

import torch
import pytorch_lightning as pl
import random
from pathlib import Path
from datetime import datetime

import sys
import os

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

from rankers.gems.argument_parser import MainParser
from rankers.gems.item_embeddings import MFEmbeddings
from paths import ONLINE_DATASETS_DIR, MF_EMBEDDINGS_DIR

argparser = MainParser()  # Program-wide parameters
argparser = MFEmbeddings.add_model_specific_args(argparser)  # MF-specific parameters
args = argparser.parse_args()
arg_dict = vars(args)

# Print execution command and parameters at the very beginning
print("=" * 100)
print("=== MF EMBEDDING TRAINING - EXECUTION RECORD ===")
print("=" * 100)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nFull Command:")
print(f"python {' '.join(sys.argv)}")
print(f"\nAll Parameters:")
for key, value in sorted(arg_dict.items()):
    print(f"  {key}: {value}")
print("=" * 100)
print()

# Seeds for reproducibility
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# Use project paths
dataset_dir = str(ONLINE_DATASETS_DIR) + "/"
output_dir = str(MF_EMBEDDINGS_DIR) + "/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset_path = dataset_dir + args.MF_dataset

print("=" * 80)
print("=== Matrix Factorization Training ===")
print("=" * 80)
print(f"Dataset: {dataset_path}")
if args.output_path:
    print(f"Output: {args.output_path} (custom path)")
else:
    print(f"Output: {output_dir} (default directory)")
print("=" * 80)

item_embeddings = MFEmbeddings(**arg_dict)
item_embeddings.train(dataset_path, output_dir, output_path=args.output_path)

if args.output_path:
    print(f"### MF training complete. Embeddings saved to: {args.output_path}")
else:
    print(f"### MF training complete. Embeddings saved to: {output_dir}")
