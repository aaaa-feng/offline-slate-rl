"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

Matrix Factorization Training Script - Adapted for offline-slate-rl project structure
"""

import torch
import pytorch_lightning as pl
import random
from pathlib import Path

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
print(f"Output: {output_dir}")
print("=" * 80)

item_embeddings = MFEmbeddings(**arg_dict)
item_embeddings.train(dataset_path, output_dir)

print(f"### MF training complete. Embeddings saved to: {output_dir}")
