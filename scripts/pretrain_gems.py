"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

GeMS VAE Pretrain Script - Adapted for offline-slate-rl project structure
"""

import torch
import pytorch_lightning as pl
import random
from pathlib import Path

import sys
import os
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from argparse import ArgumentParser

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# Import path configuration
from paths import (
    get_online_dataset_path, get_gems_checkpoint_path,
    ONLINE_DATASETS_DIR, MF_EMBEDDINGS_DIR, GEMS_CKPT_DIR
)

# Import modules from new project structure
from rankers.gems.data_utils import SlateDataModule
from rankers.gems.rankers import GeMS
from rankers.gems.argument_parser import MainParser
from rankers.gems.item_embeddings import ItemEmbeddings, MFEmbeddings
from common.logger import SwanlabLogger

# Save original command line arguments for logging
_original_argv = sys.argv.copy()

main_parser = ArgumentParser()
main_parser.add_argument("--ranker", type=str, required=True, choices=["GeMS"], help="Ranker type")
main_parser.add_argument("--dataset", type=str, default=str(ONLINE_DATASETS_DIR / "focused_topdown.pt"), help="Path to dataset")
main_parser.add_argument("--item_embedds", type=str, required=True, choices=["scratch", "mf_init", "mf_fixed"], help="Item embeddings.")

def get_elem(l, ch):
    for i, el in enumerate(l):
        if el.startswith(ch):
            return el
    return None

ranker_name = get_elem(sys.argv, "--ranker=")
dataset_path = get_elem(sys.argv, "--dataset=")
item_embedds = get_elem(sys.argv, "--item_embedds=")

if ranker_name is None or item_embedds is None:
    print("Usage: python pretrain_gems.py --ranker=GeMS --dataset=<path> --item_embedds=scratch")
    print("Example:")
    print("  python scripts/pretrain_gems.py --ranker=GeMS --dataset=data/datasets/online/diffuse_topdown.pt --item_embedds=scratch --seed=58407201 --max_epochs=10")
    sys.exit(1)

main_args = main_parser.parse_args([ranker_name, dataset_path, item_embedds])
sys.argv.remove(ranker_name)
sys.argv.remove(dataset_path)
sys.argv.remove(item_embedds)

if main_args.ranker == "GeMS":
    ranker_class = GeMS
else:
    raise NotImplementedError("This ranker is not trainable or has not been implemented yet.")

if main_args.item_embedds in ["scratch"]:
    item_embedd_class = ItemEmbeddings
elif main_args.item_embedds in ["mf_init", "mf_fixed"]:
    item_embedd_class = MFEmbeddings
else:
    raise NotImplementedError("This type of item embeddings has not been implemented yet.")

argparser = MainParser()  # Program-wide parameters
argparser = ranker_class.add_model_specific_args(argparser)  # Ranker-specific parameters
argparser = item_embedd_class.add_model_specific_args(argparser)  # Item embeddings-specific parameters
args = argparser.parse_args(sys.argv[1:])
args.MF_dataset = main_args.dataset.split("/")[-1]

# Use project paths - MF embeddings go to mf_embeddings directory
embedd_dir = str(MF_EMBEDDINGS_DIR) + "/"
Path(embedd_dir).mkdir(parents=True, exist_ok=True)

if os.path.isfile(embedd_dir + args.MF_dataset):  # Check if the MF checkpoint already exists
    args.MF_checkpoint = args.MF_dataset
else:
    args.MF_checkpoint = None
arg_dict = vars(args)

# Print full command at the beginning
def print_full_command():
    """Print the full command that was used to run this script."""
    print("=" * 80)
    print("=== Full Command ===")
    print("=" * 80)
    full_cmd_parts = ["python", os.path.basename(__file__)]
    for arg in _original_argv[1:]:
        full_cmd_parts.append(arg)
    full_cmd = " ".join(full_cmd_parts)
    print(full_cmd)
    print("=" * 80)
    print("=== Starting Execution ===")
    print("=" * 80)
    print()

print_full_command()

# Seeds for reproducibility
seed = int(args.seed)
pl.seed_everything(seed)

logger_arg_dict = {**vars(args), **vars(main_args)}
logger_kwargs = {
    "project": args.swan_project or arg_dict["exp_name"],
    "experiment_name": args.run_name,
    "workspace": args.swan_workspace,
    "description": args.swan_description,
    "tags": args.swan_tags,
    "config": logger_arg_dict,
    "mode": args.swan_mode,
    "logdir": args.swan_logdir,
    "run_id": args.swan_run_id,
    "resume": args.swan_resume,
}
exp_logger = SwanlabLogger(**logger_kwargs)
exp_logger.log_hyperparams(logger_arg_dict)

# Item embeddings
arg_dict["item_embedds"] = main_args.item_embedds
if arg_dict["item_embedds"][-5:] == "fixed":
    arg_dict["fixed_embedds"] = True
else:
    arg_dict["fixed_embedds"] = False

if main_args.item_embedds in ["scratch"]:
    item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device=args.device)
elif main_args.item_embedds.startswith("mf"):
    if args.MF_checkpoint is None:
        item_embeddings = MFEmbeddings(**arg_dict)
        print("Pre-training MF embeddings ...")
        # Use the same dataset path as provided
        item_embeddings.train(main_args.dataset, embedd_dir)
        arg_dict["MF_checkpoint"] = args.MF_dataset
        print("Pre-training done.")
    item_embeddings = ItemEmbeddings.from_pretrained(embedd_dir + arg_dict["MF_checkpoint"], args.device)
    if main_args.item_embedds == "mf_fixed":
        item_embeddings.freeze()
else:
    raise NotImplementedError("This type of item embeddings have not been implemented yet.")

ranker = ranker_class(item_embeddings=item_embeddings, **arg_dict)

# Use project checkpoint directory
ckpt_dir = str(GEMS_CKPT_DIR) + "/"
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

ckpt_name = main_args.ranker + "_" + main_args.dataset.split("/")[-1][:-3] + "_latentdim" + str(arg_dict["latent_dim"]) + \
            "_beta" + str(arg_dict["lambda_KL"]) + "_lambdaclick" + str(arg_dict["lambda_click"]) + \
            "_lambdaprior" + str(arg_dict["lambda_prior"]) + "_" + arg_dict["item_embedds"] + "_seed" + str(args.seed)

trainer = pl.Trainer(
    enable_progress_bar=arg_dict["progress_bar"],
    logger=exp_logger,
    callbacks=[
        RichProgressBar(),
        ModelCheckpoint(monitor='val_loss', dirpath=ckpt_dir, filename=ckpt_name)
    ],
    accelerator="gpu" if arg_dict["device"] == "cuda" else "cpu",
    devices=1 if arg_dict["device"] == "cuda" else None,
    max_epochs=args.max_epochs
)

print("### Loading data and initializing DataModule ...")
data = torch.load(main_args.dataset, map_location=arg_dict["device"])
datamod = SlateDataModule(env=None, data=data, full_traj=False, **arg_dict)

print("### Launch training")
trainer.fit(ranker, datamod)

print(f"### Training complete. Checkpoint saved to: {ckpt_dir}{ckpt_name}.ckpt")
