"""
Offline GeMS Training Script

Train GeMS VAE on offline V3 datasets collected from expert/medium policies.
Uses memory-efficient NPZ data loading with OfflineSlateDataModule.

Usage:
    python scripts/train_gems_offline.py \\
        --env_name diffuse_topdown \\
        --quality expert \\
        --embedding_path data/embeddings/item_embeddings_diffuse.pt \\
        --latent_dim 32 \\
        --lambda_KL 1.0 \\
        --lambda_click 0.2 \\
        --max_epochs 50 \\
        --batch_size 256 \\
        --seed 42

Author: Architecture Team
Date: 2026-01-05
"""

import torch
import pytorch_lightning as pl
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
from paths import GEMS_CKPT_DIR

# Import GeMS modules
from rankers.gems.rankers import GeMS
from rankers.gems.item_embeddings import ItemEmbeddings
from rankers.gems.offline_data_utils import OfflineSlateDataModule, load_item_embeddings

# Import logger
from common.online.logger import SwanlabLogger


def create_parser():
    """Create argument parser for offline GeMS training."""
    parser = ArgumentParser(description="Train GeMS on offline V3 datasets")

    # Data arguments
    parser.add_argument("--data_dir", type=str,
                       default="data/datasets/offline_v2",
                       help="Directory containing offline datasets")
    parser.add_argument("--env_name", type=str, required=True,
                       choices=["diffuse_topdown", "diffuse_mix", "diffuse_divpen"],
                       help="Environment name")
    parser.add_argument("--quality", type=str, default="expert",
                       choices=["expert", "medium", "random"],
                       help="Data quality level")

    # Model arguments
    parser.add_argument("--embedding_path", type=str, required=True,
                       help="Path to pre-trained item embeddings (.pt file)")
    parser.add_argument("--latent_dim", type=int, default=32,
                       help="Latent dimension for VAE")
    parser.add_argument("--hidden_layers_infer", type=int, nargs="+",
                       default=[512, 256],
                       help="Hidden layers for inference network")
    parser.add_argument("--hidden_layers_decoder", type=int, nargs="+",
                       default=[256, 512],
                       help="Hidden layers for decoder network")

    # Loss function weights
    parser.add_argument("--lambda_KL", type=float, default=1.0,
                       help="Weight for KL divergence loss")
    parser.add_argument("--lambda_click", type=float, default=1.0,
                       help="Weight for click prediction loss")
    parser.add_argument("--lambda_prior", type=float, default=1.0,
                       help="Weight for prior regularization")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum number of training epochs")
    parser.add_argument("--ranker_lr", type=float, default=3e-3,
                       help="Learning rate for ranker")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")

    # System arguments
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of DataLoader workers (use 0 to avoid NPZ multi-process issues)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--progress_bar", action="store_true",
                       help="Enable progress bar")

    # Model configuration
    parser.add_argument("--num_items", type=int, default=1000,
                       help="Number of items in catalog")
    parser.add_argument("--item_embedd_dim", type=int, default=20,
                       help="Item embedding dimension")
    parser.add_argument("--rec_size", type=int, default=10,
                       help="Recommendation slate size")
    parser.add_argument("--fixed_embedds", type=str, default="scratch",
                       help="Embedding mode (scratch/mf_fixed)")
    parser.add_argument("--ranker_sample", type=bool, default=False,
                       help="Whether to sample from ranker")

    # Logging arguments
    parser.add_argument("--exp_name", type=str, default="gems_offline",
                       help="Experiment name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name (auto-generated if not provided)")
    parser.add_argument("--swan_project", type=str, default=None,
                       help="Swanlab project name")
    parser.add_argument("--swan_workspace", type=str, default=None,
                       help="Swanlab workspace")
    parser.add_argument("--swan_description", type=str, default=None,
                       help="Swanlab description")
    parser.add_argument("--swan_tags", type=str, nargs="+", default=None,
                       help="Swanlab tags")
    parser.add_argument("--swan_mode", type=str, default="cloud",
                       choices=["cloud", "local", "disabled"],
                       help="Swanlab logging mode")
    parser.add_argument("--swan_logdir", type=str, default=None,
                       help="Swanlab log directory")
    parser.add_argument("--swan_run_id", type=str, default=None,
                       help="Swanlab run ID")
    parser.add_argument("--swan_resume", type=bool, default=False,
                       help="Resume Swanlab run")

    return parser


def main():
    """Main training function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"gems_{args.env_name}_{args.quality}_latent{args.latent_dim}_seed{args.seed}"

    # Print configuration
    print("=" * 80)
    print("=== Offline GeMS Training ===")
    print("=" * 80)
    print(f"Environment: {args.env_name}")
    print(f"Quality: {args.quality}")
    print(f"Data directory: {args.data_dir}")
    print(f"Embedding path: {args.embedding_path}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print()

    # Set random seed for reproducibility
    pl.seed_everything(args.seed)

    # Setup logger
    logger_kwargs = {
        "project": args.swan_project or args.exp_name,
        "experiment_name": args.run_name,
        "workspace": args.swan_workspace,
        "description": args.swan_description,
        "tags": args.swan_tags,
        "config": vars(args),
        "mode": args.swan_mode,
        "logdir": args.swan_logdir,
        "run_id": args.swan_run_id,
        "resume": args.swan_resume,
    }
    exp_logger = SwanlabLogger(**logger_kwargs)

    # Load item embeddings
    print("### Loading item embeddings...")
    embedding_tensor = load_item_embeddings(args.embedding_path)
    print(f"✓ Loaded embeddings: shape={embedding_tensor.shape}")

    # Create ItemEmbeddings wrapper
    item_embeddings = ItemEmbeddings.from_pretrained(
        args.embedding_path,
        device=torch.device(args.device)
    )

    # Setup DataModule
    print("### Setting up DataModule...")
    data_module = OfflineSlateDataModule(
        data_dir=args.data_dir,
        env_name=args.env_name,
        quality=args.quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        load_oracle=False
    )
    print("✓ DataModule created")

    # Create GeMS model
    print("### Creating GeMS model...")
    model_kwargs = {
        'item_embeddings': item_embeddings,
        'item_embedd_dim': args.item_embedd_dim,
        'device': torch.device(args.device),
        'rec_size': args.rec_size,
        'latent_dim': args.latent_dim,
        'lambda_click': args.lambda_click,
        'lambda_KL': args.lambda_KL,
        'lambda_prior': args.lambda_prior,
        'ranker_lr': args.ranker_lr,
        'fixed_embedds': args.fixed_embedds,
        'ranker_sample': args.ranker_sample,
        'hidden_layers_infer': args.hidden_layers_infer,
        'hidden_layers_decoder': args.hidden_layers_decoder,
    }
    ranker = GeMS(**model_kwargs)
    print("✓ GeMS model created")

    # Setup checkpoint directory and filename
    ckpt_dir = Path(GEMS_CKPT_DIR) / "offline"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = (f"GeMS_{args.env_name}_{args.quality}_"
                f"latent{args.latent_dim}_"
                f"beta{args.lambda_KL}_"
                f"click{args.lambda_click}_"
                f"seed{args.seed}")

    print(f"### Checkpoint will be saved to: {ckpt_dir}/{ckpt_name}.ckpt")

    # Create PyTorch Lightning Trainer
    print("### Creating Trainer...")
    trainer = pl.Trainer(
        enable_progress_bar=args.progress_bar,
        logger=exp_logger,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=str(ckpt_dir),
                filename=ckpt_name,
                save_top_k=1,
                mode='min'
            )
        ],
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1 if args.device == "cuda" else None,
        max_epochs=args.max_epochs
    )
    print("✓ Trainer created")

    # Start training
    print("\n" + "=" * 80)
    print("=== Starting Training ===")
    print("=" * 80)
    trainer.fit(ranker, data_module)

    print("\n" + "=" * 80)
    print("=== Training Complete ===")
    print("=" * 80)
    print(f"✓ Best checkpoint saved to: {ckpt_dir}/{ckpt_name}.ckpt")
    print("=" * 80)


if __name__ == "__main__":
    main()
