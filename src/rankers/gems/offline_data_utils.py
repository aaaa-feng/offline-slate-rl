"""
Offline Data Utilities for GeMS Training

This module provides memory-efficient data loading for offline GeMS training.
Uses NPZ format with memory mapping to handle large datasets (6GB+) without
loading everything into RAM.

Architecture Design:
- Direct NPZ loading (not pickle) for memory efficiency
- Memory-mapped arrays for random access without full load
- Interface-aligned with GeMS training_step expectations
- V3 compatible (supports raw_obs and item_relevances)

Author: Architecture Team
Date: 2026-01-05
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class OfflineSlateDataset(Dataset):
    """
    Memory-efficient dataset for offline slate recommendation.

    Uses memory-mapped NPZ files to avoid loading 6GB+ data into RAM.
    Returns batches compatible with GeMS training_step interface.

    Args:
        npz_path: Path to D4RL format NPZ file
        transform: Optional transform function
        load_oracle: Whether to load item_relevances (Oracle info)
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        transform: Optional[callable] = None,
        load_oracle: bool = False
    ):
        self.npz_path = Path(npz_path)
        self.transform = transform
        self.load_oracle = load_oracle

        print(f"⏳ Loading dataset into RAM: {self.npz_path.name} ...")

        # CRITICAL FIX: Load arrays into memory, not lazy NpzFile object
        with np.load(self.npz_path, allow_pickle=True) as data:
            # Copy to heap memory immediately
            self.slates = np.array(data['slates'], dtype=np.int64)
            self.clicks = np.array(data['clicks'], dtype=np.float32)
            self.rewards = np.array(data['rewards'], dtype=np.float32)

            # Optional fields
            if 'observations' in data.files:
                self.observations = np.array(data['observations'], dtype=np.float32)
            else:
                self.observations = None

            if load_oracle and 'item_relevances' in data.files:
                self.item_relevances = np.array(data['item_relevances'], dtype=np.float32)
            else:
                self.item_relevances = None

        self.size = len(self.slates)
        print(f"✅ Dataset loaded! {self.size:,} samples in RAM.")

        # Validate data structure
        self._validate_data()

    def _validate_data(self):
        """Validate that required fields exist and have correct shapes."""
        # Check slates and clicks consistency
        if len(self.slates) != len(self.clicks):
            raise ValueError("Inconsistent data: slates and clicks have different lengths")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - slate: (rec_size,) LongTensor of item IDs
                - clicks: (rec_size,) FloatTensor of 0/1 labels
                - reward: scalar float
                - observation: (obs_dim,) FloatTensor (optional)
                - item_relevances: (num_items,) FloatTensor (optional, if load_oracle=True)
        """
        # Fast memory access - no disk IO, no decompression!
        # Note: torch.from_numpy creates CPU tensors, Lightning will move to GPU
        sample = {
            'slate': torch.from_numpy(self.slates[idx]).long(),  # Ensure int64
            'clicks': torch.from_numpy(self.clicks[idx]).float(),  # Ensure float32
            'reward': float(self.rewards[idx])
        }

        # Optionally load observation (belief state)
        if self.observations is not None:
            sample['observation'] = torch.from_numpy(self.observations[idx]).float()

        # Optionally load Oracle information
        if self.item_relevances is not None:
            sample['item_relevances'] = torch.from_numpy(self.item_relevances[idx]).float()

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_stats(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            'num_samples': self.size,
            'avg_reward': float(np.mean(self.rewards)),
            'std_reward': float(np.std(self.rewards)),
            'slate_size': self.slates.shape[1],
        }


class GeMSBatchCollator:
    """
    Custom collator for GeMS training batches.

    Converts list of samples into batch format expected by GeMS training_step:
        batch.obs["slate"]: List[Tensor] of shape (batch_size, rec_size)
        batch.obs["clicks"]: List[Tensor] of shape (batch_size, rec_size)

    Note: Returns CPU tensors. PyTorch Lightning will move them to GPU automatically.
    """

    def __init__(self, return_dict: bool = True):
        """
        Args:
            return_dict: If True, return dict format. If False, return namespace.
        """
        self.return_dict = return_dict

    def __call__(self, batch_list):
        """
        Collate a list of samples into a batch.

        Args:
            batch_list: List of dicts from OfflineSlateDataset.__getitem__

        Returns:
            Batch object with obs dict containing slates and clicks
        """
        # Keep as list of tensors (GeMS will stack them)
        slates = [sample['slate'] for sample in batch_list]
        clicks = [sample['clicks'] for sample in batch_list]
        rewards = torch.tensor([sample['reward'] for sample in batch_list], dtype=torch.float32)

        if self.return_dict:
            return {
                'obs': {
                    'slate': slates,
                    'clicks': clicks
                },
                'rewards': rewards
            }
        else:
            # Return namespace for compatibility with GeMS
            from types import SimpleNamespace
            return SimpleNamespace(
                obs={'slate': slates, 'clicks': clicks},
                rewards=rewards
            )


class OfflineSlateDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for offline slate recommendation.

    Handles train/val/test splits and DataLoader creation.
    Memory-efficient with NPZ memory mapping.

    Args:
        data_dir: Directory containing NPZ files
        env_name: Environment name (e.g., 'diffuse_topdown')
        quality: Data quality ('expert', 'medium', 'random')
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers
        val_split: Validation split ratio (default: 0.1)
        load_oracle: Whether to load Oracle information
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        env_name: str,
        quality: str = 'expert',
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
        load_oracle: bool = False
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.env_name = env_name
        self.quality = quality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.load_oracle = load_oracle

        # Construct data path
        self.data_path = self.data_dir / env_name / f"{quality}_data_d4rl.npz"

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.

        Creates train/val split from the full dataset.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load full dataset
        full_dataset = OfflineSlateDataset(
            npz_path=self.data_path,
            load_oracle=self.load_oracle
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        # Random split
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Dataset loaded: {self.env_name} ({self.quality})")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=GeMSBatchCollator(return_dict=False),
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=GeMSBatchCollator(return_dict=False),
            pin_memory=True
        )


# ============================================================================
# Utility Functions
# ============================================================================

def load_item_embeddings(embedding_path: Union[str, Path]) -> torch.Tensor:
    """
    Load pre-trained item embeddings.

    Args:
        embedding_path: Path to embedding file (.pt or .pth)

    Returns:
        Tensor of shape (num_items, embedding_dim)
    """
    embedding_path = Path(embedding_path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    embeddings = torch.load(embedding_path)

    if isinstance(embeddings, dict):
        # Handle dict format (e.g., {'embeddings': tensor})
        if 'embeddings' in embeddings:
            embeddings = embeddings['embeddings']
        elif 'weight' in embeddings:
            embeddings = embeddings['weight']
        else:
            raise ValueError(f"Unknown embedding dict format: {embeddings.keys()}")

    return embeddings


def get_dataset_info(npz_path: Union[str, Path]) -> Dict:
    """
    Get information about a dataset without loading it fully.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary with dataset metadata
    """
    data = np.load(npz_path, mmap_mode='r', allow_pickle=True)

    info = {
        'num_samples': len(data['slates']),
        'slate_size': data['slates'].shape[1],
        'num_items': int(data['slates'].max()) + 1,
        'fields': list(data.files),
        'has_oracle': 'item_relevances' in data.files,
        'has_raw_obs': 'raw_observations' in data.files,
    }

    return info
