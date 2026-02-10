"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

import torch
import pytorch_lightning as pl

import os
from torch.nn import Embedding
from typing import List, Tuple, Dict
from collections import namedtuple
from pathlib import Path

from .matrix_factorization.models import BPRMatrixFactorization
from .data_utils import MFDataset
from .argument_parser import MyParser


class ItemEmbeddings(pl.LightningModule):
    '''
        Base Embedding class.
    '''
    def __init__(self, num_items : int, item_embedd_dim : int, device : torch.device, weights = None, **kwargs) -> None:
        super().__init__()

        self.num_items = num_items
        self.embedd_dim = item_embedd_dim
        self.embedd = Embedding(num_items, item_embedd_dim, _weight = weights).to(device)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--MF_checkpoint', type=str, default = None)
        arguments = [action.option_strings[0] for action in parser._actions]
        if '--num_items' not in arguments:
            parser.add_argument('--num_items', type=int, default = 1000)
        if '--item_embedd_dim' not in arguments:
            parser.add_argument('--item_embedd_dim', type=int, default = 20)
        return parser

    def forward(self, items : torch.LongTensor) -> torch.FloatTensor:
        return self.embedd(items)

    @classmethod
    def from_pretrained(cls, checkpoint_path : str, device : torch.device):
        weights = torch.load(checkpoint_path)
        num_items, embedd_dim = weights.size()
        return cls(num_items, embedd_dim, weights = weights, device = device)

    @classmethod
    def get_from_env(cls, env, device, data_dir : str = None, embedd_path : str = None):
        embedd_weights = env.get_item_embeddings()
        num_items, embedd_dim = embedd_weights.size()
        return cls(num_items, embedd_dim, weights = embedd_weights, device = device)

    @classmethod
    def from_scratch(cls, num_items : int, embedd_dim : int, device : torch.device):
        return cls(num_items, embedd_dim, device = device)

    def clone_weights(self):
        return self.embedd.weight.data.clone()

    def get_weights(self):
        return self.embedd.weight.data

    def freeze(self):
        self.embedd.requires_grad_(False)

class MFEmbeddings(ItemEmbeddings):
    '''
        Matrix factorization with a BPR loss and trained with SGD. Courtesy of Thibaut Thonet.
    '''
    def __init__(self, train_val_split_MF : float, batch_size_MF : int, lr_MF : float, num_neg_sample_MF : int,
                    weight_decay_MF : float, patience_MF : int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.train_val_split = train_val_split_MF
        self.batch_size = batch_size_MF
        self.lr = lr_MF
        self.num_neg_sample = num_neg_sample_MF
        self.weight_decay = weight_decay_MF
        self.patience = patience_MF

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[ItemEmbeddings.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--MF_dataset', type=str, default = None)
        parser.add_argument('--output_path', type=str, default = None,
                          help='Custom output path for the trained embedding (full path including filename). If not provided, uses default naming.')
        parser.add_argument('--train_val_split_MF', type=float, default = 0.1)
        parser.add_argument('--batch_size_MF', type=int, default = 256)
        parser.add_argument('--lr_MF', type=float, default = 1e-4)
        parser.add_argument('--num_neg_sample_MF', type=int, default = 1)
        parser.add_argument('--weight_decay_MF', type=float, default = 0)
        parser.add_argument('--patience_MF', type=int, default = 3)
        return parser

    def collate_fn(self, batch : List[Tuple]) -> Dict:
        return {"user_ids" : torch.tensor([b[0] for b in batch], dtype = torch.long, device = self.device),
                "item_ids" : torch.tensor([b[1] for b in batch], dtype = torch.long, device = self.device)}

    def train(self, dataset_path : str, data_dir : str, output_path : str = None) -> None:
        '''
            Train MF item embeddings on pre-collected dataset.

            Args:
                dataset_path: Path to the training dataset
                data_dir: Default output directory (used if output_path is None)
                output_path: Custom output path (full path including filename). If provided, overrides data_dir.
        '''
        from datetime import datetime

        train_start_time = datetime.now()
        print("\n" + "=" * 80)
        print("=== MF TRAINING STARTED ===")
        print("=" * 80)
        print(f"Start Time: {train_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: {dataset_path}")
        if output_path:
            print(f"Output Path: {output_path} (custom)")
        else:
            print(f"Output Directory: {data_dir} (default)")
        print(f"Embedding Dimension: {self.embedd_dim}")
        print(f"Learning Rate: {self.lr}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Patience: {self.patience}")
        print("=" * 80)
        print()

        ### Loading the data and pre-processing
        print("Loading dataset...")
        data = torch.load(dataset_path)
        num_user = len(data)

        train_data = {k : val for k, val in enumerate(list(data.values())[int(num_user * self.train_val_split):])}
        val_data = {k : val for k, val in enumerate(list(data.values())[:int(num_user * self.train_val_split)])}

        train_dataset = MFDataset(data = train_data)
        val_dataset = MFDataset(data = val_data)

        print("\nDataset Statistics:")
        print(f"  Total users: {num_user}")
        print(f"  Training users: {len(train_data)}")
        print(f"  Validation users: {len(val_data)}")
        print(f"  Training interactions: {len(train_dataset)}")
        print(f"  Validation interactions: {len(val_dataset)}")
        print()

        train_gen = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size,
                                                    shuffle = True, collate_fn = self.collate_fn)
        val_gen = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size,
                                                    shuffle = True, collate_fn = self.collate_fn)

        Options = namedtuple("Options", ["lr_embedd", "embedd_dim", "num_neg_sample", "weight_decay_embedd"])
        options = Options(self.lr, self.embedd_dim, self.num_neg_sample, self.weight_decay)
        model = BPRMatrixFactorization(num_user, self.num_items, options, self.device, self.device)

        ### Training
        print("Starting training...")
        epoch = 0
        min_val_loss = 1e10
        count = 0
        best_epoch = 0

        # Determine final output path
        if output_path:
            # Use custom output path
            final_output_path = output_path
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(final_output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            # Use default naming (original behavior)
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            final_output_path = data_dir + dataset_path.split("/")[-1]

        while True :
            model.train()
            epoch_loss = 0.0
            train_loss = 0.0
            for (n, batch) in enumerate(train_gen):
                if n % 1000 == 0:
                    model.eval()
                    val_loss = 0.0
                    for (k, val_batch) in enumerate(val_gen):
                        loss = model(val_batch)
                        val_loss += loss.item()
                    val_loss /= (k + 1) # Divide by the number of batches
                    model.train()
                    print("  [Epoch %d, Batch %d] train_loss = %.4f | val_loss = %.4f | patience = %d/%d" %
                          (epoch, n, train_loss / 1000, val_loss, count, self.patience))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_epoch = epoch
                        torch.save(model.item_embeddings.weight.data, final_output_path)
                        print(f"  ✓ New best model saved! val_loss = {val_loss:.4f}")
                        print(f"  ✓ Saved to: {final_output_path}")
                        count = 0
                    else:
                        count += 1
                    if count == self.patience:
                        print(f"  Early stopping triggered (patience = {self.patience})")
                        break

                    train_loss = 0.0

                loss = model(batch)

                epoch_loss += loss.item()
                train_loss += loss.item()

            if count == self.patience:
                break

            epoch_loss /= (n + 1) # Divide by the number of batches

            print('Epoch {}: train_loss = {:.4f}'.format(epoch, epoch_loss))
            epoch += 1

        # Training summary
        train_end_time = datetime.now()
        train_duration = train_end_time - train_start_time

        # Load best model and compute statistics
        best_embeddings = torch.load(final_output_path)

        print("\n" + "=" * 80)
        print("=== MF TRAINING COMPLETED ===")
        print("=" * 80)
        print(f"End Time: {train_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training Duration: {train_duration}")
        print(f"\nTraining Summary:")
        print(f"  Total Epochs: {epoch}")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best Validation Loss: {min_val_loss:.4f}")
        print(f"\nEmbedding Statistics:")
        print(f"  Shape: {best_embeddings.shape}")
        print(f"  Mean: {best_embeddings.mean().item():.4f}")
        print(f"  Std: {best_embeddings.std().item():.4f}")
        print(f"  Min: {best_embeddings.min().item():.4f}")
        print(f"  Max: {best_embeddings.max().item():.4f}")
        print(f"\n✓ Final output saved to: {final_output_path}")
        print("=" * 80)
        print()
