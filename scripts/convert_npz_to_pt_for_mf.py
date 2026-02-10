"""
Convert NPZ format offline datasets to PT format for MF training.

This script converts offline RL datasets from NPZ format to the PT format
expected by the MF training pipeline (train_mf.py).

Input format (NPZ):
    - slates: (num_transitions, slate_size) - item IDs in each slate
    - clicks: (num_transitions, slate_size) - click indicators (0/1)
    - episode_ids: (num_transitions,) - episode ID for each transition

Output format (PT):
    {
        episode_id: {
            "slate": tensor([item_ids...]),  # all items from all transitions in this episode
            "clicks": tensor([0, 1, 0, ...])  # corresponding click indicators
        },
        ...
    }
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import os


def convert_npz_to_pt(npz_path: str, output_path: str, verbose: bool = False):
    """
    Convert NPZ dataset to PT format for MF training.

    Args:
        npz_path: Path to input NPZ file
        output_path: Path to output PT file
        verbose: Whether to print detailed information
    """
    if verbose:
        print("=" * 80)
        print("NPZ to PT Conversion for MF Training")
        print("=" * 80)
        print(f"Input:  {npz_path}")
        print(f"Output: {output_path}")
        print()

    # Load NPZ data
    if verbose:
        print("Loading NPZ data...")

    data = np.load(npz_path)
    slates = data['slates']  # (num_transitions, slate_size)
    clicks = data['clicks']  # (num_transitions, slate_size)
    episode_ids = data['episode_ids']  # (num_transitions,)

    num_transitions = len(episode_ids)
    slate_size = slates.shape[1]

    if verbose:
        print(f"  Transitions: {num_transitions:,}")
        print(f"  Slate size: {slate_size}")
        print(f"  Episode IDs range: [{episode_ids.min()}, {episode_ids.max()}]")
        print()

    # Get unique episode IDs
    unique_episodes = np.unique(episode_ids)
    num_episodes = len(unique_episodes)

    if verbose:
        print(f"Processing {num_episodes:,} episodes...")
        print()

    # Build PT format dictionary
    pt_data = {}

    for i, episode_id in enumerate(unique_episodes):
        # Find all transitions belonging to this episode
        episode_mask = episode_ids == episode_id
        episode_slates = slates[episode_mask]  # (num_transitions_in_episode, slate_size)
        episode_clicks = clicks[episode_mask]  # (num_transitions_in_episode, slate_size)

        # Flatten to create long tensors
        # Each episode's slates and clicks are concatenated into a single long tensor
        flat_slates = episode_slates.flatten()  # (num_transitions_in_episode * slate_size,)
        flat_clicks = episode_clicks.flatten()  # (num_transitions_in_episode * slate_size,)

        # Convert to tensors
        pt_data[int(episode_id)] = {
            "slate": torch.from_numpy(flat_slates).long(),
            "clicks": torch.from_numpy(flat_clicks).long()
        }

        # Progress indicator
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,} / {num_episodes:,} episodes ({100 * (i + 1) / num_episodes:.1f}%)")

    if verbose:
        print(f"  ✓ Processed all {num_episodes:,} episodes")
        print()

    # Save to PT file
    if verbose:
        print("Saving PT file...")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pt_data, output_path)

    # Check file size
    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_mb / 1024

    if verbose:
        print(f"  ✓ Saved to: {output_path}")
        print()

    # Print file size information
    print(f"✓ Conversion complete!")
    print(f"  Output file size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")

    if file_size_mb > 1024:
        print(f"  ⚠️  Warning: File size exceeds 1GB. Monitor memory usage during training.")

    print()
    print("Summary:")
    print(f"  Episodes (users): {num_episodes:,}")
    print(f"  Transitions: {num_transitions:,}")
    print(f"  Items per episode: ~{num_transitions * slate_size // num_episodes}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ offline datasets to PT format for MF training"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to input NPZ file (e.g., data/datasets/offline/mix_divpen/mix_divpen_v2_b3_data_d4rl.npz)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output PT file (e.g., data/datasets/offline/mix_divpen/mix_divpen_v2_b3.pt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during conversion"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.npz_path).exists():
        print(f"Error: Input file not found: {args.npz_path}")
        return 1

    # Run conversion
    try:
        convert_npz_to_pt(args.npz_path, args.output_path, args.verbose)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
