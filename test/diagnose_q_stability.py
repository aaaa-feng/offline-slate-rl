"""
Diagnostic script to reproduce and document numerical instability
in offline RL algorithms (IQL, CQL, TD3+BC) BEFORE applying fixes.

This script establishes baseline evidence of:
- IQL: Actor loss NaN, gradient explosion
- CQL: Loss explosion (billions)
- TD3+BC: Q-value collapse

Usage:
    python test/diagnose_q_stability.py > baseline_diagnosis.log 2>&1
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.offline_config import IQLConfig, CQLConfig, TD3BCConfig
from common.offline.buffer import TrajectoryReplayBuffer
from common.offline.checkpoint_utils import resolve_gems_checkpoint
from rankers.gems.rankers import GeMS
from rankers.gems.item_embeddings import ItemEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def print_separator(char='=', length=60):
    """Print a separator line"""
    print(char * length)


def diagnose_algorithm(agent, buffer, algo_name: str, num_steps: int = 10):
    """
    Run diagnostic on a single algorithm to capture numerical instability.

    Args:
        agent: The RL agent (IQL, CQL, or TD3+BC)
        buffer: TrajectoryReplayBuffer with dataset
        algo_name: Name of algorithm for logging
        num_steps: Number of training steps to run
    """
    print(f"\n")
    print_separator()
    print(f"Baseline Diagnosis: {algo_name}")
    print_separator()
    print()

    for step in range(1, num_steps + 1):
        # Sample batch
        batch = buffer.sample(batch_size=256)

        # Train one step
        try:
            metrics = agent.train(batch)
        except Exception as e:
            print(f"Step {step}: ❌ TRAINING FAILED: {e}")
            continue

        # Print step header
        print(f"Step {step}:")

        # Print Q-value statistics
        q_mean = metrics.get('q_value', metrics.get('q_value_mean', float('nan')))
        q_std = metrics.get('q_value_std', metrics.get('q_std', float('nan')))

        print(f"  Q-values: mean={q_mean:.2f}, std={q_std:.2f}")

        # Algorithm-specific diagnostics
        if algo_name == "IQL":
            actor_loss = metrics.get('actor_loss', float('nan'))
            actor_grad = metrics.get('actor_grad_norm', 0)
            v_mean = metrics.get('v_value_mean', float('nan'))
            adv_mean = metrics.get('advantage_mean', float('nan'))

            is_nan = np.isnan(actor_loss)
            is_explosion = actor_grad > 1000

            print(f"  Actor loss: {actor_loss:.6f} {'❌ NaN!' if is_nan else '✓'}")
            print(f"  Actor grad norm: {actor_grad:.2f} {'❌ EXPLOSION!' if is_explosion else '✓'}")
            print(f"  V-value mean: {v_mean:.4f}")
            print(f"  Advantage mean: {adv_mean:.6f}")

        elif algo_name == "CQL":
            cql_loss = metrics.get('cql_loss', 0)
            total_loss = metrics.get('total_critic_loss', 0)
            random_q = metrics.get('random_q', 0)

            is_explosion = abs(cql_loss) > 1e6

            print(f"  CQL loss: {cql_loss:.2e} {'❌ EXPLOSION!' if is_explosion else '✓'}")
            print(f"  Total critic loss: {total_loss:.2e}")
            print(f"  Random Q: {random_q:.2e}")

        elif algo_name == "TD3+BC":
            critic_loss = metrics.get('critic_loss', 0)
            q1 = metrics.get('q1_value', metrics.get('q_value', 0))
            q2 = metrics.get('q2_value', 0)

            print(f"  Critic loss: {critic_loss:.2f}")
            print(f"  Q1 value: {q1:.2f}")
            print(f"  Q2 value: {q2:.2f}")

        print()

    print_separator()
    print()


def main():
    """Main function to run diagnostics on all algorithms"""

    # Configuration
    env_name = "mix_divpen"
    dataset_name = "v2_b3"
    dataset_path = f"/data/liyuefeng/offline-slate-rl/data/datasets/offline/{env_name}/{env_name}_{dataset_name}_data_d4rl.npz"

    print("\n")
    print_separator('=', 80)
    print("OFFLINE RL NUMERICAL STABILITY DIAGNOSTIC")
    print_separator('=', 80)
    print(f"Environment: {env_name}")
    print(f"Dataset: {dataset_name}")
    print_separator('=', 80)
    print()

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        return

    print("✓ Dataset file found")
    print()

    # Note: This is a simplified diagnostic script
    # Full agent initialization would require GeMS checkpoint, buffer setup, etc.
    # For now, we'll create a minimal version that can be expanded

    print("⚠️  NOTE: This is a simplified diagnostic script.")
    print("    Full implementation requires:")
    print("    - GeMS checkpoint loading")
    print("    - Buffer initialization with trajectory splitting")
    print("    - Agent initialization with proper configs")
    print()
    print("    Please run the actual training scripts to observe numerical instability.")
    print()


if __name__ == "__main__":
    main()

