"""
Test script for checkpoint_utils module
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from common.offline.checkpoint_utils import resolve_gems_checkpoint, resolve_dataset_path

print("=" * 80)
print("Testing checkpoint_utils module")
print("=" * 80)

# Test new benchmarks
print("\n### Testing NEW benchmarks (V4 format)")
print("-" * 80)

try:
    path, lambda_click = resolve_gems_checkpoint('mix_divpen', 'v2_b5')
    print(f"✓ mix_divpen v2_b5:")
    print(f"  Checkpoint: {Path(path).name}")
    print(f"  Lambda click: {lambda_click}")
    print(f"  File exists: {Path(path).exists()}")
except Exception as e:
    print(f"✗ mix_divpen v2_b5 failed: {e}")

try:
    path, lambda_click = resolve_gems_checkpoint('topdown_divpen', 'v2_b3')
    print(f"\n✓ topdown_divpen v2_b3:")
    print(f"  Checkpoint: {Path(path).name}")
    print(f"  Lambda click: {lambda_click}")
    print(f"  File exists: {Path(path).exists()}")
except Exception as e:
    print(f"✗ topdown_divpen v2_b3 failed: {e}")

# Test old benchmarks
print("\n### Testing OLD benchmarks (V3 format)")
print("-" * 80)

try:
    path, lambda_click = resolve_gems_checkpoint('diffuse_mix', 'expert')
    print(f"✓ diffuse_mix expert:")
    print(f"  Checkpoint: {Path(path).name}")
    print(f"  Lambda click: {lambda_click}")
    print(f"  File exists: {Path(path).exists()}")
except Exception as e:
    print(f"✗ diffuse_mix expert failed: {e}")

# Test dataset paths
print("\n### Testing dataset path resolution")
print("-" * 80)

try:
    path = resolve_dataset_path('mix_divpen', 'v2_b5')
    print(f"✓ mix_divpen v2_b5 dataset:")
    print(f"  Path: {Path(path).name}")
    print(f"  File exists: {Path(path).exists()}")
except Exception as e:
    print(f"✗ mix_divpen v2_b5 dataset failed: {e}")

try:
    path = resolve_dataset_path('diffuse_mix', 'expert')
    print(f"\n✓ diffuse_mix expert dataset:")
    print(f"  Path: {Path(path).name}")
    print(f"  File exists: {Path(path).exists()}")
except Exception as e:
    print(f"✗ diffuse_mix expert dataset failed: {e}")

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
