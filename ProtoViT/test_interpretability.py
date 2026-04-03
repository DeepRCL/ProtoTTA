#!/usr/bin/env python3
"""
Quick test script for the interpretability visualization system.
This verifies that all imports work and the basic structure is correct.
"""

import sys
import os

print("Testing interpretability visualization system...")
print("=" * 60)

# Test 1: Import checks
print("\n[Test 1] Checking imports...")
try:
    import torch
    print("✓ torch imported")
except ImportError as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    import interpretability_viz
    print("✓ interpretability_viz imported")
except ImportError as e:
    print(f"✗ interpretability_viz import failed: {e}")
    sys.exit(1)

try:
    from noise_utils import get_corrupted_transform
    print("✓ noise_utils.get_corrupted_transform imported")
except ImportError as e:
    print(f"✗ noise_utils import failed: {e}")
    sys.exit(1)

try:
    from preprocess import mean, std, undo_preprocess_input_function
    print("✓ preprocess utilities imported")
except ImportError as e:
    print(f"✗ preprocess import failed: {e}")
    sys.exit(1)

# Test 2: Check required functions exist
print("\n[Test 2] Checking functions exist...")
required_functions = [
    'run_comprehensive_interpretability',
    'run_batch_interpretability',
    'create_prototype_visualization',
    'get_top_k_prototypes',
    'save_prototype_patches',
    'draw_bounding_boxes',
]

for func_name in required_functions:
    if hasattr(interpretability_viz, func_name):
        print(f"✓ {func_name} exists")
    else:
        print(f"✗ {func_name} NOT FOUND")
        sys.exit(1)

# Test 3: Check directory structure
print("\n[Test 3] Checking directory structure...")
required_dirs = [
    './saved_models',
    './datasets',
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path} exists")
    else:
        print(f"⚠ {dir_path} not found (may be expected)")

# Test 4: Check settings module
print("\n[Test 4] Checking settings module...")
try:
    from settings import test_dir, img_size
    print(f"✓ test_dir: {test_dir}")
    print(f"✓ img_size: {img_size}")
    
    if os.path.exists(test_dir):
        print(f"✓ test_dir exists")
        # Count images
        import glob
        images = glob.glob(os.path.join(test_dir, '*/*.jpg')) + \
                 glob.glob(os.path.join(test_dir, '*/*.png')) + \
                 glob.glob(os.path.join(test_dir, '*/*.JPEG'))
        print(f"  Found {len(images)} images in test_dir")
    else:
        print(f"⚠ test_dir does not exist: {test_dir}")
except ImportError as e:
    print(f"✗ settings import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run the interpretability analysis:")
print("\nOption 1 (Easy):")
print("  ./run_interpretability.sh")
print("\nOption 2 (Full control):")
print("  python run_inference.py \\")
print("      -corruption gaussian_noise \\")
print("      -severity 5 \\")
print("      -mode normal,eata,proto_importance_confidence \\")
print("      --use-geometric-filter \\")
print("      --geo-filter-threshold 0.92 \\")
print("      --consensus-strategy top_k_mean \\")
print("      --adaptation-mode layernorm_attn_bias \\")
print("      --interpretability-num-images 3")
print()

