#!/usr/bin/env python3
"""
Create Stanford Dogs-C: 13 ImageNet-C style corruptions of the Dogs test set.

Usage:
    python create_dogs_c.py \
        --input_dir  datasets/stanford_dogs \
        --output_dir datasets/stanford_dogs_c \
        --corruption all \
        --severity 1 2 3 4 5

Output structure:
    datasets/stanford_dogs_c/
        gaussian_noise/
            1/ ... 5/   ← ImageFolder-compatible (class subfolders)
        shot_noise/
            ...
"""
import os
import argparse
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from noise_utils import CORRUPTION_DICT, CORRUPTION_TYPES

warnings.filterwarnings('ignore')


def _get_test_image_paths(dogs_root):
    """Collect all test images from the Stanford Dogs dataset.

    The Dogs dataset exposes test images via test_list.mat.
    We load  'Images/<breed>/<file>.jpg'  for every entry in that list.
    Falls back to scanning all of Images/ if .mat files are absent.
    """
    import scipy.io
    root = Path(dogs_root)

    mat_path = root / 'test_list.mat'
    img_root = root / 'Images'

    if mat_path.exists():
        mat = scipy.io.loadmat(str(mat_path))
        # file_list: array of arrays, each containing a relative path string
        file_list = mat['file_list'].squeeze()
        # Each element is an ndarray wrapping the string
        paths = [img_root / str(f[0]) for f in file_list]
        print(f"Loaded {len(paths)} test paths from test_list.mat")
    else:
        # Fallback: all images under Images/
        paths = sorted(img_root.rglob('*.jpg')) + sorted(img_root.rglob('*.JPEG'))
        print(f"test_list.mat not found — using all {len(paths)} images under Images/")

    return paths


def create_dogs_c(dogs_root, output_dir, corruption_types, severities,
                  skip_existing=True):
    """Apply each corruption × severity to the Dogs test set.

    Output is ImageFolder-compatible: output_dir/corruption/severity/breed/file.jpg
    """
    root = Path(dogs_root)
    out = Path(output_dir)

    test_paths = _get_test_image_paths(dogs_root)
    if not test_paths:
        raise ValueError(f"No test images found in {dogs_root}")

    total_jobs = len(corruption_types) * len(severities)
    print(f"\nCorruptions : {len(corruption_types)}")
    print(f"Severities  : {severities}")
    print(f"Test images : {len(test_paths)}")
    print(f"Total jobs  : {total_jobs}  ({total_jobs * len(test_paths):,} images)\n")

    for corruption in corruption_types:
        if corruption not in CORRUPTION_DICT:
            print(f"WARNING: unknown corruption '{corruption}', skipping.")
            continue
        fn = CORRUPTION_DICT[corruption]

        for sev in severities:
            sev_dir = out / corruption / str(sev)

            # Pre-create all breed subdirs
            breeds = set()
            for p in test_paths:
                breeds.add(p.parent.name)
            for breed in breeds:
                (sev_dir / breed).mkdir(parents=True, exist_ok=True)

            pbar = tqdm(test_paths,
                        desc=f"{corruption}-{sev}",
                        unit='img', ncols=90)

            for img_path in pbar:
                breed = img_path.parent.name
                dst = sev_dir / breed / img_path.name

                if skip_existing and dst.exists():
                    continue

                try:
                    img = Image.open(img_path).convert('RGB')
                    corrupted = fn(img, sev)
                    if isinstance(corrupted, np.ndarray):
                        corrupted = Image.fromarray(
                            np.clip(corrupted, 0, 255).astype(np.uint8))
                    elif not isinstance(corrupted, Image.Image):
                        corrupted = img
                    corrupted.save(dst, quality=95)
                except Exception as e:
                    print(f"\n  ERROR {img_path.name}: {e} — saving clean copy")
                    img.save(dst)

            pbar.close()
            print(f"  ✓  {corruption} severity {sev} → {sev_dir}")

    print(f"\n{'='*60}")
    print("Stanford Dogs-C creation complete!")
    print(f"Output: {out}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Create Stanford Dogs-C (13 ImageNet-C corruptions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available corruptions:\n  " + "\n  ".join(CORRUPTION_TYPES)
    )
    parser.add_argument('--input_dir',  default='datasets/stanford_dogs',
                        help='Root of stanford_dogs dataset (contains Images/, test_list.mat, …)')
    parser.add_argument('--output_dir', default='datasets/stanford_dogs_c',
                        help='Output root for corrupted dataset')
    parser.add_argument('--corruption', nargs='+', default=['all'],
                        help='Corruption name(s) or "all"')
    parser.add_argument('--severity', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Severity levels 1-5')
    parser.add_argument('--no_skip', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()

    corruptions = CORRUPTION_TYPES if 'all' in args.corruption else args.corruption
    for s in args.severity:
        assert 1 <= s <= 5, f"Severity must be 1-5, got {s}"

    print("=" * 60)
    print("Stanford Dogs-C Dataset Creation")
    print("=" * 60)
    print(f"Input  : {args.input_dir}")
    print(f"Output : {args.output_dir}")
    print(f"Corruptions ({len(corruptions)}): {', '.join(corruptions)}")
    print(f"Severities  : {args.severity}")
    print("=" * 60)

    if not Path(args.input_dir).exists():
        raise FileNotFoundError(f"Input not found: {args.input_dir}")

    create_dogs_c(
        dogs_root=args.input_dir,
        output_dir=args.output_dir,
        corruption_types=corruptions,
        severities=args.severity,
        skip_existing=not args.no_skip,
    )


if __name__ == '__main__':
    main()
