#!/usr/bin/env python3
"""
Script to augment training images using the Augmentor library.
Applies rotation, skew, and shear transformations with random flips.
"""

import argparse
import os
import sys
from pathlib import Path

import Augmentor
from PIL import Image


def makedir(path):
    """
    Create directory if it does not exist in the file system.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Augment CUB-200-2011 training images")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets/cub200_cropped/train_cropped",
        help="Directory containing training images to augment (default: ./datasets/cub200_cropped/train_cropped)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/cub200_cropped/train_cropped_augmented",
        help="Output directory for augmented images (default: ./datasets/cub200_cropped/train_cropped_augmented)",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=10,
        help="Number of augmentations per transformation (default: 10)",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set up paths
    input_dir = Path(args.input_dir).absolute()
    output_dir = Path(args.output_dir).absolute()

    # Validate input path
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run split_dataset.py first to create training images.")
        sys.exit(1)

    # Create output directory
    makedir(str(output_dir))

    # Get all class folders
    folders = [
        os.path.join(input_dir, folder) for folder in next(os.walk(input_dir))[1]
    ]
    target_folders = [
        os.path.join(output_dir, folder) for folder in next(os.walk(input_dir))[1]
    ]

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(folders)} class folders to augment.")

    for idx in range(len(folders)):
        fd = folders[idx]
        tfd = target_folders[idx]
        print(f"\nProcessing folder {idx+1}/{len(folders)}: {fd}")
        makedir(tfd)

        # --- Check valid images ---
        valid_images = []
        for img_name in os.listdir(fd):
            img_path = os.path.join(fd, img_name)
            try:
                with Image.open(img_path) as img:
                    if img.width > 0 and img.height > 0:
                        valid_images.append(img_name)
                    else:
                        print(f"Skipping zero-size image: {img_path}")
            except Exception as e:
                print(f"Could not open image {img_path}: {e}")

        print(f"Found {len(valid_images)} valid images in folder.")

        if not valid_images:
            print("No valid images found, skipping folder.")
            continue

        # --- Rotation ---
        print("Applying rotation + flip...")
        p = Augmentor.Pipeline(
            source_directory=fd, output_directory=os.path.relpath(tfd, start=fd)
        )
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(args.num_augmentations):
            try:
                p.process()
            except Exception as e:
                print(f"Error during rotation augmentation: {e}")
        del p

        # --- Skew ---
        print("Applying skew + flip...")
        p = Augmentor.Pipeline(
            source_directory=fd, output_directory=os.path.relpath(tfd, start=fd)
        )
        p.skew(probability=1, magnitude=0.2)
        p.flip_left_right(probability=0.5)
        for i in range(args.num_augmentations):
            try:
                p.process()
            except Exception as e:
                print(f"Error during skew augmentation: {e}")
        del p

        # --- Shear ---
        print("Applying shear + flip...")
        p = Augmentor.Pipeline(
            source_directory=fd, output_directory=os.path.relpath(tfd, start=fd)
        )
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(args.num_augmentations):
            try:
                p.process()
            except Exception as e:
                print(f"Error during shear augmentation: {e}")
        del p

        print(f"Completed augmentations for folder: {fd}")

    print("\nAll augmentations complete!")
    print(f"Augmented images saved to: {output_dir}")


if __name__ == "__main__":
    main()
