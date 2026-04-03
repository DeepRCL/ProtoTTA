#!/usr/bin/env python3
"""
Script to split cropped images into train/test sets using train_test_split.txt.
This script reads the train_test_split.txt file and organizes cropped images accordingly.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

def load_train_test_split(split_file):
    """Load train/test split information from file."""
    train_images = []
    test_images = []
    
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_id = int(parts[0])
                is_training = int(parts[1])
                
                if is_training == 1:
                    train_images.append(image_id)
                else:
                    test_images.append(image_id)
    
    return train_images, test_images

def load_image_mapping(images_file):
    """Load image ID to filename mapping."""
    image_mapping = {}
    with open(images_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_id = int(parts[0])
                image_path = parts[1]
                image_mapping[image_id] = image_path
    return image_mapping

def copy_images_to_split(image_ids, image_mapping, source_dir, target_dir):
    """Copy images to train or test directory maintaining class structure."""
    copied = 0
    failed = 0
    
    for image_id in image_ids:
        if image_id in image_mapping:
            image_path = image_mapping[image_id]
            source_path = os.path.join(source_dir, image_path)
            target_path = os.path.join(target_dir, image_path)
            
            try:
                # Create target directory if it doesn't exist
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_path, target_path)
                copied += 1
                
                if copied % 100 == 0:
                    print(f"Copied {copied} images...")
                    
            except Exception as e:
                print(f"Error copying {source_path} to {target_path}: {e}")
                failed += 1
        else:
            print(f"Warning: No image mapping found for image ID {image_id}")
            failed += 1
    
    return copied, failed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split cropped CUB-200-2011 images into train/test sets"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./datasets/CUB_200_2011',
        help='Path to CUB-200-2011 dataset directory (default: ./datasets/CUB_200_2011)'
    )
    parser.add_argument(
        '--cropped_dir',
        type=str,
        default='./datasets/cub200_cropped/cropped',
        help='Directory containing cropped images (default: ./datasets/cub200_cropped/cropped)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./datasets/cub200_cropped',
        help='Output directory for split datasets (default: ./datasets/cub200_cropped)'
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set up paths
    dataset_dir = Path(args.dataset_dir)
    cropped_dir = Path(args.cropped_dir)
    output_dir = Path(args.output_dir)
    split_file = dataset_dir / "train_test_split.txt"
    images_file = dataset_dir / "images.txt"
    
    train_dir = output_dir / "train_cropped"
    test_dir = output_dir / "test_cropped"
    
    # Validate input paths
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please download the CUB-200-2011 dataset and specify the correct path.")
        sys.exit(1)
    
    if not cropped_dir.exists():
        print(f"Error: Cropped images directory not found: {cropped_dir}")
        print("Please run crop_images.py first to create cropped images.")
        sys.exit(1)
    
    if not split_file.exists():
        print(f"Error: Train/test split file not found: {split_file}")
        sys.exit(1)
    
    if not images_file.exists():
        print(f"Error: Images file not found: {images_file}")
        sys.exit(1)
    
    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Cropped images directory: {cropped_dir}")
    print(f"Output directory: {output_dir}")
    print("\nLoading train/test split...")
    train_images, test_images = load_train_test_split(str(split_file))
    print(f"Found {len(train_images)} training images and {len(test_images)} test images")
    
    print("Loading image mappings...")
    image_mapping = load_image_mapping(str(images_file))
    print(f"Loaded {len(image_mapping)} image mappings")
    
    # Copy training images
    print("\nCopying training images...")
    train_copied, train_failed = copy_images_to_split(
        train_images, image_mapping, str(cropped_dir), str(train_dir)
    )
    
    # Copy test images
    print("\nCopying test images...")
    test_copied, test_failed = copy_images_to_split(
        test_images, image_mapping, str(cropped_dir), str(test_dir)
    )
    
    print(f"\nDataset splitting completed!")
    print(f"Training set: {train_copied} images copied, {train_failed} failed")
    print(f"Test set: {test_copied} images copied, {test_failed} failed")
    print(f"Training images saved to: {train_dir}")
    print(f"Test images saved to: {test_dir}")


if __name__ == "__main__":
    main()
