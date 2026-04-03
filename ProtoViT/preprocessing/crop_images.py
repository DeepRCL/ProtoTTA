#!/usr/bin/env python3
"""
Script to crop CUB-200-2011 images using bounding box information.
This script reads bounding_boxes.txt and crops each image according to its bounding box.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image

def load_bounding_boxes(bbox_file):
    """Load bounding box information from file."""
    bboxes = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                image_id = int(parts[0])
                x, y, width, height = map(float, parts[1:5])
                bboxes[image_id] = (x, y, width, height)
    return bboxes

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

def crop_image(image_path, bbox, output_path):
    """Crop image using bounding box coordinates."""
    try:
        # Open the image
        with Image.open(image_path) as img:
            x, y, width, height = bbox
            
            # Convert to integers
            x, y, width, height = int(x), int(y), int(width), int(height)
            
            # Calculate crop coordinates (x, y, x+width, y+height)
            left = x
            top = y
            right = x + width
            bottom = y + height
            
            # Ensure coordinates are within image bounds
            img_width, img_height = img.size
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the cropped image
            cropped_img.save(output_path)
            return True
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crop CUB-200-2011 images using bounding box information"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./datasets/CUB_200_2011',
        help='Path to CUB-200-2011 dataset directory (default: ./datasets/CUB_200_2011)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./datasets/cub200_cropped',
        help='Output directory for cropped images (default: ./datasets/cub200_cropped)'
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set up paths
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    bbox_file = dataset_dir / "bounding_boxes.txt"
    images_file = dataset_dir / "images.txt"
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please download the CUB-200-2011 dataset and specify the correct path.")
        sys.exit(1)
    
    if not bbox_file.exists():
        print(f"Error: Bounding boxes file not found: {bbox_file}")
        sys.exit(1)
    
    if not images_file.exists():
        print(f"Error: Images file not found: {images_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print("\nLoading bounding boxes...")
    bboxes = load_bounding_boxes(str(bbox_file))
    print(f"Loaded {len(bboxes)} bounding boxes")
    
    print("Loading image mappings...")
    image_mapping = load_image_mapping(str(images_file))
    print(f"Loaded {len(image_mapping)} image mappings")
    
    # Process each image
    processed = 0
    failed = 0
    
    for image_id, image_path in image_mapping.items():
        if image_id in bboxes:
            # Construct full paths
            full_image_path = images_dir / image_path
            output_path = output_dir / "cropped" / image_path
            
            # Crop the image
            if crop_image(str(full_image_path), bboxes[image_id], str(output_path)):
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed} images...")
            else:
                failed += 1
        else:
            print(f"Warning: No bounding box found for image ID {image_id}")
            failed += 1
    
    print(f"\nCropping completed!")
    print(f"Successfully processed: {processed} images")
    print(f"Failed: {failed} images")
    print(f"Cropped images saved to: {output_dir / 'cropped'}")


if __name__ == "__main__":
    main()
