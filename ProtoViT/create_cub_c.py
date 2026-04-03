#!/usr/bin/env python3
"""
Script to create CUB-200-C (CUB with corruptions).
Generates corrupted versions of the CUB test dataset for robustness evaluation.

Usage:
    python create_cub_c.py --input_dir ./datasets/cub200_cropped/test_cropped/ \
                           --output_dir ./datasets/cub200_c/ \
                           --corruption all \
                           --severity 1,2,3,4,5

This will create a directory structure like:
    ./datasets/cub200_c/
        gaussian_noise/
            1/  (severity 1)
            2/
            ...
            5/
        shot_noise/
            1/
            ...
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

# Import corruption functions
from noise_utils import CORRUPTION_DICT, get_all_corruption_types

warnings.filterwarnings('ignore')


def create_corrupted_dataset(input_dir, output_dir, corruption_types, severities, skip_existing=True):
    """
    Create corrupted versions of a dataset.
    
    Args:
        input_dir: Path to clean dataset (ImageFolder format)
        output_dir: Path to save corrupted datasets
        corruption_types: List of corruption types to apply
        severities: List of severity levels (1-5)
        skip_existing: Whether to skip already corrupted images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files organized by class
    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {input_dir}")
    
    print(f"Found {len(class_dirs)} classes")
    
    # Process each corruption type
    for corruption_type in corruption_types:
        print(f"\n{'='*60}")
        print(f"Processing corruption: {corruption_type}")
        print(f"{'='*60}")
        
        if corruption_type not in CORRUPTION_DICT:
            print(f"Warning: Unknown corruption type '{corruption_type}', skipping...")
            continue
        
        corruption_fn = CORRUPTION_DICT[corruption_type]
        
        # Process each severity level
        for severity in severities:
            print(f"\n  Severity level: {severity}")
            
            corruption_output_dir = output_path / corruption_type / str(severity)
            
            # Create directory structure for all classes
            for class_dir in class_dirs:
                class_name = class_dir.name
                (corruption_output_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Process all images
            total_images = sum(len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + 
                                  list(class_dir.glob('*.JPEG'))) for class_dir in class_dirs)
            
            pbar = tqdm(total=total_images, desc=f"  {corruption_type}-{severity}")
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                
                # Get all image files
                image_files = (list(class_dir.glob('*.jpg')) + 
                             list(class_dir.glob('*.png')) + 
                             list(class_dir.glob('*.JPEG')))
                
                for img_path in image_files:
                    output_img_path = corruption_output_dir / class_name / img_path.name
                    
                    # Skip if already exists
                    if skip_existing and output_img_path.exists():
                        pbar.update(1)
                        continue
                    
                    try:
                        # Load image
                        img = Image.open(img_path).convert('RGB')
                        
                        # Apply corruption
                        corrupted = corruption_fn(img, severity)
                        
                        # Handle different return types
                        if isinstance(corrupted, np.ndarray):
                            corrupted = Image.fromarray(corrupted.astype(np.uint8))
                        
                        # Save corrupted image
                        corrupted.save(output_img_path, quality=95)
                        
                    except Exception as e:
                        print(f"\n  Error processing {img_path}: {e}")
                        # Save original image as fallback
                        img.save(output_img_path)
                    
                    pbar.update(1)
            
            pbar.close()
            print(f"  ✓ Completed {corruption_type} severity {severity}")
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"Corrupted datasets saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Create CUB-200-C dataset with ImageNet-C style corruptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all corruptions with all severities
  python create_cub_c.py --corruption all
  
  # Create specific corruptions
  python create_cub_c.py --corruption gaussian_noise shot_noise --severity 3 5
  
  # Specify custom directories
  python create_cub_c.py --input_dir ./my_dataset/ --output_dir ./my_dataset_c/
  
Available corruption types:
  """ + '\n  '.join(get_all_corruption_types())
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./datasets/cub200_cropped/test_cropped/',
        help='Path to clean CUB test dataset (ImageFolder format)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./datasets/cub200_c/',
        help='Path to save corrupted datasets'
    )
    
    parser.add_argument(
        '--corruption',
        nargs='+',
        default=['all'],
        help='Corruption types to apply. Use "all" for all types, or specify: gaussian_noise shot_noise etc.'
    )
    
    parser.add_argument(
        '--severity',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Severity levels to generate (1-5). Default: all levels'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip images that already exist (useful for resuming)'
    )
    
    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='Regenerate all images even if they exist'
    )
    
    args = parser.parse_args()
    
    # Determine corruption types
    if 'all' in args.corruption:
        corruption_types = get_all_corruption_types()
    else:
        corruption_types = args.corruption
    
    # Validate severities
    for sev in args.severity:
        if sev < 1 or sev > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {sev}")
    
    skip_existing = args.skip_existing and not args.no_skip
    
    # Print configuration
    print("="*60)
    print("CUB-200-C Dataset Creation")
    print("="*60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Corruptions:      {len(corruption_types)} types")
    print(f"Severities:       {args.severity}")
    print(f"Skip existing:    {skip_existing}")
    print("="*60)
    
    if len(corruption_types) > 10:
        print("\nCorruption types (showing first 10):")
        for ct in corruption_types[:10]:
            print(f"  - {ct}")
        print(f"  ... and {len(corruption_types) - 10} more")
    else:
        print("\nCorruption types:")
        for ct in corruption_types:
            print(f"  - {ct}")
    
    # Verify input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Create corrupted dataset
    create_corrupted_dataset(
        args.input_dir,
        args.output_dir,
        corruption_types,
        args.severity,
        skip_existing=skip_existing
    )


if __name__ == '__main__':
    main()

