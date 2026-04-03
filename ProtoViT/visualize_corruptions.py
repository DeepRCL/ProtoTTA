#!/usr/bin/env python3
"""
Visualize corruption examples on CUB dataset.
Creates a grid showing the same image with different corruptions applied.

Usage:
    python visualize_corruptions.py --image path/to/image.jpg
    python visualize_corruptions.py --random  # Use random image from test set
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

from noise_utils import CORRUPTION_DICT, get_all_corruption_types


def apply_corruption_to_pil(img, corruption_fn, severity):
    """Apply corruption to PIL image and return PIL image."""
    try:
        corrupted = corruption_fn(img, severity)
        if isinstance(corrupted, np.ndarray):
            corrupted = Image.fromarray(corrupted.astype(np.uint8))
        return corrupted
    except Exception as e:
        print(f"Warning: Corruption failed: {e}")
        return img


def create_corruption_grid(image_path, corruption_types, severities, output_path=None):
    """
    Create a grid visualization of corruptions.
    
    Args:
        image_path: Path to input image
        corruption_types: List of corruption types to visualize
        severities: List of severity levels
        output_path: Where to save the grid (if None, just display)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize for visualization
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Calculate grid size
    n_corruptions = len(corruption_types)
    n_severities = len(severities)
    
    # Create figure
    fig, axes = plt.subplots(n_corruptions, n_severities + 1, 
                            figsize=(3 * (n_severities + 1), 3 * n_corruptions))
    
    if n_corruptions == 1:
        axes = axes.reshape(1, -1)
    
    # Process each corruption type
    for i, corruption_type in enumerate(corruption_types):
        print(f"Processing {corruption_type}...")
        
        corruption_fn = CORRUPTION_DICT[corruption_type]
        
        # Show original in first column
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '', fontsize=10)
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(corruption_type.replace('_', ' ').title(), 
                             fontsize=9, rotation=0, ha='right', va='center')
        
        # Show corrupted versions
        for j, severity in enumerate(severities):
            corrupted_img = apply_corruption_to_pil(img, corruption_fn, severity)
            axes[i, j + 1].imshow(corrupted_img)
            if i == 0:
                axes[i, j + 1].set_title(f'Severity {severity}', fontsize=10)
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_corruptions_overview(image_path, severity=3, output_path=None):
    """
    Create a large grid showing all 19 corruption types at a single severity.
    
    Args:
        image_path: Path to input image
        severity: Severity level to use (1-5)
        output_path: Where to save the grid
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    
    corruption_types = get_all_corruption_types()
    
    # Create 5x4 grid (19 corruptions + 1 original = 20 images)
    n_rows = 5
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 15))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(img)
    axes[0].set_title('Original (Clean)', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Show all corruptions
    for idx, corruption_type in enumerate(corruption_types):
        print(f"Processing {idx+1}/{len(corruption_types)}: {corruption_type}")
        
        corruption_fn = CORRUPTION_DICT[corruption_type]
        corrupted_img = apply_corruption_to_pil(img, corruption_fn, severity)
        
        axes[idx + 1].imshow(corrupted_img)
        axes[idx + 1].set_title(corruption_type.replace('_', ' ').title(), fontsize=9)
        axes[idx + 1].axis('off')
    
    # Hide the last empty subplot
    axes[-1].axis('off')
    
    plt.suptitle(f'All ImageNet-C Corruptions (Severity {severity})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_random_test_image(test_dir):
    """Get a random image from the test directory."""
    test_path = Path(test_dir)
    
    # Get all class directories
    class_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {test_dir}")
    
    # Pick random class
    random_class = random.choice(class_dirs)
    
    # Get all images in that class
    images = (list(random_class.glob('*.jpg')) + 
             list(random_class.glob('*.png')) + 
             list(random_class.glob('*.JPEG')))
    
    if not images:
        raise ValueError(f"No images found in {random_class}")
    
    # Pick random image
    random_image = random.choice(images)
    
    print(f"Selected random image: {random_image}")
    print(f"Class: {random_class.name}")
    
    return str(random_image)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize corruption effects on CUB images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all corruptions on a specific image
  python visualize_corruptions.py --image datasets/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg --all

  # Visualize specific corruptions with all severities
  python visualize_corruptions.py --image my_image.jpg --corruptions gaussian_noise shot_noise

  # Use random test image
  python visualize_corruptions.py --random --all

  # Create overview of all corruptions at severity 3
  python visualize_corruptions.py --random --all --severity 3
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                           help='Path to input image')
    input_group.add_argument('--random', action='store_true',
                           help='Use random image from test set')
    
    # Corruption options
    parser.add_argument('--corruptions', nargs='+',
                       help='Specific corruption types to visualize')
    parser.add_argument('--all', action='store_true',
                       help='Visualize all corruption types')
    parser.add_argument('--severity', type=int, nargs='+', default=[1, 3, 5],
                       help='Severity levels to show (default: 1 3 5)')
    
    # Data directory (for random selection)
    parser.add_argument('--test_dir', type=str,
                       default='./datasets/cub200_cropped/test_cropped/',
                       help='Test data directory (for --random)')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file path (if not specified, displays interactively)')
    parser.add_argument('--output_dir', type=str, default='./corruption_visualizations/',
                       help='Output directory for saved visualizations')
    
    args = parser.parse_args()
    
    # Get image path
    if args.random:
        image_path = get_random_test_image(args.test_dir)
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine corruption types
    if args.all:
        corruption_types = get_all_corruption_types()
    elif args.corruptions:
        corruption_types = args.corruptions
        # Validate
        for ct in corruption_types:
            if ct not in CORRUPTION_DICT:
                raise ValueError(f"Unknown corruption type: {ct}")
    else:
        # Default: show a few representative ones
        corruption_types = ['gaussian_noise', 'defocus_blur', 'fog', 'contrast']
        print(f"No corruptions specified. Using default set: {corruption_types}")
    
    # Create output directory if needed
    if args.output or not args.output:  # If saving or default behavior
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate filename
        img_name = Path(image_path).stem
        if args.all and len(args.severity) == 1:
            output_path = os.path.join(args.output_dir, 
                                      f'{img_name}_all_corruptions_sev{args.severity[0]}.png')
        else:
            output_path = os.path.join(args.output_dir, 
                                      f'{img_name}_corruptions.png')
    
    print("="*60)
    print("Corruption Visualization")
    print("="*60)
    print(f"Input image:  {image_path}")
    print(f"Corruptions:  {len(corruption_types)} types")
    print(f"Severities:   {args.severity}")
    print(f"Output:       {output_path}")
    print("="*60)
    
    # Create visualization
    if args.all and len(args.severity) == 1:
        # Create overview of all corruptions at single severity
        print("\nCreating overview of all corruptions...")
        create_all_corruptions_overview(image_path, args.severity[0], output_path)
    else:
        # Create grid with multiple severities
        print("\nCreating corruption comparison grid...")
        create_corruption_grid(image_path, corruption_types, args.severity, output_path)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

