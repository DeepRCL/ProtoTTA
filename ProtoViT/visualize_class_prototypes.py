"""
Visualize Top Prototypes by Weight for Specific Classes

This script loads a trained ProtoViT model and visualizes the top-k prototypes
(with highest weights in the final layer) for specified classes.

Usage:
    python visualize_class_prototypes.py \
        --model saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
        --classes 10 13 17 20 \
        --top_k 10 \
        --prototype_dir saved_models/deit_small_patch16_224/exp1/img \
        --output_dir ./plots/class_prototypes
"""

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import model (needed for torch.load)
import model
import push_greedy


def load_model(model_path, device='cuda'):
    """Load the trained model."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model_obj = torch.load(model_path, weights_only=False)
    model_obj = model_obj.to(device)
    model_obj.eval()
    
    print(f"Model loaded successfully")
    print(f"  Number of prototypes: {model_obj.num_prototypes}")
    print(f"  Number of classes: {model_obj.num_classes}")
    
    return model_obj


def get_top_k_prototypes_by_weight(model, class_idx, k=10):
    """
    Get top-k prototypes with highest weights for a given class.
    
    Args:
        model: PPNet model
        class_idx: Class index (0-based)
        k: Number of prototypes to return
    
    Returns:
        List of dicts with: proto_idx, weight, proto_class
    """
    # Get weights for this class: [num_prototypes]
    weights = model.last_layer.weight[class_idx, :].detach().cpu().numpy()
    
    # Get prototype class identities: [num_prototypes, num_classes]
    proto_class_identity = model.prototype_class_identity.detach().cpu().numpy()
    
    # Sort by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]
    
    # Filter to only positive weights
    positive_mask = weights > 0
    positive_indices = np.where(positive_mask)[0]
    
    if len(positive_indices) == 0:
        print(f"  WARNING: No prototypes with positive weight for class {class_idx}")
        # Still return top-k even if negative
        top_indices = sorted_indices[:k]
    else:
        # Sort positive weights
        positive_weights = weights[positive_indices]
        positive_sorted = np.argsort(positive_weights)[::-1]
        top_indices = positive_indices[positive_sorted][:k]
    
    results = []
    for proto_idx in top_indices:
        weight = weights[proto_idx]
        proto_class = np.argmax(proto_class_identity[proto_idx])
        
        results.append({
            'proto_idx': int(proto_idx),
            'weight': float(weight),
            'proto_class': int(proto_class)
        })
    
    return results


def find_prototype_image(prototype_img_dir, proto_idx):
    """Find prototype image file."""
    possible_patterns = [
        f'prototype-imgbbox-original{proto_idx}.png',
        f'prototype-img_vis_{proto_idx}.png',
        f'prototype-img-original{proto_idx}.png',
        f'prototype-img{proto_idx}.png',
        f'prototype{proto_idx}.png',
    ]
    
    # Try direct directory
    for pattern in possible_patterns:
        path = os.path.join(prototype_img_dir, pattern)
        if os.path.exists(path):
            return path
    
    # Try subdirectories
    for subdir in ['epoch-4', 'epoch-10', '']:
        base = os.path.join(prototype_img_dir, subdir) if subdir else prototype_img_dir
        for pattern in possible_patterns:
            path = os.path.join(base, pattern)
            if os.path.exists(path):
                return path
    
    return None


def visualize_class_prototypes(model, class_idx, prototype_img_dir, output_dir, top_k=10):
    """
    Visualize top-k prototypes for a class.
    
    Args:
        model: PPNet model
        class_idx: Class index
        prototype_img_dir: Directory containing prototype images
        output_dir: Output directory
        top_k: Number of prototypes to show
    """
    print(f"\n{'='*60}")
    print(f"Visualizing prototypes for Class {class_idx}")
    print(f"{'='*60}")
    
    # Get top-k prototypes
    proto_results = get_top_k_prototypes_by_weight(model, class_idx, k=top_k)
    
    if len(proto_results) == 0:
        print(f"  No prototypes found for class {class_idx}")
        return
    
    # Print summary
    print(f"\nTop {len(proto_results)} prototypes for class {class_idx}:")
    print(f"{'Rank':<6} {'Proto ID':<10} {'Weight':<12} {'Trained Class':<15}")
    print("-" * 50)
    for i, proto in enumerate(proto_results, 1):
        print(f"{i:<6} {proto['proto_idx']:<10} {proto['weight']:<12.6f} {proto['proto_class']:<15}")
    
    # Count how many prototypes are actually trained for this class
    same_class_count = sum(1 for p in proto_results if p['proto_class'] == class_idx)
    print(f"\n  Prototypes trained for class {class_idx}: {same_class_count}/{len(proto_results)}")
    
    # Create visualization
    n_cols = 5
    n_rows = (len(proto_results) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Top {len(proto_results)} Prototypes by Weight for Class {class_idx}', 
                 fontsize=16, fontweight='bold')
    
    for i, proto in enumerate(proto_results):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        proto_idx = proto['proto_idx']
        weight = proto['weight']
        proto_class = proto['proto_class']
        
        # Try to load prototype image
        proto_img_path = find_prototype_image(prototype_img_dir, proto_idx)
        
        if proto_img_path and os.path.exists(proto_img_path):
            try:
                img = plt.imread(proto_img_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f'Proto {proto_idx}\n(Load error)', 
                       ha='center', va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'Proto {proto_idx}\n(Not found)', 
                   ha='center', va='center', fontsize=10)
        
        # Title with weight and class info
        class_match = "✓" if proto_class == class_idx else "✗"
        title = f"Proto {proto_idx}\nWeight: {weight:.4f}\nClass: {proto_class} {class_match}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(proto_results), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'class_{class_idx}_top{len(proto_results)}_prototypes.png')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved visualization: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize top prototypes by weight for specific classes'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pth)'
    )
    
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        required=True,
        help='Class indices to visualize (e.g., --classes 10 13 17)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top prototypes to show per class (default: 10)'
    )
    
    parser.add_argument(
        '--prototype_dir',
        type=str,
        default=None,
        help='Directory containing prototype images. If not provided, will try to infer from model path.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots/class_prototypes',
        help='Output directory for visualizations (default: ./plots/class_prototypes)'
    )
    
    parser.add_argument(
        '--gpuid',
        type=str,
        default='0',
        help='GPU ID to use (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model_obj = load_model(args.model, device)
    
    # Determine prototype image directory
    if args.prototype_dir:
        prototype_img_dir = args.prototype_dir
    else:
        # Try to infer from model path
        model_dir = os.path.dirname(args.model)
        possible_dirs = [
            os.path.join(model_dir, 'img'),
            os.path.join(model_dir, 'prototype_imgs'),
            os.path.join(model_dir, 'prototype-img'),
        ]
        prototype_img_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                prototype_img_dir = d
                break
        
        if prototype_img_dir is None:
            print(f"WARNING: Could not find prototype image directory.")
            print(f"  Searched in: {possible_dirs}")
            print(f"  Please specify --prototype_dir")
            prototype_img_dir = model_dir  # Fallback
    
    print(f"\nPrototype image directory: {prototype_img_dir}")
    
    # Visualize each class
    output_paths = []
    for class_idx in args.classes:
        if class_idx < 0 or class_idx >= model_obj.num_classes:
            print(f"\n⚠ Skipping class {class_idx} (out of range [0, {model_obj.num_classes-1}])")
            continue
        
        output_path = visualize_class_prototypes(
            model_obj, class_idx, prototype_img_dir, 
            args.output_dir, top_k=args.top_k
        )
        if output_path:
            output_paths.append(output_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Visualized {len(output_paths)} classes")
    print(f"Output directory: {args.output_dir}")
    for path in output_paths:
        print(f"  - {path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

