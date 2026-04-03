"""
Generate Amazon-C (Corrupted Amazon) Dataset for TTA Evaluation.

Creates corrupted versions of Amazon reviews using WildNLP library.
Uses MORE SEVERE corruptions (70%, 80%, 90%) to create meaningful test scenarios.

Usage:
    python generate_amazon_c.py
    python generate_amazon_c.py --num_samples 4000 --severities 70 80 90

Output:
    Datasets/Amazon-C/amazon_c_<corruption>_s<severity>.csv
    Datasets/Amazon-C/amazon_c_clean.csv
    Datasets/Amazon-C/metadata.json
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add WildNLP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'WildNLP'))

from wildnlp.aspects import QWERTY, Swap, RemoveChar


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def create_qwerty_aspect(severity: int, seed: int = 42) -> QWERTY:
    """Create QWERTY aspect with given severity.
    
    severity: percentage of words to affect (0-100)
    Each affected word will have ~50% of characters modified.
    """
    return QWERTY(
        words_percentage=severity,  # % of words to modify
        characters_percentage=50,   # % of chars per word
        seed=seed
    )


def create_swap_aspect(severity: int, seed: int = 42) -> Swap:
    """Create Swap aspect with given severity."""
    return Swap(
        transform_percentage=severity,
        seed=seed
    )


def create_remove_char_aspect(severity: int, seed: int = 42) -> RemoveChar:
    """Create RemoveChar aspect with given severity."""
    return RemoveChar(
        words_percentage=severity,    # % of words to affect
        characters_percentage=30,     # % of chars per word to remove
        seed=seed
    )


def apply_qwerty_corruption(text: str, aspect: QWERTY) -> str:
    """Apply QWERTY keyboard typo corruption."""
    try:
        result = aspect(text)
        return result if result else text
    except Exception:
        return text


def apply_swap_corruption(text: str, aspect: Swap) -> str:
    """Apply character swap corruption."""
    try:
        result = aspect(text)
        return result if result else text
    except Exception:
        return text


def apply_remove_char_corruption(text: str, aspect: RemoveChar) -> str:
    """Apply character removal corruption."""
    try:
        result = aspect(text)
        return result if result else text
    except Exception:
        return text


def apply_mixed_corruption(text: str, severity: int, seed: int = 42) -> str:
    """Apply mixed corruption (all three types in sequence)."""
    try:
        # Apply each corruption with reduced severity
        per_type_severity = max(20, severity // 3)
        
        qwerty = create_qwerty_aspect(per_type_severity, seed)
        swap = create_swap_aspect(per_type_severity, seed + 1)
        remove = create_remove_char_aspect(per_type_severity, seed + 2)
        
        text = apply_qwerty_corruption(text, qwerty)
        text = apply_swap_corruption(text, swap)
        text = apply_remove_char_corruption(text, remove)
        
        return text
    except Exception:
        return text


def apply_aggressive_corruption(text: str, severity: int, seed: int = 42) -> str:
    """Apply aggressive corruption - all types at high severity."""
    try:
        # Apply all corruptions aggressively
        qwerty = create_qwerty_aspect(severity, seed)
        swap = create_swap_aspect(severity // 2, seed + 1)
        remove = create_remove_char_aspect(severity // 3, seed + 2)
        
        text = apply_qwerty_corruption(text, qwerty)
        text = apply_swap_corruption(text, swap)
        text = apply_remove_char_corruption(text, remove)
        
        return text
    except Exception:
        return text


def generate_corrupted_dataset(
    df: pd.DataFrame,
    corruption_type: str,
    severity: int,
    seed: int = 42,
    text_column: str = 'review',
    label_column: str = 'sentiment'
) -> Tuple[pd.DataFrame, Dict]:
    """Generate corrupted version of dataset."""
    
    corrupted_texts = []
    original_texts = []
    was_modified = []
    
    # Create aspects once for efficiency
    if corruption_type == 'qwerty':
        aspect = create_qwerty_aspect(severity, seed)
    elif corruption_type == 'swap':
        aspect = create_swap_aspect(severity, seed)
    elif corruption_type == 'remove_char':
        aspect = create_remove_char_aspect(severity, seed)
    else:
        aspect = None  # Mixed and aggressive create aspects per-text
    
    for idx, row in tqdm(df.iterrows(), total=len(df), 
                         desc=f"{corruption_type} s{severity}"):
        original_text = str(row[text_column])
        
        if corruption_type == 'qwerty':
            corrupted_text = apply_qwerty_corruption(original_text, aspect)
        elif corruption_type == 'swap':
            corrupted_text = apply_swap_corruption(original_text, aspect)
        elif corruption_type == 'remove_char':
            corrupted_text = apply_remove_char_corruption(original_text, aspect)
        elif corruption_type == 'mixed':
            corrupted_text = apply_mixed_corruption(original_text, severity, seed + idx)
        elif corruption_type == 'aggressive':
            corrupted_text = apply_aggressive_corruption(original_text, severity, seed + idx)
        else:
            corrupted_text = original_text
        
        corrupted_texts.append(corrupted_text)
        original_texts.append(original_text)
        was_modified.append(corrupted_text != original_text)
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        'review': corrupted_texts,
        'sentiment': df[label_column].values,
        'original_review': original_texts,
        'was_modified': was_modified,
        'corruption_type': corruption_type,
        'severity': severity
    })
    
    stats = {
        'corruption_type': corruption_type,
        'severity': severity,
        'total_samples': len(df),
        'modified_samples': sum(was_modified),
        'modification_rate': sum(was_modified) / len(df) * 100
    }
    
    return result_df, stats


def main():
    parser = argparse.ArgumentParser(description='Generate Amazon-C corrupted dataset')
    parser.add_argument('--source_file', type=str, 
                       default='Datasets/Amazon/test.csv',
                       help='Source Amazon dataset file')
    parser.add_argument('--output_dir', type=str, 
                       default='Datasets/Amazon-C',
                       help='Output directory for corrupted datasets')
    parser.add_argument('--num_samples', type=int, default=4000,
                       help='Number of samples (balanced 50/50)')
    parser.add_argument('--severities', type=int, nargs='+', 
                       default=[70, 80, 90],
                       help='Severity levels (percentage of words to corrupt)')
    parser.add_argument('--corruption_types', type=str, nargs='+',
                       default=['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive'],
                       help='Corruption types to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("Amazon-C Dataset Generator (Severe Corruptions)")
    print("=" * 80)
    print(f"Source: {args.source_file}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Severities: {args.severities}% (of words to corrupt)")
    print(f"Corruptions: {args.corruption_types}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load source data
    print("\nLoading source Amazon dataset...")
    df_source = pd.read_csv(args.source_file)
    print(f"Total available samples: {len(df_source)}")
    print(f"Label distribution: {df_source['sentiment'].value_counts().to_dict()}")
    
    # Sample balanced dataset
    samples_per_class = args.num_samples // 2
    df_neg = df_source[df_source['sentiment'] == 0].sample(n=samples_per_class, random_state=args.seed)
    df_pos = df_source[df_source['sentiment'] == 1].sample(n=samples_per_class, random_state=args.seed)
    df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    print(f"\nSampled {len(df)} balanced samples")
    print(f"  Negative: {len(df_neg)}")
    print(f"  Positive: {len(df_pos)}")
    
    # Save clean reference
    clean_path = os.path.join(args.output_dir, 'amazon_c_clean.csv')
    df[['review', 'sentiment']].to_csv(clean_path, index=False)
    print(f"\nSaved clean reference: {clean_path}")
    
    # Generate corrupted versions
    all_stats = []
    
    for corruption_type in args.corruption_types:
        for severity in args.severities:
            print(f"\nGenerating {corruption_type} at {severity}% severity...")
            
            corrupted_df, stats = generate_corrupted_dataset(
                df, corruption_type, severity, args.seed
            )
            
            # Save corrupted dataset
            output_file = f"amazon_c_{corruption_type}_s{severity}.csv"
            output_path = os.path.join(args.output_dir, output_file)
            corrupted_df.to_csv(output_path, index=False)
            
            print(f"  Saved: {output_file}")
            print(f"  Modified: {stats['modified_samples']}/{stats['total_samples']} ({stats['modification_rate']:.1f}%)")
            
            all_stats.append(stats)
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'source_file': args.source_file,
        'num_samples': args.num_samples,
        'severities': args.severities,
        'corruption_types': args.corruption_types,
        'seed': args.seed,
        'dataset_stats': all_stats
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")
    
    # Show sample comparisons
    print("\n" + "=" * 80)
    print("Sample Comparisons (original vs corrupted)")
    print("=" * 80)
    
    sample_idx = 0
    original = df.iloc[sample_idx]['review'][:150]
    
    print(f"\nOriginal ({len(original)} chars):")
    print(f"  {original}")
    
    for corruption_type in ['qwerty', 'aggressive']:
        for severity in [90]:
            if corruption_type == 'qwerty':
                aspect = create_qwerty_aspect(severity, args.seed)
                corrupted = apply_qwerty_corruption(original, aspect)
            else:
                corrupted = apply_aggressive_corruption(original, severity, args.seed)
            
            print(f"\n{corruption_type} s{severity}:")
            print(f"  {corrupted}")
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated {len(args.corruption_types) * len(args.severities)} corrupted datasets")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
