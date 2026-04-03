"""
Generate Yelp-C (Corrupted Yelp) Dataset for Test-Time Adaptation Evaluation.

This script creates corrupted versions of the Yelp test set using WildNLP library.
It simulates realistic text corruptions like typos, keyboard errors, character swaps, etc.

Output:
- Datasets/Yelp-C/yelp_c_{corruption_type}_{severity}.csv: Corrupted datasets
- Datasets/Yelp-C/sample_comparison.json: Sample comparisons (original vs corrupted)
- Datasets/Yelp-C/metadata.json: Dataset metadata and corruption info

Usage:
    python generate_yelp_c.py --num_samples 4000 --severity 30
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

# Add WildNLP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'WildNLP'))

from wildnlp.aspects import (
    QWERTY,
    RemoveChar,
    Swap,
    Punctuation,
    WhiteSpaces,
    Misspelling,
)
from wildnlp.aspects.utils import compose


# ============================================================================
# Corruption Types Configuration
# ============================================================================

CORRUPTION_TYPES = {
    'qwerty': {
        'name': 'QWERTY Keyboard Errors',
        'description': 'Simulates typos from adjacent keys on QWERTY keyboard',
        'factory': lambda severity, seed: QWERTY(
            words_percentage=severity,  # % of words to corrupt
            characters_percentage=20,   # % of chars in each word
            seed=seed
        )
    },
    'swap': {
        'name': 'Character Swap',
        'description': 'Randomly swaps adjacent characters within words',
        'factory': lambda severity, seed: Swap(
            transform_percentage=severity,
            seed=seed
        )
    },
    'remove_char': {
        'name': 'Character Removal',
        'description': 'Randomly removes characters from words',
        'factory': lambda severity, seed: RemoveChar(
            words_percentage=severity,
            characters_percentage=15,
            seed=seed
        )
    },
    'remove_space': {
        'name': 'Space Removal',
        'description': 'Randomly removes spaces between words',
        'factory': lambda severity, seed: RemoveChar(
            char=' ',
            words_percentage=severity,
            seed=seed
        )
    },
    'punctuation': {
        'name': 'Punctuation Errors',
        'description': 'Randomly adds or removes punctuation marks',
        'factory': lambda severity, seed: Punctuation(
            transform_percentage=severity,
            seed=seed
        )
    },
    'misspelling': {
        'name': 'Common Misspellings',
        'description': 'Replaces words with common misspellings',
        'factory': lambda severity, seed: Misspelling(
            transform_percentage=severity,
            seed=seed
        )
    },
    'mixed': {
        'name': 'Mixed Corruptions',
        'description': 'Combination of QWERTY, Swap, and RemoveChar',
        'factory': lambda severity, seed: compose(
            QWERTY(words_percentage=severity//3, characters_percentage=15, seed=seed),
            Swap(transform_percentage=severity//3, seed=seed+1),
            RemoveChar(words_percentage=severity//3, characters_percentage=10, seed=seed+2)
        )
    }
}


def load_yelp_test_data(data_path: str, num_samples: int = 4000) -> pd.DataFrame:
    """Load Yelp test data with balanced class sampling.
    
    Args:
        data_path: Path to test.csv
        num_samples: Total number of samples (will be split equally between classes)
    
    Returns:
        DataFrame with balanced samples
    """
    print(f"Loading Yelp test data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Get class distribution
    class_counts = df['sentiment'].value_counts()
    print(f"  Class distribution: {dict(class_counts)}")
    
    # Balance classes: take num_samples/2 from each class
    samples_per_class = num_samples // 2
    
    # Separate by class
    positive_samples = df[df['sentiment'] == 1].sample(n=min(samples_per_class, len(df[df['sentiment'] == 1])), 
                                                        random_state=42)
    negative_samples = df[df['sentiment'] == 0].sample(n=min(samples_per_class, len(df[df['sentiment'] == 0])), 
                                                        random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([positive_samples, negative_samples], ignore_index=False)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=False)  # Keep original index as 'index'
    balanced_df = balanced_df.rename(columns={'index': 'original_idx'})
    
    print(f"  Selected {len(balanced_df)} samples ({samples_per_class} per class)")
    
    return balanced_df


def apply_corruption(text: str, corruptor, max_retries: int = 3) -> Tuple[str, bool]:
    """Apply corruption to a single text with error handling.
    
    Returns:
        Tuple of (corrupted_text, was_modified)
    """
    try:
        corrupted = corruptor(text)
        was_modified = (corrupted != text)
        return corrupted, was_modified
    except Exception as e:
        # If corruption fails, return original
        return text, False


def generate_corrupted_dataset(
    df: pd.DataFrame,
    corruption_type: str,
    severity: int,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a corrupted version of the dataset.
    
    Args:
        df: Original DataFrame with 'review' and 'sentiment' columns
        corruption_type: Key from CORRUPTION_TYPES
        severity: Corruption severity (0-100)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (corrupted_df, statistics_dict)
    """
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"Unknown corruption type: {corruption_type}. "
                        f"Available: {list(CORRUPTION_TYPES.keys())}")
    
    config = CORRUPTION_TYPES[corruption_type]
    corruptor = config['factory'](severity, seed)
    
    print(f"\nApplying corruption: {config['name']}")
    print(f"  Severity: {severity}%")
    
    corrupted_reviews = []
    original_reviews = []
    was_modified_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Corrupting"):
        original_text = row['review']
        corrupted_text, was_modified = apply_corruption(original_text, corruptor)
        
        corrupted_reviews.append(corrupted_text)
        original_reviews.append(original_text)
        was_modified_list.append(was_modified)
    
    # Create result DataFrame
    result_df = df.copy()
    result_df['original_review'] = original_reviews
    result_df['review'] = corrupted_reviews
    result_df['was_modified'] = was_modified_list
    
    # Calculate statistics
    stats = {
        'corruption_type': corruption_type,
        'corruption_name': config['name'],
        'description': config['description'],
        'severity': severity,
        'total_samples': len(df),
        'modified_samples': sum(was_modified_list),
        'modification_rate': sum(was_modified_list) / len(df) * 100,
        'seed': seed
    }
    
    print(f"  Modified: {stats['modified_samples']}/{stats['total_samples']} "
          f"({stats['modification_rate']:.1f}%)")
    
    return result_df, stats


def create_sample_comparison(df: pd.DataFrame, num_examples: int = 50) -> List[Dict]:
    """Create sample comparisons for visualization.
    
    Args:
        df: DataFrame with 'original_review', 'review', 'sentiment' columns
        num_examples: Number of examples to include
    
    Returns:
        List of comparison dictionaries
    """
    examples = []
    
    # Get examples that were modified
    modified_df = df[df['was_modified'] == True].head(num_examples)
    
    for idx, row in modified_df.iterrows():
        # Truncate very long texts for readability
        original = row['original_review'][:500] + ('...' if len(row['original_review']) > 500 else '')
        corrupted = row['review'][:500] + ('...' if len(row['review']) > 500 else '')
        
        examples.append({
            'id': int(row.get('original_idx', idx)),
            'sentiment': int(row['sentiment']),
            'sentiment_label': 'positive' if row['sentiment'] == 1 else 'negative',
            'original': original,
            'corrupted': corrupted,
        })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description='Generate Yelp-C (Corrupted Yelp) dataset')
    parser.add_argument('--data_path', type=str, 
                       default='Datasets/Yelp/test.csv',
                       help='Path to Yelp test.csv')
    parser.add_argument('--output_dir', type=str,
                       default='Datasets/Yelp-C',
                       help='Output directory for corrupted datasets')
    parser.add_argument('--num_samples', type=int, default=4000,
                       help='Total number of samples (2K per class)')
    parser.add_argument('--severity', type=int, default=30,
                       help='Default corruption severity (0-100)')
    parser.add_argument('--corruption_types', type=str, nargs='+',
                       default=['qwerty', 'swap', 'remove_char', 'mixed'],
                       help='Corruption types to generate')
    parser.add_argument('--severities', type=int, nargs='+',
                       default=None,
                       help='Multiple severity levels to generate (overrides --severity)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Yelp-C Dataset Generator")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Corruption types: {args.corruption_types}")
    
    # Load original data
    df = load_yelp_test_data(args.data_path, args.num_samples)
    
    # Save original (clean) version for reference
    clean_path = os.path.join(args.output_dir, 'yelp_c_clean.csv')
    df_clean = df[['original_idx', 'review', 'sentiment']].copy()
    df_clean.to_csv(clean_path, index=False)
    print(f"\nSaved clean reference: {clean_path}")
    
    # Determine severities to generate
    if args.severities:
        severities = args.severities
    else:
        severities = [args.severity]
    
    print(f"Severities: {severities}")
    
    # Generate corrupted datasets
    all_stats = []
    all_comparisons = {}
    
    for corruption_type in args.corruption_types:
        for severity in severities:
            print(f"\n{'─' * 60}")
            
            # Generate corrupted version
            try:
                corrupted_df, stats = generate_corrupted_dataset(
                    df, corruption_type, severity, seed=args.seed
                )
            except Exception as e:
                print(f"  ERROR: Failed to apply {corruption_type}: {e}")
                continue
            
            # Save corrupted dataset
            output_filename = f'yelp_c_{corruption_type}_s{severity}.csv'
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Save only necessary columns for inference
            save_df = corrupted_df[['original_idx', 'review', 'original_review', 'sentiment', 'was_modified']].copy()
            save_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")
            
            # Add to stats
            stats['filename'] = output_filename
            stats['generated_at'] = datetime.now().isoformat()
            all_stats.append(stats)
            
            # Create sample comparisons
            key = f'{corruption_type}_s{severity}'
            all_comparisons[key] = create_sample_comparison(corrupted_df, num_examples=20)
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'source_file': args.data_path,
        'num_samples': args.num_samples,
        'samples_per_class': args.num_samples // 2,
        'seed': args.seed,
        'corruption_types_available': list(CORRUPTION_TYPES.keys()),
        'datasets': all_stats
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")
    
    # Save sample comparisons
    comparison_path = os.path.join(args.output_dir, 'sample_comparisons.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)
    print(f"Saved sample comparisons: {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated {len(all_stats)} corrupted datasets:")
    for stat in all_stats:
        print(f"  - {stat['filename']}: {stat['corruption_name']} "
              f"(severity={stat['severity']}%, modified={stat['modification_rate']:.1f}%)")
    
    print(f"\nFiles saved to: {args.output_dir}/")
    print("\nTo run TTA inference on Yelp-C:")
    print(f"  python run_inference_yelp_c.py --corruption_type mixed --severity {args.severity}")


if __name__ == '__main__':
    main()
