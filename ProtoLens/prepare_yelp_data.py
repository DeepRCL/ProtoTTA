"""
Script to prepare Yelp dataset for ProtoLens training.
Converts 5-star ratings to binary classification:
- Negative: 1-2 stars (label 0)
- Positive: 4-5 stars (label 1)
- Neutral: 3 stars (excluded as per paper)

Target: 580k reviews total (550k train, 30k test)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_yelp_reviews(file_path, max_reviews=None):
    """Load Yelp reviews from JSON file"""
    reviews = []
    labels = []
    
    print(f"Loading reviews from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_reviews and i >= max_reviews:
                break
            
            if i % 100000 == 0 and i > 0:
                print(f"Processed {i} reviews...")
            
            try:
                review_data = json.loads(line)
                stars = review_data['stars']
                text = review_data['text'].strip()
                
                # Skip empty reviews
                if not text:
                    continue
                
                # Binary classification: skip 3-star reviews (neutral)
                if stars <= 2.0:
                    label = 0  # Negative
                elif stars >= 4.0:
                    label = 1  # Positive
                else:
                    continue  # Skip 3-star (neutral)
                
                reviews.append(text)
                labels.append(label)
                
            except json.JSONDecodeError:
                print(f"Error decoding line {i}")
                continue
    
    print(f"Loaded {len(reviews)} reviews (excluding 3-star neutral reviews)")
    return reviews, labels

def balance_and_sample_data(reviews, labels, target_size=580000, test_size=30000):
    """
    Balance the dataset and sample to target size
    """
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    
    # Count distribution
    print("\nOriginal distribution:")
    print(df['sentiment'].value_counts())
    
    # Separate by class
    neg_df = df[df['sentiment'] == 0]
    pos_df = df[df['sentiment'] == 1]
    
    # Calculate samples per class
    samples_per_class = target_size // 2
    train_per_class = (target_size - test_size) // 2
    test_per_class = test_size // 2
    
    print(f"\nTarget: {target_size} total ({target_size - test_size} train, {test_size} test)")
    print(f"Samples per class: {samples_per_class}")
    print(f"Train per class: {train_per_class}, Test per class: {test_per_class}")
    
    # Sample from each class
    if len(neg_df) >= samples_per_class:
        neg_sample = neg_df.sample(n=samples_per_class, random_state=42)
    else:
        print(f"Warning: Not enough negative samples ({len(neg_df)}), using all")
        neg_sample = neg_df
    
    if len(pos_df) >= samples_per_class:
        pos_sample = pos_df.sample(n=samples_per_class, random_state=42)
    else:
        print(f"Warning: Not enough positive samples ({len(pos_df)}), using all")
        pos_sample = pos_df
    
    # Combine and shuffle
    balanced_df = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print("Balanced distribution:")
    print(balanced_df['sentiment'].value_counts())
    
    # Split into train and test
    test_df = balanced_df.iloc[:test_size]
    train_df = balanced_df.iloc[test_size:]
    
    print(f"\nTrain size: {len(train_df)}")
    print("Train distribution:")
    print(train_df['sentiment'].value_counts())
    
    print(f"\nTest size: {len(test_df)}")
    print("Test distribution:")
    print(test_df['sentiment'].value_counts())
    
    return train_df, test_df

def main():
    # Paths
    base_dir = Path(__file__).parent / "Datasets"
    yelp_json = base_dir / "yelp_academic_dataset_review.json"
    yelp_output_dir = base_dir / "Yelp"
    
    # Create output directory
    yelp_output_dir.mkdir(exist_ok=True)
    
    # Check if files already exist
    train_file = yelp_output_dir / "train.csv"
    test_file = yelp_output_dir / "test.csv"
    
    if train_file.exists() and test_file.exists():
        print(f"\nDataset files already exist:")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # Load reviews (limit to ~800k for efficiency, we'll sample 580k from these)
    print("\n" + "="*60)
    print("STEP 1: Loading Yelp reviews")
    print("="*60)
    reviews, labels = load_yelp_reviews(yelp_json, max_reviews=1000000)
    
    # Balance and sample
    print("\n" + "="*60)
    print("STEP 2: Balancing and sampling to target size")
    print("="*60)
    train_df, test_df = balance_and_sample_data(reviews, labels, target_size=580000, test_size=30000)
    
    # Save to CSV
    print("\n" + "="*60)
    print("STEP 3: Saving to CSV files")
    print("="*60)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✓ Train data saved to: {train_file}")
    print(f"✓ Test data saved to: {test_file}")
    
    # Show sample reviews
    print("\n" + "="*60)
    print("Sample reviews from train set:")
    print("="*60)
    print("\nNegative examples:")
    print(train_df[train_df['sentiment'] == 0]['review'].head(2).values)
    print("\nPositive examples:")
    print(train_df[train_df['sentiment'] == 1]['review'].head(2).values)
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
