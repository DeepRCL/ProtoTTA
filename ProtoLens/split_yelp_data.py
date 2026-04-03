"""
Split Yelp data into proper train/validation/test sets.

Current data:
- train.csv: 476,537 samples (will split into train + val)
- test.csv: 30,000 samples (keep as final test set)

New split:
- train.csv: ~380,000 samples (80% of original train)
- val.csv: ~96,000 samples (20% of original train)  
- test.csv: 30,000 samples (unchanged - final evaluation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data():
    base_dir = Path(__file__).parent / "Datasets" / "Yelp"
    
    # Load current train data
    train_file = base_dir / "train.csv"
    test_file = base_dir / "test.csv"
    
    print("="*80)
    print("Splitting Yelp Data into Train/Val/Test")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Original train: {len(train_df)} samples")
    print(f"Original test: {len(test_df)} samples")
    print(f"\nTrain distribution:\n{train_df['sentiment'].value_counts()}")
    
    # Split train into train + validation (80/20 split)
    print("\n" + "="*80)
    print("Creating train/validation split (80/20)...")
    print("="*80)
    
    train_new, val_new = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['sentiment']  # Keep class balance
    )
    
    print(f"\nNew train: {len(train_new)} samples")
    print(f"New validation: {len(val_new)} samples")
    print(f"Final test: {len(test_df)} samples (unchanged)")
    
    print(f"\nNew train distribution:\n{train_new['sentiment'].value_counts()}")
    print(f"\nNew validation distribution:\n{val_new['sentiment'].value_counts()}")
    print(f"\nFinal test distribution:\n{test_df['sentiment'].value_counts()}")
    
    # Save new splits
    print("\n" + "="*80)
    print("Saving new splits...")
    print("="*80)
    
    # Backup original train file
    backup_file = base_dir / "train_original_backup.csv"
    if not backup_file.exists():
        print(f"Backing up original train.csv to {backup_file.name}")
        train_df.to_csv(backup_file, index=False)
    
    # Save new splits
    train_new_file = base_dir / "train.csv"
    val_new_file = base_dir / "val.csv"
    
    train_new.to_csv(train_new_file, index=False)
    val_new.to_csv(val_new_file, index=False)
    
    print(f"✓ Saved train set: {train_new_file} ({len(train_new)} samples)")
    print(f"✓ Saved validation set: {val_new_file} ({len(val_new)} samples)")
    print(f"✓ Test set unchanged: {test_file} ({len(test_df)} samples)")
    
    print("\n" + "="*80)
    print("Data Split Complete!")
    print("="*80)
    print("\nFinal structure:")
    print(f"  - train.csv: {len(train_new):,} samples (for training)")
    print(f"  - val.csv: {len(val_new):,} samples (for validation during training)")
    print(f"  - test.csv: {len(test_df):,} samples (for final evaluation)")
    print(f"  - train_original_backup.csv: {len(train_df):,} samples (backup)")
    
    # Show sample reviews
    print("\n" + "="*80)
    print("Sample Reviews:")
    print("="*80)
    print("\nTrain - Negative:")
    print(train_new[train_new['sentiment'] == 0]['review'].iloc[0][:200] + "...")
    print("\nTrain - Positive:")
    print(train_new[train_new['sentiment'] == 1]['review'].iloc[0][:200] + "...")
    
    return train_new, val_new, test_df

if __name__ == "__main__":
    split_data()
