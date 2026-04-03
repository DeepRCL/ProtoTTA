"""
Test ProtoLens model on Hotel dataset with and without TENT adaptation.

Usage:
    python test_adaptation.py \
        --model_path log_folder/Yelp/.../model.pth \
        --test_file Datasets/Hotel/hotels.csv \
        --text_column Description \
        --label_column HotelRating \
        --batch_size 16 \
        --method tent
"""

import torch
from transformers import AutoTokenizer
import os
import argparse
import pandas as pd
from utils import load_data, TextClassificationDataset
from experiment import evaluate
from PLens import BERTClassifier
from torch.utils.data import DataLoader
import tent


def load_hotel_data(file_path, text_column='Description', label_column='HotelRating', 
                    label_mapping=None, max_samples=None):
    """
    Load hotel review data from CSV.
    
    Args:
        file_path: Path to hotels.csv
        text_column: Column containing text reviews/descriptions
        label_column: Column containing ratings
        label_mapping: Dict mapping ratings to binary labels (e.g., {'FourStar': 1, 'ThreeStar': 0})
        max_samples: Maximum number of samples to load (for testing)
    
    Returns:
        texts: List of text strings
        labels: List of integer labels
    """
    print(f"Loading hotel data from: {file_path}")
    
    # Read CSV with latin-1 encoding (hotels.csv has encoding issues)
    df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean column names (remove leading spaces)
    df.columns = df.columns.str.strip()
    
    # Check if required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {df.columns.tolist()}")
    
    # Get texts and labels
    texts = df[text_column].fillna("").astype(str).tolist()
    ratings = df[label_column].fillna("Unknown").astype(str).tolist()
    
    # Apply label mapping if provided
    if label_mapping is None:
        # Default: Map FourStar/FiveStar to 1 (positive), others to 0 (negative)
        label_mapping = {
            'FiveStar': 1,
            'FourStar': 1,
            'ThreeStar': 0,
            'TwoStar': 0,
            'OneStar': 0,
            'Unknown': 0
        }
        print(f"Using default label mapping: {label_mapping}")
    
    labels = [label_mapping.get(rating, 0) for rating in ratings]
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(texts) if len(text.strip()) > 0]
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    print(f"Loaded {len(texts)} valid samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Limit samples if specified
    if max_samples and max_samples < len(texts):
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    return texts, labels


def test_with_adaptation(model_path, test_file, text_column='Description', 
                        label_column='HotelRating', batch_size=16, method='tent',
                        learning_rate=0.001, max_samples=None):
    """
    Test ProtoLens model with adaptation method.
    
    Args:
        model_path: Path to trained ProtoLens model
        test_file: Path to test data CSV
        text_column: Column name for text data
        label_column: Column name for labels
        batch_size: Batch size for evaluation
        method: Adaptation method ('none', 'tent')
        learning_rate: Learning rate for adaptation
        max_samples: Max samples to test (None = all)
    """
    print("="*80)
    print(f"Testing ProtoLens with Adaptation")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")
    print(f"Method: {method}")
    print("="*80)
    
    # Load checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint['pnfrl_args']
    
    print(f"Model configuration:")
    print(f"  - bert_model: {saved_args['bert_model_name']}")
    print(f"  - num_classes: {saved_args['num_classes']}")
    print(f"  - num_prototypes: {saved_args['prototype_num']}")
    print(f"  - hidden_dim: {saved_args['hidden_dim']}")
    print(f"  - max_length: {saved_args['max_length']}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create args object for model initialization
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Use Yelp dataset for sentence pool since model was trained on Yelp
    # This allows us to test domain adaptation (Yelp -> Hotel)
    args = Args(
        bert_model_name=saved_args['bert_model_name'],
        num_classes=saved_args['num_classes'],
        prototype_num=saved_args['prototype_num'],
        batch_size=saved_args['batch_size'],
        hidden_dim=saved_args['hidden_dim'],
        max_length=saved_args['max_length'],
        data_set='Yelp',  # Use Yelp for sentence pool (model's training dataset)
        base_folder="Datasets",
        gaussian_num=6,
        window_size=5
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    model = BERTClassifier(
        args=args,
        bert_model_name=saved_args['bert_model_name'],
        num_classes=saved_args['num_classes'],
        num_prototype=saved_args['prototype_num'],
        batch_size=saved_args['batch_size'],
        hidden_dim=saved_args['hidden_dim'],
        max_length=saved_args['max_length'],
        tokenizer=tokenizer
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    test_texts, test_labels = load_hotel_data(
        test_file, 
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples
    )
    
    # Create test dataloader
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, saved_args['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # --- Baseline (No Adaptation) ---
    print("\n" + "="*80)
    print("BASELINE: No Adaptation")
    print("="*80)
    model.eval()
    baseline_acc, baseline_report = evaluate(model, test_loader, device)
    
    print(f"\nBaseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"\nBaseline Classification Report:")
    print(baseline_report)
    
    # --- Test Adaptation Method ---
    results = {'Baseline': baseline_acc}
    
    if method.lower() == 'tent':
        print("\n" + "="*80)
        print("TENT Adaptation")
        print("="*80)
        
        # Reload model fresh
        model_tent = BERTClassifier(
            args=args,
            bert_model_name=saved_args['bert_model_name'],
            num_classes=saved_args['num_classes'],
            num_prototype=saved_args['prototype_num'],
            batch_size=saved_args['batch_size'],
            hidden_dim=saved_args['hidden_dim'],
            max_length=saved_args['max_length'],
            tokenizer=tokenizer
        ).to(device)
        model_tent.load_state_dict(checkpoint['model_state_dict'])
        
        # Configure for TENT
        print("Configuring model for TENT...")
        model_tent = tent.configure_model(model_tent)
        params, param_names = tent.collect_params(model_tent)
        print(f"Adapting {len(params)} parameters:")
        for name in param_names[:5]:  # Show first 5
            print(f"  - {name}")
        if len(param_names) > 5:
            print(f"  ... and {len(param_names)-5} more")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        # Wrap with TENT
        tent_model = tent.Tent(model_tent, optimizer, steps=1, episodic=False)
        
        # Evaluate with TENT
        print("\nRunning TENT adaptation...")
        tent_acc, tent_report = evaluate(tent_model, test_loader, device)
        
        print(f"\nTENT Accuracy: {tent_acc:.4f} ({tent_acc*100:.2f}%)")
        print(f"Improvement: {(tent_acc-baseline_acc)*100:+.2f}%")
        print(f"\nTENT Classification Report:")
        print(tent_report)
        
        # Print adaptation stats
        if hasattr(tent_model, 'adaptation_stats'):
            stats = tent_model.adaptation_stats
            print(f"\nAdaptation Statistics:")
            print(f"  - Total samples: {stats['total_samples']}")
            print(f"  - Adapted samples: {stats['adapted_samples']}")
            print(f"  - Total updates: {stats['total_updates']}")
        
        results['TENT'] = tent_acc
    
    # --- Summary ---
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    for method_name, acc in results.items():
        print(f"{method_name:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    if 'TENT' in results:
        improvement = (results['TENT'] - results['Baseline']) * 100
        print(f"\nTENT Improvement: {improvement:+.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test ProtoLens with adaptation on Hotel dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained ProtoLens model checkpoint (.pth file)')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--text_column', type=str, default='Description',
                        help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='HotelRating',
                        help='Column name for labels')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--method', type=str, default='tent',
                        choices=['none', 'tent'],
                        help='Adaptation method to test')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for adaptation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (for quick testing)')
    
    args = parser.parse_args()
    
    test_with_adaptation(
        model_path=args.model_path,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        method=args.method,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
