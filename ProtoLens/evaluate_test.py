"""
Evaluate trained ProtoLens model on held-out test set.

Usage:
    python evaluate_test.py --model_path log_folder/Yelp/.../model.pth --dataset Yelp
"""

import torch
from transformers import AutoTokenizer
import os
import argparse
from utils import get_data_loader, load_data, TextClassificationDataset
from experiment import evaluate
from PLens import BERTClassifier
from torch.utils.data import DataLoader
import pandas as pd

def evaluate_test_set(model_path, dataset='Yelp', batch_size=16):
    """
    Evaluate model on final test set (test.csv)
    """
    print("="*80)
    print(f"Evaluating Model on {dataset} Test Set")
    print("="*80)
    print(f"\nModel: {model_path}")
    
    # Load checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint['pnfrl_args']
    
    print(f"Model configuration:")
    print(f"  - bert_model: {saved_args['bert_model_name']}")
    print(f"  - num_classes: {saved_args['num_classes']}")
    print(f"  - num_prototypes: {saved_args['prototype_num']}")
    print(f"  - batch_size: {saved_args['batch_size']}")
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
    
    args = Args(
        bert_model_name=saved_args['bert_model_name'],
        num_classes=saved_args['num_classes'],
        prototype_num=saved_args['prototype_num'],
        batch_size=saved_args['batch_size'],
        hidden_dim=saved_args['hidden_dim'],
        max_length=saved_args['max_length'],
        data_set=dataset,
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
    print(f"\nLoading {dataset} test set...")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
    test_file = os.path.join(base_dir, dataset, "test.csv")
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return
    
    test_texts, test_labels = load_data(test_file)
    print(f"✓ Loaded {len(test_texts)} test samples")
    
    # Create test dataloader
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, saved_args['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Evaluate
    print("\n" + "="*80)
    print("Running Evaluation...")
    print("="*80)
    accuracy, report = evaluate(model, test_loader, device)
    
    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS on {dataset}")
    print(f"{'='*80}")
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(report)
    
    # Save results
    results_dir = os.path.dirname(model_path)
    results_file = os.path.join(results_dir, f'test_results_{dataset}.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Test Set Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return accuracy, report

def main():
    parser = argparse.ArgumentParser(description='Evaluate ProtoLens on test set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--dataset', type=str, default='Yelp',
                        choices=['Yelp', 'Amazon', 'Hotel', 'IMDB'],
                        help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    evaluate_test_set(args.model_path, args.dataset, args.batch_size)

if __name__ == '__main__':
    main()
