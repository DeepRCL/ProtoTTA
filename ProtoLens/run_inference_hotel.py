"""
Run inference with ProtoLens on a chosen dataset (e.g., Hotel or Yelp) and compare
baseline vs test-time adaptation methods (TENT / EATA / ProtoTTA).

Examples:
    # Evaluate on Hotel (raw hotels.csv with 1-2 vs 4-5 mapping)
    python run_inference_hotel.py \\
        --model_path log_folder/Yelp/.../model.pth \\
        --dataset hotel \\
        --data_file Datasets/Hotel/hotels.csv \\
        --split test \\
        --num_samples 4000 \\
        --methods baseline tent eata prototta

    # Evaluate on Yelp itself (standard ProtoLens CSV: Datasets/Yelp/test.csv)
    python run_inference_hotel.py \\
        --model_path log_folder/Yelp/.../model.pth \\
        --dataset yelp \\
        --split test \\
        --num_samples 4000 \\
        --methods baseline tent eata prototta
"""

import torch
from transformers import AutoTokenizer
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import TextClassificationDataset
from PLens import BERTClassifier
from torch.utils.data import DataLoader, Subset
import tent
import eata
import proto_tta
import adapt_utils
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def load_standard_binary_csv_sampled(
    csv_path,
    num_samples=2000,
    seed=42,
    text_column="review",
    label_column="sentiment",
):
    """
    Load a standard ProtoLens dataset CSV (columns: review/sentiment) and optionally
    take a reproducible balanced subset (50% class 0, 50% class 1).
    """
    print(f"\n{'='*80}")
    print(f"Loading Standard Dataset CSV")
    print(f"{'='*80}")
    print(f"File: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{text_column}' and '{label_column}'. "
            f"Found: {list(df.columns)}"
        )

    valid_mask = (
        df[text_column].notna()
        & (df[text_column].astype(str).str.strip().str.len() > 1)
        & df[label_column].notna()
    )
    df_valid = df[valid_mask].copy()
    df_valid[label_column] = df_valid[label_column].astype(int)
    df_valid = df_valid[df_valid[label_column].isin([0, 1])].copy()

    print(f"Valid samples: {len(df_valid):,}")
    label_counts = df_valid[label_column].value_counts().sort_index()
    print("Label distribution:")
    for label, count in label_counts.items():
        label_name = "Positive (1)" if label == 1 else "Negative (0)"
        print(f"  {label_name}: {count:,} ({count/len(df_valid)*100:.1f}%)")

    if num_samples and num_samples < len(df_valid):
        samples_per_class = num_samples // 2
        print(f"\nBalanced sampling: {num_samples:,} total samples (seed={seed})")
        print(f"  → {samples_per_class:,} negative + {samples_per_class:,} positive")

        df_negative = df_valid[df_valid[label_column] == 0]
        df_positive = df_valid[df_valid[label_column] == 1]

        if len(df_negative) < samples_per_class:
            print(f"  ⚠️  Warning: Only {len(df_negative):,} negative samples available")
            samples_per_class_neg = len(df_negative)
        else:
            samples_per_class_neg = samples_per_class

        if len(df_positive) < samples_per_class:
            print(f"  ⚠️  Warning: Only {len(df_positive):,} positive samples available")
            samples_per_class_pos = len(df_positive)
        else:
            samples_per_class_pos = samples_per_class

        df_neg_sampled = df_negative.sample(n=samples_per_class_neg, random_state=seed)
        df_pos_sampled = df_positive.sample(n=samples_per_class_pos, random_state=seed)
        df_sampled = pd.concat([df_neg_sampled, df_pos_sampled], ignore_index=True)
        df_sampled = df_sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        df_sampled = df_valid

    print(f"\nFinal set: {len(df_sampled):,} samples")
    final_counts = df_sampled[label_column].value_counts().sort_index()
    print("Final label distribution:")
    for label, count in final_counts.items():
        label_name = "Positive (1)" if label == 1 else "Negative (0)"
        print(f"  {label_name}: {count:,} ({count/len(df_sampled)*100:.1f}%)")
    print(f"{'='*80}\n")

    texts = df_sampled[text_column].astype(str).tolist()
    labels = df_sampled[label_column].astype(int).tolist()
    return texts, labels


def resolve_dataset_file(dataset, split, data_file):
    """Resolve a dataset file path based on dataset name and split."""
    dataset_norm = dataset.strip().lower()
    split_norm = split.strip().lower()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")

    if data_file:
        return data_file

    if dataset_norm == "hotel":
        return os.path.join(base, "Hotel", "hotels.csv")
    elif dataset_norm == "amazon":
        return os.path.join(base, "Amazon", f"{split_norm}.csv")
    elif dataset_norm == "yelp":
        return os.path.join(base, "Yelp", f"{split_norm}.csv")
    else:
        # Generic fallback
        return os.path.join(base, dataset, f"{split_norm}.csv")


def load_hotel_data_sampled(file_path, num_samples=2000, seed=42, 
                            text_column='Description', label_column='HotelRating'):
    """
    Load and sample hotel data with balanced classes.
    
    Args:
        file_path: Path to hotels.csv
        num_samples: Number of samples to load (None = all). Will be split 50/50 between classes.
        seed: Random seed for reproducible sampling
        text_column: Column with text
        label_column: Column with labels
    
    Returns:
        texts, labels
    """
    print(f"\n{'='*80}")
    print(f"Loading Hotel Dataset")
    print(f"{'='*80}")
    print(f"File: {file_path}")
    
    # Read CSV
    print("Reading CSV file...")
    df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    
    print(f"Total entries: {len(df):,}")
    
    # Filter valid samples
    print("Filtering valid samples...")
    valid_mask = (
        df[text_column].notna() & 
        (df[text_column].astype(str).str.strip().str.len() > 10) &  # At least 10 chars
        df[label_column].notna()
    )
    df_valid = df[valid_mask].copy()
    print(f"Valid samples: {len(df_valid):,}")
    
    # Label mapping: 1-2 star → negative (0), 4-5 star → positive (1), exclude 3-star
    label_mapping = {
        'FiveStar': 1,
        'FourStar': 1,
        'TwoStar': 0,
        'OneStar': 0,
    }
    
    # Map labels
    df_valid['binary_label'] = df_valid[label_column].map(label_mapping)
    # Drop any unmapped labels (including ThreeStar)
    df_valid = df_valid[df_valid['binary_label'].notna()].copy()
    
    print(f"Samples after label mapping (excluding 3-star): {len(df_valid):,}")
    print(f"Full dataset label distribution:")
    label_counts = df_valid['binary_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Positive (4-5 star)" if label == 1 else "Negative (1-2 star)"
        print(f"  {label_name}: {count:,} ({count/len(df_valid)*100:.1f}%)")
    
    # Balanced sampling: 50% negative, 50% positive
    if num_samples and num_samples < len(df_valid):
        samples_per_class = num_samples // 2
        print(f"\nBalanced sampling: {num_samples:,} total samples (seed={seed})")
        print(f"  → {samples_per_class:,} negative + {samples_per_class:,} positive")
        
        # Sample from each class
        df_negative = df_valid[df_valid['binary_label'] == 0]
        df_positive = df_valid[df_valid['binary_label'] == 1]
        
        # Check if we have enough samples
        if len(df_negative) < samples_per_class:
            print(f"  ⚠️  Warning: Only {len(df_negative):,} negative samples available")
            samples_per_class_neg = len(df_negative)
        else:
            samples_per_class_neg = samples_per_class
            
        if len(df_positive) < samples_per_class:
            print(f"  ⚠️  Warning: Only {len(df_positive):,} positive samples available")
            samples_per_class_pos = len(df_positive)
        else:
            samples_per_class_pos = samples_per_class
        
        df_neg_sampled = df_negative.sample(n=samples_per_class_neg, random_state=seed)
        df_pos_sampled = df_positive.sample(n=samples_per_class_pos, random_state=seed)
        
        df_sampled = pd.concat([df_neg_sampled, df_pos_sampled], ignore_index=True)
        # Shuffle the combined dataset
        df_sampled = df_sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        df_sampled = df_valid
        num_samples = len(df_sampled)
    
    print(f"\nFinal test set: {len(df_sampled):,} samples")
    final_counts = df_sampled['binary_label'].value_counts().sort_index()
    print(f"Test set label distribution:")
    for label, count in final_counts.items():
        label_name = "Positive (4-5 star)" if label == 1 else "Negative (1-2 star)"
        print(f"  {label_name}: {count:,} ({count/len(df_sampled)*100:.1f}%)")
    
    # Extract texts and labels
    texts = df_sampled[text_column].astype(str).tolist()
    labels = df_sampled['binary_label'].astype(int).tolist()
    
    print(f"{'='*80}\n")
    
    return texts, labels


def evaluate_model(model, test_loader, device, description="Inference"):
    """Evaluate model on test data with progress bar."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=description, ncols=100)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            original_text = batch['original_text']
            
            # Forward pass - handle both 3-value and 4-value returns
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=original_text,
                current_batch_num=batch_idx
            )
            # ProtoLens now returns 4 values: (logits, loss_mu, augmented_loss, similarity)
            # TTA wrappers may return 3 values
            outputs = result[0]  # First element is always logits
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            # Update progress bar with running accuracy
            acc = accuracy_score(all_labels, all_predictions)
            pbar.set_postfix({'Acc': f'{acc:.4f}'})
    
    # Final metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, digits=4)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, conf_matrix


def run_inference(
    model_path,
    dataset,
    split,
    data_file,
    num_samples=2000,
    methods=['baseline', 'tent'],
    batch_size=32,
    learning_rate=0.001,
    seed=42,
):
    """
    Run inference with multiple adaptation methods.
    
    Args:
        model_path: Path to trained ProtoLens model
        test_file: Path to Hotel CSV file
        num_samples: Number of test samples
        methods: List of methods to test ('baseline', 'tent')
        batch_size: Batch size
        learning_rate: LR for adaptation
        seed: Random seed
    """
    set_seed(seed)
    
    print("\n" + "="*80)
    print("ProtoLens Inference (Dataset-Agnostic)")
    print("="*80)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Eval dataset: {dataset} (split={split})")
    print(f"Samples: {num_samples:,}" if num_samples else "Samples: ALL")
    print(f"Methods: {', '.join(methods)}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print("="*80)
    
    # Load model
    print("\n>>> Loading trained model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint['pnfrl_args']
    
    print(f"Model config:")
    print(f"  - BERT: {saved_args['bert_model_name']}")
    print(f"  - Classes: {saved_args['num_classes']}")
    print(f"  - Prototypes: {saved_args['prototype_num']}")
    print(f"  - Hidden dim: {saved_args['hidden_dim']}")
    print(f"  - Max length: {saved_args['max_length']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Device: {device}")
    
    # Create args for model
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    source_dataset = saved_args.get('data_set', 'Yelp')
    args = Args(
        bert_model_name=saved_args['bert_model_name'],
        num_classes=saved_args['num_classes'],
        prototype_num=saved_args['prototype_num'],
        batch_size=saved_args['batch_size'],
        hidden_dim=saved_args['hidden_dim'],
        max_length=saved_args['max_length'],
        data_set=source_dataset,  # Use the model's prototype pool (typically Yelp)
        base_folder="Datasets",
        gaussian_num=6,
        window_size=5
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    # Resolve dataset file and load data
    resolved_file = resolve_dataset_file(dataset=dataset, split=split, data_file=data_file)
    if dataset.lower() == "hotel":
        test_texts, test_labels = load_hotel_data_sampled(
            resolved_file,
            num_samples=num_samples,
            seed=seed
        )
    else:
        test_texts, test_labels = load_standard_binary_csv_sampled(
            resolved_file,
            num_samples=num_samples,
            seed=seed,
            text_column="review",
            label_column="sentiment",
        )
    
    # Create dataloader
    print("Creating test dataloader...")
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, 
                                            saved_args['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=4)
    print(f"Test batches: {len(test_loader)}")
    
    # Results storage
    results = {}
    
    # ===== BASELINE =====
    if 'baseline' in methods:
        print("\n" + "="*80)
        print("METHOD: Baseline (No Adaptation)")
        print("="*80)
        
        # Load fresh model
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
        model.eval()
        
        acc, report, conf_matrix = evaluate_model(model, test_loader, device, 
                                                  description="Baseline")
        
        print(f"\n{'─'*80}")
        print(f"Baseline Results")
        print(f"{'─'*80}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        results['Baseline'] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ===== TENT =====
    if 'tent' in methods:
        print("\n" + "="*80)
        print("METHOD: TENT (Entropy Minimization)")
        print("="*80)
        
        # Load fresh model
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
        
        # Configure for TENT (use layernorm_attn_bias as per user request)
        print("Configuring TENT...")
        model = adapt_utils.configure_model(model, adaptation_mode='layernorm_attn_bias')
        params, param_names = adapt_utils.collect_params(model, adaptation_mode='layernorm_attn_bias')
        print(f"  Adaptation mode: layernorm_attn_bias")
        print(f"  Adapting {len(params)} parameters (LayerNorm + Attention biases)")
        print(f"  Learning rate: {learning_rate}")
        
        # Setup optimizer and wrap with TENT
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        tent_model = tent.Tent(model, optimizer, steps=1, episodic=False)
        
        acc, report, conf_matrix = evaluate_model(tent_model, test_loader, device,
                                                  description="TENT")
        
        print(f"\n{'─'*80}")
        print(f"TENT Results")
        print(f"{'─'*80}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Adaptation stats
        if hasattr(tent_model, 'adaptation_stats'):
            stats = tent_model.adaptation_stats
            print(f"\nAdaptation Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Adapted samples: {stats['adapted_samples']}")
            print(f"  Total updates: {stats['total_updates']}")
        
        results['TENT'] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        del tent_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ===== EATA =====
    if 'eata' in methods:
        print("\n" + "="*80)
        print("METHOD: EATA (Efficient Anti-Catastrophic TTA)")
        print("="*80)
        
        # Load fresh model
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
        
        # Configure for EATA (use layernorm_attn_bias as per user request)
        print("Configuring EATA...")
        model = adapt_utils.configure_model(model, adaptation_mode='layernorm_attn_bias')
        params, param_names = adapt_utils.collect_params(model, adaptation_mode='layernorm_attn_bias')
        print(f"  Adaptation mode: layernorm_attn_bias (same as TENT)")
        print(f"  Adapting {len(params)} parameters (LayerNorm + Attention biases)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Entropy margin: 0.4 (keeps confident predictions)")
        print(f"  Diversity margin: 0.05")
        
        # Setup optimizer and wrap with EATA
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        eata_model = eata.EATA(model, optimizer, fishers=None, steps=1, episodic=False)
        
        acc, report, conf_matrix = evaluate_model(eata_model, test_loader, device,
                                                  description="EATA")
        
        print(f"\n{'─'*80}")
        print(f"EATA Results")
        print(f"{'─'*80}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Adaptation stats
        if hasattr(eata_model, 'adaptation_stats'):
            stats = eata_model.adaptation_stats
            print(f"\nAdaptation Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Reliable samples (entropy filter): {stats['reliable_samples']}")
            print(f"  Adapted samples (both filters): {stats['adapted_samples']}")
            print(f"  Total updates: {stats['total_updates']}")
            print(f"  Adaptation rate: {stats['adapted_samples']/stats['total_samples']*100:.1f}%")
        
        results['EATA'] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        del eata_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ===== ProtoTTA =====
    if 'prototta' in methods:
        print("\n" + "="*80)
        print("METHOD: ProtoTTA (V3 Config)")
        print("="*80)
        
        # Load fresh model
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
        
        # Configure for ProtoTTA (V3 uses layernorm_attn_bias mode)
        print("Configuring ProtoTTA (V3)...")
        model = adapt_utils.configure_model(model, adaptation_mode='layernorm_attn_bias')
        params, param_names = adapt_utils.collect_params(model, adaptation_mode='layernorm_attn_bias')
        print(f"  Adaptation mode: layernorm_attn_bias")
        print(f"  Adapting {len(params)} parameters (LayerNorm + Attention biases)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Geometric filter: threshold=0.1 (based on avg_similarity ~0.1)")
        print(f"  Aggregation: max similarity (no sub-prototypes in ProtoLens)")
        
        # Setup optimizer and wrap with ProtoTTA
        # NOTE: Lower threshold (0.1) and max strategy for ProtoLens
        # ProtoLens doesn't have sub-prototypes like ProtoViT, so consensus doesn't apply
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        prototta_model = proto_tta.ProtoTTA(
            model, optimizer, steps=1, episodic=False,
            use_geometric_filter=True,
            geo_filter_threshold=0.1,  # Lower threshold based on actual avg_similarity (~0.1)
            consensus_strategy='max',   # Use max similarity (no sub-prototypes)
            consensus_ratio=0.5  # Not used with 'max' strategy
        )
        
        acc, report, conf_matrix = evaluate_model(prototta_model, test_loader, device,
                                                  description="ProtoTTA")
        
        print(f"\n{'─'*80}")
        print(f"ProtoTTA Results")
        print(f"{'─'*80}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Adaptation stats
        if hasattr(prototta_model, 'adaptation_stats'):
            stats = prototta_model.adaptation_stats
            print(f"\nAdaptation Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Adapted samples: {stats['adapted_samples']}")
            print(f"  Total updates: {stats['total_updates']}")
            print(f"  Adaptation rate: {stats['adapted_samples']/stats['total_samples']*100:.1f}%")
            if 'filter_stats' in stats:
                fstats = stats['filter_stats']
                print(f"  Filtered out: {fstats['filtered_out']}")
                if len(fstats['avg_similarity']) > 0:
                    avg_sim = sum(fstats['avg_similarity']) / len(fstats['avg_similarity'])
                    print(f"  Avg similarity: {avg_sim:.4f}")
        
        results['ProtoTTA'] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        del prototta_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Accuracy':<12} {'Change':<12}")
    print("─"*80)
    
    baseline_acc = results.get('Baseline', {}).get('accuracy', None)
    
    for method_name, result in results.items():
        acc = result['accuracy']
        if baseline_acc and method_name != 'Baseline':
            change = (acc - baseline_acc) * 100
            change_str = f"{change:+.2f}%"
        else:
            change_str = "─"
        print(f"{method_name:<20} {acc:.4f} ({acc*100:5.2f}%)  {change_str}")
    
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run ProtoLens inference on a dataset (Hotel/Yelp) with adaptation methods'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ProtoLens model (.pth)')
    parser.add_argument('--dataset', type=str, default='hotel',
                       choices=['hotel', 'yelp', 'amazon'],
                       help="Which dataset to evaluate on. 'hotel' uses hotels.csv mapping; 'yelp'/'amazon' use Datasets/<Dataset>/<split>.csv")
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help="Which split to use for standard datasets (e.g., Yelp). Ignored for hotel CSV.")
    parser.add_argument('--data_file', type=str, default=None,
                       help="Optional path to dataset file. If omitted: hotel -> Datasets/Hotel/hotels.csv, yelp -> Datasets/Yelp/<split>.csv")
    # Backward compatible flags
    parser.add_argument('--test_file', type=str, default=None,
                       help='[Deprecated] Use --data_file. Kept for backward compatibility.')
    parser.add_argument('--num_samples', type=int, default=4000,
                       help='Number of samples (balanced 50/50 when possible). Use 0 to use ALL.')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['baseline', 'tent', 'eata', 'prototta'],
                       choices=['baseline', 'tent', 'eata', 'prototta'],
                       help='Methods to test')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                       help='Learning rate for adaptation (use low values like 0.00001 to prevent collapse)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Backward-compat: allow --test_file to behave like --data_file
    data_file = args.data_file or args.test_file
    num_samples = None if (args.num_samples is not None and args.num_samples <= 0) else args.num_samples
    
    run_inference(
        model_path=args.model_path,
        dataset=args.dataset,
        split=args.split,
        data_file=data_file,
        num_samples=num_samples,
        methods=args.methods,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
