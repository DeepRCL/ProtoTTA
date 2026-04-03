"""
Run Test-Time Adaptation (TTA) Inference on Yelp-C (Corrupted Yelp) Dataset.

Evaluates ProtoLens model with various TTA methods on corrupted Yelp data.
Supports various corruption types and severity levels for meaningful TTA testing.

Usage:
    python run_inference_yelp_c.py --methods baseline tent prototta --corruption_type mixed --severity 30
    python run_inference_yelp_c.py --methods prototta --geo_filter --geo_threshold 0.05

Output:
    - Console: Accuracy, classification report, confusion matrix
    - Datasets/Yelp-C/results/: Detailed results JSON
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset
import tent
import eata
import proto_tta
import adapt_utils


# ============================================================================
# Configuration (matches run_inference_amazon_c.py)
# ============================================================================

class Cfg:
    """Configuration for TTA optimization."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.000005  # Low LR to prevent collapse
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


def setup_optimizer(params):
    """Set up optimizer for TTA adaptation."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD)
    return optim.SGD(params, lr=cfg.OPTIM.LR, momentum=0.9, weight_decay=cfg.OPTIM.WD)


def parse_args():
    parser = argparse.ArgumentParser(description='TTA evaluation on Yelp-C')
    
    # Dataset settings
    parser.add_argument('--data_dir', type=str, default='Datasets/Yelp-C',
                       help='Directory containing Yelp-C datasets')
    parser.add_argument('--corruption_type', type=str, default='mixed',
                       choices=['qwerty', 'swap', 'remove_char', 'remove_space', 
                               'punctuation', 'misspelling', 'mixed', 'aggressive', 'clean'],
                       help='Type of corruption')
    parser.add_argument('--severity', type=int, default=30,
                       help='Corruption severity (%%)')
    
    # Model settings
    parser.add_argument('--model_path', type=str,
                       default='log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth',
                       help='Path to trained ProtoLens model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    
    # TTA settings
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['baseline', 'tent', 'eata', 'prototta'],
                       help='TTA methods to evaluate')
    parser.add_argument('--learning_rate', type=float, default=0.000005,
                       help='Learning rate for TTA')
    parser.add_argument('--adaptation_mode', type=str, default='layernorm_attn_bias',
                       help='Which parameters to adapt')
    
    # ProtoTTA specific
    parser.add_argument('--geo_filter', action='store_true', default=True,
                       help='Use geometric filtering for ProtoTTA')
    parser.add_argument('--geo_threshold', type=float, default=0.05,
                       help='Geometric filter threshold')
    parser.add_argument('--importance_mode', type=str, default='global',
                       choices=['global', 'class_specific'],
                       help='Prototype importance weighting mode: global (avg across classes) or class_specific (for predicted class)')
    
    # EATA specific
    parser.add_argument('--e_margin', type=float, default=0.4,
                       help='EATA entropy margin')
    parser.add_argument('--d_margin', type=float, default=0.05,
                       help='EATA diversity margin')
    
    # Output
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON')
    
    return parser.parse_args()


def load_corrupted_data(data_dir: str, corruption_type: str, severity: int) -> pd.DataFrame:
    """Load corrupted Yelp-C dataset."""
    if corruption_type == 'clean':
        filename = 'yelp_c_clean.csv'
    else:
        filename = f'yelp_c_{corruption_type}_s{severity}.csv'
    
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}\nRun generate_yelp_c.py first!")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filename}")
    
    if 'was_modified' in df.columns:
        modified_count = df['was_modified'].sum()
        print(f"  Modified samples: {modified_count} ({modified_count/len(df)*100:.1f}%)")
    
    return df


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, AutoTokenizer]:
    """Load trained ProtoLens model."""
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint.get('pnfrl_args', {})
    
    print(f"Model config:")
    for k, v in saved_args.items():
        print(f"  - {k}: {v}")
    
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(
        bert_model_name=saved_args.get('bert_model_name', 'all-mpnet-base-v2'),
        num_classes=saved_args.get('num_classes', 2),
        prototype_num=saved_args.get('prototype_num', 50),
        batch_size=saved_args.get('batch_size', 32),
        hidden_dim=saved_args.get('hidden_dim', 768),
        max_length=saved_args.get('max_length', 512),
        data_set='Yelp',
        base_folder='Datasets',
        gaussian_num=6,
        window_size=5
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    print("Initializing model...")
    model = BERTClassifier(
        args=args,
        bert_model_name='sentence-transformers/all-mpnet-base-v2',
        num_classes=args.num_classes,
        num_prototype=args.prototype_num,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        max_length=args.max_length,
        tokenizer=tokenizer
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  Model loaded successfully!")
    
    return model, tokenizer, args


def create_dataloader(df: pd.DataFrame, tokenizer, max_length: int, batch_size: int) -> DataLoader:
    """Create DataLoader from DataFrame."""
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                     pin_memory=True, num_workers=4)


# ============================================================================
# TTA Method Setup
# ============================================================================

def setup_tent(model, adaptation_mode: str):
    """Set up TENT adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"TENT: Adapting {len(params)} parameter groups")
    
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_eata(model, adaptation_mode: str, e_margin: float, d_margin: float):
    """Set up EATA adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"EATA: Adapting {len(params)} parameter groups")
    print(f"  E-margin: {e_margin}, D-margin: {d_margin}")
    
    optimizer = setup_optimizer(params)
    return eata.EATA(model, optimizer, fishers=None, steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC, e_margin=e_margin, d_margin=d_margin)


def setup_prototta(model, adaptation_mode: str, use_geo_filter: bool, geo_threshold: float,
                   importance_mode: str = 'global'):
    """Set up ProtoTTA adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"ProtoTTA: Adapting {len(params)} parameter groups")
    print(f"  Geometric filter: {use_geo_filter}, Threshold: {geo_threshold}")
    print(f"  Importance mode: {importance_mode}")
    
    optimizer = setup_optimizer(params)
    return proto_tta.ProtoTTA(model, optimizer, steps=cfg.OPTIM.STEPS,
                              episodic=cfg.MODEL.EPISODIC,
                              use_geometric_filter=use_geo_filter,
                              geo_filter_threshold=geo_threshold,
                              consensus_strategy='max',
                              importance_mode=importance_mode)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_baseline(model, dataloader, device, description="Baseline") -> Dict:
    """Evaluate model without adaptation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description, ncols=100)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            original_text = batch['original_text']
            
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=original_text,
                current_batch_num=0
            )
            outputs = result[0] if isinstance(result, tuple) else result
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'Acc': f'{acc:.4f}'})
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }


def evaluate_tta_method(tta_model, dataloader, device, description="TTA") -> Dict:
    """Evaluate with TTA adaptation."""
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=description, ncols=100)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        original_text = batch['original_text']
        
        try:
            result = tta_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=original_text,
                current_batch_num=0
            )
            outputs = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
        
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'Acc': f'{acc:.4f}'})
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # Get adaptation stats if available
    adaptation_stats = {}
    if hasattr(tta_model, 'adaptation_stats'):
        adaptation_stats = tta_model.adaptation_stats.copy()
        if adaptation_stats.get('total_samples', 0) > 0:
            adaptation_stats['adaptation_rate'] = (
                adaptation_stats.get('adapted_samples', 0) / 
                adaptation_stats['total_samples'] * 100
            )
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'adaptation_stats': adaptation_stats
    }


def print_results(results: Dict, corruption_type: str, severity: int):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print(f"RESULTS: Yelp-C ({corruption_type}, severity={severity}%)")
    print("=" * 80)
    
    baseline_acc = results.get('baseline', {}).get('accuracy', 0)
    
    print(f"\n{'Method':<16} {'Accuracy':<16} {'Adapted':<12} {'Improvement':<10}")
    print("-" * 55)
    
    for method, data in results.items():
        acc = data.get('accuracy', 0)
        improvement = (acc - baseline_acc) * 100 if method != 'baseline' else 0
        
        adapted = 'N/A'
        if 'adaptation_stats' in data and data['adaptation_stats']:
            stats = data['adaptation_stats']
            adapted = f"{stats.get('adapted_samples', 0)}/{stats.get('total_samples', 0)}"
        
        imp_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
        if method == 'baseline':
            imp_str = "+0.00%"
        
        print(f"{method:<16} {acc:.4f} ({acc*100:.1f}%) {adapted:<12} {imp_str}")
    
    # Detailed results
    for method, data in results.items():
        print(f"\n{'─' * 40}")
        print(f"{method.upper()} Detailed Results")
        print(f"{'─' * 40}")
        print(f"Accuracy: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
        
        if data.get('adaptation_stats'):
            print(f"\nAdaptation Statistics:")
            for k, v in data['adaptation_stats'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, int):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        if isinstance(v2, float):
                            print(f"    {k2}: {v2:.4f}")
                        elif isinstance(v2, int):
                            print(f"    {k2}: {v2}")
                        elif isinstance(v2, list) and len(v2) > 0:
                            if isinstance(v2[0], float):
                                print(f"    {k2}: min={min(v2):.4f}, max={max(v2):.4f}, mean={sum(v2)/len(v2):.4f}")
                            elif isinstance(v2[0], int):
                                print(f"    {k2}: {len(v2)} values, sum={sum(v2)}")
                            else:
                                print(f"    {k2}: {len(v2)} values")
                        else:
                            print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
        
        print(f"\nConfusion Matrix:")
        conf = data['confusion_matrix']
        print(f"  Predicted:  Neg    Pos")
        print(f"  Actual Neg:{conf[0][0]:>6} {conf[0][1]:>6}")
        print(f"  Actual Pos:{conf[1][0]:>6} {conf[1][1]:>6}")


def main():
    args = parse_args()
    
    # Update cfg with args
    cfg.OPTIM.LR = args.learning_rate
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "=" * 80)
    print("ProtoLens TTA Evaluation on Yelp-C")
    print("=" * 80)
    print(f"Corruption type: {args.corruption_type}")
    print(f"Severity: {args.severity}%")
    print(f"Methods: {args.methods}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Adaptation mode: {args.adaptation_mode}")
    if 'prototta' in args.methods:
        print(f"ProtoTTA importance mode: {args.importance_mode}")
    
    # Load data
    df = load_corrupted_data(args.data_dir, args.corruption_type, args.severity)
    
    # Load model
    model, tokenizer, model_args = load_model(args.model_path, device)
    
    # Create dataloader
    dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
    print(f"\nDataLoader: {len(dataloader)} batches of size {args.batch_size}")
    
    # Evaluate each method
    results = {}
    
    for method in args.methods:
        print(f"\n{'═' * 60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'═' * 60}")
        
        # Reload model for clean state
        model, tokenizer, model_args = load_model(args.model_path, device)
        
        if method == 'baseline':
            results['baseline'] = evaluate_baseline(model, dataloader, device)
        
        elif method == 'tent':
            tent_model = setup_tent(model, args.adaptation_mode)
            results['tent'] = evaluate_tta_method(tent_model, dataloader, device, "TENT")
        
        elif method == 'eata':
            eata_model = setup_eata(model, args.adaptation_mode, args.e_margin, args.d_margin)
            results['eata'] = evaluate_tta_method(eata_model, dataloader, device, "EATA")
        
        elif method == 'prototta':
            prototta_model = setup_prototta(model, args.adaptation_mode, args.geo_filter, 
                                            args.geo_threshold, args.importance_mode)
            results['prototta'] = evaluate_tta_method(prototta_model, dataloader, device, "ProtoTTA")
    
    # Print results
    print_results(results, args.corruption_type, args.severity)
    
    # Save results
    if args.save_results:
        results_dir = os.path.join(args.data_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results_{args.corruption_type}_s{args.severity}_{timestamp}.json"
        results_path = os.path.join(results_dir, results_file)
        
        # Convert to serializable format
        output = {
            'corruption_type': args.corruption_type,
            'severity': args.severity,
            'methods': args.methods,
            'learning_rate': args.learning_rate,
            'importance_mode': args.importance_mode if 'prototta' in args.methods else None,
            'results': {}
        }
        for method, data in results.items():
            output['results'][method] = {
                'accuracy': data['accuracy'],
                'confusion_matrix': data['confusion_matrix'],
                'adaptation_stats': data.get('adaptation_stats', {})
            }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
