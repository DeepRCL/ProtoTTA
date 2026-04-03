"""
Compare two ProtoTTA configurations across all corruptions.

This script is a DIRECT COPY of run_inference_amazon_c.py logic,
just comparing two specific configs instead of multiple methods.

Usage:
    CUDA_VISIBLE_DEVICES=0 python compare_prototta_configs.py
    CUDA_VISIBLE_DEVICES=0 python compare_prototta_configs.py --quick
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer

from PLens import BERTClassifier
from utils import TextClassificationDataset
import proto_tta
import adapt_utils


# ============================================================================
# Two configurations to compare
# ============================================================================
CONFIGS = [
    {'name': 'Config1_Temp5', 'sigmoid_temperature': 5.0, 'geo_threshold': 0.1, 'learning_rate': 5e-6, 'use_geo_filter': True},
    {'name': 'Config2_Temp7', 'sigmoid_temperature': 7.0, 'geo_threshold': 0.1, 'learning_rate': 5e-6, 'use_geo_filter': True},
]

# All corruption types and severities
ALL_CORRUPTIONS = ['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive']
ALL_SEVERITIES = [20, 40, 60, 80]

QUICK_CORRUPTIONS = ['qwerty', 'aggressive']
QUICK_SEVERITIES = [40, 60]


# ============================================================================
# Configuration class - EXACTLY as in run_inference_amazon_c.py
# ============================================================================
class Cfg:
    """Configuration for TTA optimization."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.00001  # Low LR to prevent collapse - will be updated per config
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


# ============================================================================
# Functions COPIED from run_inference_amazon_c.py
# ============================================================================
def setup_optimizer(params):
    """Set up optimizer for TTA adaptation."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.OPTIM.METHOD} not supported")


def load_model(model_path, device):
    """Load trained ProtoLens model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint.get('pnfrl_args', {})
    
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
    
    return model, tokenizer, args


def create_dataloader(df, tokenizer, max_length, batch_size):
    """Create DataLoader from DataFrame."""
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                     pin_memory=True, num_workers=4)


def setup_prototta(model, adaptation_mode: str, use_geo_filter: bool, geo_threshold: float,
                   importance_mode: str = 'global', sigmoid_temperature: float = 5.0):
    """Set up ProtoTTA adaptation - EXACT COPY from run_inference_amazon_c.py."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"ProtoTTA: Adapting {len(params)} parameter groups")
    print(f"  Geometric filter: {use_geo_filter}, Threshold: {geo_threshold}")
    print(f"  Sigmoid temperature: {sigmoid_temperature}")
    
    optimizer = setup_optimizer(params)
    return proto_tta.ProtoTTA(model, optimizer, steps=cfg.OPTIM.STEPS,
                              episodic=cfg.MODEL.EPISODIC,
                              use_geometric_filter=use_geo_filter,
                              geo_filter_threshold=geo_threshold,
                              consensus_strategy='max',
                              importance_mode=importance_mode,
                              sigmoid_temperature=sigmoid_temperature)


def evaluate_baseline(model, dataloader, device) -> Dict:
    """Evaluate model without adaptation - EXACT COPY from run_inference_amazon_c.py."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Baseline", ncols=100)
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
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }


def evaluate_tta_method(tta_model, dataloader, device, description="TTA") -> Dict:
    """Evaluate with TTA adaptation - EXACT COPY from run_inference_amazon_c.py."""
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
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'adaptation_stats': adaptation_stats
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='Datasets/Amazon-C')
    parser.add_argument('--model_path', default='log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--adaptation_mode', default='layernorm_attn_bias')
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corruptions = QUICK_CORRUPTIONS if args.quick else ALL_CORRUPTIONS
    severities = QUICK_SEVERITIES if args.quick else ALL_SEVERITIES
    
    print("=" * 80)
    print("ProtoTTA Configuration Comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Config 1: Temp={CONFIGS[0]['sigmoid_temperature']}, GeoTh={CONFIGS[0]['geo_threshold']}, LR={CONFIGS[0]['learning_rate']}")
    print(f"Config 2: Temp={CONFIGS[1]['sigmoid_temperature']}, GeoTh={CONFIGS[1]['geo_threshold']}, LR={CONFIGS[1]['learning_rate']}")
    print(f"Corruptions: {corruptions}")
    print(f"Severities: {severities}")
    print("=" * 80)
    
    # Results storage
    results = {'baseline': [], **{cfg['name']: [] for cfg in CONFIGS}}
    all_details = []
    
    for corruption in corruptions:
        for severity in severities:
            config_name = f"{corruption}_s{severity}"
            filepath = os.path.join(args.data_dir, f'amazon_c_{corruption}_s{severity}.csv')
            
            if not os.path.exists(filepath):
                print(f"\n[{config_name}] SKIPPED (file not found)")
                continue
            
            df = pd.read_csv(filepath)
            print(f"\n{'='*60}")
            print(f"[{config_name}] ({len(df)} samples)")
            print(f"{'='*60}")
            
            # ===== Baseline =====
            print("\n--- Baseline ---")
            model, tokenizer, model_args = load_model(args.model_path, device)
            dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
            baseline_result = evaluate_baseline(model, dataloader, device)
            baseline_acc = baseline_result['accuracy']
            results['baseline'].append(baseline_acc)
            print(f"Baseline accuracy: {baseline_acc:.4f}")
            
            detail = {
                'corruption': corruption,
                'severity': severity,
                'baseline': baseline_acc
            }
            
            # ===== Test each ProtoTTA config =====
            for config in CONFIGS:
                print(f"\n--- {config['name']} ---")
                
                # CRITICAL: Update cfg.OPTIM.LR BEFORE loading model
                cfg.OPTIM.LR = config['learning_rate']
                
                # Load fresh model for each config
                model, tokenizer, model_args = load_model(args.model_path, device)
                dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                
                # Set up ProtoTTA using EXACT same function as run_inference_amazon_c.py
                tta_model = setup_prototta(
                    model, 
                    args.adaptation_mode,
                    config['use_geo_filter'],
                    config['geo_threshold'],
                    importance_mode='global',
                    sigmoid_temperature=config['sigmoid_temperature']
                )
                
                # Evaluate
                tta_result = evaluate_tta_method(tta_model, dataloader, device, config['name'])
                acc = tta_result['accuracy']
                improvement = (acc - baseline_acc) * 100
                
                # Get adaptation stats
                stats = tta_result.get('adaptation_stats', {})
                adapted = stats.get('adapted_samples', 0)
                total = stats.get('total_samples', len(df))
                
                results[config['name']].append(acc)
                detail[config['name']] = {
                    'accuracy': acc,
                    'improvement': improvement,
                    'adapted': adapted,
                    'total': total
                }
                
                print(f"\n{config['name']} Results:")
                print(f"  Accuracy: {acc:.4f} ({improvement:+.2f}% vs baseline)")
                print(f"  Adapted: {adapted}/{total}")
            
            all_details.append(detail)
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY - All Corruptions")
    print("=" * 80)
    
    print(f"\n{'Config':<25} {'Avg Acc':<12} {'Avg Improvement':<18} {'Std Imp':<10}")
    print("-" * 70)
    
    baseline_avg = np.mean(results['baseline'])
    print(f"{'Baseline':<25} {baseline_avg:.4f}")
    
    for config in CONFIGS:
        accs = results[config['name']]
        if accs:
            avg_acc = np.mean(accs)
            improvements = [(a - b) * 100 for a, b in zip(accs, results['baseline'])]
            avg_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            print(f"{config['name']:<25} {avg_acc:.4f}       {avg_imp:+.2f}%              {std_imp:.2f}%")
    
    # Winner
    if results[CONFIGS[0]['name']] and results[CONFIGS[1]['name']]:
        avg1 = np.mean(results[CONFIGS[0]['name']])
        avg2 = np.mean(results[CONFIGS[1]['name']])
        winner = CONFIGS[0] if avg1 > avg2 else CONFIGS[1]
        print(f"\n=> WINNER: {winner['name']} (Temp={winner['sigmoid_temperature']})")
    
    # Save results
    results_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"config_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'configs': CONFIGS,
        'corruptions': corruptions,
        'severities': severities,
        'results': {k: [float(v) for v in vals] for k, vals in results.items()},
        'details': all_details,
        'summary': {
            'baseline_avg': float(baseline_avg),
            'config1_avg': float(np.mean(results[CONFIGS[0]['name']])) if results[CONFIGS[0]['name']] else 0,
            'config2_avg': float(np.mean(results[CONFIGS[1]['name']])) if results[CONFIGS[1]['name']] else 0,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
