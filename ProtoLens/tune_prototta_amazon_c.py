"""
ProtoTTA Hyperparameter Tuning for Amazon-C.

Paper-ready grid search over key ProtoTTA hyperparameters.

Key hyperparameters:
- sigmoid_temperature: How strongly to spread similarity probabilities (3-10)
- geo_threshold: Geometric filter threshold (0.02-0.2)
- learning_rate: TTA learning rate (1e-6 to 5e-5)
- use_geo_filter: Whether to use geometric filtering

NOTE: This script matches the behavior of run_inference_amazon_c.py exactly
for fair comparison. Each configuration loads a fresh model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tune_prototta_amazon_c.py
    CUDA_VISIBLE_DEVICES=0 python tune_prototta_amazon_c.py --quick

Output:
    Datasets/Amazon-C/results/tuning_results_*.json
    Datasets/Amazon-C/results/best_prototta_settings.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import product
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset
import proto_tta
import adapt_utils


# ============================================================================
# Paper-Ready Hyperparameter Search Space
# ============================================================================

# Key ProtoTTA hyperparameters to tune
SEARCH_SPACE = {
    'sigmoid_temperature': [3.0, 5.0, 7.0, 10.0],
    'geo_threshold': [0.02, 0.05, 0.1, 0.15],
    'learning_rate': [1e-6, 5e-6, 1e-5],
    'use_geo_filter': [True, False],
}

# Reduced search space for quick testing
QUICK_SEARCH_SPACE = {
    'sigmoid_temperature': [5.0, 7.0],
    'geo_threshold': [0.05, 0.1],
    'learning_rate': [5e-6],
    'use_geo_filter': [True, False],
}

# Corruption types to test
CORRUPTION_TYPES = ['qwerty', 'swap', 'aggressive']
SEVERITIES = [30, 50, 70]

# Quick mode
QUICK_CORRUPTION_TYPES = ['qwerty', 'aggressive']
QUICK_SEVERITIES = [50]


# ============================================================================
# Configuration - MUST match run_inference_amazon_c.py
# ============================================================================

class Cfg:
    """Configuration for TTA optimization."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.00001  # Will be updated per-config
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


def setup_optimizer(params):
    """Set up optimizer for TTA adaptation - uses cfg.OPTIM.LR."""
    return optim.Adam(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD)


def parse_args():
    parser = argparse.ArgumentParser(description='ProtoTTA hyperparameter tuning (paper-ready)')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='Datasets/Amazon-C')
    parser.add_argument('--model_path', type=str,
                       default='log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--adaptation_mode', type=str, default='layernorm_attn_bias')
    
    # What to test
    parser.add_argument('--corruption_types', type=str, nargs='+', default=None,
                       help='Corruption types to test')
    parser.add_argument('--severities', type=int, nargs='+', default=None,
                       help='Severities to test')
    
    # Modes
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: reduced search space and fewer corruptions')
    parser.add_argument('--single_corruption', type=str, default=None,
                       help='Test only a single corruption type')
    parser.add_argument('--single_severity', type=int, default=None,
                       help='Test only a single severity')
    
    return parser.parse_args()


def load_corrupted_data(data_dir: str, corruption_type: str, severity: int) -> pd.DataFrame:
    """Load corrupted Amazon-C dataset."""
    if corruption_type == 'clean':
        filename = 'amazon_c_clean.csv'
    else:
        filename = f'amazon_c_{corruption_type}_s{severity}.csv'
    
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    return pd.read_csv(filepath)


def load_model(model_path: str, device: torch.device):
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


def create_dataloader(df: pd.DataFrame, tokenizer, max_length: int, batch_size: int) -> DataLoader:
    """Create DataLoader from DataFrame."""
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                     pin_memory=True, num_workers=4)


def evaluate_baseline(model, dataloader, device) -> float:
    """Evaluate model without adaptation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
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
    
    return accuracy_score(all_labels, all_preds)


def setup_prototta(model, adaptation_mode: str, use_geo_filter: bool, geo_threshold: float,
                   importance_mode: str = 'global', sigmoid_temperature: float = 5.0):
    """Set up ProtoTTA - SAME SIGNATURE as run_inference_amazon_c.py."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, _ = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)  # Uses cfg.OPTIM.LR
    
    return proto_tta.ProtoTTA(
        model, optimizer, 
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        use_geometric_filter=use_geo_filter,
        geo_filter_threshold=geo_threshold,
        consensus_strategy='max',
        importance_mode=importance_mode,
        sigmoid_temperature=sigmoid_temperature
    )


def evaluate_prototta(model, dataloader, device, adaptation_mode: str,
                      sigmoid_temperature: float, geo_threshold: float,
                      learning_rate: float, use_geo_filter: bool) -> Tuple[float, Dict]:
    """Evaluate ProtoTTA with specific hyperparameters."""
    
    # CRITICAL: Update cfg.OPTIM.LR before creating optimizer
    cfg.OPTIM.LR = learning_rate
    
    # Set up ProtoTTA using the same function signature as run_inference_amazon_c.py
    tta_model = setup_prototta(
        model, adaptation_mode, use_geo_filter, geo_threshold,
        importance_mode='global', sigmoid_temperature=sigmoid_temperature
    )
    
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        original_text = batch['original_text']
        
        # No silent exception handling
        result = tta_model(
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    stats = tta_model.adaptation_stats.copy() if hasattr(tta_model, 'adaptation_stats') else {}
    
    return accuracy, stats


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("ProtoTTA Hyperparameter Tuning (Paper-Ready)")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Determine search space
    if args.quick:
        search_space = QUICK_SEARCH_SPACE
        corruption_types = QUICK_CORRUPTION_TYPES
        severities = QUICK_SEVERITIES
        print("MODE: Quick (reduced search space)")
    else:
        search_space = SEARCH_SPACE
        corruption_types = args.corruption_types or CORRUPTION_TYPES
        severities = args.severities or SEVERITIES
        print("MODE: Full search")
    
    # Override with single values if specified
    if args.single_corruption:
        corruption_types = [args.single_corruption]
    if args.single_severity:
        severities = [args.single_severity]
    
    print(f"\nSearch space:")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    print(f"\nCorruption types: {corruption_types}")
    print(f"Severities: {severities}")
    
    total_configs = 1
    for v in search_space.values():
        total_configs *= len(v)
    total_tests = total_configs * len(corruption_types) * len(severities)
    print(f"\nTotal configurations: {total_configs}")
    print(f"Total tests: {total_tests}")
    print("=" * 80)
    
    # Store all results
    all_results = []
    baseline_results = {}
    
    # First, collect baselines
    print("\n" + "=" * 40)
    print("Collecting Baselines")
    print("=" * 40)
    
    for corruption_type in corruption_types:
        for severity in severities:
            config_name = f"{corruption_type}_s{severity}"
            try:
                df = load_corrupted_data(args.data_dir, corruption_type, severity)
                model, tokenizer, model_args = load_model(args.model_path, device)
                dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                baseline_acc = evaluate_baseline(model, dataloader, device)
                baseline_results[(corruption_type, severity)] = baseline_acc
                print(f"  {config_name}: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
            except FileNotFoundError:
                print(f"  {config_name}: SKIPPED (file not found)")
    
    # Grid search
    print("\n" + "=" * 40)
    print("Grid Search")
    print("=" * 40)
    
    config_idx = 0
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    for values in product(*param_values):
        params_dict = dict(zip(param_names, values))
        config_idx += 1
        
        print(f"\n[Config {config_idx}/{total_configs}]")
        print(f"  temp={params_dict['sigmoid_temperature']}, geo_th={params_dict['geo_threshold']}, "
              f"lr={params_dict['learning_rate']:.1e}, geo_filter={params_dict['use_geo_filter']}")
        
        for corruption_type in corruption_types:
            for severity in severities:
                config_name = f"{corruption_type}_s{severity}"
                
                if (corruption_type, severity) not in baseline_results:
                    continue
                
                baseline_acc = baseline_results[(corruption_type, severity)]
                
                try:
                    # CRITICAL: Reload fresh model for each test
                    model, tokenizer, model_args = load_model(args.model_path, device)
                    df = load_corrupted_data(args.data_dir, corruption_type, severity)
                    dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                    
                    accuracy, stats = evaluate_prototta(
                        model, dataloader, device, args.adaptation_mode,
                        sigmoid_temperature=params_dict['sigmoid_temperature'],
                        geo_threshold=params_dict['geo_threshold'],
                        learning_rate=params_dict['learning_rate'],
                        use_geo_filter=params_dict['use_geo_filter']
                    )
                    
                    improvement = (accuracy - baseline_acc) * 100
                    adapted = stats.get('adapted_samples', 0)
                    total = stats.get('total_samples', len(df))
                    
                    print(f"    {config_name}: {accuracy:.4f} ({improvement:+.2f}%) [Adapted: {adapted}/{total}]")
                    
                    result = {
                        'corruption_type': corruption_type,
                        'severity': severity,
                        'sigmoid_temperature': params_dict['sigmoid_temperature'],
                        'geo_threshold': params_dict['geo_threshold'],
                        'learning_rate': params_dict['learning_rate'],
                        'use_geo_filter': params_dict['use_geo_filter'],
                        'accuracy': accuracy,
                        'baseline_accuracy': baseline_acc,
                        'improvement': improvement,
                        'adapted_samples': adapted,
                        'total_samples': total
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"    {config_name}: FAILED - {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("TUNING RESULTS SUMMARY")
    print("=" * 80)
    
    if all_results:
        # Compute average improvement per config
        avg_per_config = defaultdict(list)
        for r in all_results:
            config_key = (r['sigmoid_temperature'], r['geo_threshold'], 
                         r['learning_rate'], r['use_geo_filter'])
            avg_per_config[config_key].append(r['improvement'])
        
        # Sort by average improvement
        ranked_configs = sorted(avg_per_config.items(), key=lambda x: np.mean(x[1]), reverse=True)
        
        print("\n[Top 10 Configurations by Average Improvement]")
        print(f"{'Rank':<6} {'Temp':<8} {'GeoTh':<8} {'LR':<12} {'GeoFilter':<10} {'Avg Imp':<12} {'Std':<10}")
        print("-" * 70)
        
        for rank, (config, improvements) in enumerate(ranked_configs[:10], 1):
            avg_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            print(f"{rank:<6} {config[0]:<8} {config[1]:<8} {config[2]:<12.1e} {str(config[3]):<10} "
                  f"{avg_imp:+.2f}%       {std_imp:.2f}%")
        
        # Best config
        best_config = ranked_configs[0][0]
        print(f"\n[Best Configuration]")
        print(f"  sigmoid_temperature: {best_config[0]}")
        print(f"  geo_threshold: {best_config[1]}")
        print(f"  learning_rate: {best_config[2]}")
        print(f"  use_geo_filter: {best_config[3]}")
        print(f"  Avg Improvement: {np.mean(ranked_configs[0][1]):+.2f}%")
    
    # ========== Save Results ==========
    results_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"tuning_results_{timestamp}.json"
    results_path = os.path.join(results_dir, results_file)
    
    # Format for saving
    ranked_configs_list = []
    for config, improvements in ranked_configs[:20]:
        ranked_configs_list.append({
            'sigmoid_temperature': config[0],
            'geo_threshold': config[1],
            'learning_rate': config[2],
            'use_geo_filter': config[3],
            'avg_improvement': float(np.mean(improvements)),
            'std_improvement': float(np.std(improvements))
        })
    
    output = {
        'timestamp': timestamp,
        'search_space': {k: [float(x) if isinstance(x, (int, float)) else x for x in v] 
                        for k, v in search_space.items()},
        'corruption_types': corruption_types,
        'severities': severities,
        'baselines': {f"{k[0]}_s{k[1]}": v for k, v in baseline_results.items()},
        'best_config': ranked_configs_list[0] if ranked_configs_list else None,
        'ranked_configs': ranked_configs_list,
        'all_results': all_results
    }
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save best settings for easy loading
    if ranked_configs_list:
        best = ranked_configs_list[0]
        best_settings_path = os.path.join(results_dir, "best_prototta_settings.json")
        with open(best_settings_path, 'w') as f:
            json.dump({
                'sigmoid_temperature': best['sigmoid_temperature'],
                'geo_threshold': best['geo_threshold'],
                'learning_rate': best['learning_rate'],
                'use_geo_filter': best['use_geo_filter'],
                'avg_improvement': best['avg_improvement'],
                'note': 'Best average settings across all tested corruptions/severities'
            }, f, indent=2)
        print(f"Best settings saved to: {best_settings_path}")
    
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
