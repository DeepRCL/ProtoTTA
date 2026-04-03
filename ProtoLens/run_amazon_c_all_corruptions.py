"""
Run TTA Evaluation on ALL Amazon-C corruptions with Best Tuned Settings.

Evaluates ProtoLens TTA methods (Baseline, TENT, EATA, ProtoTTA) across multiple
corruption types and severities. Saves results in a comprehensive JSON format
similar to ProtoViT's robustness_results.json for paper comparisons.

NOTE: This script is designed to match the behavior of run_inference_amazon_c.py
exactly. Each method/corruption combination loads a fresh model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_amazon_c_all_corruptions.py
    CUDA_VISIBLE_DEVICES=0 python run_amazon_c_all_corruptions.py --quick

Output:
    Datasets/Amazon-C/results/robustness_results_<timestamp>.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
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

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset
import tent
import eata
import proto_tta
import adapt_utils
from prototype_metrics import ProtoLensMetricsEvaluator, EfficiencyTracker


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
            self.LR = 0.00001  # Will be updated from args
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


# Default corruptions and severities for paper
ALL_CORRUPTION_TYPES = ['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive']
ALL_SEVERITIES = [20, 40, 60, 80]

# Quick mode settings
QUICK_CORRUPTION_TYPES = ['qwerty', 'aggressive']
QUICK_SEVERITIES = [30, 50]


def setup_optimizer(params):
    """Set up optimizer for TTA adaptation - uses cfg.OPTIM.LR (same as run_inference_amazon_c.py)."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.OPTIM.METHOD} not supported")


def parse_args():
    parser = argparse.ArgumentParser(description='TTA evaluation on ALL Amazon-C corruptions')
    
    # Dataset settings
    parser.add_argument('--data_dir', type=str, default='Datasets/Amazon-C',
                       help='Directory containing Amazon-C datasets')
    parser.add_argument('--corruption_types', type=str, nargs='+', default=None,
                       help='Corruption types to evaluate (default: all)')
    parser.add_argument('--severities', type=int, nargs='+', default=None,
                       help='Severity levels to evaluate (default: 20-80)')
    
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
                       help='Learning rate for ALL TTA methods')
    parser.add_argument('--adaptation_mode', type=str, default='layernorm_attn_bias',
                       help='Which parameters to adapt')
    
    # EATA specific
    parser.add_argument('--e_margin', type=float, default=0.6,
                       help='EATA entropy margin (higher=adapt more samples)')
    parser.add_argument('--d_margin', type=float, default=0.05,
                       help='EATA diversity margin')
    
    # ProtoTTA specific - matching run_inference_amazon_c.py defaults
    parser.add_argument('--geo_filter', action='store_true', default=True,
                       help='Use geometric filtering for ProtoTTA')
    parser.add_argument('--no_geo_filter', action='store_true', default=False,
                       help='Disable geometric filtering')
    parser.add_argument('--geo_threshold', type=float, default=0.1,
                       help='Geometric filter threshold')
    parser.add_argument('--sigmoid_temperature', type=float, default=5.0,
                       help='Sigmoid temperature for ProtoTTA')
    parser.add_argument('--importance_mode', type=str, default='global',
                       choices=['global', 'class_specific'],
                       help='Prototype importance weighting mode')
    
    # Prototype metrics - matching ProtoViT evaluate_robustness.py
    parser.add_argument('--prototype-metrics', action='store_true', default=True,
                       help='Compute prototype-based metrics (PAC, PCA, Sparsity, etc.)')
    parser.add_argument('--no-prototype-metrics', action='store_true', default=False,
                       help='Disable prototype metrics computation')
    parser.add_argument('--track-efficiency', action='store_true', default=True,
                       help='Track computational efficiency metrics')
    
    # Modes
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer corruptions and severities')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without .json)')
    
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


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, any, any]:
    """Load trained ProtoLens model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint.get('pnfrl_args', {})
    
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():\
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


# ============================================================================
# TTA Method Setup - MUST match run_inference_amazon_c.py exactly
# ============================================================================

def setup_tent(model, adaptation_mode: str):
    """Set up TENT adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_eata(model, adaptation_mode: str, e_margin: float, d_margin: float):
    """Set up EATA adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)
    return eata.EATA(model, optimizer, fishers=None, steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC, e_margin=e_margin, d_margin=d_margin)


def setup_prototta(model, adaptation_mode: str, use_geo_filter: bool, geo_threshold: float,
                   importance_mode: str = 'global', sigmoid_temperature: float = 5.0):
    """Set up ProtoTTA adaptation - same signature as run_inference_amazon_c.py."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    
    optimizer = setup_optimizer(params)
    return proto_tta.ProtoTTA(model, optimizer, steps=cfg.OPTIM.STEPS,
                              episodic=cfg.MODEL.EPISODIC,
                              use_geometric_filter=use_geo_filter,
                              geo_filter_threshold=geo_threshold,
                              consensus_strategy='max',
                              importance_mode=importance_mode,
                              sigmoid_temperature=sigmoid_temperature)


# ============================================================================
# Evaluation Functions - MUST match run_inference_amazon_c.py exactly
# ============================================================================

def evaluate_baseline(model, dataloader, device, 
                       metrics_evaluator: Optional[ProtoLensMetricsEvaluator] = None,
                       track_efficiency: bool = False) -> Dict:
    """Evaluate model without adaptation.
    
    Args:
        model: Base model
        dataloader: Test data loader
        device: Device
        metrics_evaluator: Optional evaluator for prototype metrics
        track_efficiency: Track computational efficiency
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_similarities = []
    
    # Setup efficiency tracker
    efficiency_tracker = None
    if track_efficiency:
        efficiency_tracker = EfficiencyTracker('baseline', device=str(device))
        # Baseline has NO adapted parameters
        efficiency_tracker.count_adapted_parameters(model, adapted_params=[])
    
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch['label'].size(0)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            original_text = batch['original_text']
            
            if efficiency_tracker:
                with efficiency_tracker.track_inference(batch_size):
                    result = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        special_tokens_mask=special_tokens_mask,
                        mode="test",
                        original_text=original_text,
                        current_batch_num=0
                    )
            else:
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    mode="test",
                    original_text=original_text,
                    current_batch_num=0
                )
            
            if isinstance(result, tuple) and len(result) >= 4:
                outputs = result[0]
                similarities = result[3]
                all_similarities.append(similarities.cpu())
            else:
                outputs = result[0] if isinstance(result, tuple) else result
            
            all_logits.append(outputs.cpu())
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Collect baseline for metrics evaluator
    if metrics_evaluator is not None and all_similarities:
        metrics_evaluator.clean_activations = torch.cat(all_similarities, dim=0)
        metrics_evaluator.clean_logits = torch.cat(all_logits, dim=0)
        metrics_evaluator.clean_predictions = torch.tensor(all_preds)
        metrics_evaluator.clean_labels = torch.tensor(all_labels)
    
    result_dict = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'num_samples': len(all_labels)
    }
    
    # Add efficiency metrics
    if efficiency_tracker:
        result_dict['efficiency'] = efficiency_tracker.get_metrics()
    
    # For baseline, no adaptation stats (set defaults)
    result_dict['adaptation_rate'] = 0.0
    result_dict['avg_updates_per_sample'] = 0.0
    result_dict['adaptation_stats'] = {
        'total_samples': len(all_labels),
        'adapted_samples': 0,
        'total_updates': 0
    }
    
    return result_dict


def evaluate_tta_method(tta_model, dataloader, device, method_name="TTA",
                        metrics_evaluator: Optional[ProtoLensMetricsEvaluator] = None,
                        track_efficiency: bool = False,
                        compute_proto_metrics: bool = True) -> Dict:
    """Evaluate with TTA adaptation and compute all metrics.
    
    Args:
        tta_model: TTA-adapted model
        dataloader: Test data loader
        device: Device
        method_name: Name of the method
        metrics_evaluator: Optional evaluator for prototype metrics
        track_efficiency: Track computational efficiency
        compute_proto_metrics: Compute PAC, PCA, etc.
    """
    all_preds = []
    all_labels = []
    all_logits = []
    all_similarities = []
    
    # Setup efficiency tracker
    efficiency_tracker = None
    if track_efficiency:
        efficiency_tracker = EfficiencyTracker(method_name, device=str(device))
        # Count adapted parameters
        if hasattr(tta_model, 'model'):
            adapted_params = [p for p in tta_model.model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(tta_model.model, adapted_params)
        else:
            adapted_params = [p for p in tta_model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(tta_model, adapted_params)
    
    num_batches = 0
    for batch in dataloader:
        batch_size = batch['label'].size(0)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        original_text = batch['original_text']
        
        # Forward with efficiency tracking
        if efficiency_tracker:
            with efficiency_tracker.track_inference(batch_size):
                result = tta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    mode="test",
                    original_text=original_text,
                    current_batch_num=num_batches
                )
        else:
            result = tta_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=original_text,
                current_batch_num=num_batches
            )
        
        # Parse result - extract similarities
        if isinstance(result, tuple) and len(result) >= 4:
            outputs = result[0]
            similarities = result[3]
            all_similarities.append(similarities.detach().cpu())
        else:
            outputs = result[0] if isinstance(result, tuple) else result
        
        all_logits.append(outputs.detach().cpu())
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        num_batches += 1
    
    # Record adaptation steps
    if efficiency_tracker:
        efficiency_tracker.record_adaptation_step(num_batches)
    
    # Build result dictionary
    result_dict = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # Compute prototype metrics if evaluator is provided
    if compute_proto_metrics and metrics_evaluator is not None and all_similarities:
        adapted_activations = torch.cat(all_similarities, dim=0)
        adapted_logits = torch.cat(all_logits, dim=0)
        adapted_predictions = torch.tensor(all_preds)
        labels_tensor = torch.tensor(all_labels)
        
        # PAC (Prototype Activation Consistency)
        pac_metrics = metrics_evaluator.compute_pac(adapted_activations)
        result_dict.update(pac_metrics)
        
        # PCA (Prototype Class Alignment)
        pca_metrics = metrics_evaluator.compute_pca(adapted_activations, labels_tensor, top_k=10)
        result_dict.update(pca_metrics)
        
        # Sparsity
        sparsity_metrics = metrics_evaluator.compute_sparsity(adapted_activations)
        result_dict.update(sparsity_metrics)
        
        # PCA Weighted
        pca_weighted_metrics = metrics_evaluator.compute_pca_weighted(adapted_activations, labels_tensor, top_k=10)
        result_dict.update(pca_weighted_metrics)
        
        # Calibration
        calibration_metrics = metrics_evaluator.compute_calibration(adapted_predictions, adapted_logits)
        result_dict.update(calibration_metrics)
        
        # GT Class Contribution
        gt_contrib_metrics = metrics_evaluator.compute_gt_class_contribution(adapted_activations, labels_tensor)
        result_dict.update(gt_contrib_metrics)
    
    # Adaptation stats from model
    if hasattr(tta_model, 'adaptation_stats'):
        stats = tta_model.adaptation_stats
        total_samples = stats.get('total_samples', len(all_labels))
        adapted_samples = stats.get('adapted_samples', 0)
        total_updates = stats.get('total_updates', adapted_samples)
        
        result_dict['adaptation_rate'] = adapted_samples / max(total_samples, 1)
        result_dict['avg_updates_per_sample'] = total_updates / max(total_samples, 1)
        result_dict['adaptation_stats'] = {
            'total_samples': total_samples,
            'adapted_samples': adapted_samples,
            'total_updates': total_updates
        }
    else:
        # Default adaptation stats for methods without tracking
        result_dict['adaptation_rate'] = 1.0  # Assume all samples adapted
        result_dict['avg_updates_per_sample'] = 1.0
        result_dict['adaptation_stats'] = {
            'total_samples': len(all_labels),
            'adapted_samples': len(all_labels),
            'total_updates': num_batches
        }
    
    # Efficiency metrics
    if efficiency_tracker:
        result_dict['efficiency'] = efficiency_tracker.get_metrics()
    
    return result_dict


def main():
    args = parse_args()
    
    # CRITICAL: Update cfg.OPTIM.LR from args - same as run_inference_amazon_c.py
    cfg.OPTIM.LR = args.learning_rate
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("ProtoLens TTA Robustness Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Determine corruption types and severities
    if args.quick:
        corruption_types = QUICK_CORRUPTION_TYPES
        severities = QUICK_SEVERITIES
        print("MODE: Quick")
    else:
        corruption_types = args.corruption_types or ALL_CORRUPTION_TYPES
        severities = args.severities or ALL_SEVERITIES
        print("MODE: Full")
    
    print(f"\nCorruption types: {corruption_types}")
    print(f"Severities: {severities}")
    print(f"Methods: {args.methods}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Adaptation mode: {args.adaptation_mode}")
    
    # ProtoTTA settings
    use_geo = args.geo_filter and not args.no_geo_filter
    print(f"\nProtoTTA settings:")
    print(f"  geo_filter: {use_geo}")
    print(f"  geo_threshold: {args.geo_threshold}")
    print(f"  sigmoid_temperature: {args.sigmoid_temperature}")
    print(f"  importance_mode: {args.importance_mode}")
    
    print(f"\nEATA settings:")
    print(f"  e_margin: {args.e_margin}")
    print(f"  d_margin: {args.d_margin}")
    
    # Prototype metrics settings
    compute_proto_metrics = getattr(args, 'prototype_metrics', True) and not getattr(args, 'no_prototype_metrics', False)
    track_efficiency = getattr(args, 'track_efficiency', True)
    print(f"\nMetrics settings:")
    print(f"  prototype_metrics: {compute_proto_metrics}")
    print(f"  track_efficiency: {track_efficiency}")
    
    print("=" * 80)
    
    # Store results in ProtoViT-like format
    results = {method: {} for method in args.methods}
    method_accuracies = {method: [] for method in args.methods}
    
    total_tests = len(corruption_types) * len(severities)
    test_idx = 0
    
    for corruption_type in corruption_types:
        for severity in severities:
            test_idx += 1
            config_name = f"{corruption_type}_s{severity}"
            
            print(f"\n[{test_idx}/{total_tests}] {config_name}")
            print("-" * 40)
            
            try:
                df = load_corrupted_data(args.data_dir, corruption_type, severity)
            except FileNotFoundError as e:
                print(f"  SKIPPED: {e}")
                continue
            
            # Get baseline accuracy first (needed for improvement calculation)
            baseline_acc = None
            metrics_evaluator = None  # Will be initialized from baseline
            
            # Evaluate each method
            for method in args.methods:
                # CRITICAL: Load fresh model for EACH method
                model, tokenizer, model_args = load_model(args.model_path, device)
                dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                
                if method == 'baseline':
                    # Initialize metrics evaluator from baseline model
                    if compute_proto_metrics:
                        metrics_evaluator = ProtoLensMetricsEvaluator(model, device=str(device))
                    
                    result = evaluate_baseline(
                        model, dataloader, device,
                        metrics_evaluator=metrics_evaluator,
                        track_efficiency=track_efficiency
                    )
                    baseline_acc = result['accuracy']
                
                elif method == 'tent':
                    # Need to collect baseline first if not done
                    if compute_proto_metrics and metrics_evaluator is None:
                        base_model, _, _ = load_model(args.model_path, device)
                        metrics_evaluator = ProtoLensMetricsEvaluator(base_model, device=str(device))
                        base_dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                        metrics_evaluator.collect_baseline(base_model, base_dataloader, verbose=False)
                    
                    tta_model = setup_tent(model, args.adaptation_mode)
                    result = evaluate_tta_method(
                        tta_model, dataloader, device, "tent",
                        metrics_evaluator=metrics_evaluator,
                        track_efficiency=track_efficiency,
                        compute_proto_metrics=compute_proto_metrics
                    )
                
                elif method == 'eata':
                    # Need to collect baseline first if not done
                    if compute_proto_metrics and metrics_evaluator is None:
                        base_model, _, _ = load_model(args.model_path, device)
                        metrics_evaluator = ProtoLensMetricsEvaluator(base_model, device=str(device))
                        base_dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                        metrics_evaluator.collect_baseline(base_model, base_dataloader, verbose=False)
                    
                    tta_model = setup_eata(model, args.adaptation_mode, args.e_margin, args.d_margin)
                    result = evaluate_tta_method(
                        tta_model, dataloader, device, "eata",
                        metrics_evaluator=metrics_evaluator,
                        track_efficiency=track_efficiency,
                        compute_proto_metrics=compute_proto_metrics
                    )
                
                elif method == 'prototta':
                    # Need to collect baseline first if not done
                    if compute_proto_metrics and metrics_evaluator is None:
                        base_model, _, _ = load_model(args.model_path, device)
                        metrics_evaluator = ProtoLensMetricsEvaluator(base_model, device=str(device))
                        base_dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
                        metrics_evaluator.collect_baseline(base_model, base_dataloader, verbose=False)
                    
                    tta_model = setup_prototta(
                        model, args.adaptation_mode, use_geo, args.geo_threshold,
                        args.importance_mode, args.sigmoid_temperature
                    )
                    result = evaluate_tta_method(
                        tta_model, dataloader, device, "prototta",
                        metrics_evaluator=metrics_evaluator,
                        track_efficiency=track_efficiency,
                        compute_proto_metrics=compute_proto_metrics
                    )
                
                else:
                    print(f"  Unknown method: {method}")
                    continue
                
                # Store result
                if corruption_type not in results[method]:
                    results[method][corruption_type] = {}
                results[method][corruption_type][str(severity)] = result
                method_accuracies[method].append(result['accuracy'])
                
                # Print result
                acc = result['accuracy']
                if baseline_acc is not None and method != 'baseline':
                    improvement = (acc - baseline_acc) * 100
                    imp_str = f"({improvement:+.2f}%)"
                else:
                    imp_str = ""
                
                # Build metrics string
                metrics_str = ""
                if result.get('adaptation_rate') is not None and method != 'baseline':
                    metrics_str += f" adapt:{result['adaptation_rate']*100:.1f}%"
                if result.get('PAC_mean') is not None:
                    metrics_str += f" PAC:{result['PAC_mean']*100:.1f}%"
                if result.get('PCA_mean') is not None:
                    metrics_str += f" PCA:{result['PCA_mean']*100:.1f}%"
                    
                print(f"  {method}: {acc:.4f} {imp_str}{metrics_str}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<16} {'Avg Accuracy':<16} {'Improvement':<12}")
    print("-" * 45)
    
    baseline_avg = np.mean(method_accuracies.get('baseline', [0])) if method_accuracies.get('baseline') else 0
    
    for method in args.methods:
        if method_accuracies[method]:
            avg_acc = np.mean(method_accuracies[method])
            std_acc = np.std(method_accuracies[method])
            improvement = (avg_acc - baseline_avg) * 100 if method != 'baseline' else 0
            imp_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            print(f"{method:<16} {avg_acc:.4f} ± {std_acc:.4f}   {imp_str if method != 'baseline' else ''}")
    
    # ========== Save Results ==========
    if args.save_results:
        results_dir = os.path.join(args.data_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.output_name:
            results_file = f"{args.output_name}.json"
        else:
            results_file = f"robustness_results_{timestamp}.json"
        results_path = os.path.join(results_dir, results_file)
        
        # Build output in ProtoViT-like format
        output = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'model_path': args.model_path,
                'data_dir': args.data_dir,
                'batch_size': args.batch_size,
                'corruption_types': corruption_types,
                'severities': severities,
                'methods': args.methods,
                'learning_rate': args.learning_rate,
                'adaptation_mode': args.adaptation_mode,
                'mode_configs': {
                    'baseline': {},
                    'tent': {
                        'learning_rate': args.learning_rate,
                        'adaptation_mode': args.adaptation_mode
                    },
                    'eata': {
                        'learning_rate': args.learning_rate,
                        'e_margin': args.e_margin,
                        'd_margin': args.d_margin,
                        'adaptation_mode': args.adaptation_mode
                    },
                    'prototta': {
                        'learning_rate': args.learning_rate,
                        'sigmoid_temperature': args.sigmoid_temperature,
                        'geo_filter': use_geo,
                        'geo_threshold': args.geo_threshold,
                        'importance_mode': args.importance_mode,
                        'adaptation_mode': args.adaptation_mode
                    }
                }
            },
            'results': results,
            'summary': {
                method: {
                    'avg_accuracy': float(np.mean(accs)) if accs else 0,
                    'std_accuracy': float(np.std(accs)) if accs else 0,
                    'min_accuracy': float(np.min(accs)) if accs else 0,
                    'max_accuracy': float(np.max(accs)) if accs else 0,
                    'improvement_over_baseline': float(np.mean(accs) - baseline_avg) * 100 if accs else 0,
                    'num_tests': len(accs)
                }
                for method, accs in method_accuracies.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
