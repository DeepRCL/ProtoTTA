#!/usr/bin/env python3
"""
Enhanced script to add MEMO results with full prototype metrics to an existing robustness evaluation JSON.
This allows you to run MEMO separately without re-running all other methods.

Usage:
    python evaluate_robustness_memo_only.py --model ./saved_models/best_model.pth \
                                             --data_dir ./datasets/cub200_c/ \
                                             --input ./robustness_results_sev5_metrics.json \
                                             --output ./robustness_results_sev5_with_memo.json
"""

import os
import sys
import argparse
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm

# Import ProtoViT modules
import model  # Necessary for torch.load
import train_and_test as tnt
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import memo_adapt
from prototype_tta_metrics import PrototypeMetricsEvaluator
from enhanced_prototype_metrics import EnhancedPrototypeMetrics
from efficiency_metrics import EfficiencyTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def setup_memo(model, lr=0.00025, batch_size=16, steps=1):
    """Set up MEMO adaptation with optimized settings for speed."""
    model = memo_adapt.configure_model(model)
    params, param_names = memo_adapt.collect_params(model)
    
    # MEMO uses SGD optimizer with momentum
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    
    memo_model = memo_adapt.MEMO(
        model,
        optimizer,
        steps=steps,
        batch_size=batch_size,
        episodic=True
    )
    return memo_model


def evaluate_model(model, loader, description="Inference", verbose=True,
                   proto_evaluator=None, compute_proto_metrics=False,
                   efficiency_tracker=None):
    """Run evaluation and return accuracy and prototype metrics.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        description: Description for progress
        verbose: Whether to print progress
        proto_evaluator: PrototypeMetricsEvaluator instance (optional)
        compute_proto_metrics: Whether to compute prototype metrics
        efficiency_tracker: EfficiencyTracker instance (optional)
    
    Returns:
        Tuple of (accuracy, proto_metrics_dict)
    """
    if verbose:
        print(f'\n{description}...')
    
    # MEMO evaluation with batch_size=1
    model.eval()
    n_examples = 0
    n_correct = 0
    
    # Track efficiency if enabled
    if efficiency_tracker is not None:
        for batch_idx, (images, labels) in enumerate(tqdm(loader, disable=not verbose)):
            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.size(0)
            
            # Track this batch
            with efficiency_tracker.track_inference(batch_size):
                with torch.no_grad():
                    outputs = model(images)
                    
                    # Get predictions - handle both tuple and single output
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    _, predicted = logits.max(1)
                    batch_correct = predicted.eq(labels).sum().item()
                    
                    n_correct += batch_correct
                    n_examples += batch_size
    else:
        # Standard evaluation without per-batch timing
        for batch_idx, (images, labels) in enumerate(tqdm(loader, disable=not verbose)):
            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.size(0)
            
            with torch.no_grad():
                outputs = model(images)
                
                # Get predictions - handle both tuple and single output
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                _, predicted = logits.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                
                n_correct += batch_correct
                n_examples += batch_size
    
    accu = n_correct / n_examples
    if verbose:
        print(f'Accuracy: {accu*100:.2f}%')
    
    # Optionally compute prototype metrics
    proto_metrics = {}
    if compute_proto_metrics and proto_evaluator is not None:
        if verbose:
            print('Computing prototype-based metrics...')
        if isinstance(proto_evaluator, EnhancedPrototypeMetrics):
            proto_metrics = proto_evaluator.evaluate_tta_method_enhanced(
                model, loader, top_k=10, max_samples=None, verbose=False,
                track_adaptation_rate=True
            )
        else:
            proto_metrics = proto_evaluator.evaluate_tta_method(
                model, loader, top_k=10, max_samples=None, verbose=False
            )
        if verbose:
            print(f'  PAC: {proto_metrics.get("PAC_mean", 0)*100:.2f}%')
            print(f'  PCA: {proto_metrics.get("PCA_mean", 0)*100:.2f}%')
            print(f'  Sparsity: {proto_metrics.get("sparsity_gini_mean", 0):.3f}')
    
    return accu, proto_metrics


def load_corrupted_dataset(data_dir, corruption_type, severity, batch_size=1):
    """Load a pre-generated corrupted dataset."""
    corruption_path = Path(data_dir) / corruption_type / str(severity)
    
    if not corruption_path.exists():
        raise FileNotFoundError(f"Corrupted dataset not found at {corruption_path}")
    
    # Simple transform: just resize, to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = datasets.ImageFolder(corruption_path, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    return loader


def load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size=1):
    """Load clean dataset and apply corruption on-the-fly."""
    transform = get_corrupted_transform(img_size, mean, std, corruption_type, severity)
    
    dataset = datasets.ImageFolder(clean_data_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    return loader


def main():
    parser = argparse.ArgumentParser(
        description='Add MEMO results with full metrics to existing robustness evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument('--model', type=str, 
                       default='./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth',
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str,
                       default='./datasets/cub200_c/',
                       help='Path to pre-generated CUB-C dataset')
    parser.add_argument('--clean_data_dir', type=str,
                       default=None,
                       help='Path to clean test data (for on-the-fly). Default: uses test_dir from settings.')
    
    # Data loading
    parser.add_argument('--on_the_fly', action='store_true',
                       help='Generate corruptions on-the-fly instead of loading pre-generated')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Path to existing results JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save updated results JSON')
    
    # Hardware
    parser.add_argument('--gpuid', type=str, default='0',
                       help='GPU ID to use')
    
    # MEMO settings
    parser.add_argument('--memo-lr', type=float, default=0.00025,
                       help='MEMO learning rate (default: 0.00025)')
    parser.add_argument('--memo-batch-size', type=int, default=16,
                       help='MEMO augmented views per step (default: 16, reduced for speed)')
    parser.add_argument('--memo-steps', type=int, default=1,
                       help='MEMO adaptation steps per sample (default: 1)')
    
    # Metrics settings
    parser.add_argument('--prototype-metrics', action='store_true', default=True,
                       help='Compute prototype-based metrics (PAC, PCA, sparsity, etc.)')
    parser.add_argument('--use-enhanced-metrics', action='store_true', default=True,
                       help='Use enhanced prototype metrics (PCA-weighted, calibration, GT contribution)')
    parser.add_argument('--proto-baseline-samples', type=int, default=1000,
                       help='Number of clean samples to use for baseline (default: 1000)')
    parser.add_argument('--track-efficiency', action='store_true', default=True,
                       help='Track computational efficiency metrics')
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Use clean_data_dir from args or fallback to settings
    clean_data_dir = args.clean_data_dir if args.clean_data_dir else test_dir
    
    # Verify model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Load existing results
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input results file not found: {args.input}")
    
    print(f"\nLoading existing results from {args.input}")
    with open(args.input, 'r') as f:
        existing_data = json.load(f)
    
    results_dict = existing_data.get('results', {})
    metadata = existing_data.get('metadata', {})
    
    # Define corruption types (severity 5 only)
    corruption_types = [
        'gaussian_noise', 'fog', 'gaussian_blur', 'elastic_transform', 
        'brightness', 'jpeg_compression', 'contrast', 'defocus_blur', 
        'frost', 'impulse_noise', 'pixelate', 
        'shot_noise', 
        'speckle_noise'
    ]
    severity = 5
    severity_key = str(severity)
    
    # Initialize MEMO results structure
    if 'memo' not in results_dict:
        results_dict['memo'] = {}
    
    for corruption_type in corruption_types:
        if corruption_type not in results_dict['memo']:
            results_dict['memo'][corruption_type] = {}
        if severity_key not in results_dict['memo'][corruption_type]:
            results_dict['memo'][corruption_type][severity_key] = None
    
    # Setup prototype evaluator if requested
    proto_evaluator = None
    if args.prototype_metrics:
        print("\n" + "="*80)
        print("Setting up Prototype Metrics Evaluator")
        print("="*80)
        
        # Load base model
        base_model = torch.load(args.model, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()
        
        # Initialize evaluator (enhanced if requested)
        if args.use_enhanced_metrics:
            proto_evaluator = EnhancedPrototypeMetrics(base_model, device=device)
            print("Using EnhancedPrototypeMetrics (includes PCA-Weighted, Calibration, GT Δ)")
        else:
            proto_evaluator = PrototypeMetricsEvaluator(base_model, device=device)
            print("Using standard PrototypeMetricsEvaluator (PAC, PCA, Sparsity)")
        
        # Load clean data for baseline
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(clean_data_dir, transform_clean)
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset, batch_size=test_batch_size, shuffle=False, 
            num_workers=8, pin_memory=False
        )
        
        # Collect clean baseline (enhanced if requested)
        if args.use_enhanced_metrics:
            proto_evaluator.collect_clean_baseline_enhanced(
                clean_loader, 
                max_samples=args.proto_baseline_samples, 
                verbose=True
            )
        else:
            proto_evaluator.collect_clean_baseline(
                clean_loader, 
                max_samples=args.proto_baseline_samples, 
                verbose=True
            )
        
        print(f"✓ Prototype evaluator ready")
        print("="*80)
        
        del base_model
        torch.cuda.empty_cache()
    
    # Print configuration
    print("="*80)
    print("MEMO-ONLY ROBUSTNESS EVALUATION (WITH FULL METRICS)")
    print("="*80)
    print(f"Model:           {args.model}")
    print(f"Data directory:  {args.data_dir if not args.on_the_fly else 'On-the-fly generation'}")
    print(f"Clean data dir:  {clean_data_dir}")
    print(f"Corruptions:     {len(corruption_types)} types")
    print(f"Severity:        {severity}")
    print(f"MEMO settings:   lr={args.memo_lr}, batch_size={args.memo_batch_size}, steps={args.memo_steps}")
    print(f"Prototype metrics: {args.prototype_metrics}")
    print(f"Enhanced metrics:  {args.use_enhanced_metrics}")
    print(f"Efficiency tracking: {args.track_efficiency}")
    print(f"Input file:      {args.input}")
    print(f"Output file:     {args.output}")
    print("="*80)
    
    # Create list of corruptions to evaluate
    corruptions_to_eval = []
    for corruption_type in corruption_types:
        # Check if already completed (with full metrics)
        existing_result = results_dict['memo'].get(corruption_type, {}).get(severity_key)
        if existing_result is not None:
            # Check if it has full metrics (not just accuracy)
            if isinstance(existing_result, dict) and len(existing_result) > 1:
                print(f"✓ Skipping {corruption_type} (already completed with full metrics)")
                continue
            elif isinstance(existing_result, dict) and 'accuracy' in existing_result:
                print(f"⚠ {corruption_type} has only accuracy, will recompute with full metrics")
        corruptions_to_eval.append(corruption_type)
    
    if len(corruptions_to_eval) == 0:
        print("\nAll MEMO evaluations already completed with full metrics!")
        return
    
    print(f"\nEvaluating MEMO on {len(corruptions_to_eval)} corruptions...")
    
    # Evaluate each corruption
    start_time = time.time()
    
    pbar = tqdm(corruptions_to_eval, desc="MEMO Evaluation", unit="corruption")
    for corruption_type in pbar:
        pbar.set_description(f"MEMO on {corruption_type}")
        
        try:
            # Load data - MEMO requires batch_size=1
            if args.on_the_fly:
                loader = load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size=1)
            else:
                loader = load_corrupted_dataset(args.data_dir, corruption_type, severity, batch_size=1)
            
            # Load fresh model
            base_model = torch.load(args.model, weights_only=False)
            base_model = base_model.to(device)
            base_model.eval()
            
            # Setup MEMO
            memo_model = setup_memo(base_model, lr=args.memo_lr, 
                                   batch_size=args.memo_batch_size, 
                                   steps=args.memo_steps)
            
            # Setup efficiency tracker
            efficiency_tracker = None
            if args.track_efficiency:
                efficiency_tracker = EfficiencyTracker('memo', device=str(device))
                # Count adapted parameters
                if hasattr(memo_model, 'model'):
                    adapted_params = [p for p in memo_model.model.parameters() if p.requires_grad]
                    efficiency_tracker.count_adapted_parameters(memo_model.model, adapted_params)
                else:
                    adapted_params = [p for p in memo_model.parameters() if p.requires_grad]
                    efficiency_tracker.count_adapted_parameters(memo_model, adapted_params)
            
            # Evaluate
            acc, proto_metrics = evaluate_model(
                memo_model, loader, 
                description=f"MEMO on {corruption_type}-{severity}",
                verbose=False,
                proto_evaluator=proto_evaluator,
                compute_proto_metrics=args.prototype_metrics,
                efficiency_tracker=efficiency_tracker
            )
            
            # Record adaptation steps (MEMO: 1 step per sample)
            if efficiency_tracker:
                num_samples = len(loader.dataset)
                efficiency_tracker.record_adaptation_step(num_samples * args.memo_steps)
            
            # Build result dictionary
            result = {'accuracy': float(acc)}
            
            # Add prototype metrics if computed
            if args.prototype_metrics and proto_metrics:
                # Standard metrics
                result.update({
                    'PAC_mean': proto_metrics.get('PAC_mean'),
                    'PAC_std': proto_metrics.get('PAC_std'),
                    'PCA_mean': proto_metrics.get('PCA_mean'),
                    'PCA_std': proto_metrics.get('PCA_std'),
                    'sparsity_gini_mean': proto_metrics.get('sparsity_gini_mean'),
                    'sparsity_active_mean': proto_metrics.get('sparsity_active_mean'),
                })
                
                # Enhanced metrics (if available)
                if 'PCA_weighted_mean' in proto_metrics:
                    result.update({
                        'PCA_weighted_mean': proto_metrics.get('PCA_weighted_mean'),
                        'PCA_weighted_std': proto_metrics.get('PCA_weighted_std'),
                    })
                if 'calibration_agreement' in proto_metrics:
                    result.update({
                        'calibration_agreement': proto_metrics.get('calibration_agreement'),
                        'calibration_logit_corr': proto_metrics.get('calibration_logit_corr'),
                    })
                if 'gt_class_contrib_improvement' in proto_metrics:
                    result.update({
                        'gt_class_contrib_improvement': proto_metrics.get('gt_class_contrib_improvement'),
                        'gt_class_contrib_change_mean': proto_metrics.get('gt_class_contrib_change_mean'),
                    })
                if 'adaptation_rate' in proto_metrics:
                    result.update({
                        'adaptation_rate': proto_metrics.get('adaptation_rate'),
                        'avg_updates_per_sample': proto_metrics.get('avg_updates_per_sample'),
                    })
            
            # Add efficiency metrics if tracked
            if efficiency_tracker:
                efficiency_metrics = efficiency_tracker.get_metrics()
                result['efficiency'] = efficiency_metrics
                
                # Extract adaptation stats from model if available
                if hasattr(memo_model, 'adaptation_stats'):
                    result['adaptation_stats'] = memo_model.adaptation_stats.copy()
                elif hasattr(memo_model, 'model') and hasattr(memo_model.model, 'adaptation_stats'):
                    result['adaptation_stats'] = memo_model.model.adaptation_stats.copy()
            
            # Store result
            results_dict['memo'][corruption_type][severity_key] = result
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{acc*100:.2f}%'})
            
            # Save after each corruption (iterative saving)
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata,
                'results': results_dict
            }
            
            # Atomic write
            temp_file = args.output + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            os.replace(temp_file, args.output)
            
            # Cleanup
            del memo_model
            del base_model
            if efficiency_tracker:
                del efficiency_tracker
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate MEMO on {corruption_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results_dict['memo'][corruption_type][severity_key] = None
    
    pbar.close()
    
    # Final summary
    print("\n" + "="*80)
    print("MEMO EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("MEMO RESULTS SUMMARY")
    print("="*80)
    
    for corruption_type in corruption_types:
        if (corruption_type in results_dict['memo'] and 
            severity_key in results_dict['memo'][corruption_type]):
            result = results_dict['memo'][corruption_type][severity_key]
            if result is not None:
                if isinstance(result, dict):
                    acc = result.get('accuracy', 0)
                    has_metrics = len(result) > 1
                    status = "✓" if has_metrics else "⚠"
                    print(f"{status} {corruption_type:<25s} {acc*100:>6.2f}% {'(full metrics)' if has_metrics else '(accuracy only)'}")
                else:
                    acc = result
                    print(f"⚠ {corruption_type:<25s} {acc*100:>6.2f}% (accuracy only)")
            else:
                print(f"✗ {corruption_type:<25s} {'FAILED':>6s}")
    
    print("="*80)


if __name__ == '__main__':
    main()
