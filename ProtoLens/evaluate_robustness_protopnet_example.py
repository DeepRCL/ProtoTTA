#!/usr/bin/env python3
"""
Robustness evaluation script for ProtoPNet TTA.

Evaluates ProtoPNet with various TTA methods across multiple corruption types
and severities. Results are saved iteratively to a JSON file for resumability.

Usage:
    python -m protopnet_tta.evaluate_robustness \
        --model ./saved_models/vgg19/sicapv2_001/epoch_10_last_0.pth \
        --data_dir ./datasets/SICAPv2_c/ \
        --output ./robustness_results.json
"""

import os
import sys
import argparse
import json
import time
import math
import torch
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ProtoPNet
from proto_baseline import ProtoPNetModel, ModelConfig, HeadConfig

# Import TTA methods
from . import tent
from . import proto_entropy
from . import proto_entropy_enhanced
from . import eata_adapt
from . import sar_adapt
from . import loss_adapt
from .settings import (
    img_size, test_dir, test_batch_size, num_classes, k, sum_cls,
    base_architecture, prototype_depth, prototype_activation_function,
    add_on_layers_type
)
from .preprocess import mean, std
from .noise_utils import get_all_corruption_types
from .prototype_metrics import PrototypeMetricsEvaluator
from .enhanced_prototype_metrics import EnhancedPrototypeMetrics
from .efficiency_metrics import EfficiencyTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)


# Default corruption types for histopathology
HISTOPATHOLOGY_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'gaussian_blur', 'defocus_blur', 'fog', 'frost',
    'jpeg_compression', 'pixelate', 'contrast', 'brightness',
    'elastic_transform'
]

# All available corruptions
ALL_CORRUPTIONS = get_all_corruption_types()


class OptimConfig:
    """Optimizer configuration for TTA."""
    LR = 0.001
    BETA = 0.9
    WD = 0.0
    STEPS = 1


cfg_optim = OptimConfig()


def setup_optimizer(params):
    """Set up Adam optimizer for TTA."""
    return optim.Adam(
        params,
        lr=cfg_optim.LR,
        betas=(cfg_optim.BETA, 0.999),
        weight_decay=cfg_optim.WD
    )


def setup_tent(model):
    """Set up Tent adaptation."""
    model = tent.configure_model(model, adaptation_mode='batchnorm_addon')
    params, _ = tent.collect_params(model, adaptation_mode='batchnorm_addon')
    
    if not params:
        logger.warning("No BatchNorm params for Tent. Returning eval model.")
        model.eval()
        return model
    
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg_optim.STEPS, episodic=False)


def setup_eata(model, test_loader, device):
    """Set up EATA adaptation."""
    model = eata_adapt.configure_model(model, adaptation_mode='batchnorm_addon')
    params, _ = eata_adapt.collect_params(model, adaptation_mode='batchnorm_addon')
    
    if not params:
        logger.warning("No params for EATA. Returning eval model.")
        model.eval()
        return model
    
    # Compute Fisher information on test samples (first 500)
    fishers = eata_adapt.compute_fishers(model, test_loader, device, num_samples=500)
    
    optimizer = setup_optimizer(params)
    
    # Use fixed e_margin formula from ProtoViT
    e_margin = math.log(1000)/2 - 1  # = 2.45, regardless of num_classes
    
    return eata_adapt.EATA(
        model, optimizer, fishers=fishers, fisher_alpha=2000.0,
        steps=cfg_optim.STEPS, episodic=False,
        e_margin=e_margin, d_margin=0.05, num_classes=5
    )


def setup_proto_entropy(model, geo_filter_threshold=0.993, adaptation_mode='all_adapt'):
    """Set up ProtoEntropy with configurable adaptation mode."""
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, _ = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)
    
    optimizer = setup_optimizer(params) if params else None
    
    return proto_entropy.ProtoEntropy(
        model, optimizer,
        steps=cfg_optim.STEPS,
        episodic=False,
        use_prototype_importance=True,
        use_confidence_weighting=True,
        confidence_threshold=0.7,
        use_geometric_filter=True,
        geo_filter_threshold=geo_filter_threshold
    )


def setup_sar(model):
    """Set up SAR adaptation."""
    model = sar_adapt.configure_model(model)
    params, _ = sar_adapt.collect_params(model)
    
    if not params:
        logger.warning("No params for SAR. Returning eval model.")
        model.eval()
        return model
    
    # SAR uses SAM optimizer with SGD base
    base_optimizer = torch.optim.SGD
    optimizer = sar_adapt.SAM(params, base_optimizer, lr=cfg_optim.LR, momentum=0.9)
    return sar_adapt.SAR(model, optimizer, steps=cfg_optim.STEPS, episodic=False)


def setup_memo(model):
    """Set up MEMO (Loss-based) adaptation."""
    model = loss_adapt.configure_model(model)
    params, _ = loss_adapt.collect_params(model)
    
    if not params:
        logger.warning("No params for MEMO. Returning eval model.")
        model.eval()
        return model
    
    optimizer = setup_optimizer(params)
    return loss_adapt.LossAdapt(model, optimizer, steps=cfg_optim.STEPS, episodic=False)


def setup_proto_hybrid(model, geo_filter_threshold=0.7, alpha_proto=0.7, alpha_softmax=0.3):
    """Set up Proto++Hybrid (ProtoEntropy + Softmax Entropy blend)."""
    return proto_entropy_enhanced.setup_proto_entropy_enhanced(
        model,
        lr=cfg_optim.LR,
        use_sam=False,
        alpha_proto=alpha_proto,
        alpha_softmax=alpha_softmax,
        use_entropy_filter=True,
        entropy_margin_scale=0.4,
        use_geometric_filter=True,
        geo_filter_threshold=geo_filter_threshold,
        adaptation_mode='batchnorm_addon',
        steps=cfg_optim.STEPS
    )


def load_model(model_path, device):
    """
    Robust model loading helper.
    Handles both full model saves and checkpoint dictionaries.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
        
    logger.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Case 1: Full model object
    if isinstance(checkpoint, torch.nn.Module):
        logger.info("Detected full model object.")
        return checkpoint.to(device)
        
    # Case 2: Checkpoint dictionary
    if isinstance(checkpoint, dict):
        logger.info("Detected checkpoint dictionary.")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Reconstruct architecture from settings and state dict
        arch = base_architecture
        if 'vgg19_bn' in model_path: arch = 'vgg19_bn'
        elif 'vgg19' in model_path: arch = 'vgg19'
        elif 'vgg16_bn' in model_path: arch = 'vgg16_bn'
        elif 'vgg16' in model_path: arch = 'vgg16'
        
        # Determine number of prototypes from state dict
        num_prototypes = 2000  # default
        for key in state_dict.keys():
            if 'prototype_vectors' in key:
                num_prototypes = state_dict[key].shape[0]
                break
        
        logger.info(f"Reconfiguring model: {arch}, {num_prototypes} prototypes, {num_classes} classes")
        
        blueprint = ModelConfig(
            base_architecture=arch,
            img_size=img_size,
            prototype_shape=(num_prototypes, prototype_depth, 1, 1),
            num_classes=num_classes,
            prototype_activation=prototype_activation_function,
            add_on_layers_type=add_on_layers_type,
            pretrained=False
        )
        
        # Head config - guess from keys
        head_type = 'linear'
        if any('kan' in k for k in state_dict.keys()): head_type = 'kan'
        elif any('mlp' in k for k in state_dict.keys()): head_type = 'mlp'
        
        head_config = HeadConfig(name=head_type)
        
        model = ProtoPNetModel(blueprint, head_config)
        
        # Load state dict (handle potential 'module.' prefix from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        return model.to(device)
        
    raise ValueError(f"Unknown checkpoint format at {model_path}")


def evaluate_model(model, loader, device, description="Inference", 
                   verbose=True, efficiency_tracker=None):
    """
    Run evaluation and return accuracy.
    
    Args:
        efficiency_tracker: Optional EfficiencyTracker to record per-batch timing
    
    Returns:
        accuracy: Classification accuracy
    """
    if verbose:
        print(f'\n{description}...')
    
    model.eval()
    n_examples = 0
    n_correct = 0
    
    iterator = tqdm(loader, desc=description) if verbose else loader
    
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        # Track this batch if efficiency tracking enabled
        if efficiency_tracker is not None:
            with efficiency_tracker.track_inference(batch_size):
                # Note: For TTA methods, we call forward WITH gradients enabled
                # because they need to adapt. For normal inference, we use no_grad.
                if hasattr(model, 'forward_and_adapt'):
                    outputs = model(images)
                else:
                    with torch.no_grad():
                        outputs = model(images)
        else:
            # No efficiency tracking
            if hasattr(model, 'forward_and_adapt'):
                outputs = model(images)
            else:
                with torch.no_grad():
                    outputs = model(images)
        
        # Get predictions
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        _, predicted = logits.max(1)
        n_correct += predicted.eq(labels).sum().item()
        n_examples += batch_size
    
    accuracy = n_correct / n_examples
    
    if verbose:
        print(f'Accuracy: {accuracy*100:.2f}%')
    
    return accuracy


def evaluate_single_combination(model_path, corruption_type, severity, 
                                data_dir, clean_data_dir, mode_name, mode_config, 
                                device, batch_size, proto_evaluator=None,
                                compute_proto_metrics=False, track_efficiency=False):
    """
    Evaluate a single combination of corruption type and TTA method.
    
    Returns:
        dict with 'accuracy' and other metrics, or None on failure
    """
    # Initialize efficiency tracker
    efficiency_tracker = EfficiencyTracker(mode_name, device=str(device)) if track_efficiency else None
    base_model = None
    eval_model = None
    
    try:
        # Load data
        if corruption_type == 'clean':
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(clean_data_dir, transform)
        else:
            corrupted_path = Path(data_dir) / corruption_type / str(severity)
            if not corrupted_path.exists():
                logger.warning(f"Corrupted dataset not found: {corrupted_path}")
                return None
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corrupted_path), transform)
        
        # TTA methods require shuffled batches to avoid class-imbalanced batches
        # (ImageFolder orders by class, causing BatchNorm statistics to collapse)
        # Using deterministic generator for reproducibility
        g = torch.Generator()
        g.manual_seed(0)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,  # Required for TTA methods to work properly
            num_workers=4,
            pin_memory=True,
            generator=g  # Deterministic shuffling across runs
        )
        
        # Load model
        base_model = load_model(model_path, device)
        base_model.eval()
        
        # Setup TTA method
        if mode_name == 'Normal':
            eval_model = base_model
        elif mode_name == 'Tent':
            eval_model = setup_tent(base_model)
        elif mode_name == 'EATA':
            eval_model = setup_eata(base_model, test_loader, device)
        elif mode_name == 'ProtoEntropy' or mode_name.startswith('ProtoEntropy-BN'):
            geo_threshold = mode_config.get('geo_filter_threshold', 0.993)
            adapt_mode = mode_config.get('adaptation_mode', 'all_adapt')
            eval_model = setup_proto_entropy(base_model, geo_threshold, adapt_mode)
        elif mode_name == 'ProtoHybrid':
            geo_threshold = mode_config.get('geo_filter_threshold', 0.8)
            alpha_proto = mode_config.get('alpha_proto', 0.7)
            alpha_softmax = mode_config.get('alpha_softmax', 0.3)
            eval_model = setup_proto_hybrid(base_model, geo_threshold, alpha_proto, alpha_softmax)
        elif mode_name == 'SAR':
            eval_model = setup_sar(base_model)
        elif mode_name == 'MEMO':
            eval_model = setup_memo(base_model)
        else:
            logger.warning(f"Unknown mode: {mode_name}. Using eval mode.")
            eval_model = base_model
        
        # Track adapted parameters if efficiency tracking is enabled
        if efficiency_tracker:
            if mode_name == 'Normal':
                # Normal inference has NO adapted parameters
                efficiency_tracker.count_adapted_parameters(eval_model, adapted_params=[])
            elif hasattr(eval_model, 'model'):
                # Wrapper model
                adapted_params = [p for p in eval_model.model.parameters() if p.requires_grad]
                efficiency_tracker.count_adapted_parameters(eval_model.model, adapted_params)
            else:
                # Direct model
                adapted_params = [p for p in eval_model.parameters() if p.requires_grad]
                efficiency_tracker.count_adapted_parameters(eval_model, adapted_params)
        
        # Evaluate with efficiency tracking
        accuracy = evaluate_model(
            eval_model, test_loader, device,
            description=f"{mode_name} on {corruption_type}-{severity}",
            verbose=False,
            efficiency_tracker=efficiency_tracker
        )
        
        # Record adaptation steps AFTER evaluation
        if efficiency_tracker and mode_name != 'Normal':
            num_batches = len(test_loader)
            efficiency_tracker.record_adaptation_step(num_batches * cfg_optim.STEPS)
        
        # Evaluate prototype metrics if requested
        result = {'accuracy': float(accuracy)}
        
        if compute_proto_metrics and proto_evaluator is not None:
            # Use enhanced evaluation if available
            if isinstance(proto_evaluator, EnhancedPrototypeMetrics):
                proto_metrics = proto_evaluator.evaluate_tta_method_enhanced(
                    eval_model, test_loader,
                    top_k=10, max_samples=None, verbose=False
                )
            else:
                proto_metrics = proto_evaluator.evaluate_tta_method(
                    eval_model, test_loader,
                    top_k=10, max_samples=None, verbose=False
                )
            
            # Add all metrics (includes both standard and enhanced)
            result.update({
                'PAC_mean': proto_metrics.get('PAC_mean'),
                'PAC_std': proto_metrics.get('PAC_std'),
                'PCA_mean': proto_metrics.get('PCA_mean'),
                'PCA_std': proto_metrics.get('PCA_std'),
                'sparsity_gini_mean': proto_metrics.get('sparsity_gini_mean'),
                'sparsity_active_mean': proto_metrics.get('sparsity_active_mean'),
                # Enhanced metrics
                'PCA_weighted_mean': proto_metrics.get('PCA_weighted_mean'),
                'PCA_weighted_std': proto_metrics.get('PCA_weighted_std'),
                'calibration_agreement': proto_metrics.get('calibration_agreement'),
                'calibration_logit_corr': proto_metrics.get('calibration_logit_corr'),
                'gt_class_contrib_improvement': proto_metrics.get('gt_class_contrib_improvement'),
                'gt_class_contrib_change_mean': proto_metrics.get('gt_class_contrib_change_mean'),
            })
        
        # Add efficiency metrics if tracking
        if efficiency_tracker:
            efficiency_metrics = efficiency_tracker.get_metrics()
            result['efficiency'] = efficiency_metrics
            
            # Also extract adaptation stats from the model if available
            if hasattr(eval_model, 'adaptation_stats'):
                result['adaptation_stats'] = eval_model.adaptation_stats.copy()
            
            # Extract geo_filter_stats for ProtoEntropy methods
            if hasattr(eval_model, 'get_geo_filter_stats'):
                geo_stats = eval_model.get_geo_filter_stats()
                # Convert to adaptation_stats format for consistency
                if geo_stats.get('total_samples', 0) > 0:
                    result['adaptation_stats'] = {
                        'total_samples': geo_stats['total_samples'],
                        'adapted_samples': geo_stats['total_samples'] - geo_stats['filtered_samples'],
                        'total_updates': geo_stats.get('total_updates', 0),
                        'filtered_samples': geo_stats['filtered_samples'],
                        'filter_rate': geo_stats.get('filter_rate', 0.0)
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to evaluate {mode_name} on {corruption_type}-{severity}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Cleanup
        try:
            if eval_model is not None:
                del eval_model
            if base_model is not None:
                del base_model
        except Exception:
            pass
        torch.cuda.empty_cache()


def load_existing_results(output_path):
    """Load existing results from JSON file."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            return data.get('results', {}), data.get('metadata', {})
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load existing results: {e}")
    return {}, {}


def save_results_json(output_path, results, metadata):
    """Save results to JSON file."""
    output = {
        'metadata': metadata,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ProtoPNet robustness with TTA methods',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='./saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
        help='Path to saved ProtoPNet model'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/ext/SICAPv2_c/',
        help='Path to corrupted dataset directory'
    )
    
    parser.add_argument(
        '--clean_data_dir',
        type=str,
        default='/home/mahdi/prototta/kan-head/datasets/SICAPv2_cropped/test_cropped',
        help='Path to clean test dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results.json',
        help='Path to save results JSON (default: results.json to append to existing)'
    )
    
    parser.add_argument(
        '--severity',
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help='Corruption severity to evaluate'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--corruptions',
        nargs='+',
        default=None,
        help='Specific corruption types to evaluate (default: histopathology-relevant)'
    )
    
    parser.add_argument(
        '--modes',
        nargs='+',
        default=None,
        help='TTA modes to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--gpuid',
        type=str,
        default='0',
        help='GPU ID'
    )
    
    parser.add_argument(
        '--prototype-metrics',
        action='store_true',
        default=False,
        help='Compute prototype-based metrics (PAC, PCA, Sparsity)'
    )
    
    parser.add_argument(
        '--proto-baseline-samples',
        type=int,
        default=1000,
        help='Number of clean samples for prototype baseline (default: 1000)'
    )
    
    parser.add_argument(
        '--track-efficiency',
        action='store_true',
        default=False,
        help='Track computational efficiency metrics'
    )
    
    parser.add_argument(
        '--geo_filter_threshold',
        type=float,
        default=0.993,
        help='ProtoEntropy geometric filtering threshold (for ProtoEntropy mode)'
    )
    
    parser.add_argument(
        '--adaptation_mode',
        type=str,
        default='all_adapt',
        choices=['batchnorm_only', 'batchnorm_addon', 'batchnorm_proto', 'all_adapt'],
        help='ProtoEntropy adaptation mode (for ProtoEntropy mode). '
             'Options: batchnorm_only, batchnorm_addon, batchnorm_proto, all_adapt. '
             'Default: all_adapt. Note: ProtoEntropy-BN uses batchnorm_addon at 0.995 threshold.'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define corruption types
    if args.corruptions:
        if 'all' in args.corruptions:
            corruption_types = ALL_CORRUPTIONS
        elif 'histopathology' in args.corruptions:
            corruption_types = HISTOPATHOLOGY_CORRUPTIONS
        else:
            corruption_types = args.corruptions
    else:
        corruption_types = HISTOPATHOLOGY_CORRUPTIONS
    
    # Just use the corruption types without adding 'clean'
    corruption_types = list(corruption_types)
    
    # Define TTA modes
    all_modes = {
        'Normal': {},
        'Tent': {},
        'EATA': {},
        'ProtoEntropy': {
            'geo_filter_threshold': args.geo_filter_threshold,
            'adaptation_mode': args.adaptation_mode
        },
        'ProtoEntropy-BN-0.98': {
            'geo_filter_threshold': 0.98,
            'adaptation_mode': 'batchnorm_addon'
        },
        'ProtoHybrid': {
            'geo_filter_threshold': 0.8,
            'alpha_proto': 0.7,
            'alpha_softmax': 0.3
        },
        'SAR': {},
        'MEMO': {},
    }
    
    if args.modes:
        modes = {k: v for k, v in all_modes.items() if k in args.modes}
    else:
        modes = all_modes
    
    severity = args.severity
    severity_key = str(severity)  # Use "5" not "severity_5" to match ProtoViT format
    batch_size = args.batch_size
    
    # Setup prototype metrics evaluator if requested
    proto_evaluator = None
    if args.prototype_metrics:
        print("\n" + "="*80)
        print("Setting up Prototype Metrics Evaluator")
        print("="*80)
        
        # Load base model
        base_model = load_model(args.model, device)
        base_model.eval()
        
        # Use EnhancedPrototypeMetrics for full metric set
        proto_evaluator = EnhancedPrototypeMetrics(base_model, device=device)
        print("Using EnhancedPrototypeMetrics (includes PCA-Weighted, Calibration, GT Contribution)")
        
        # Load clean data for baseline
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(args.clean_data_dir, transform_clean)
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=False
        )
        
        # Collect enhanced clean baseline (includes predictions and logits)
        proto_evaluator.collect_clean_baseline_enhanced(
            clean_loader, 
            max_samples=args.proto_baseline_samples, 
            verbose=True
        )
        
        print(f"✓ Prototype evaluator ready")
        print("="*80)
        
        del base_model
        torch.cuda.empty_cache()
    
    # Evaluate unadapted model on clean data
    print("\n" + "="*80)
    print("UNADAPTED MODEL PERFORMANCE ON CLEAN DATA")
    print("="*80)
    base_model = load_model(args.model, device)
    base_model.eval()
    
    transform_clean = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    clean_dataset = datasets.ImageFolder(args.clean_data_dir, transform_clean)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=False
    )
    
    clean_accuracy = evaluate_model(base_model, clean_loader, device, 
                                    description="Clean data evaluation", verbose=True)
    print(f"Clean accuracy (unadapted): {clean_accuracy*100:.2f}%")
    print("="*80 + "\n")
    
    del base_model
    torch.cuda.empty_cache()
    
    # Load existing results for resumability
    results_dict, _ = load_existing_results(args.output)
    
    # Initialize result structure
    for mode_name in modes:
        if mode_name not in results_dict:
            results_dict[mode_name] = {}
    
    # Evaluate "Normal" method on "clean" data and store as baseline
    print("="*80)
    print("EVALUATING NORMAL METHOD ON CLEAN DATA (Baseline)")
    print("="*80)
    if 'Normal' in modes:
        if 'clean' not in results_dict.get('Normal', {}):
            print("Evaluating Normal on clean data...")
            normal_clean_result = evaluate_single_combination(
                args.model,
                'clean',  # Special case for clean data
                severity,
                args.data_dir,
                args.clean_data_dir,
                'Normal',
                {},
                device,
                batch_size,
                proto_evaluator=proto_evaluator,
                compute_proto_metrics=args.prototype_metrics,
                track_efficiency=args.track_efficiency
            )
            
            if 'Normal' not in results_dict:
                results_dict['Normal'] = {}
            results_dict['Normal']['clean'] = normal_clean_result
            print(f"Normal on clean: {normal_clean_result.get('accuracy', 0)*100:.2f}%")
        else:
            print("Normal on clean already evaluated (skipping)")
    print("="*80 + "\n")
    
    # Print configuration
    print("="*80)
    print("PROTOPNET ROBUSTNESS EVALUATION - CORRUPTIONS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Clean data: {args.clean_data_dir}")
    print(f"Corrupted data: {args.data_dir}")
    print(f"Severity: {severity}")
    print(f"Corruption types: {len(corruption_types)}")
    print(f"TTA modes: {list(modes.keys())}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Calculate total combinations (all modes on all corruptions)
    total_combinations = len(corruption_types) * len(modes)
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc="Evaluating", unit="combo")
    
    # Evaluate all combinations (all methods on corruptions)
    for corruption_type in corruption_types:
        for mode_name, mode_config in modes.items():
            # Check if already computed
            if (corruption_type in results_dict.get(mode_name, {}) and
                severity_key in results_dict[mode_name].get(corruption_type, {})):
                pbar.update(1)
                existing = results_dict[mode_name][corruption_type][severity_key]
                acc = existing.get('accuracy', existing) if isinstance(existing, dict) else existing
                pbar.set_postfix({'acc': f'{acc*100:.2f}%', 'skip': '✓'})
                continue
            
            pbar.set_description(f"{mode_name[:15]:15s} | {corruption_type[:15]:15s}")
            
            # Run evaluation
            result = evaluate_single_combination(
                args.model,
                corruption_type,
                severity,
                args.data_dir,
                args.clean_data_dir,
                mode_name,
                mode_config,
                device,
                batch_size,
                proto_evaluator=proto_evaluator,
                compute_proto_metrics=args.prototype_metrics,
                track_efficiency=args.track_efficiency
            )
            
            # Store result
            if mode_name not in results_dict:
                results_dict[mode_name] = {}
            if corruption_type not in results_dict[mode_name]:
                results_dict[mode_name][corruption_type] = {}
            
            results_dict[mode_name][corruption_type][severity_key] = result
            
            # Save iteratively
            metadata = {
                'model_path': args.model,
                'data_dir': args.data_dir,
                'clean_data_dir': args.clean_data_dir,
                'severity': severity,
                'batch_size': batch_size,
                'corruption_types': corruption_types,
                'modes': list(modes.keys()),
                'mode_configs': modes,  # Save full mode configurations
                'clean_accuracy_unadapted': clean_accuracy,
                'prototype_metrics_enabled': args.prototype_metrics,
                'proto_baseline_samples': args.proto_baseline_samples if args.prototype_metrics else None,
                'efficiency_tracking_enabled': args.track_efficiency,
            }
            save_results_json(args.output, results_dict, metadata)
            
            # Update progress
            if result:
                pbar.set_postfix({'acc': f'{result["accuracy"]*100:.2f}%', 'saved': '✓'})
            else:
                pbar.set_postfix({'acc': 'N/A', 'saved': '✓'})
            
            pbar.update(1)
    
    pbar.close()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Corruption':<25s}"
    for mode_name in modes:
        header += f" {mode_name[:15]:>15s}"
    print(header)
    print("-" * (25 + 16 * len(modes)))
    
    # Show clean baseline first (Normal method only)
    if 'Normal' in results_dict and 'clean' in results_dict['Normal']:
        line = f"{'clean (baseline)':<25s}"
        for mode_name in modes:
            if mode_name == 'Normal':
                clean_result = results_dict['Normal']['clean']
                if clean_result is not None:
                    acc = clean_result.get('accuracy', 0) if isinstance(clean_result, dict) else clean_result
                    line += f" {acc*100:14.2f}%"
                else:
                    line += f" {'N/A':>15s}"
            else:
                line += f" {'-':>15s}"  # Not applicable for TTA methods
        print(line)
        print("-" * (25 + 16 * len(modes)))
    
    # Results for corruptions
    for corruption_type in corruption_types:
        line = f"{corruption_type:<25s}"
        for mode_name in modes:
            if (corruption_type in results_dict.get(mode_name, {}) and
                severity_key in results_dict[mode_name].get(corruption_type, {})):
                result = results_dict[mode_name][corruption_type][severity_key]
                if result is not None:
                    acc = result.get('accuracy', 0) if isinstance(result, dict) else result
                    line += f" {acc*100:14.2f}%"
                else:
                    line += f" {'N/A':>15s}"
            else:
                line += f" {'N/A':>15s}"
        print(line)
    
    print("="*80)
    
    # Print prototype metrics summary if enabled
    if args.prototype_metrics:
        print("\n" + "="*80)
        print("PROTOTYPE METRICS SUMMARY (Mean across corruptions)")
        print("="*80)
        
        for mode_name in modes.keys():
            pac_values = []
            pca_values = []
            sparsity_values = []
            
            for corruption_type in corruption_types:
                if (corruption_type in results_dict[mode_name] and 
                    severity_key in results_dict[mode_name][corruption_type]):
                    result = results_dict[mode_name][corruption_type][severity_key]
                    if result is not None and isinstance(result, dict):
                        if 'PAC_mean' in result:
                            pac_values.append(result['PAC_mean'])
                        if 'PCA_mean' in result:
                            pca_values.append(result['PCA_mean'])
                        if 'sparsity_gini_mean' in result:
                            sparsity_values.append(result['sparsity_gini_mean'])
            
            if pac_values or pca_values or sparsity_values:
                print(f"\n{mode_name}:")
                if pac_values:
                    print(f"  PAC (Consistency):     {np.mean(pac_values)*100:.2f}%")
                if pca_values:
                    print(f"  PCA (Alignment):       {np.mean(pca_values)*100:.2f}%")
                if sparsity_values:
                    print(f"  Sparsity (Gini):       {np.mean(sparsity_values):.3f}")
        
        print("="*80)


if __name__ == '__main__':
    main()
