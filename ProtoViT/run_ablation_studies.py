#!/usr/bin/env python3
"""
Comprehensive ablation study script for ProtoEntropy method.
Tests different variations across 5 experiments on a fixed subset of data.

Experiments:
1. Consensus Strategy: max, median, mean, top_k_mean
2. Adaptation Mode: layernorm_only, layernorm_attn_bias, all_adaptive
3. Target Prototypes: target_only (requires code mod for "all prototypes")
4. Geometric Filter: with/without filtering
5. Weighting: none, importance, confidence, both

Features:
- Creates fixed subset (500 samples per corruption, 13 corruptions)
- Caches best method results for reuse
- Resumable (saves after each evaluation)
- Each experiment in separate JSON file

Usage:
    # Run all experiments
    python run_ablation_studies.py --model ./saved_models/best_model.pth \
                                   --data_dir ./datasets/cub200_c/ \
                                   --output_dir ./ablation_studies/
    
    # Skip Experiment 3 (requires code modification)
    python run_ablation_studies.py --skip_exp3 ...
    
    # Skip running best method (use cached)
    python run_ablation_studies.py --skip_best ...
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
import shutil

# Import ProtoViT modules
import model  # Necessary for torch.load
import train_and_test as tnt
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import proto_entropy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

class Cfg:
    """Configuration for test-time adaptation methods."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.001
            self.BETA = 0.9
            self.WD = 0.000
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()

# Best method configuration (proto_imp_conf_v3)
BEST_CONFIG = {
    'use_geometric_filter': True,
    'geo_filter_threshold': 0.92,
    'consensus_strategy': 'top_k_mean',
    'adaptation_mode': 'layernorm_attn_bias',
    'use_ensemble_entropy': False,
    'reset_mode': None,
    'reset_frequency': 10,
    'confidence_threshold': 0.7,
    'ema_alpha': 0.999,
    'consensus_ratio': 0.5,
    'use_prototype_importance': True,
    'use_confidence_weighting': True,
}

# Corruption types (13 total)
CORRUPTION_TYPES = [
    'gaussian_noise', 'fog', 'gaussian_blur', 'elastic_transform', 
    'brightness', 'jpeg_compression', 'contrast', 'defocus_blur', 
    'frost', 'impulse_noise', 'pixelate', 'shot_noise', 
    'speckle_noise'
]

SEVERITY = 5
SAMPLES_PER_CORRUPTION = 500


def setup_optimizer(params):
    """Set up optimizer for adaptation."""
    if cfg.OPTIM.METHOD == 'Adam':
        return torch.optim.Adam(params, lr=cfg.OPTIM.LR,
                               betas=(cfg.OPTIM.BETA, 0.999),
                               weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError


def setup_proto_entropy(model, use_importance=False, use_confidence=False, 
                        reset_mode=None, reset_frequency=10, 
                        confidence_threshold=0.7, ema_alpha=0.999,
                        use_geometric_filter=False, geo_filter_threshold=0.3,
                        consensus_strategy='max', consensus_ratio=0.5,
                        adaptation_mode='layernorm_only',
                        use_ensemble_entropy=False,
                        source_proto_stats=None, alpha_source_kl=0.0,
                        adapt_all_prototypes=False):
    """Set up Prototype Entropy adaptation."""
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)
    
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        use_prototype_importance=use_importance,
        use_confidence_weighting=use_confidence,
        reset_mode=reset_mode,
        reset_frequency=reset_frequency,
        confidence_threshold=confidence_threshold,
        ema_alpha=ema_alpha,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        consensus_strategy=consensus_strategy,
        consensus_ratio=consensus_ratio,
        use_ensemble_entropy=use_ensemble_entropy,
        source_proto_stats=source_proto_stats,
        alpha_source_kl=alpha_source_kl,
        adapt_all_prototypes=adapt_all_prototypes
    )
    return proto_model


def evaluate_model(model, loader, description="Inference", verbose=True):
    """Run evaluation and return accuracy."""
    if verbose:
        print(f'\n{description}...')
    
    class_specific = True
    accu, test_loss_dict = tnt.test(
        model=model, 
        dataloader=loader,
        class_specific=class_specific,
        log=print if verbose else lambda x: None,
        clst_k=k,
        sum_cls=sum_cls
    )
    
    if verbose:
        print(f'Accuracy: {accu*100:.2f}%')
    
    return accu


def create_subset_indices(data_dir, corruption_types, severity, samples_per_corruption, output_file, use_all_samples=False):
    """Create and save subset indices for each corruption type.
    
    Args:
        use_all_samples: If True, use all available samples (ignores samples_per_corruption)
    
    Returns:
        dict: {corruption_type: list of indices} or None if use_all_samples=True
    """
    # If using all samples, return None (indicates to use full dataset)
    if use_all_samples:
        print("Using ALL samples (no subset indices needed)")
        return None
    
    # Check if existing file has all corruption types
    if os.path.exists(output_file):
        print(f"Loading existing subset indices from {output_file}")
        with open(output_file, 'r') as f:
            existing_indices = json.load(f)
        
        # Check if all corruption types are present
        missing_types = [c for c in corruption_types if c not in existing_indices]
        if not missing_types:
            print(f"All {len(corruption_types)} corruption types found in existing file")
            return existing_indices
        else:
            print(f"WARNING: {len(missing_types)} corruption types missing from existing file: {missing_types}")
            print(f"Regenerating subset indices...")
    
    print(f"Creating subset indices ({samples_per_corruption} samples per corruption)...")
    subset_indices = {}
    missing_corruptions = []
    
    for corruption_type in tqdm(corruption_types, desc="Creating subsets"):
        corruption_path = Path(data_dir) / corruption_type / str(severity)
        
        if not corruption_path.exists():
            logger.warning(f"Corruption path not found: {corruption_path}")
            missing_corruptions.append(corruption_type)
            continue
        
        try:
            # Load dataset to get all samples
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            dataset = datasets.ImageFolder(str(corruption_path), transform)
            
            total_samples = len(dataset)
            if total_samples == 0:
                logger.warning(f"No samples found for {corruption_type}")
                missing_corruptions.append(corruption_type)
                continue
            
            if total_samples < samples_per_corruption:
                logger.warning(f"Only {total_samples} samples available for {corruption_type}, using all")
                indices = list(range(total_samples))
            else:
                # Randomly select indices
                indices = np.random.choice(total_samples, size=samples_per_corruption, replace=False).tolist()
                indices.sort()  # Sort for reproducibility
            
            subset_indices[corruption_type] = indices
            
        except Exception as e:
            logger.error(f"Error loading {corruption_type}: {e}")
            missing_corruptions.append(corruption_type)
            continue
    
    # Save indices
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(subset_indices, f, indent=2)
    
    print(f"Saved subset indices to {output_file}")
    print(f"Successfully created indices for {len(subset_indices)}/{len(corruption_types)} corruption types")
    if missing_corruptions:
        print(f"WARNING: Missing corruption types: {missing_corruptions}")
        print(f"These will be skipped during evaluation.")
    return subset_indices


def load_subset_dataset(data_dir, corruption_type, severity, indices=None):
    """Load a subset dataset using pre-selected indices.
    
    Args:
        indices: List of indices to use, or None to use all samples
    """
    corruption_path = Path(data_dir) / corruption_type / str(severity)
    
    if not corruption_path.exists():
        raise FileNotFoundError(f"Corruption path not found: {corruption_path}")
    
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    full_dataset = datasets.ImageFolder(str(corruption_path), transform)
    
    # If indices is None, use all samples
    if indices is None:
        dataset = full_dataset
    else:
        dataset = torch.utils.data.Subset(full_dataset, indices)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    return loader


def load_results_json(output_file):
    """Load existing results from JSON file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load existing results from {output_file}: {e}")
            logger.warning("Starting fresh...")
    return None


def save_results_json(output_file, results_dict, metadata=None):
    """Save results to JSON file."""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'results': results_dict
    }
    
    # Create backup before writing
    if os.path.exists(output_file):
        backup_file = output_file + '.backup'
        try:
            shutil.copy2(output_file, backup_file)
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
    
    # Write to temporary file first, then rename (atomic write)
    temp_file = output_file + '.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        os.replace(temp_file, output_file)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def evaluate_single_config(model_path, corruption_type, severity, data_dir, 
                           subset_indices, device, config_name, config):
    """Evaluate a single configuration on a single corruption type."""
    try:
        # Determine which indices to use
        if subset_indices is None:
            # Use all samples
            indices = None
        elif corruption_type not in subset_indices:
            logger.error(f"Corruption type '{corruption_type}' not in subset_indices. Skipping.")
            return None
        else:
            indices = subset_indices[corruption_type]
        
        # Load dataset (subset or full)
        loader = load_subset_dataset(data_dir, corruption_type, severity, indices)
        
        # Load fresh model
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()
        
        # Setup ProtoEntropy with config
        eval_model = setup_proto_entropy(
            base_model,
            use_importance=config.get('use_prototype_importance', False),
            use_confidence=config.get('use_confidence_weighting', False),
            reset_mode=config.get('reset_mode', None),
            reset_frequency=config.get('reset_frequency', 10),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            ema_alpha=config.get('ema_alpha', 0.999),
            use_geometric_filter=config.get('use_geometric_filter', False),
            geo_filter_threshold=config.get('geo_filter_threshold', 0.3),
            consensus_strategy=config.get('consensus_strategy', 'max'),
            consensus_ratio=config.get('consensus_ratio', 0.5),
            adaptation_mode=config.get('adaptation_mode', 'layernorm_only'),
            use_ensemble_entropy=config.get('use_ensemble_entropy', False),
            source_proto_stats=config.get('source_proto_stats', None),
            alpha_source_kl=config.get('alpha_source_kl', 0.0),
            adapt_all_prototypes=config.get('adapt_all_prototypes', False)
        )
        
        # Reset geo filter stats if applicable
        if config.get('use_geometric_filter', False) and hasattr(eval_model, 'reset_geo_filter_stats'):
            eval_model.reset_geo_filter_stats()
        
        # Evaluate
        acc = evaluate_model(eval_model, loader, 
                           description=f"{config_name} on {corruption_type}",
                           verbose=False)
        
        # Cleanup
        del eval_model
        del base_model
        torch.cuda.empty_cache()
        
        return float(acc)
        
    except Exception as e:
        logger.error(f"Failed to evaluate {config_name} on {corruption_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_experiment(experiment_name, experiment_configs, model_path, data_dir, 
                  subset_indices, device, output_dir, best_results=None,
                  samples_per_corruption=500, use_all_samples=False):
    """Run a single ablation experiment."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    output_file = os.path.join(output_dir, f"{experiment_name}_results.json")
    
    # Load existing results
    existing_results = load_results_json(output_file)
    if existing_results:
        results_dict = existing_results.get('results', {})
        print(f"Loaded existing results from {output_file}")
    else:
        results_dict = {}
    
    # Initialize results structure
    for config_name in experiment_configs.keys():
        if config_name not in results_dict:
            results_dict[config_name] = {}
        for corruption_type in CORRUPTION_TYPES:
            if corruption_type not in results_dict[config_name]:
                results_dict[config_name][corruption_type] = None
    
    # If best_results provided and config matches best, use cached results
    if best_results:
        for config_name, config in experiment_configs.items():
            # Check if this config matches the best config (all relevant keys)
            is_best = all(
                config.get(k) == BEST_CONFIG.get(k) 
                for k in ['consensus_strategy', 'adaptation_mode', 'use_geometric_filter',
                         'use_prototype_importance', 'use_confidence_weighting',
                         'use_ensemble_entropy', 'geo_filter_threshold', 'consensus_ratio']
            )
            
            if is_best:
                print(f"\n{config_name} matches best config - using cached results")
                cached_count = 0
                for corruption_type in CORRUPTION_TYPES:
                    if corruption_type in best_results and best_results[corruption_type] is not None:
                        results_dict[config_name][corruption_type] = best_results[corruption_type]
                        cached_count += 1
                print(f"  Cached {cached_count}/{len(CORRUPTION_TYPES)} corruption types")
    
    # Create list of combinations to evaluate
    all_combinations = []
    for config_name, config in experiment_configs.items():
        for corruption_type in CORRUPTION_TYPES:
            # Skip if using subset_indices and corruption type not found
            if subset_indices is not None and corruption_type not in subset_indices:
                logger.warning(f"Skipping {corruption_type} (not in subset_indices)")
                continue
            
            # Check if already completed
            if (corruption_type in results_dict[config_name] and 
                results_dict[config_name][corruption_type] is not None):
                continue  # Skip already completed
            
            # Skip if using cached best results
            is_best = all(
                config.get(k) == BEST_CONFIG.get(k) 
                for k in ['consensus_strategy', 'adaptation_mode', 'use_geometric_filter',
                         'use_prototype_importance', 'use_confidence_weighting',
                         'use_ensemble_entropy', 'geo_filter_threshold', 'consensus_ratio']
            )
            if is_best and best_results and corruption_type in best_results and best_results[corruption_type] is not None:
                continue
            
            all_combinations.append((config_name, config, corruption_type))
    
    total_combinations = len(all_combinations)
    print(f"\nTotal combinations to evaluate: {total_combinations}")
    if total_combinations == 0:
        print("All combinations already completed!")
        return results_dict
    
    # Evaluate each combination
    start_time = time.time()
    pbar = tqdm(all_combinations, desc="Evaluating", unit="comb")
    
    for config_name, config, corruption_type in pbar:
        pbar.set_description(f"Evaluating {config_name} on {corruption_type}")
        
        acc = evaluate_single_config(
            model_path,
            corruption_type,
            SEVERITY,
            data_dir,
            subset_indices,
            device,
            config_name,
            config
        )
        
        # Update results
        results_dict[config_name][corruption_type] = acc
        
        # Save after each combination (iterative saving)
        try:
            metadata = {
                'experiment_name': experiment_name,
                'model_path': model_path,
                'data_dir': data_dir,
                'severity': SEVERITY,
                'samples_per_corruption': samples_per_corruption,
                'use_all_samples': use_all_samples,
                'configs': experiment_configs
            }
            save_results_json(output_file, results_dict, metadata)
            
            if acc is not None:
                pbar.set_postfix({'acc': f'{acc*100:.2f}%', 'saved': '✓'})
            else:
                pbar.set_postfix({'acc': 'FAILED', 'saved': '✓'})
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            pbar.set_postfix({'acc': 'ERROR', 'saved': '✗'})
    
    pbar.close()
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\nExperiment {experiment_name} completed in {elapsed_time:.1f} minutes")
    
    return results_dict


def run_best_method(model_path, data_dir, subset_indices, device, output_dir):
    """Run best method once and cache results."""
    print(f"\n{'='*80}")
    print("RUNNING BEST METHOD (for caching)")
    print(f"{'='*80}")
    
    output_file = os.path.join(output_dir, "best_method_results.json")
    
    # Check if already cached
    if os.path.exists(output_file):
        print(f"Loading cached best method results from {output_file}")
        with open(output_file, 'r') as f:
            data = json.load(f)
            return data.get('results', {})
    
    print("Running best method on all corruptions...")
    best_results = {}
    
    for corruption_type in tqdm(CORRUPTION_TYPES, desc="Best method"):
        # Determine indices to use
        if subset_indices is None:
            indices = None  # Use all samples
        elif corruption_type not in subset_indices:
            logger.warning(f"Skipping {corruption_type} (not in subset_indices)")
            best_results[corruption_type] = None
            continue
        else:
            indices = subset_indices[corruption_type]
        
        try:
            loader = load_subset_dataset(data_dir, corruption_type, SEVERITY, indices)
            
            base_model = torch.load(model_path, weights_only=False)
            base_model = base_model.to(device)
            base_model.eval()
            
            eval_model = setup_proto_entropy(
                base_model,
                use_importance=BEST_CONFIG['use_prototype_importance'],
                use_confidence=BEST_CONFIG['use_confidence_weighting'],
                reset_mode=BEST_CONFIG['reset_mode'],
                reset_frequency=BEST_CONFIG['reset_frequency'],
                confidence_threshold=BEST_CONFIG['confidence_threshold'],
                ema_alpha=BEST_CONFIG['ema_alpha'],
                use_geometric_filter=BEST_CONFIG['use_geometric_filter'],
                geo_filter_threshold=BEST_CONFIG['geo_filter_threshold'],
                consensus_strategy=BEST_CONFIG['consensus_strategy'],
                consensus_ratio=BEST_CONFIG['consensus_ratio'],
                adaptation_mode=BEST_CONFIG['adaptation_mode'],
                use_ensemble_entropy=BEST_CONFIG['use_ensemble_entropy'],
                adapt_all_prototypes=False  # Best config uses target only
            )
            
            if hasattr(eval_model, 'reset_geo_filter_stats'):
                eval_model.reset_geo_filter_stats()
            
            acc = evaluate_model(eval_model, loader, 
                               description=f"Best method on {corruption_type}",
                               verbose=False)
            
            best_results[corruption_type] = float(acc)
            
            del eval_model
            del base_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed on {corruption_type}: {e}")
            best_results[corruption_type] = None
    
    # Save cached results
    with open(output_file, 'w') as f:
        json.dump({'results': best_results, 'config': BEST_CONFIG}, f, indent=2)
    
    print(f"Saved best method results to {output_file}")
    return best_results


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation studies for ProtoEntropy method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, 
                       default='./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth',
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str,
                       default='./datasets/cub200_c/',
                       help='Path to CUB-C dataset')
    parser.add_argument('--output_dir', type=str, default='./ablation_studies',
                       help='Output directory for ablation results')
    parser.add_argument('--gpuid', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--samples_per_corruption', type=int, default=500,
                       help='Number of samples per corruption type (default: 500). Ignored if --use_all_samples is set.')
    parser.add_argument('--use_all_samples', action='store_true',
                       help='Use all available samples instead of subset (ignores --samples_per_corruption)')
    parser.add_argument('--skip_best', action='store_true', default=True,
                       help='Skip running best method (use cached if available)')
    parser.add_argument('--skip_exp3', action='store_true',
                       help='Skip Experiment 3 (requires code modification)')
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subset indices (or None if using all samples)
    subset_indices_file = os.path.join(args.output_dir, 'subset_indices.json')
    subset_indices = create_subset_indices(
        args.data_dir, CORRUPTION_TYPES, SEVERITY, 
        args.samples_per_corruption, subset_indices_file,
        use_all_samples=args.use_all_samples
    )
    
    if args.use_all_samples:
        print("NOTE: Using ALL available samples (no subset)")
        actual_samples_per_corruption = "all"
    else:
        actual_samples_per_corruption = args.samples_per_corruption
    
    # Run best method first (for caching)
    best_results = None
    if not args.skip_best:
        best_results = run_best_method(
            args.model, args.data_dir, subset_indices, device, args.output_dir
        )
    else:
        best_file = os.path.join(args.output_dir, "best_method_results.json")
        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                data = json.load(f)
                best_results = data.get('results', {})
            print(f"Loaded cached best method results")
    
    # Define experiments
    experiments = {}
    
    # Experiment 1: Consensus Strategy
    experiments['exp1_consensus'] = {
        'max': {
            **BEST_CONFIG,
            'consensus_strategy': 'max',
        },
        'median': {
            **BEST_CONFIG,
            'consensus_strategy': 'median',
        },
        'mean': {
            **BEST_CONFIG,
            'consensus_strategy': 'mean',
        },
        'top_k_mean': {
            **BEST_CONFIG,
            'consensus_strategy': 'top_k_mean',  # Current best
        },
    }
    
    # Experiment 2: Adaptation Mode
    experiments['exp2_adaptation'] = {
        'layernorm_only': {
            **BEST_CONFIG,
            'adaptation_mode': 'layernorm_only',
        },
        'layernorm_attn_bias': {
            **BEST_CONFIG,
            'adaptation_mode': 'layernorm_attn_bias',  # Current best
        },
        'all_adaptive': {
            **BEST_CONFIG,
            'adaptation_mode': 'all_adaptive',
        },
    }
    
    # Experiment 3: Target Prototypes Only vs All Prototypes
    if not args.skip_exp3:
        experiments['exp3_target_prototypes'] = {
            'target_only': {
                **BEST_CONFIG,
                'adapt_all_prototypes': False,  # Current: only target prototypes
            },
            'all_prototypes': {
                **BEST_CONFIG,
                'adapt_all_prototypes': True,  # Alternative: all prototypes
            },
        }
    
    # Experiment 4: Geometric Filter
    experiments['exp4_geometric_filter'] = {
        'no_filter': {
            **BEST_CONFIG,
            'use_geometric_filter': False,
        },
        'with_filter': {
            **BEST_CONFIG,
            'use_geometric_filter': True,  # Current best
        },
    }
    
    # Experiment 5: Weighting Strategies
    experiments['exp5_weighting'] = {
        'no_weighting': {
            **BEST_CONFIG,
            'use_prototype_importance': False,
            'use_confidence_weighting': False,
        },
        'importance_only': {
            **BEST_CONFIG,
            'use_prototype_importance': True,
            'use_confidence_weighting': False,
        },
        'confidence_only': {
            **BEST_CONFIG,
            'use_prototype_importance': False,
            'use_confidence_weighting': True,
        },
        'both': {
            **BEST_CONFIG,
            'use_prototype_importance': True,  # Current best
            'use_confidence_weighting': True,  # Current best
        },
    }
    
    # Run all experiments
    all_results = {}
    for exp_name, exp_configs in experiments.items():
        results = run_experiment(
            exp_name, exp_configs, args.model, args.data_dir,
            subset_indices, device, args.output_dir, best_results,
            samples_per_corruption=actual_samples_per_corruption if isinstance(actual_samples_per_corruption, int) else args.samples_per_corruption,
            use_all_samples=args.use_all_samples
        )
        all_results[exp_name] = results
    
    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDIES COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nSummary of experiments:")
    for exp_name in experiments.keys():
        print(f"  - {exp_name}")
    print(f"{'='*80}")
    
    # Create README with instructions
    readme_path = os.path.join(args.output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("""# Ablation Studies Results

This directory contains results from ablation studies on ProtoEntropy method.

## Experiments

1. **exp1_consensus**: Impact of aggregation method (max, median, mean, top_k_mean)
2. **exp2_adaptation**: Impact of adaptation mode (layernorm_only, layernorm_attn_bias, all_adaptive)
3. **exp3_target_prototypes**: Target prototypes only vs all prototypes (requires code modification)
4. **exp4_geometric_filter**: With/without geometric filtering
5. **exp5_weighting**: Weighting strategies (none, importance, confidence, both)

## Experiment 3

Tests adapting target prototypes only vs all prototypes. The `adapt_all_prototypes` parameter
has been implemented in `proto_entropy.py` to support this experiment.

## Data

- Subset indices: `subset_indices.json` (500 samples per corruption, 13 corruptions)
- Best method results: `best_method_results.json` (cached for reuse)

## Results Format

Each experiment has a JSON file with structure:
```json
{
  "results": {
    "config_name": {
      "corruption_type": accuracy_float
    }
  }
}
```
""")
    print(f"\nCreated README at: {readme_path}")


if __name__ == '__main__':
    main()

