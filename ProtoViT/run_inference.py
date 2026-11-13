import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import torch.optim as optim
import logging
import train_and_test as tnt
import model # Necessary for torch.load to find the class definition
import push_greedy # Importing just in case the model object depends on it
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import tent
import proto_entropy
import eata_adapt
from pathlib import Path
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import interpretability_viz
from prototype_tta_metrics import PrototypeMetricsEvaluator
from enhanced_prototype_metrics import EnhancedPrototypeMetrics
from efficiency_metrics import EfficiencyTracker, print_efficiency_comparison

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

class Cfg:
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

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def evaluate_with_optional_metrics(model, test_loader, proto_evaluator, description="Evaluation", 
                                   track_per_batch=False, store_predictions=False, 
                                   store_proto_details=False, efficiency_tracker=None,
                                   adaptation_stats_storage=None, method_name=None):
    """
    Wrapper to evaluate model with optional prototype metrics and efficiency tracking.
    
    Args:
        adaptation_stats_storage: Optional dict to store adaptation stats from the model
        method_name: Name of the method (for storing stats)
    
    Returns:
        If proto_evaluator is None: returns (acc, batch_accs) or (acc, batch_accs, preds, details)
        If proto_evaluator is provided: returns (metrics_dict, batch_accs) or (metrics_dict, batch_accs, preds, details)
            where metrics_dict contains: {'accuracy': acc, 'PAC_mean': ..., 'PCA_mean': ..., 'Sparsity_mean': ...}
    """
    # Start efficiency tracking if enabled
    if efficiency_tracker is not None:
        # Count total samples
        total_samples = len(test_loader.dataset)
        with efficiency_tracker.track_inference(total_samples):
            eval_results = _evaluate_core(model, test_loader, proto_evaluator, description,
                                          track_per_batch, store_predictions, store_proto_details)
    else:
        eval_results = _evaluate_core(model, test_loader, proto_evaluator, description,
                                      track_per_batch, store_predictions, store_proto_details)
    
    # Extract and store adaptation stats if available
    if adaptation_stats_storage is not None and method_name is not None:
        if hasattr(model, 'adaptation_stats'):
            adaptation_stats_storage[method_name] = model.adaptation_stats.copy()
        elif hasattr(model, 'model') and hasattr(model.model, 'adaptation_stats'):
            # For wrapped models
            adaptation_stats_storage[method_name] = model.model.adaptation_stats.copy()
    
    return eval_results


def _evaluate_core(model, test_loader, proto_evaluator, description, 
                   track_per_batch, store_predictions, store_proto_details):
    """Core evaluation logic without efficiency tracking wrapper."""
    if proto_evaluator is None:
        # Original behavior
        return evaluate_model(model, test_loader, description, track_per_batch, 
                            store_predictions, store_proto_details)
    else:
        # First get accuracy and other info
        eval_results = evaluate_model(model, test_loader, description, track_per_batch, 
                                     store_predictions, store_proto_details)
        
        # Extract accuracy and batch_accs
        if store_predictions or store_proto_details:
            if len(eval_results) == 4:
                acc, batch_accs, preds_dict, proto_details = eval_results
            else:
                acc, batch_accs = eval_results[:2]
                preds_dict = eval_results[2] if len(eval_results) > 2 else None
                proto_details = eval_results[3] if len(eval_results) > 3 else None
        else:
            acc, batch_accs = eval_results
            preds_dict = None
            proto_details = None
        
        # Compute prototype metrics (standard or enhanced)
        print(f"Computing prototype metrics for {description}...")
        
        # Check if using enhanced metrics
        if isinstance(proto_evaluator, EnhancedPrototypeMetrics):
            proto_metrics = proto_evaluator.evaluate_tta_method_enhanced(
                model, test_loader, top_k=10, max_samples=None, verbose=False
            )
        else:
            proto_metrics = proto_evaluator.evaluate_tta_method(
                model, test_loader, top_k=10, max_samples=None, verbose=False
            )
        
        # Merge accuracy with proto metrics
        metrics_dict = {
            'accuracy': acc,
            'PAC_mean': proto_metrics.get('PAC_mean', 0.0),
            'PCA_mean': proto_metrics.get('PCA_mean', 0.0),
            'Sparsity_mean': proto_metrics.get('sparsity_gini_mean', 0.0)
        }
        
        # Add enhanced metrics if available
        if 'PCA_weighted_mean' in proto_metrics:
            metrics_dict['PCA_weighted_mean'] = proto_metrics['PCA_weighted_mean']
        if 'calibration_agreement' in proto_metrics:
            metrics_dict['calibration_agreement'] = proto_metrics['calibration_agreement']
            metrics_dict['calibration_logit_corr'] = proto_metrics.get('calibration_logit_corr', 0.0)
        if 'gt_class_contrib_improvement' in proto_metrics:
            metrics_dict['gt_class_contrib_improvement'] = proto_metrics['gt_class_contrib_improvement']
            metrics_dict['gt_class_contrib_change_mean'] = proto_metrics.get('gt_class_contrib_change_mean', 0.0)
        
        # Return in the same format as input
        if store_predictions and store_proto_details:
            return metrics_dict, batch_accs, preds_dict, proto_details
        elif store_predictions:
            return metrics_dict, batch_accs, preds_dict
        elif store_proto_details:
            return metrics_dict, batch_accs, proto_details
        else:
            return metrics_dict, batch_accs


def setup_tent(model):
    """Set up tent adaptation.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def get_adapted_params_from_wrapper(model_wrapper):
    """Extract adapted parameters from a TTA wrapper model."""
    if hasattr(model_wrapper, 'model'):
        # Wrapper has a .model attribute
        base_model = model_wrapper.model
        adapted_params = [p for p in base_model.parameters() if p.requires_grad]
        return base_model, adapted_params
    else:
        # Direct model
        adapted_params = [p for p in model_wrapper.parameters() if p.requires_grad]
        return model_wrapper, adapted_params


def setup_proto_entropy(model, use_importance=False, use_confidence=False,
                        use_geometric_filter=False, geo_filter_threshold=0.3,
                        consensus_strategy='max', consensus_ratio=0.5,
                        adaptation_mode='layernorm_only',
                        adapt_all_prototypes=False,
                        logit_weight=0.0):
    """Set up Prototype Entropy adaptation (without threshold).
    
    Args:
        use_geometric_filter: Use geometric similarity to filter unreliable samples
        geo_filter_threshold: Minimum similarity to ANY prototype to be considered reliable
        consensus_strategy: How to aggregate sub-prototypes ('max', 'mean', 'top_k_mean')
        consensus_ratio: For 'top_k_mean', fraction of sub-prototypes to use
        adaptation_mode: What parameters to adapt ('layernorm_only' or 'layernorm_attn_bias')
        adapt_all_prototypes: If True, adapt all prototypes (not just target)
    """
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)
    
    print(f"Adapting {len(params)} parameter groups: {', '.join(param_names[:5])}...")
    
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        use_prototype_importance=use_importance,
        use_confidence_weighting=use_confidence,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        consensus_strategy=consensus_strategy,
        consensus_ratio=consensus_ratio,
        adapt_all_prototypes=adapt_all_prototypes,
        logit_weight=logit_weight,
    )
    return proto_model


def setup_eata(model, fishers):
    """Set up EATA adaptation."""
    model = eata_adapt.configure_model(model)
    params, param_names = eata_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    eata_model = eata_adapt.EATA(
        model,
        optimizer,
        fishers=fishers,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC
    )
    return eata_model


def create_performance_plots(results_dict, output_dir, corruption_info=""):
    """Create bar plots showing per-batch performance for each method.
    
    Args:
        results_dict: Dict mapping method_name -> {'accuracy': float, 'batch_accuracies': list}
        output_dir: Directory to save plots
        corruption_info: String describing corruption (for title)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    methods = []
    batch_accs_list = []
    final_accs = []
    
    for method_name, data in results_dict.items():
        if 'batch_accuracies' in data and data['batch_accuracies'] is not None:
            methods.append(method_name)
            batch_accs_list.append(data['batch_accuracies'])
            final_accs.append(data['accuracy'] * 100)
    
    if not methods:
        print("No per-batch data to plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar plot
    num_batches = max(len(batch_accs) for batch_accs in batch_accs_list)
    x = np.arange(num_batches)
    width = 0.8 / len(methods)
    
    # Plot bars for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, (method, batch_accs, final_acc, color) in enumerate(zip(methods, batch_accs_list, final_accs, colors)):
        # Pad with NaN if needed
        padded_accs = batch_accs + [np.nan] * (num_batches - len(batch_accs))
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, [acc * 100 if not np.isnan(acc) else 0 for acc in padded_accs], 
               width, label=f'{method} (Final: {final_acc:.2f}%)', 
               color=color, alpha=0.7)
    
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    title = f'Per-Batch Performance Comparison'
    if corruption_info:
        title += f'\n{corruption_info}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x[::max(1, num_batches//20)])  # Show every Nth batch label
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_batch_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-batch performance plot to: {plot_path}")
    
    # Also create a line plot for easier trend visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    for method, batch_accs, final_acc, color in zip(methods, batch_accs_list, final_accs, colors):
        batch_indices = np.arange(len(batch_accs))
        ax.plot(batch_indices, [acc * 100 for acc in batch_accs], 
               label=f'{method} (Final: {final_acc:.2f}%)', 
               color=color, linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    title = f'Per-Batch Performance Trends'
    if corruption_info:
        title += f'\n{corruption_info}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_batch_trends.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-batch trends plot to: {plot_path}")


def parse_modes(mode_arg: str):
    """
    Parse a comma-separated list of modes.

    If empty or contains 'all', all modes are enabled.
    Valid individual modes: normal, proto_importance_confidence, eata.
    """
    if not mode_arg:
        return {"normal", "proto_importance_confidence", "eata"}

    raw = [m.strip().lower() for m in mode_arg.split(",") if m.strip()]
    modes = set(raw)
    if "all" in modes:
        return {"normal", "proto_importance_confidence", "eata"}

    valid = {"normal", "proto_importance_confidence", "eata"}
    selected = modes & valid
    return selected or valid

def evaluate_model(model, loader, description="Inference", track_per_batch=False, store_predictions=False, store_proto_details=False):
    """Helper to run evaluation loop.
    
    Args:
        track_per_batch: If True, returns per-batch accuracies for visualization
        store_predictions: If True, stores all predictions and labels for later analysis
        store_proto_details: If True, stores top-k prototype details for all samples (heavy!)
    """
    print(f'\nStarting {description}...')
    class_specific = True 
    
    # Check if we can store details
    if store_proto_details:
        import interpretability_viz
    
    if track_per_batch or store_predictions or store_proto_details:
        # Custom evaluation with per-batch tracking
        model.eval()
        batch_accuracies = []
        n_examples = 0
        n_correct = 0
        
        # Storage for predictions (if requested)
        all_predictions = [] if store_predictions else None
        all_labels = [] if store_predictions else None
        all_indices = [] if store_predictions else None
        all_proto_details = [] if store_proto_details else None # List of lists of dicts
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                
                # Forward pass
                outputs = model(images)
                
                # Capture prototype details immediately after forward pass (while model state is fresh)
                if store_proto_details:
                    # Extract underlying PPNet
                    if hasattr(model, 'model'):
                        ppnet = model.model
                    else:
                        ppnet = model
                    
                    # IMPORTANT: Pass precomputed outputs to avoid calling model twice (which would cause double adaptation)
                    # outputs could be tuple (logits, min_distances, values) or just logits
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        precomputed = outputs  # Already in correct format
                    else:
                        precomputed = None  # Would need another forward pass, but this shouldn't happen for PPNet
                    
                    # Use batch version to preserve batch statistics (crucial for TTA methods)
                    # IMPORTANT: Save ALL prototypes (k=None) so we can properly sort by weight later
                    # We'll filter to top-k during visualization, but need all data for proper sorting
                    batch_details_list = interpretability_viz.get_top_k_prototypes_batch(
                        ppnet, images, k=None,  # k=None means save ALL prototypes
                        precomputed_outputs=precomputed, 
                        sort_by='activation'
                    )
                    
                    batch_details = []
                    for results, pred_cls in batch_details_list:
                        batch_details.append({'proto_results': results, 'pred_class': pred_cls})
                    
                    all_proto_details.extend(batch_details)
                
                # Get predictions - handle both tuple and single output
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is logits
                else:
                    logits = outputs
                
                _, predicted = logits.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_size = labels.size(0)
                
                if track_per_batch:
                    batch_acc = batch_correct / batch_size
                    batch_accuracies.append(batch_acc)
                
                if store_predictions:
                    all_predictions.append(predicted.cpu())
                    all_labels.append(labels.cpu())
                    # Store global indices
                    start_idx = batch_idx * loader.batch_size
                    batch_indices = torch.arange(start_idx, start_idx + batch_size)
                    all_indices.append(batch_indices)
                
                n_correct += batch_correct
                n_examples += batch_size
        
        accu = n_correct / n_examples
        print('-' * 20)
        print(f'{description} Complete.')
        print(f'Final Accuracy: {accu*100:.2f}%')
        print('-' * 20)
        
        ret_val = [accu]
        if track_per_batch:
            ret_val.append(batch_accuracies)
        
        if store_predictions:
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            all_indices = torch.cat(all_indices)
            predictions_dict = {
                'predictions': all_predictions,
                'labels': all_labels,
                'indices': all_indices
            }
            ret_val.append(predictions_dict)
            
        if store_proto_details:
            ret_val.append(all_proto_details)
            
        if len(ret_val) == 1:
            return ret_val[0]
        return tuple(ret_val)
    else:
        accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                        class_specific=class_specific, log=print, 
                                        clst_k=k, sum_cls=sum_cls)
        print('-' * 20)
        print(f'{description} Complete.')
        print(f'Final Accuracy: {accu*100:.2f}%')
        print('-' * 20)
        return accu


def run_unified_inference(model_path, gpu_id='0', corruption=None, severity=1, mode='all', 
                         use_pre_generated=True,
                         use_geometric_filter=False, geo_filter_threshold=0.3, output_dir='./plots',
                         consensus_strategy='max', consensus_ratio=0.5,
                         adaptation_mode='layernorm_only',
                         interpretability_images=None, interpretability_num_images=0,
                         interpretability_mode='random',
                         use_prototype_metrics=False, proto_baseline_samples=None,
                         track_efficiency=False, use_enhanced_metrics=False,
                         proto_lambda=1.0):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Prepare interpretability analysis
    corruption_str = f"{corruption}_sev{severity}" if corruption else "clean"
    interpretability_enabled = interpretability_images is not None or interpretability_num_images > 0
    
    # Determine which images to analyze for interpretability
    image_paths_to_analyze = []
    smart_samples = None  # Will store (path, true_class) tuples if using smart selection
    
    if interpretability_enabled:
        if interpretability_images:
            # Use provided image paths
            image_paths_to_analyze = interpretability_images if isinstance(interpretability_images, list) else [interpretability_images]
        else:
            # Will select images after model evaluation (smart or random)
            pass

    # Data loading
    using_pre_generated_images = False
    
    if corruption:
        # Check if pre-generated corrupted dataset exists
        corrupted_data_dir = Path('./datasets/cub200_c')
        corruption_path = corrupted_data_dir / corruption / str(severity)
        
        if use_pre_generated and corruption_path.exists():
            print(f'Using PRE-GENERATED corrupted images from: {corruption_path}')
            # Load pre-generated corrupted dataset (corruption already applied)
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corruption_path), transform)
            using_pre_generated_images = True
        else:
            if use_pre_generated:
                print(f'Pre-generated corrupted images not found at {corruption_path}')
            print(f'Generating corruption ON-THE-FLY: {corruption} (Severity: {severity})')
            print(f'Loading clean images from: {test_dir}')
            transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
            test_dataset = datasets.ImageFolder(test_dir, transform)
    else:
        print('Applying NO corruption (Clean Data)')
        print(f'Loading test data from: {test_dir}')
        transform = get_corrupted_transform(img_size, mean, std, None, severity)
        test_dataset = datasets.ImageFolder(test_dir, transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    print(f'Test set size: {len(test_loader.dataset)}')
    
    # We'll select samples AFTER running all methods if using smart selection
    # (need model predictions first)
    
    results = {}
    results_with_batches = {}  # Store per-batch data for plotting
    
    # Store efficiency trackers for each method
    efficiency_trackers = {} if track_efficiency else None
    adaptation_stats_storage = {} if track_efficiency else None  # Store adaptation stats from TTA methods
    
    # Store model wrappers for interpretability
    models_for_interpretability = {}
    
    # Store predictions for smart sample selection (without re-running inference)
    predictions_storage = {}
    
    # Store detailed prototype info for later visualization (avoids re-inference issues)
    detailed_proto_storage = {}
    
    # --- INITIALIZE PROTOTYPE METRICS EVALUATOR (if enabled) ---
    proto_metrics_evaluator = None
    if use_prototype_metrics or use_enhanced_metrics:
        print("\n" + "="*50)
        print("PROTOTYPE METRICS INITIALIZATION")
        print("="*50)
        print(f"Loading model for clean baseline collection...")
        
        # Load clean model
        clean_model = torch.load(model_path, weights_only=False)
        clean_model = clean_model.to(device)
        clean_model.eval()
        
        # Initialize evaluator (enhanced or standard)
        if use_enhanced_metrics:
            print("Using ENHANCED metrics evaluator (with PCA-Weighted, Calibration, etc.)")
            proto_metrics_evaluator = EnhancedPrototypeMetrics(clean_model, device=str(device))
        else:
            print("Using standard metrics evaluator")
            proto_metrics_evaluator = PrototypeMetricsEvaluator(clean_model, device=str(device))
        
        # Collect clean baseline on clean data
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(test_dir, transform_clean)
        
        # Use subset or full dataset based on proto_baseline_samples
        if proto_baseline_samples is not None and len(clean_dataset) > proto_baseline_samples:
            print(f"Collecting clean baseline (using {proto_baseline_samples} samples)...")
            baseline_indices = torch.randperm(len(clean_dataset))[:proto_baseline_samples]
            baseline_subset = torch.utils.data.Subset(clean_dataset, baseline_indices)
        else:
            print(f"Collecting clean baseline (using all {len(clean_dataset)} samples)...")
            baseline_subset = clean_dataset
        
        baseline_loader = torch.utils.data.DataLoader(
            baseline_subset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # Collect baseline (enhanced version if using enhanced metrics)
        if use_enhanced_metrics:
            proto_metrics_evaluator.collect_clean_baseline_enhanced(
                baseline_loader, 
                max_samples=proto_baseline_samples,
                verbose=True
            )
        else:
            proto_metrics_evaluator.collect_clean_baseline(
                baseline_loader, 
                max_samples=proto_baseline_samples,
                verbose=True
            )
        
        print("Clean baseline collected successfully.")
        print("-" * 50)
        
        # Clean up
        del clean_model
        torch.cuda.empty_cache()

    # Determine which modes to run (supports comma-separated list).
    selected_modes = parse_modes(mode)
    run_normal = "normal" in selected_modes
    run_proto_importance_confidence = "proto_importance_confidence" in selected_modes
    run_eata = "eata" in selected_modes

    # --- NORMAL INFERENCE ---
    if run_normal:
        print(f'\n>>> Loading model for NORMAL inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('Normal', device=str(device)) if track_efficiency else None
        if tracker:
            tracker.count_adapted_parameters(base_model, adapted_params=[])  # No adaptation
            efficiency_trackers['Normal'] = tracker

        # Store predictions if we need them for smart selection
        if interpretability_enabled:
            # For interpretability, we now ALWAYS capture proto details during the first pass
            # This ensures we visualize exactly what happened during inference
            result = evaluate_with_optional_metrics(
                base_model, test_loader, proto_metrics_evaluator,
                description="Normal Inference (No Adaptation)",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['Normal'] = preds_dict
            detailed_proto_storage['Normal'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                base_model, test_loader, proto_metrics_evaluator,
                description="Normal Inference (No Adaptation)",
                track_per_batch=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs = result
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['Normal'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['Normal'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['Normal'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        # Run interpretability analysis (Legacy local analysis, not the comprehensive one)
        # if interpretability_enabled:
        #    run_interpretability_analysis(...)
        
        # Clean up to free memory (unless we keep it for legacy local analysis, but we are using precomputed now)
        if interpretability_enabled:
             models_for_interpretability['Normal'] = base_model
        else:
             del base_model
             torch.cuda.empty_cache()

    # --- PROTO ENTROPY with IMPORTANCE+CONFIDENCE WEIGHTING ---
    if run_proto_importance_confidence:
        print(f'\n>>> Loading model for PROTO ENTROPY (Importance+Confidence-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with both importance and confidence weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        print(f"Setting up ProtoEntropy Imp+Conf ({filter_info.lstrip(', ')}{consensus_info}{adapt_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=True,
                                         use_geometric_filter=use_geometric_filter,
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         logit_weight=1.0 - proto_lambda)

        # Reset stats before evaluation
        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('ProtoEntropy-Imp+Conf', device=str(device)) if track_efficiency else None
        if tracker:
            base_model_inner, adapted_params = get_adapted_params_from_wrapper(proto_model)
            tracker.count_adapted_parameters(base_model_inner, adapted_params)
            efficiency_trackers['ProtoEntropy-Imp+Conf'] = tracker
        
        # Store predictions if we need them for smart selection
        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Importance+Confidence-Weighted) Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['ProtoEntropy-Imp+Conf'] = preds_dict
            detailed_proto_storage['ProtoEntropy-Imp+Conf'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Importance+Confidence-Weighted) Inference",
                track_per_batch=True,
                efficiency_tracker=tracker,
                adaptation_stats_storage=adaptation_stats_storage,
                method_name='ProtoEntropy-Imp+Conf'
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['ProtoEntropy-Imp+Conf'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['ProtoEntropy-Imp+Conf'] = metrics_or_acc
            acc = metrics_or_acc
        
        results_with_batches['ProtoEntropy-Imp+Conf'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        # Get geometric filter statistics
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Imp+Conf']['geo_stats'] = geo_stats
            print(f"\n--- Geometric Filter Statistics (ProtoEntropy-Imp+Conf) ---")
            print(f"Total samples processed: {geo_stats['total_samples']}")
            print(f"Samples filtered: {geo_stats['filtered_samples']} ({geo_stats['filter_rate']*100:.2f}%)")
            if geo_stats['overall_min_sim'] is not None:
                print(f"Minimum similarity across all batches: {geo_stats['overall_min_sim']:.4f}")
                print(f"Maximum similarity across all batches: {geo_stats['overall_max_sim']:.4f}")
                print(f"Average similarity across all batches: {geo_stats['overall_avg_sim']:.4f}")
                print(f"Threshold used: {geo_filter_threshold}")
            print("-" * 50)

        if interpretability_enabled:
            models_for_interpretability['ProtoEntropy-Imp+Conf'] = proto_model
        
        # Clean up
        # del proto_model
        # torch.cuda.empty_cache()

    # --- EATA INFERENCE ---
    if run_eata:
        print(f'\n>>> Loading model for EATA inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        print("Computing Fisher Information Matrix on TEST data (online estimation)...")
        # Use test loader (corrupted data) to estimate Fishers
        base_model = eata_adapt.configure_model(base_model)
        fishers = eata_adapt.compute_fishers(base_model, test_loader, device, num_samples=500)
        
        print("Fisher information computed.")

        # Setup EATA
        print("Setting up EATA adaptation...")
        eata_model = setup_eata(base_model, fishers)
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('EATA', device=str(device)) if track_efficiency else None
        if tracker:
            base_model_inner, adapted_params = get_adapted_params_from_wrapper(eata_model)
            tracker.count_adapted_parameters(base_model_inner, adapted_params)
            efficiency_trackers['EATA'] = tracker

        # Store predictions if we need them for smart selection
        if interpretability_enabled:
            # Always capture prediction and details
            result = evaluate_with_optional_metrics(
                eata_model, test_loader, proto_metrics_evaluator,
                description="EATA Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['EATA'] = preds_dict
            detailed_proto_storage['EATA'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                eata_model, test_loader, proto_metrics_evaluator,
                description="EATA Adaptation Inference",
                track_per_batch=True,
                efficiency_tracker=tracker,
                adaptation_stats_storage=adaptation_stats_storage,
                method_name='EATA'
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['EATA'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['EATA'] = metrics_or_acc
            acc = metrics_or_acc
        
        results_with_batches['EATA'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Store for interpretability
        if interpretability_enabled:
            models_for_interpretability['EATA'] = eata_model
        
        # Clean up
        # del eata_model
        # torch.cuda.empty_cache()

    # --- RUN COMPREHENSIVE INTERPRETABILITY ANALYSIS ---
    if interpretability_enabled and models_for_interpretability:
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE INTERPRETABILITY ANALYSIS")
        print(f"{'='*60}")
        
        # Smart sample selection if requested
        if not image_paths_to_analyze and interpretability_mode == 'proto_wins':
            print(f"\n>>> Using SMART selection: ProtoEntropy correct, Normal AND EATA both wrong")
            print(f">>> Using PRE-STORED predictions from evaluation phase (no re-inference)...")
            
            # Check if we have all required predictions
            if all(k in predictions_storage for k in ['Normal', 'EATA', 'ProtoEntropy-Imp+Conf']):
                smart_samples = interpretability_viz.select_smart_samples_from_predictions(
                    predictions_storage=predictions_storage,
                    test_dataset=test_loader.dataset,
                    num_samples=interpretability_num_images
                )
            else:
                print(f"⚠ Missing stored predictions. Available: {list(predictions_storage.keys())}")
                print(f"  Cannot perform smart selection. Falling back to random.")
                smart_samples = None
            
            if smart_samples and len(smart_samples) > 0:
                image_paths_to_analyze = smart_samples  # Already includes true classes
                print(f"\n✓ Selected {len(smart_samples)} smart samples from diverse classes")
                print(f"  (ProtoEntropy correct, but BOTH Normal AND EATA wrong)")
                for path, true_class in smart_samples:
                    print(f"  - {path} (true class: {true_class})")
            else:
                print(f"\n⚠ Smart selection didn't find enough samples, falling back to random...")
        
        # Random selection fallback
        if not image_paths_to_analyze:
            print(f"\n>>> Using RANDOM selection")
            transform_clean = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            clean_test_dataset = datasets.ImageFolder(test_dir, transform_clean)
            
            dataset_size = len(clean_test_dataset)
            num_to_select = min(interpretability_num_images, dataset_size)
            selected_indices = np.random.choice(dataset_size, size=num_to_select, replace=False)
            
            for idx in selected_indices:
                img_path, true_class = clean_test_dataset.samples[idx]
                rel_path = os.path.relpath(img_path, test_dir)
                image_paths_to_analyze.append((rel_path, true_class))
            
            print(f'✓ Selected {len(image_paths_to_analyze)} random images')
            for path, true_class in image_paths_to_analyze:
                print(f'  - {path} (true class: {true_class})')
        
        # Determine prototype image directory
        model_dir = os.path.dirname(model_path)
        possible_img_dirs = [
            os.path.join(model_dir, 'img'),
            os.path.join(model_dir, 'prototype_imgs'),
            os.path.join(model_dir, 'prototype-img'),
        ]
        prototype_img_dir = None
        for img_dir in possible_img_dirs:
            if os.path.exists(img_dir):
                prototype_img_dir = img_dir
                break
        
        if prototype_img_dir is None:
            print("WARNING: Could not find prototype image directory. Skipping interpretability.")
            print(f"  Searched in: {possible_img_dirs}")
        else:
            # Prepare experimental settings dict
            experimental_settings = {
                'model_path': model_path,
                'corruption': corruption if corruption else 'None',
                'severity': severity if corruption else 'N/A',
                'modes': list(models_for_interpretability.keys()),
                'geometric_filter': use_geometric_filter,
                'geo_filter_threshold': geo_filter_threshold if use_geometric_filter else 'N/A',
                'consensus_strategy': consensus_strategy,
                'adaptation_mode': adaptation_mode,
                'interpretability_mode': interpretability_mode,
            }
            
            # Prepare batch of precomputed results for the selected images
            # Need to map image paths to their index in the dataset to extract results
            # image_paths_to_analyze contains (rel_path, true_class) tuples
            
            # Create a map from rel_path to index
            # This requires iterating the dataset samples
            # Since test_loader is sequential, index corresponds to dataset index
            print("Preparing precomputed results for visualization...")
            path_to_idx = {}
            if hasattr(test_dataset, 'samples'):
                for idx, (path, _) in enumerate(test_dataset.samples):
                    rel = os.path.relpath(path, test_dataset.root)
                    path_to_idx[rel] = idx
            
            batch_precomputed_results = []
            for path, _ in image_paths_to_analyze:
                if path in path_to_idx:
                    idx = path_to_idx[path]
                    
                    # Create a dict mapping {method_name: results_dict} for this image
                    img_results = {}
                    for method_name, details_list in detailed_proto_storage.items():
                         if idx < len(details_list):
                             img_results[method_name] = details_list[idx]
                    
                    batch_precomputed_results.append(img_results)
                else:
                    print(f"⚠ Warning: Could not find index for {path}")
                    batch_precomputed_results.append(None)
            
            # Run batch interpretability
            output_dirs = interpretability_viz.run_batch_interpretability(
                models_dict=models_for_interpretability,
                image_paths=image_paths_to_analyze,
                test_dir=test_dir,
                prototype_img_dir=prototype_img_dir,
                output_base_dir=output_dir,
                corruption_name=corruption,
                severity=severity,
                experimental_settings=experimental_settings,
                use_pre_corrupted=using_pre_generated_images,
                data_root=test_dataset.root if hasattr(test_dataset, 'root') else test_dir,
                batch_precomputed_results=batch_precomputed_results
            )
            
            print(f"\n{'='*60}")
            print(f"Interpretability analysis complete!")
            print(f"Generated {len(output_dirs)} visualization sets")
            for out_dir in output_dirs:
                print(f"  - {out_dir}")
            print(f"{'='*60}\n")
        
        # Clean up models after interpretability
        for model_wrapper in models_for_interpretability.values():
            del model_wrapper
        torch.cuda.empty_cache()
    
    # --- CREATE PLOTS ---
    if results_with_batches:
        corruption_str = f"{corruption}_sev{severity}" if corruption else "clean"
        plot_subdir = os.path.join(output_dir, corruption_str)
        os.makedirs(plot_subdir, exist_ok=True)
        
        # Create per-batch performance plots
        create_performance_plots(results_with_batches, plot_subdir, 
                                corruption_info=f"Corruption: {corruption if corruption else 'None'}, Severity: {severity if corruption else 'N/A'}")
        
        # Print geometric filter statistics for all methods that used it
        print("\n" + "="*50)
        print("GEOMETRIC FILTER STATISTICS")
        print("="*50)
        for method_name, data in results_with_batches.items():
            if 'geo_stats' in data:
                geo_stats = data['geo_stats']
                print(f"\n{method_name}:")
                print(f"  Total samples: {geo_stats['total_samples']}")
                print(f"  Filtered: {geo_stats['filtered_samples']} ({geo_stats['filter_rate']*100:.2f}%)")
                if geo_stats['overall_min_sim'] is not None:
                    print(f"  Min similarity: {geo_stats['overall_min_sim']:.4f}")
                    print(f"  Max similarity: {geo_stats['overall_max_sim']:.4f}")
                    print(f"  Avg similarity: {geo_stats['overall_avg_sim']:.4f}")
                    print(f"  Threshold: {geo_filter_threshold}")
        print("="*50)
    
    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset Corruption: {corruption if corruption else 'None'}")
    if corruption:
        print(f"Severity: {severity}")
    print("-" * 50)
    
    # Helper to extract accuracy from results (handles both float and dict)
    def get_acc(method_name):
        if method_name in results:
            val = results[method_name]
            return val['accuracy'] if isinstance(val, dict) else val
        return None
    
    # Helper to get prototype metrics from results (handles both float and dict)
    def get_metrics(method_name):
        if method_name in results and isinstance(results[method_name], dict):
            return results[method_name]
        return None
    
    # Print results with metrics if available
    method_display_names = {
        'Normal': 'Normal',
        'ProtoEntropy-Imp+Conf': 'ProtoEntropy-Imp+Conf',
        'EATA': 'EATA',
    }
    
    for method_key, display_name in method_display_names.items():
        acc = get_acc(method_key)
        if acc is not None:
            metrics = get_metrics(method_key)
            if metrics and (use_prototype_metrics or use_enhanced_metrics):
                # Basic metrics
                output = f"{display_name:25} Acc: {acc*100:5.2f}%  |  "
                output += f"PAC: {metrics.get('PAC_mean', 0):.4f}  |  "
                output += f"PCA: {metrics.get('PCA_mean', 0):.4f}  |  "
                output += f"Sparsity: {metrics.get('Sparsity_mean', 0):.4f}"
                
                # Enhanced metrics (if available)
                if use_enhanced_metrics and 'PCA_weighted_mean' in metrics:
                    output += f"\n{'':<25}   PCA-Weighted: {metrics.get('PCA_weighted_mean', 0):.4f}"
                    
                    if 'calibration_agreement' in metrics:
                        output += f"  |  Calib: {metrics.get('calibration_agreement', 0)*100:.1f}%"
                    
                    if 'gt_class_contrib_improvement' in metrics:
                        output += f"  |  GT Δ: {metrics.get('gt_class_contrib_improvement', 0):+.2f}"
                
                print(output)
            else:
                print(f"{display_name:25} Accuracy: {acc*100:.2f}%")
    
    print("="*50)
    
    # --- EFFICIENCY COMPARISON ---
    if track_efficiency and efficiency_trackers:
        # adaptation_stats_storage was populated during evaluation
        print_efficiency_comparison(efficiency_trackers, baseline_method='Normal', 
                                   adaptation_stats=adaptation_stats_storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference for ProtoViT with multiple TTA methods')
    
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default='gaussian_noise', 
                       help='Type of corruption to apply (e.g., gaussian_noise). Use None or empty string for clean data (no corruption).')
    parser.add_argument('-severity', type=int, choices=[5], default=5, help='Severity of corruption (5 in the paper)')
    parser.add_argument('--no-corruption', action='store_true', 
                       help='Run inference without any corruption (clean test data)')
    parser.add_argument(
        '-mode',
        type=str,
        default='all',
        help=(
            'Inference mode(s): '
            'use a comma-separated list of any of [normal, proto_importance_confidence, eata], '
            'or "all" (default) to run every mode.'
        ),
    )  
    parser.add_argument(
        '--on-the-fly',
        action='store_true',
        default=False,
        help='Force on-the-fly corruption generation (ignore pre-generated images). By default, uses pre-generated images if available.'
    )
    
    parser.add_argument(
        '--use-geometric-filter',
        action='store_true',
        default=False,
        help='Enable geometric filtering: filter out samples with low similarity to ALL prototypes (noisy/unreliable samples)'
    )
    
    parser.add_argument(
        '--geo-filter-threshold',
        type=float,
        default=0.3,
        help='Geometric filter threshold: minimum max similarity to ANY prototype to be considered reliable (default: 0.3). '
             'Higher = more strict filtering, lower = accept more samples. Range: [-1, 1]'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Directory to save performance plots (default: ./plots)'
    )
    
    parser.add_argument(
        '--consensus-strategy',
        type=str,
        default='max',
        choices=['max', 'mean', 'top_k_mean'],
        help='Strategy for aggregating sub-prototypes: '
             'max (best sub-proto, default), mean (all must agree), '
             'top_k_mean (soft consensus)'
    )
    
    parser.add_argument(
        '--consensus-ratio',
        type=float,
        default=0.5,
        help='For top_k_mean consensus: fraction of sub-prototypes to use (default: 0.5 = top 50%%)'
    )
    
    parser.add_argument(
        '--adaptation-mode',
        type=str,
        default='layernorm_only',
        choices=['layernorm_only', 'layernorm_attn_bias'],
        help='What parameters to adapt during TTA:\n'
             'layernorm_only (default) - Only LayerNorms\n'
             'layernorm_attn_bias - LayerNorms + Attention biases'
    )
    
    parser.add_argument(
        '--interpretability-images',
        type=str,
        nargs='+',
        default=None,
        help='Image paths (relative to test_dir) for interpretability analysis. '
             'Example: "class1/img1.jpg" "class2/img2.jpg". '
             'If not provided and --interpretability-num-images > 0, random images will be selected.'
    )
    
    parser.add_argument(
        '--interpretability-num-images',
        type=int,
        default=0,
        help='Number of images to select for interpretability analysis (default: 0 = disabled). '
             'Ignored if --interpretability-images is provided.'
    )
    
    parser.add_argument(
        '--interpretability-mode',
        type=str,
        default='random',
        choices=['random', 'proto_wins'],
        help='How to select images for interpretability: '
             'random (default) - select random images, '
             'proto_wins - select images where ProtoEntropy is correct but EATA is wrong'
    )
    
    parser.add_argument(
        '--prototype-metrics',
        action='store_true',
        default=True,
        help='Enable the prototype metrics (PAC, PCA-W, prediction stability) (default: enabled)'
    )
    
    parser.add_argument(
        '--no-prototype-metrics',
        dest='prototype_metrics',
        action='store_false',
        help='Disable prototype metrics (use this flag to turn off)'
    )
    
    parser.add_argument(
        '--proto-baseline-samples',
        type=int,
        default=None,
        help='Number of clean samples to use for prototype metrics baseline (default: None = use all samples)'
    )
    
    parser.add_argument(
        '--track-efficiency',
        action='store_true',
        default=True,
        help='Track and report computational efficiency metrics (default: enabled)'
    )
    
    parser.add_argument(
        '--no-track-efficiency',
        dest='track_efficiency',
        action='store_false',
        help='Disable efficiency tracking (use this flag to turn off)'
    )
    
    parser.add_argument(
        '--use-enhanced-metrics',
        action='store_true',
        default=True,
        help='Use the enhanced prototype metrics (PCA-W, prediction stability) (default: enabled)'
    )
    
    parser.add_argument(
        '--no-enhanced-metrics',
        dest='use_enhanced_metrics',
        action='store_false',
        help='Disable enhanced metrics (use this flag to turn off)'
    )

    parser.add_argument(
        '--proto-lambda',
        type=float,
        default=1.0,
        help='Unified ProtoTTA interpolation: 1.0 = pure prototype entropy (ProtoTTA), '
             '0.0 = pure logit entropy (Tent-style), 0.7 = ProtoTTA+ default. '
             'Loss = λ*proto_loss + (1-λ)*logit_entropy. (default: 1.0)'
    )

    args = parser.parse_args()
    
    # Handle no-corruption flag
    if args.no_corruption:
        corruption = None
    elif args.corruption and args.corruption.lower() in ['none', 'null', '']:
        corruption = None
    else:
        corruption = args.corruption
    
    # Determine whether to use pre-generated images (default: True, unless --on-the-fly is set)
    use_pre_generated = not args.on_the_fly
    
    run_unified_inference(args.model, args.gpuid, corruption, args.severity, args.mode,
                         use_pre_generated,
                         args.use_geometric_filter, args.geo_filter_threshold, args.output_dir,
                         args.consensus_strategy, args.consensus_ratio, args.adaptation_mode,
                         args.interpretability_images, args.interpretability_num_images,
                         args.interpretability_mode,
                         args.prototype_metrics, args.proto_baseline_samples,
                         args.track_efficiency, args.use_enhanced_metrics,
                         proto_lambda=args.proto_lambda)
