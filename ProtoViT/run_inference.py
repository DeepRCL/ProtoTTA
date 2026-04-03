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
import loss_adapt
import fisher_proto
import eata_adapt
import memo_adapt
import sar_adapt
from pathlib import Path
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import local_analysis
from log import create_logger
import re
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
                        reset_mode=None, reset_frequency=10, 
                        confidence_threshold=0.7, ema_alpha=0.999,
                        use_geometric_filter=False, geo_filter_threshold=0.3,
                        consensus_strategy='max', consensus_ratio=0.5,
                        adaptation_mode='layernorm_only',
                        use_ensemble_entropy=False,
                        source_proto_stats=None, alpha_source_kl=0.0,
                        adapt_all_prototypes=False):
    """Set up Prototype Entropy adaptation (without threshold).
    
    Args:
        reset_mode: 'episodic', 'periodic', 'confidence', 'hybrid', 'ema', 'none'
                   If None, infers from cfg.MODEL.EPISODIC
        reset_frequency: How often to reset in 'periodic'/'hybrid' modes (in BATCHES, not samples)
        confidence_threshold: Confidence threshold for 'confidence'/'hybrid' modes
        ema_alpha: EMA decay for 'ema' mode
        use_geometric_filter: Use geometric similarity to filter unreliable samples
        geo_filter_threshold: Minimum similarity to ANY prototype to be considered reliable
        consensus_strategy: How to aggregate sub-prototypes ('max', 'mean', 'median', 'top_k_mean', 'weighted_mean')
        consensus_ratio: For 'top_k_mean', fraction of sub-prototypes to use
        adaptation_mode: What parameters to adapt ('layernorm_only', 'layernorm_proto', etc.)
        use_ensemble_entropy: Treat sub-prototypes as ensemble (compute entropy per sub-proto, then average)
        source_proto_stats: Pre-computed source prototype statistics
        alpha_source_kl: Weight for source KL regularization
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


def setup_proto_entropy_eata(model, entropy_threshold=0.4):
    """Set up Prototype Entropy adaptation with EATA-style thresholding."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropyEATA(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        entropy_threshold=entropy_threshold
    )
    return proto_model


def setup_loss_adapt(model):
    """Set up Loss-based adaptation."""
    model = loss_adapt.configure_model(model)
    params, param_names = loss_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    loss_model = loss_adapt.LossAdapt(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC
    )
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return loss_model


def setup_fisher_proto(model):
    """Set up Fisher-guided prototype adaptation."""
    model = fisher_proto.configure_model(model)
    params, param_names = fisher_proto.collect_params(model)
    optimizer = setup_optimizer(params)
    fisher_model = fisher_proto.FisherProto(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
    )
    # Ensure wrapper (and its buffers such as fisher_scores) are on the
    # same device as the underlying model parameters.
    device = next(model.parameters()).device
    fisher_model = fisher_model.to(device)
    # logger.info(f"model for adaptation (Fisher): %s", model)
    # logger.info(f"params for adaptation (Fisher): %s", param_names)
    # logger.info(f"optimizer for adaptation (Fisher): %s", optimizer)
    return fisher_model


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


def setup_sar(model):
    """Set up SAR adaptation."""
    model = sar_adapt.configure_model(model)
    params, param_names = sar_adapt.collect_params(model)
    # SAR uses SAM optimizer with SGD base
    base_optimizer = torch.optim.SGD
    optimizer = sar_adapt.SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    return sar_adapt.SAR(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_memo(model, lr=0.00025, batch_size=64, steps=1):
    """Set up MEMO adaptation.
    
    MEMO (Test Time Robustness via Adaptation and Augmentation) adapts
    the model to each test sample by minimizing the entropy of the marginal
    distribution over multiple augmented views.
    """
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


def extract_underlying_model(adapted_model):
    """Extract the underlying PPNet model from adaptation wrappers.
    
    Args:
        adapted_model: Can be a wrapped model (Tent, EATA, ProtoEntropy, etc.) or raw model
    
    Returns:
        The underlying PPNet model
    """
    # If it's a wrapped model, access the .model attribute
    if hasattr(adapted_model, 'model'):
        return adapted_model.model
    # Otherwise, assume it's already the raw model
    return adapted_model


def run_interpretability_analysis(model_or_wrapper, method_name, image_paths, 
                                  model_path, test_image_dir, output_dir, 
                                  corruption_str, log=None):
    """Run local analysis for interpretability comparison.
    
    Args:
        model_or_wrapper: The model (may be wrapped in adaptation method)
                         IMPORTANT: The model should be in its adapted state (after running on test set)
        method_name: Name of the method (e.g., 'Normal', 'EATA', 'ProtoEntropy-Imp+Conf')
        image_paths: List of image paths relative to test_image_dir
        model_path: Path to the saved model (to find prototype images)
        test_image_dir: Base directory for test images (should be clean test_dir)
        output_dir: Base output directory for analysis results
        corruption_str: String describing corruption (for subdirectory)
        log: Logger function (optional)
    """
    if log is None:
        log = print
    
    # Extract underlying model - this preserves the adapted parameters
    # The wrapper's .model attribute points to the same object, so adaptations are preserved
    ppnet = extract_underlying_model(model_or_wrapper)
    
    # For adaptation methods that use train mode (like Tent, ProtoEntropy), 
    # we need to keep them in train mode to preserve adapted parameters
    # But local_analysis will set eval mode, which is fine for most cases
    # However, for LayerNorm/BatchNorm adaptation, we might want to preserve train mode
    # For now, let's use eval mode for analysis (this is what local_analysis does anyway)
    ppnet.eval()  # Ensure eval mode for analysis
    
    # Determine prototype image directory from model path
    # Typically: saved_models/arch/exp/model.pth -> saved_models/arch/exp/img_name/
    model_dir = os.path.dirname(model_path)
    # Try common prototype image directory names
    possible_img_dirs = [
        os.path.join(model_dir, 'img'),
        os.path.join(model_dir, 'prototype_imgs'),
        os.path.join(model_dir, 'prototype-img'),
    ]
    load_img_dir = None
    for img_dir in possible_img_dirs:
        if os.path.exists(img_dir):
            load_img_dir = img_dir
            break
    
    if load_img_dir is None:
        log(f"WARNING: Could not find prototype image directory for {method_name}. Skipping interpretability analysis.")
        log(f"  Searched in: {possible_img_dirs}")
        return
    
    # Extract epoch number from model filename
    model_filename = os.path.basename(model_path)
    epoch_match = re.search(r'\d+', model_filename)
    start_epoch_number = int(epoch_match.group(0)) if epoch_match else 0
    
    # Create save directory for this method
    save_analysis_path = os.path.join(output_dir, 'interpretability', corruption_str, method_name)
    os.makedirs(save_analysis_path, exist_ok=True)
    
    # Create logger for this analysis
    log_file = os.path.join(save_analysis_path, f'{method_name}_analysis.log')
    analysis_log, logclose = create_logger(log_filename=log_file)
    
    log(f"\n>>> Running interpretability analysis for {method_name}")
    log(f"  Analysis output: {save_analysis_path}")
    log(f"  Prototype images: {load_img_dir}")
    
    # Run local analysis for each image
    for img_path in image_paths:
        try:
            analysis_log(f'\n{"="*60}')
            analysis_log(f'Analyzing image: {img_path}')
            analysis_log(f'Method: {method_name}')
            analysis_log(f'{"="*60}')
            
            local_analysis.local_analysis(
                imgs=img_path,
                ppnet=ppnet,
                save_analysis_path=save_analysis_path,
                test_image_dir=test_image_dir,
                start_epoch_number=start_epoch_number,
                load_img_dir=load_img_dir,
                log=analysis_log,
                prototype_layer_stride=1
            )
            log(f"  ✓ Analyzed: {img_path}")
        except Exception as e:
            log(f"  ✗ Error analyzing {img_path}: {e}")
            analysis_log(f"ERROR analyzing {img_path}: {e}")
            import traceback
            analysis_log(traceback.format_exc())
    
    logclose()
    log(f"  Analysis complete for {method_name}")


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
    Valid individual modes: normal, tent, proto, proto_importance, proto_confidence, proto_importance_confidence, proto_eata, loss, fisher, eata, sar, memo.
    """
    if not mode_arg:
        return {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}

    raw = [m.strip().lower() for m in mode_arg.split(",") if m.strip()]
    modes = set(raw)
    if "all" in modes:
        return {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}

    valid = {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}
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
                         use_pre_generated=True, use_clean_fisher=False, proto_threshold=None,
                         reset_mode=None, reset_frequency=10, confidence_threshold=0.7, ema_alpha=0.999,
                         use_geometric_filter=False, geo_filter_threshold=0.3, output_dir='./plots',
                         consensus_strategy='max', consensus_ratio=0.5,
                         adaptation_mode='layernorm_only',
                         use_ensemble_entropy=False,
                         use_source_stats=False, alpha_source_kl=0.0, num_source_samples=500,
                         interpretability_images=None, interpretability_num_images=0,
                         interpretability_mode='random',
                         use_prototype_metrics=False, proto_baseline_samples=None,
                         track_efficiency=False, use_enhanced_metrics=False):
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
    
    # --- Compute Source Prototype Statistics (if requested) ---
    source_proto_stats = None
    if use_source_stats and alpha_source_kl > 0:
        print(f'\n>>> Computing source prototype statistics on clean data...')
        print(f'Loading clean images from: {test_dir}')
        
        # Load a temporary model for computing source stats
        temp_model = torch.load(model_path, weights_only=False)
        temp_model = temp_model.to(device)
        temp_model.eval()
        
        # Create clean data loader
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(test_dir, transform_clean)
        
        # Use subset for efficiency
        if len(clean_dataset) > num_source_samples:
            source_indices = torch.randperm(len(clean_dataset))[:num_source_samples]
            source_subset = torch.utils.data.Subset(clean_dataset, source_indices)
        else:
            source_subset = clean_dataset
        
        source_loader = torch.utils.data.DataLoader(
            source_subset, batch_size=32, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        # Compute statistics
        source_proto_stats = proto_entropy.compute_source_proto_stats(
            temp_model, source_loader, device, num_samples=num_source_samples
        )
        
        print(f"Source prototype statistics computed successfully.")
        print("-" * 50)
        
        # Clean up
        del temp_model
        torch.cuda.empty_cache()

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
    run_tent = "tent" in selected_modes
    run_proto = "proto" in selected_modes
    run_proto_importance = "proto_importance" in selected_modes
    run_proto_confidence = "proto_confidence" in selected_modes
    run_proto_importance_confidence = "proto_importance_confidence" in selected_modes
    run_proto_eata = "proto_eata" in selected_modes
    run_loss = "loss" in selected_modes
    run_fisher = "fisher" in selected_modes
    run_eata = "eata" in selected_modes
    run_sar = "sar" in selected_modes
    run_memo = "memo" in selected_modes

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

    # --- TENT INFERENCE ---
    if run_tent:
        print(f'\n>>> Loading model for TENT inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk (IMPORTANT: start from same state)
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        # Setup Tent (this switches mode to train for norm layers)
        print("Setting up Tent adaptation...")
        tent_model = setup_tent(base_model)
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('Tent', device=str(device)) if track_efficiency else None
        if tracker:
            # Get adapted parameters from tent_model
            adapted_params = [p for p in tent_model.model.parameters() if p.requires_grad]
            tracker.count_adapted_parameters(tent_model.model, adapted_params)
            efficiency_trackers['Tent'] = tracker
        
        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                tent_model, test_loader, proto_metrics_evaluator,
                description="Tent Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['Tent'] = preds_dict
            detailed_proto_storage['Tent'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                tent_model, test_loader, proto_metrics_evaluator,
                description="Tent Adaptation Inference",
                track_per_batch=True,
                efficiency_tracker=tracker,
                adaptation_stats_storage=adaptation_stats_storage,
                method_name='Tent'
            )
            metrics_or_acc, batch_accs = result

        # Record adaptation steps AFTER evaluation (approx: 1 optimizer step per batch, repeated cfg.OPTIM.STEPS times)
        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['Tent'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['Tent'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['Tent'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if interpretability_enabled:
             models_for_interpretability['Tent'] = tent_model
        
        # Clean up
        # del tent_model
        # torch.cuda.empty_cache()

    # --- PROTO ENTROPY INFERENCE ---
    if run_proto:
        print(f'\n>>> Loading model for PROTO ENTROPY inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy (without threshold)
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        source_info = f", source_kl={alpha_source_kl}" if alpha_source_kl > 0 else ""
        print(f"Setting up ProtoEntropy (reset_mode={reset_mode}, freq={reset_frequency} batches{filter_info}{consensus_info}{adapt_info}{ensemble_info}{source_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=False,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy,
                                         source_proto_stats=source_proto_stats, alpha_source_kl=alpha_source_kl)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('ProtoEntropy', device=str(device)) if track_efficiency else None
        if tracker:
            base_model, adapted_params = get_adapted_params_from_wrapper(proto_model)
            tracker.count_adapted_parameters(base_model, adapted_params)
            efficiency_trackers['ProtoEntropy'] = tracker
        
        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['ProtoEntropy'] = preds_dict
            detailed_proto_storage['ProtoEntropy'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy Adaptation Inference",
                track_per_batch=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['ProtoEntropy'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['ProtoEntropy'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['ProtoEntropy'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy']['geo_stats'] = geo_stats

        if interpretability_enabled:
             models_for_interpretability['ProtoEntropy'] = proto_model

        # Clean up
        # del proto_model
        # torch.cuda.empty_cache()

    # --- PROTO ENTROPY with IMPORTANCE WEIGHTING ---
    if run_proto_importance:
        print(f'\n>>> Loading model for PROTO ENTROPY (Importance-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with importance weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        source_info = f", source_kl={alpha_source_kl}" if alpha_source_kl > 0 else ""
        print(f"Setting up ProtoEntropy+Importance (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info}{source_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=False,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy,
                                         source_proto_stats=source_proto_stats, alpha_source_kl=alpha_source_kl)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('ProtoEntropy-Importance', device=str(device)) if track_efficiency else None
        if tracker:
            base_model, adapted_params = get_adapted_params_from_wrapper(proto_model)
            tracker.count_adapted_parameters(base_model, adapted_params)
            efficiency_trackers['ProtoEntropy-Importance'] = tracker
        
        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Importance-Weighted) Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['ProtoEntropy-Importance'] = preds_dict
            detailed_proto_storage['ProtoEntropy-Importance'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Importance-Weighted) Inference",
                track_per_batch=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['ProtoEntropy-Importance'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['ProtoEntropy-Importance'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['ProtoEntropy-Importance'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Importance']['geo_stats'] = geo_stats

        if interpretability_enabled:
             models_for_interpretability['ProtoEntropy-Importance'] = proto_model

        # Clean up
        # del proto_model
        # torch.cuda.empty_cache()

    # --- PROTO ENTROPY with CONFIDENCE WEIGHTING ---
    if run_proto_confidence:
        print(f'\n>>> Loading model for PROTO ENTROPY (Confidence-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with confidence weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        source_info = f", source_kl={alpha_source_kl}" if alpha_source_kl > 0 else ""
        print(f"Setting up ProtoEntropy+Confidence (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info}{source_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=True,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy,
                                         source_proto_stats=source_proto_stats, alpha_source_kl=alpha_source_kl)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('ProtoEntropy-Confidence', device=str(device)) if track_efficiency else None
        if tracker:
            base_model, adapted_params = get_adapted_params_from_wrapper(proto_model)
            tracker.count_adapted_parameters(base_model, adapted_params)
            efficiency_trackers['ProtoEntropy-Confidence'] = tracker
        
        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Confidence-Weighted) Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['ProtoEntropy-Confidence'] = preds_dict
            detailed_proto_storage['ProtoEntropy-Confidence'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                proto_model, test_loader, proto_metrics_evaluator,
                description="ProtoEntropy (Confidence-Weighted) Inference",
                track_per_batch=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['ProtoEntropy-Confidence'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['ProtoEntropy-Confidence'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['ProtoEntropy-Confidence'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Confidence']['geo_stats'] = geo_stats

        if interpretability_enabled:
             models_for_interpretability['ProtoEntropy-Confidence'] = proto_model

        # Clean up
        # del proto_model
        # torch.cuda.empty_cache()

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
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        source_info = f", source_kl={alpha_source_kl}" if alpha_source_kl > 0 else ""
        print(f"Setting up ProtoEntropy Imp+Conf (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info}{source_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=True,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy,
                                         source_proto_stats=source_proto_stats, alpha_source_kl=alpha_source_kl)

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

    # --- PROTO ENTROPY+EATA INFERENCE ---
    if run_proto_eata:
        print(f'\n>>> Loading model for PROTO ENTROPY+EATA inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with EATA threshold
        print(f"Setting up ProtoEntropy+EATA adaptation (threshold={proto_threshold})...")
        proto_eata_model = setup_proto_entropy_eata(base_model, entropy_threshold=proto_threshold)

        if interpretability_enabled:
            acc, batch_accs, preds_dict, proto_details = evaluate_model(
                proto_eata_model, test_loader, 
                description="ProtoEntropy+EATA Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True
            )
            predictions_storage['ProtoEntropy+EATA'] = preds_dict
            detailed_proto_storage['ProtoEntropy+EATA'] = proto_details
        else:
            acc, batch_accs = evaluate_model(proto_eata_model, test_loader, 
                                            description="ProtoEntropy+EATA Adaptation Inference",
                                            track_per_batch=True)
            
        results['ProtoEntropy+EATA'] = acc
        results_with_batches['ProtoEntropy+EATA'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        if interpretability_enabled:
             models_for_interpretability['ProtoEntropy+EATA'] = proto_eata_model

        # Clean up
        # del proto_eata_model
        # torch.cuda.empty_cache()

    # --- LOSS ADAPT INFERENCE ---
    if run_loss:
        print(f'\n>>> Loading model for LOSS ADAPT inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        print("Setting up Loss Adapt adaptation...")
        loss_model = setup_loss_adapt(base_model)
        
        if interpretability_enabled:
            acc, batch_accs, preds_dict, proto_details = evaluate_model(
                loss_model, test_loader, 
                description="Loss Adapt Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True
            )
            predictions_storage['LossAdapt'] = preds_dict
            detailed_proto_storage['LossAdapt'] = proto_details
        else:
            acc, batch_accs = evaluate_model(loss_model, test_loader, 
                                            description="Loss Adapt Inference",
                                            track_per_batch=True)
            
        results['LossAdapt'] = acc
        results_with_batches['LossAdapt'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if interpretability_enabled:
             models_for_interpretability['LossAdapt'] = loss_model
        
        # del loss_model
        # torch.cuda.empty_cache()

    # --- FISHER-PROTO INFERENCE ---
    if run_fisher:
        print(f'\n>>> Loading model for FISHER PROTO inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup Fisher-guided prototype adaptation
        print("Setting up Fisher-guided Proto adaptation...")
        fisher_model = setup_fisher_proto(base_model)

        if interpretability_enabled:
            acc, batch_accs, preds_dict, proto_details = evaluate_model(
                fisher_model, test_loader, 
                description="FisherProto Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True
            )
            predictions_storage['FisherProto'] = preds_dict
            detailed_proto_storage['FisherProto'] = proto_details
        else:
            acc, batch_accs = evaluate_model(fisher_model, test_loader, 
                                            description="FisherProto Adaptation Inference",
                                            track_per_batch=True)
            
        results['FisherProto'] = acc
        results_with_batches['FisherProto'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        if interpretability_enabled:
             models_for_interpretability['FisherProto'] = fisher_model

        # Clean up
        # del fisher_model
        # torch.cuda.empty_cache()

    # --- EATA INFERENCE ---
    if run_eata:
        print(f'\n>>> Loading model for EATA inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        if use_clean_fisher:
            print("Computing Fisher Information Matrix on CLEAN data (using clean images)...")
            # Prepare clean data loader for Fisher computation (subset of 2000 samples)
            transform_clean = get_corrupted_transform(img_size, mean, std, None, 1)
            clean_dataset = datasets.ImageFolder(test_dir, transform_clean)
            
            num_fisher_samples = 2000
            if len(clean_dataset) > num_fisher_samples:
                fisher_indices = torch.randperm(len(clean_dataset))[:num_fisher_samples]
                fisher_subset = torch.utils.data.Subset(clean_dataset, fisher_indices)
            else:
                fisher_subset = clean_dataset
                
            fisher_loader = torch.utils.data.DataLoader(
                fisher_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
            )
            
            base_model = eata_adapt.configure_model(base_model)
            fishers = eata_adapt.compute_fishers(base_model, fisher_loader, device)
        else:
            print("Computing Fisher Information Matrix on TEST data (online estimation)...")
            # Use test loader (corrupted data) to estimate Fishers
            base_model = eata_adapt.configure_model(base_model)
            # Compute Fishers on a subset of test data (e.g., first 2000 samples)
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

    # --- SAR INFERENCE ---
    if run_sar:
        print(f'\n>>> Loading model for SAR inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup SAR
        print("Setting up SAR adaptation...")
        sar_model = setup_sar(base_model)
        
        # Initialize efficiency tracker
        tracker = EfficiencyTracker('SAR', device=str(device)) if track_efficiency else None
        if tracker:
            base_model_inner, adapted_params = get_adapted_params_from_wrapper(sar_model)
            tracker.count_adapted_parameters(base_model_inner, adapted_params)
            efficiency_trackers['SAR'] = tracker

        if interpretability_enabled:
            result = evaluate_with_optional_metrics(
                sar_model, test_loader, proto_metrics_evaluator,
                description="SAR Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True,
                efficiency_tracker=tracker
            )
            metrics_or_acc, batch_accs, preds_dict, proto_details = result
            predictions_storage['SAR'] = preds_dict
            detailed_proto_storage['SAR'] = proto_details
        else:
            result = evaluate_with_optional_metrics(
                sar_model, test_loader, proto_metrics_evaluator,
                description="SAR Adaptation Inference",
                track_per_batch=True,
                efficiency_tracker=tracker,
                adaptation_stats_storage=adaptation_stats_storage,
                method_name='SAR'
            )
            metrics_or_acc, batch_accs = result

        if tracker:
            tracker.record_adaptation_step(len(test_loader) * cfg.OPTIM.STEPS)
        
        # Handle both single accuracy and metrics dict
        if isinstance(metrics_or_acc, dict):
            results['SAR'] = metrics_or_acc
            acc = metrics_or_acc['accuracy']
        else:
            results['SAR'] = metrics_or_acc
            acc = metrics_or_acc
            
        results_with_batches['SAR'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        if interpretability_enabled:
             models_for_interpretability['SAR'] = sar_model

        # Clean up
        # del sar_model
        # torch.cuda.empty_cache()

    # --- MEMO INFERENCE ---
    if run_memo:
        print(f'\n>>> Loading model for MEMO inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        print("Setting up MEMO adaptation...")
        print("MEMO parameters: lr=0.00025, batch_size=32, steps=1")
        memo_model = setup_memo(base_model, lr=0.00025, batch_size=32, steps=1)
        
        # MEMO requires batch_size=1 (processes one image at a time)
        memo_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=False)

        if interpretability_enabled:
            acc, batch_accs, preds_dict, proto_details = evaluate_model(
                memo_model, memo_loader, 
                description="MEMO Adaptation Inference",
                track_per_batch=True,
                store_predictions=True,
                store_proto_details=True
            )
            predictions_storage['MEMO'] = preds_dict
            detailed_proto_storage['MEMO'] = proto_details
        else:
            acc, batch_accs = evaluate_model(memo_model, memo_loader, 
                                            description="MEMO Adaptation Inference",
                                            track_per_batch=True)
            
        results['MEMO'] = acc
        results_with_batches['MEMO'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        if interpretability_enabled:
             models_for_interpretability['MEMO'] = memo_model

        # Clean up
        # del memo_model
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
                'reset_mode': reset_mode,
                'reset_frequency': reset_frequency,
                'use_ensemble_entropy': use_ensemble_entropy,
                'use_source_stats': use_source_stats,
                'alpha_source_kl': alpha_source_kl if use_source_stats else 0.0,
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
        'Tent': 'Tent',
        'ProtoEntropy': 'ProtoEntropy',
        'ProtoEntropy-Importance': 'ProtoEntropy-Importance',
        'ProtoEntropy-Confidence': 'ProtoEntropy-Confidence',
        'ProtoEntropy-Imp+Conf': 'ProtoEntropy-Imp+Conf',
        'ProtoEntropy+EATA': 'ProtoEntropy+EATA',
        'LossAdapt': 'LossAdapt',
        'FisherProto': 'FisherProto',
        'EATA': 'EATA',
        'SAR': 'SAR',
        'MEMO': 'MEMO'
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
    parser.add_argument('-severity', type=int, default=3, help='Severity of corruption (1-5)')
    parser.add_argument('--no-corruption', action='store_true', 
                       help='Run inference without any corruption (clean test data)')
    parser.add_argument(
        '-mode',
        type=str,
        default='all',
        help=(
            'Inference mode(s): '
            'use a comma-separated list of any of [normal, tent, proto, proto_importance, proto_confidence, proto_importance_confidence, proto_eata, loss, fisher, eata, sar, memo], '
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
        '--use-clean-fisher',
        action='store_true',
        default=False,
        help='Use clean data to compute Fisher Information Matrix for EATA (requires access to clean data). Default is False (use test data).'
    )
    
    parser.add_argument(
        '--proto-threshold',
        type=float,
        default=0.4,
        help='Entropy threshold for ProtoEntropy+EATA adaptation (default: 0.4). Samples with higher entropy are ignored.'
    )
    
    parser.add_argument(
        '--reset-mode',
        type=str,
        default=None,
        choices=['episodic', 'periodic', 'confidence', 'hybrid', 'ema', 'none', None],
        help='Reset strategy for ProtoEntropy: episodic (every sample), periodic (every N samples), '
             'confidence (when conf drops), hybrid (periodic+confidence), ema (smooth updates), '
             'none (no resets). Default: inferred from --episodic flag.'
    )
    
    parser.add_argument(
        '--reset-frequency',
        type=int,
        default=10,
        help='How often to reset in periodic/hybrid modes in BATCHES (default: 10 batches). '
             'With batch_size=128, this means reset every 1280 samples.'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for confidence/hybrid reset modes (default: 0.7)'
    )
    
    parser.add_argument(
        '--ema-alpha',
        type=float,
        default=0.999,
        help='EMA decay factor for ema mode (default: 0.999, closer to 1 = slower adaptation)'
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
        choices=['max', 'mean', 'median', 'top_k_mean', 'weighted_mean'],
        help='Strategy for aggregating sub-prototypes: '
             'max (best sub-proto, default), mean (all must agree), median (robust), '
             'top_k_mean (soft consensus), weighted_mean (similarity-weighted)'
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
        choices=['layernorm_only', 'layernorm_proto', 'layernorm_proto_patch', 
                 'layernorm_proto_last', 'layernorm_attn_bias', 'layernorm_last_block', 
                 'full_proto', 'all_adaptive'],
        help='What parameters to adapt during TTA:\n'
             'layernorm_only (default) - Only LayerNorms (safest)\n'
             'layernorm_proto - LayerNorms + Prototype vectors\n'
             'layernorm_proto_patch - LayerNorms + Prototypes + Patch selection\n'
             'layernorm_proto_last - LayerNorms + Prototypes + Last layer\n'
             'layernorm_attn_bias - LayerNorms + Attention biases\n'
             'layernorm_last_block - LayerNorms + Last transformer blocks\n'
             'full_proto - Prototypes + Patch select + Last layer (no backbone)\n'
             'all_adaptive - Everything except backbone features'
    )
    
    parser.add_argument(
        '--use-ensemble-entropy',
        action='store_true',
        default=False,
        help='Use ensemble entropy: compute entropy for each sub-prototype independently, then average. '
             'Prevents single overconfident sub-prototype from dominating (requires sub-prototypes).'
    )
    
    parser.add_argument(
        '--use-source-stats',
        action='store_true',
        default=False,
        help='Compute prototype activation statistics on clean source data (like EATA Fisher). '
             'Requires access to clean data. Used for KL divergence regularization.'
    )
    
    parser.add_argument(
        '--alpha-source-kl',
        type=float,
        default=0.0,
        help='Weight for KL divergence regularization to source prototype distribution (default: 0.0 = disabled). '
             'Prevents test-time prototype activations from drifting too far from source distribution. '
             'Typical values: 0.01 to 1.0. Only active if --use-source-stats is set.'
    )
    
    parser.add_argument(
        '--num-source-samples',
        type=int,
        default=500,
        help='Number of clean source samples to use for computing prototype statistics (default: 500)'
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
        help='Enable prototype-based metrics (PAC, PCA, Sparsity) for TTA quality evaluation (default: enabled)'
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
        help='Use enhanced prototype metrics - PCA-Weighted, Calibration, GT Contribution (default: enabled)'
    )
    
    parser.add_argument(
        '--no-enhanced-metrics',
        dest='use_enhanced_metrics',
        action='store_false',
        help='Disable enhanced metrics (use this flag to turn off)'
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
                         use_pre_generated, args.use_clean_fisher, args.proto_threshold,
                         args.reset_mode, args.reset_frequency, args.confidence_threshold, args.ema_alpha,
                         args.use_geometric_filter, args.geo_filter_threshold, args.output_dir,
                         args.consensus_strategy, args.consensus_ratio, args.adaptation_mode,
                         args.use_ensemble_entropy, args.use_source_stats, args.alpha_source_kl, 
                         args.num_source_samples,
                         args.interpretability_images, args.interpretability_num_images,
                         args.interpretability_mode,
                         args.prototype_metrics, args.proto_baseline_samples,
                         args.track_efficiency, args.use_enhanced_metrics)
