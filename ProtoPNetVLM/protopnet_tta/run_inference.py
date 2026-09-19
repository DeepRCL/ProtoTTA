#!/usr/bin/env python3
"""
Run inference with various TTA methods on ProtoPNet.

This script evaluates ProtoPNet models with and without test-time adaptation
on clean and corrupted versions of the SICAPv2 dataset.

Usage:
    # Normal inference on clean data
    python -m protopnet_tta.run_inference --model ./saved_models/vgg19/sicapv2_001/epoch_10_last_0.pth

    # Inference with corruption
    python -m protopnet_tta.run_inference --model ./saved_models/vgg19/sicapv2_001/epoch_10_last_0.pth \
                                           --corruption gaussian_noise --severity 3

    # Compare different TTA methods
    python -m protopnet_tta.run_inference --model ./saved_models/vgg19/sicapv2_001/epoch_10_last_0.pth \
                                           --corruption gaussian_noise --severity 3 \
                                           --mode normal,tent,proto_importance_confidence
"""

import os
import sys
import gc
import argparse
import torch
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ProtoPNet model
from proto_baseline import ProtoPNetModel, ModelConfig, HeadConfig

# Import TTA methods
from . import tent
from . import proto_entropy
from . import proto_entropy_enhanced
from . import eata_adapt
from . import sar_adapt
from . import train_and_test as tnt
from .settings import (
    img_size, test_dir, test_batch_size, num_classes, k, sum_cls,
    base_architecture, prototype_depth, prototype_activation_function,
    add_on_layers_type
)
from .preprocess import mean, std
from .noise_utils import get_corrupted_transform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

# Enable deterministic algorithms for reproducibility
# This ensures CUDA operations are deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For PyTorch 1.8+: Enable deterministic algorithms
# This may impact performance but ensures reproducibility
try:
    torch.use_deterministic_algorithms(True)
except AttributeError:
    # Older PyTorch versions don't have this
    pass

# For deterministic DataLoader workers
def seed_worker(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TTAOptimConfig:
    """Optimizer configuration for TTA methods."""
    def __init__(self):
        self.METHOD = 'Adam'
        # Match ProtoViT's LR for testing
        self.LR = 0.001
        self.BETA = 0.9
        self.WD = 0.0
        self.STEPS = 1


class TTAModelConfig:
    """Model configuration for TTA methods."""
    def __init__(self):
        self.EPISODIC = False


cfg_optim = TTAOptimConfig()
cfg_model = TTAModelConfig()


def setup_optimizer(params):
    """Set up optimizer for TTA adaptation."""
    if cfg_optim.METHOD == 'Adam':
        return optim.Adam(
            params,
            lr=cfg_optim.LR,
            betas=(cfg_optim.BETA, 0.999),
            weight_decay=cfg_optim.WD
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg_optim.METHOD} not implemented")


def setup_tent(model, adaptation_mode='batchnorm_addon'):
    """Set up Tent adaptation.

    Args:
        model: Model to adapt
        adaptation_mode: 'batchnorm_addon' (default, recommended), 'batchnorm_only', 'addon_only', 'full'
    """
    model = tent.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = tent.collect_params(model, adaptation_mode=adaptation_mode)

    if not params:
        logger.warning(f"No parameters found for Tent with mode {adaptation_mode}.")
        return model

    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(
        model,
        optimizer,
        steps=cfg_optim.STEPS,
        episodic=cfg_model.EPISODIC
    )
    logger.info(f"Tent adaptation ({adaptation_mode}): {len(params)} parameters")
    return tent_model


def setup_proto_entropy(model, use_importance=False, use_confidence=False,
                        use_geometric_filter=True, geo_filter_threshold=0.3,
                        adaptation_mode='all_adapt',
                        reset_mode=None, reset_frequency=10,
                        confidence_threshold=0.7):
    """Set up ProtoEntropy adaptation."""
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)

    if not params:
        logger.warning(f"No parameters found for ProtoEntropy with mode {adaptation_mode}.")

    optimizer = setup_optimizer(params) if params else None

    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg_optim.STEPS,
        episodic=cfg_model.EPISODIC,
        use_prototype_importance=use_importance,
        use_confidence_weighting=use_confidence,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        reset_mode=reset_mode,
        reset_frequency=reset_frequency,
        confidence_threshold=confidence_threshold
    )

    mode_str = []
    if use_importance:
        mode_str.append("importance")
    if use_confidence:
        mode_str.append("confidence")
    if use_geometric_filter:
        mode_str.append(f"geo@{geo_filter_threshold}")
    logger.info(f"ProtoEntropy ({'+'.join(mode_str) if mode_str else 'basic'}, {adaptation_mode}, {len(params)} params)")

    return proto_model


def setup_sar(model):
    """Set up SAR adaptation."""
    model = sar_adapt.configure_model(model)
    params, param_names = sar_adapt.collect_params(model)

    if not params:
        logger.warning("No parameters found for SAR.")
        return model

    base_optimizer = torch.optim.SGD
    optimizer = sar_adapt.SAM(params, base_optimizer, lr=cfg_optim.LR, momentum=0.9)

    sar_model = sar_adapt.SAR(model, optimizer, steps=cfg_optim.STEPS, episodic=cfg_model.EPISODIC)
    logger.info(f"SAR adaptation: {len(params)} parameters")
    return sar_model


def setup_proto_eata(model, entropy_threshold=0.4, adaptation_mode='all_adapt'):
    """Set up ProtoEntropy with EATA-style entropy thresholding."""
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)

    if not params:
        logger.warning(f"No parameters found for ProtoEATA with mode {adaptation_mode}.")

    optimizer = setup_optimizer(params) if params else None

    proto_model = proto_entropy.ProtoEntropyEATA(
        model,
        optimizer,
        steps=cfg_optim.STEPS,
        episodic=cfg_model.EPISODIC,
        entropy_threshold=entropy_threshold
    )

    logger.info(f"ProtoEATA adaptation (threshold={entropy_threshold}): {len(params)} parameters")
    return proto_model


def setup_eata(model, d_margin=0.05, fisher_alpha=2000.0, fishers=None, adaptation_mode='batchnorm_addon'):
    """Set up standard EATA adaptation (logit-based, like Tent but with filtering).

    Matches ProtoViT's EATA implementation exactly:
    - Uses fixed e_margin = log(1000)/2 - 1 (same as original EATA paper)
    - Requires pre-computed Fisher information

    Args:
        model: Model to adapt
        d_margin: Cosine similarity margin (0.05 default, matches ProtoViT)
        fisher_alpha: Fisher regularization weight (2000.0 default, matches ProtoViT)
        fishers: Pre-computed Fisher information (required for EATA)
        adaptation_mode: 'batchnorm_addon' (default), 'batchnorm_only', 'addon_only', 'full'
    """
    model = tent.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = tent.collect_params(model, adaptation_mode=adaptation_mode)

    if not params:
        logger.warning(f"No parameters found for EATA with mode {adaptation_mode}.")

    optimizer = setup_optimizer(params) if params else None

    # Use ProtoViT's fixed e_margin formula (from original EATA paper)
    # This is independent of num_classes, unlike the scaled version
    import math
    e_margin = math.log(1000) / 2 - 1  # = 2.45 (fixed for all datasets)

    eata_model = eata_adapt.EATA(
        model,
        optimizer,
        fishers=fishers,
        fisher_alpha=fisher_alpha,
        steps=cfg_optim.STEPS,
        episodic=cfg_model.EPISODIC,
        e_margin=e_margin,
        d_margin=d_margin,
        num_classes=num_classes
    )

    logger.info(f"EATA adaptation ({adaptation_mode}, e_margin={eata_model.e_margin:.3f} [ProtoViT formula], d_margin={d_margin}, fisher_alpha={fisher_alpha}): {len(params)} parameters")
    if fishers is not None:
        logger.info(f"  Using pre-computed Fisher information on {len(fishers)} parameters")
    else:
        logger.warning("  No Fisher information provided - EATA may not work optimally")
    return eata_model


def evaluate_model(model, loader, description="Inference", track_per_batch=False, is_adaptation=False):
    """
    Run evaluation on a dataloader.

    Args:
        model: Model to evaluate
        loader: Data loader
        description: Description for logging
        track_per_batch: Whether to track per-batch accuracy
        is_adaptation: If True, allows gradients and doesn't force eval mode (for TTA)

    Returns:
        accuracy: Overall accuracy
        batch_accuracies: List of per-batch accuracies (if track_per_batch=True)
    """
    print(f'\nStarting {description}...')

    # Only force eval mode if NOT adapting (TTA methods manage their own mode)
    if not is_adaptation:
        model.eval()

    n_examples = 0
    n_correct = 0
    batch_accuracies = [] if track_per_batch else None

    # context manager: no_grad for normal inference, enable_grad for adaptation
    context = torch.enable_grad() if is_adaptation else torch.no_grad()

    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(images)

            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            _, predicted = logits.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_size = labels.size(0)

            if track_per_batch:
                batch_acc = batch_correct / batch_size
                batch_accuracies.append(batch_acc)

            n_correct += batch_correct
            n_examples += batch_size

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(loader)} - Acc: {batch_correct/batch_size:.2%}", end='\r')

    print() # Newline after progress
    accuracy = n_correct / n_examples
    print(f'{description} Complete. Accuracy: {accuracy * 100:.2f}%')

    if track_per_batch:
        return accuracy, batch_accuracies
    return accuracy


def load_model(model_path, device):
    """
    Robust model loading helper.
    Handles both full model saves and Trainer checkpoint dictionaries.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Case 1: Full model object
    if isinstance(checkpoint, torch.nn.Module):
        print("Detected full model object.")
        return checkpoint.to(device)

    # Case 2: Checkpoint dictionary (Standard Trainer output)
    if isinstance(checkpoint, dict):
        print("Detected checkpoint dictionary.")
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Reconstruct architecture from settings and state dict
        arch = base_architecture
        if 'vgg19_bn' in model_path: arch = 'vgg19_bn'
        elif 'vgg19' in model_path: arch = 'vgg19'
        elif 'vgg16_bn' in model_path: arch = 'vgg16_bn'
        elif 'vgg16' in model_path: arch = 'vgg16'

        # Determine number of prototypes from state dict
        num_prototypes = 2000 # default
        for key in state_dict.keys():
            if 'prototype_vectors' in key:
                num_prototypes = state_dict[key].shape[0]
                break

        print(f"Reconfiguring model: {arch}, {num_prototypes} prototypes, {num_classes} classes")

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


def parse_modes(mode_arg):
    """
    Parse comma-separated list of modes.

    Valid modes: normal, tent, eata, sar,
                 prototта_all (all_adapt), prototта_bn (batchnorm_addon)
    """
    if not mode_arg:
        return {"normal"}

    raw = [m.strip().lower() for m in mode_arg.split(",") if m.strip()]
    modes = set(raw)

    if "all" in modes:
        return {"normal", "tent", "eata", "sar", "prototта_all", "prototта_bn", "protopp_hybrid"}

    valid = {"normal", "tent", "eata", "sar", "prototта_all", "prototта_bn",
             "proto", "proto_importance", "proto_confidence",
             "proto_importance_confidence", "proto_eata", "protopp_hybrid"}
    selected = modes & valid

    return selected or {"normal"}


def run_inference(model_path, gpu_id='0', corruption=None, severity=1, mode='normal',
                  use_pre_generated=True, output_dir='./results',
                  geo_filter_threshold=None, compute_source_threshold=False,
                  adaptation_mode='all_adapt',
                  corrupted_data_dir='./datasets/SICAPv2_c'):
    """
    Main inference function.

    Args:
        model_path: Path to saved model
        gpu_id: GPU ID to use
        corruption: Corruption type (None for clean data)
        severity: Corruption severity (1-5)
        mode: Comma-separated list of TTA modes
        use_pre_generated: Use pre-generated corrupted images
        corrupted_data_dir: Root directory of the pre-generated corruptions
        output_dir: Directory to save results
    """
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # For some deterministic operations (like index_add in Adam optimizer)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'Deterministic mode: ENABLED (cudnn.deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    if corruption:
        # Check for pre-generated corrupted dataset
        corruption_path = Path(corrupted_data_dir) / corruption / str(severity)

        if use_pre_generated and corruption_path.exists():
            print(f'Using PRE-GENERATED corrupted images from: {corruption_path}')
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corruption_path), transform)
        else:
            if use_pre_generated:
                print(f'Pre-generated corrupted images not found at {corruption_path}')
            print(f'Generating corruption ON-THE-FLY: {corruption} (Severity: {severity})')
            transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
            test_dataset = datasets.ImageFolder(test_dir, transform)
    else:
        print('Applying NO corruption (Clean Data)')
        print(f'Loading test data from: {test_dir}')
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_dataset = datasets.ImageFolder(test_dir, transform)

    # For TTA: Use deterministic shuffling for reproducible results
    # Data is ordered by class, so we need shuffle to avoid class-imbalanced batches
    # Use generator with fixed seed for deterministic shuffling across runs
    g = torch.Generator()
    g.manual_seed(0)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,  # Shuffle to mix classes
        num_workers=4,
        pin_memory=True,
        generator=g,  # Deterministic shuffling with fixed seed
        worker_init_fn=seed_worker  # Seed workers for deterministic augmentations
    )

    print(f'Test set size: {len(test_loader.dataset)}')
    print(f'Batch size: {test_batch_size}')
    print(f'Number of batches: {len(test_loader)}')
    print(f'Number of classes: {len(test_dataset.classes)}')
    print(f'Classes: {test_dataset.classes}')

    # Compute geometric filter threshold from source/clean data if requested
    if compute_source_threshold:
        print(f'\n{"="*60}')
        print('>>> COMPUTING SOURCE THRESHOLD (Clean Data)')
        print(f'{"="*60}')
        print('Loading clean test data to compute baseline similarity statistics...')

        # Load clean data
        clean_transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(test_dir, clean_transform)
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Load model and compute similarities on clean data
        source_model = load_model(model_path, device)
        source_model.eval()

        all_max_sims = []
        with torch.no_grad():
            for images, _ in clean_loader:
                images = images.to(device)
                outputs = source_model(images)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    _, min_distances = outputs[0], outputs[1]
                    # Compute similarities (same as ProtoEntropy)
                    raw_similarities = torch.log((min_distances + 1.0) / (min_distances + 1e-4))
                    similarities = raw_similarities / 9.0
                    max_sims = similarities.max(dim=1)[0]
                    all_max_sims.append(max_sims.cpu())

        all_max_sims = torch.cat(all_max_sims)
        source_mean = all_max_sims.mean().item()
        source_std = all_max_sims.std().item()
        source_min = all_max_sims.min().item()
        source_max = all_max_sims.max().item()

        # Compute percentiles
        import numpy as np
        source_percentiles = {
            'p50': np.percentile(all_max_sims.numpy(), 50),
            'p75': np.percentile(all_max_sims.numpy(), 75),
            'p85': np.percentile(all_max_sims.numpy(), 85),
            'p90': np.percentile(all_max_sims.numpy(), 90),
            'p95': np.percentile(all_max_sims.numpy(), 95),
            'p99': np.percentile(all_max_sims.numpy(), 99),
        }

        # Strategy: Use a high bar that's achievable
        # Option 1: max - small_margin (e.g., max - 0.01)
        # Option 2: 99th percentile
        # Option 3: mean + 3*std

        threshold_option1 = min(source_max - 0.001, 0.999)  # Very high bar, near max
        threshold_option2 = source_percentiles['p99']
        threshold_option3 = min(source_mean + 3.0 * source_std, 0.999)

        # Use option 1 (near-max) as it gives the strictest, most consistent threshold
        computed_threshold = threshold_option1

        print(f'\nClean Data Similarity Statistics:')
        print(f'  Mean: {source_mean:.3f}')
        print(f'  Std:  {source_std:.3f}')
        print(f'  Range: [{source_min:.3f}, {source_max:.3f}]')
        print(f'  Percentiles: 50%={source_percentiles["p50"]:.3f}, 75%={source_percentiles["p75"]:.3f}, '
              f'85%={source_percentiles["p85"]:.3f}, 90%={source_percentiles["p90"]:.3f}, '
              f'95%={source_percentiles["p95"]:.3f}, 99%={source_percentiles["p99"]:.3f}')
        print(f'\n  Threshold Options:')
        print(f'    max - 0.001 = {threshold_option1:.3f}')
        print(f'    99th percentile = {threshold_option2:.3f}')
        print(f'    mean + 3σ = {threshold_option3:.3f}')
        print(f'  Computed threshold: {computed_threshold:.3f} (using max - 0.001)')
        print(f'  Rationale: Only adapt on samples near the best clean data performance')

        # Override user threshold with computed one
        geo_filter_threshold = computed_threshold
        print(f'\n✓ Using computed threshold from clean data: {geo_filter_threshold:.3f}')

        del source_model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f'Using user-specified threshold: {geo_filter_threshold:.3f}')

    # Parse modes
    selected_modes = parse_modes(mode)
    print(f'Selected TTA modes: {selected_modes}')

    results = {}

    # --- NORMAL INFERENCE ---
    if "normal" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> NORMAL INFERENCE (No Adaptation)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        model.eval()

        accuracy = evaluate_model(model, test_loader, "Normal Inference", is_adaptation=False)
        results['Normal'] = accuracy

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # --- TENT INFERENCE ---
    if "tent" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> TENT INFERENCE')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        tent_model = setup_tent(model)

        accuracy = evaluate_model(tent_model, test_loader, "Tent Adaptation", is_adaptation=True)
        results['Tent'] = accuracy

        # Print Tent adaptation statistics
        if hasattr(tent_model, 'adaptation_stats'):
            stats = tent_model.adaptation_stats
            print(f'\nTent Adaptation Statistics:')
            print(f'  Total samples: {stats["total_samples"]}')
            print(f'  Adapted samples: {stats["adapted_samples"]} (all samples)')
            print(f'  Total updates: {stats["total_updates"]}')

        del tent_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- EATA INFERENCE (Logit-based with filtering) ---
    if "eata" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> EATA INFERENCE (Entropy Filtering + Anti-forgetting)')
        print(f'{"="*60}')

        # Compute Fisher Information Matrix (like ProtoViT)
        print('Computing Fisher Information Matrix on 500 test samples...')
        fisher_model = load_model(model_path, device)
        fisher_model = tent.configure_model(fisher_model, adaptation_mode='batchnorm_addon')

        # Create a small loader for Fisher computation (500 samples)
        fisher_dataset = test_dataset
        fisher_loader = torch.utils.data.DataLoader(
            fisher_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            generator=g,
            worker_init_fn=seed_worker
        )

        fishers = eata_adapt.compute_fishers(fisher_model, fisher_loader, device, num_samples=500)
        print(f'Fisher computation complete: {len(fishers)} parameters tracked')

        del fisher_model
        torch.cuda.empty_cache()
        gc.collect()

        # Now set up EATA with Fisher information
        model = load_model(model_path, device)
        eata_model = setup_eata(model, fishers=fishers)

        accuracy = evaluate_model(eata_model, test_loader, "EATA Adaptation", is_adaptation=True)
        results['EATA'] = accuracy

        # Print EATA filtering statistics
        if hasattr(eata_model, 'adaptation_stats'):
            stats = eata_model.adaptation_stats
            adapt_rate = stats['adapted_samples'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
            print(f'\nEATA Adaptation Statistics:')
            print(f'  Total samples: {stats["total_samples"]}')
            print(f'  Adapted samples: {stats["adapted_samples"]} ({adapt_rate:.1f}%)')
            print(f'  Filtered out: {stats["total_samples"] - stats["adapted_samples"]} ({100-adapt_rate:.1f}%)')
            print(f'  Total updates: {stats["total_updates"]}')

        del eata_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTO ENTROPY (Basic) ---
    if "proto" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTO ENTROPY INFERENCE (Basic)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_entropy(model, use_importance=False, use_confidence=False)

        accuracy = evaluate_model(proto_model, test_loader, "ProtoEntropy", is_adaptation=True)
        results['ProtoEntropy'] = accuracy

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTO ENTROPY (Importance Weighted) ---
    if "proto_importance" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTO ENTROPY INFERENCE (Importance-Weighted)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_entropy(model, use_importance=True, use_confidence=False)

        accuracy = evaluate_model(proto_model, test_loader, "ProtoEntropy-Importance", is_adaptation=True)
        results['ProtoEntropy-Importance'] = accuracy

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTO ENTROPY (Confidence Weighted) ---
    if "proto_confidence" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTO ENTROPY INFERENCE (Confidence-Weighted)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_entropy(model, use_importance=False, use_confidence=True)

        accuracy = evaluate_model(proto_model, test_loader, "ProtoEntropy-Confidence", is_adaptation=True)
        results['ProtoEntropy-Confidence'] = accuracy

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTO ENTROPY (Importance + Confidence) ---
    if "proto_importance_confidence" in selected_modes:
        print(f'\n{"="*60}')
        print(f'>>> PROTO ENTROPY INFERENCE (Importance + Confidence + GeoFilter)')
        print(f'>>> Adaptation Mode: {adaptation_mode}')
        print(f'{"="*60}')

        model = load_model(model_path, device)

        # Full configuration with importance, confidence, and geometric filtering
        # Following ProtoViT config3: use stricter threshold for better filtering
        proto_model = setup_proto_entropy(
            model,
            use_importance=True,
            use_confidence=True,
            use_geometric_filter=True,
            geo_filter_threshold=geo_filter_threshold,
            adaptation_mode=adaptation_mode
        )

        accuracy = evaluate_model(proto_model, test_loader, "ProtoEntropy-Full", is_adaptation=True)
        results['ProtoEntropy-Full'] = accuracy

        # Print filtering statistics
        if hasattr(proto_model, 'get_geo_filter_stats'):
            stats = proto_model.get_geo_filter_stats()
            print(f'\nGeometric Filtering Statistics:')
            print(f'  Total samples: {stats["total_samples"]}')
            print(f'  Filtered out: {stats["filtered_samples"]} ({stats.get("filter_rate", 0)*100:.1f}%)')
            print(f'  Adapted samples: {stats["total_samples"] - stats["filtered_samples"]}')

            # Show similarity distribution to help tune threshold
            if stats.get('avg_similarities') and len(stats['avg_similarities']) > 0:
                import numpy as np
                sims = np.array(stats['avg_similarities'])
                print(f'\n  Normalized Max-Similarities (per-sample max, scaled to [0,1]):')
                print(f'    Mean: {sims.mean():.3f}')
                print(f'    Std:  {sims.std():.3f}')
                print(f'    Min:  {sims.min():.3f}')
                print(f'    Max:  {sims.max():.3f}')
                print(f'    Percentiles: [10%={np.percentile(sims, 10):.3f}, 25%={np.percentile(sims, 25):.3f}, '
                      f'50%={np.percentile(sims, 50):.3f}, 75%={np.percentile(sims, 75):.3f}, 90%={np.percentile(sims, 90):.3f}]')

                # Also show raw similarities for context
                if stats.get('raw_avg_similarities') and len(stats['raw_avg_similarities']) > 0:
                    raw_sims = np.array(stats['raw_avg_similarities'])
                    print(f'\n  Raw Log-Similarities (ProtoPNet activation, range ~0-9):')
                    print(f'    Mean: {raw_sims.mean():.3f}')
                    print(f'    Std:  {raw_sims.std():.3f}')
                    print(f'    Range: [{raw_sims.min():.3f}, {raw_sims.max():.3f}]')

                print(f'\n  Current threshold: {geo_filter_threshold:.2f}')
                print(f'  Samples below threshold: {(sims <= geo_filter_threshold).sum()} ({(sims <= geo_filter_threshold).mean()*100:.1f}%)')
                print(f'  Recommendation: Use threshold between {np.percentile(sims, 10):.2f} (filter 90%) and {np.percentile(sims, 50):.2f} (filter 50%)')

        del proto_model
        torch.cuda.empty_cache()

    # --- PROTO EATA (Entropy-Adaptive Thresholding) ---
    if "proto_eata" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTO EATA INFERENCE (Entropy Thresholding)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_eata(model, entropy_threshold=0.4, adaptation_mode='all_adapt')

        accuracy = evaluate_model(proto_model, test_loader, "ProtoEATA", is_adaptation=True)
        results['ProtoEATA'] = accuracy

        del proto_model
        torch.cuda.empty_cache()

    # --- SAR ---
    if "sar" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> SAR (Sharpness-Aware + Reliable Entropy Minimization)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        sar_model = setup_sar(model)

        accuracy = evaluate_model(sar_model, test_loader, "SAR", is_adaptation=True)
        results['SAR'] = accuracy

        if hasattr(sar_model, 'adaptation_stats'):
            stats = sar_model.adaptation_stats
            rate = stats['adapted_samples'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
            print(f'  Adapted: {stats["adapted_samples"]}/{stats["total_samples"]} ({rate:.1f}%)')

        del sar_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTOTTA (all_adapt) ---
    if "prototта_all" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTOTTA (all_adapt - adapts BN + addon + prototypes)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_entropy(
            model,
            use_importance=True,
            use_confidence=True,
            use_geometric_filter=True,
            geo_filter_threshold=geo_filter_threshold if geo_filter_threshold else 0.70,
            adaptation_mode='all_adapt'
        )

        accuracy = evaluate_model(proto_model, test_loader, "ProtoTTA-All", is_adaptation=True)
        results['ProtoTTA-All'] = accuracy

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTOTTA-BN (batchnorm_addon) ---
    if "prototта_bn" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTOTTA-BN (batchnorm_addon - BN + addon, prototypes FROZEN)')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = setup_proto_entropy(
            model,
            use_importance=True,
            use_confidence=True,
            use_geometric_filter=True,
            geo_filter_threshold=geo_filter_threshold if geo_filter_threshold else 0.70,
            adaptation_mode='batchnorm_addon'
        )

        accuracy = evaluate_model(proto_model, test_loader, "ProtoTTA-BN", is_adaptation=True)
        results['ProtoTTA-BN'] = accuracy

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # --- PROTO++ HYBRID (Proto Entropy + Softmax Entropy blend) ---
    if "protopp_hybrid" in selected_modes:
        print(f'\n{"="*60}')
        print('>>> PROTO++ HYBRID (Proto Entropy + Softmax Entropy blend)')
        print('>>> Combines ProtoTTA (70%) + Tent (30%) objectives')
        print(f'{"="*60}')

        model = load_model(model_path, device)
        proto_model = proto_entropy_enhanced.setup_proto_entropy_enhanced(
            model,
            lr=cfg_optim.LR,
            use_sam=False,
            alpha_proto=0.7,  # 70% binary proto entropy (ProtoTTA)
            alpha_softmax=0.3,  # 30% softmax entropy (Tent)
            use_entropy_filter=True,
            entropy_margin_scale=0.4,
            use_geometric_filter=True,
            geo_filter_threshold=geo_filter_threshold if geo_filter_threshold else 0.70,
            adaptation_mode='batchnorm_addon',
            steps=cfg_optim.STEPS
        )

        accuracy = evaluate_model(proto_model, test_loader, "Proto++Hybrid", is_adaptation=True)
        results['Proto++Hybrid'] = accuracy

        if hasattr(proto_model, 'get_stats'):
            stats = proto_model.get_stats()
            rate = stats['adapted_samples'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
            print(f'\nProto++Hybrid Statistics:')
            print(f'  Total samples: {stats["total_samples"]}')
            print(f'  Adapted samples: {stats["adapted_samples"]} ({rate:.1f}%)')
            print(f'  Filtered by entropy: {stats["filtered_by_entropy"]}')
            print(f'  Filtered by geo: {stats["filtered_by_geo"]}')
            print(f'  Geo threshold used: {geo_filter_threshold if geo_filter_threshold else 0.70}')

        del proto_model
        torch.cuda.empty_cache()
        gc.collect()

    # Print summary
    print(f'\n{"="*60}')
    print('RESULTS SUMMARY')
    print(f'{"="*60}')

    corruption_info = f"{corruption} (severity {severity})" if corruption else "Clean"
    print(f'Dataset: {corruption_info}')
    print(f'Model: {os.path.basename(model_path)}')
    print()

    for method, acc in results.items():
        print(f'  {method:30s}: {acc * 100:.2f}%')

    print(f'{"="*60}')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with TTA methods on ProtoPNet',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        # default='./saved_models/vgg19/sicapv2_001/epoch_10_last_15.pth',
        # default='./saved_models/vgg19_bn/sicapv2_002/epoch_10_last_0.pth',
        default='./saved_models/vgg19_bn/sicapv2_002/epoch_20_last_5.pth',
        help='Path to saved model'
    )

    parser.add_argument(
        '--gpuid',
        type=str,
        default='0',
        help='GPU ID to use'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./datasets/SICAPv2_c',
        help='Root directory of the pre-generated SICAPv2-C dataset'
    )

    parser.add_argument(
        '--corruption',
        type=str,
        default=None,
        help='Corruption type (e.g., gaussian_noise, fog). None for clean data.'
    )

    parser.add_argument(
        '--severity',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Corruption severity (1-5)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='normal',
        help='TTA modes (comma-separated). Options: normal, tent, eata, sar, '
             'prototта_all (all_adapt), prototта_bn (batchnorm_addon)'
    )

    parser.add_argument(
        '--use_pre_generated',
        action='store_true',
        default=True,
        help='Use pre-generated corrupted images from SICAPv2_c'
    )

    parser.add_argument(
        '--on_the_fly',
        action='store_true',
        help='Generate corruptions on-the-fly (ignores --use_pre_generated)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--geo_filter_threshold',
        type=float,
        default=0.95,
        help='Geometric filter threshold for ProtoEntropy (0.0-1.0). '
             'Higher = stricter filtering. Default: 0.95 (high bar for adaptation). '
             'Use --compute_source_threshold to compute from clean data automatically.'
    )

    parser.add_argument(
        '--compute_source_threshold',
        action='store_true',
        help='Compute geometric filter threshold from clean/source data before adaptation. '
             'Uses (max - 0.001) of clean data similarities. Overrides --geo_filter_threshold.'
    )

    parser.add_argument(
        '--adaptation_mode',
        type=str,
        default='all_adapt',
        choices=['batchnorm_only', 'batchnorm_addon', 'batchnorm_proto', 'all_adapt'],
        help='ProtoEntropy adaptation mode. Options: '
             'batchnorm_only (BN only), '
             'batchnorm_addon (BN + add-on layers), '
             'batchnorm_proto (BN + prototypes), '
             'all_adapt (BN + add-on + prototypes, default)'
    )

    args = parser.parse_args()

    use_pre_generated = args.use_pre_generated and not args.on_the_fly

    run_inference(
        model_path=args.model,
        gpu_id=args.gpuid,
        corruption=args.corruption,
        severity=args.severity,
        mode=args.mode,
        use_pre_generated=use_pre_generated,
        output_dir=args.output_dir,
        geo_filter_threshold=args.geo_filter_threshold,
        compute_source_threshold=args.compute_source_threshold,
        adaptation_mode=args.adaptation_mode,
        corrupted_data_dir=args.data_dir,
    )


if __name__ == '__main__':
    main()
