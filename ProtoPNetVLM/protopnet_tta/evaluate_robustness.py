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
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
import argparse
import hashlib
import json
import time
import math
import subprocess
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
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


def sha256_file(path):
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values):
    """Hash an ordered sequence with unambiguous newline separators."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def sha256_tensor(value):
    """Hash a captured CPU tensor including dtype and shape."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(tuple(value.shape)).encode('ascii'))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_value(*args):
    """Best-effort git provenance; evaluation remains usable outside git."""
    try:
        return subprocess.check_output(
            ['git', *args], cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluator_provenance(model_path):
    evaluator_path = Path(__file__).resolve()
    diff = git_value('diff', '--', str(evaluator_path.relative_to(
        evaluator_path.parent.parent)))
    return {
        'model_path': str(Path(model_path).resolve()),
        'model_checksum_sha256': sha256_file(model_path),
        'git_commit': git_value('rev-parse', 'HEAD'),
        'evaluator_path': str(evaluator_path),
        'evaluator_checksum_sha256': sha256_file(evaluator_path),
        'evaluator_diff_sha256': hashlib.sha256(
            (diff or '').encode('utf-8')
        ).hexdigest(),
        'git_worktree_dirty': bool(git_value('status', '--porcelain')),
    }


def dataset_identity(dataset):
    """Return stable class-relative IDs and ImageFolder labels in load order."""
    root = Path(dataset.root).resolve()
    ids = [str(Path(path).resolve().relative_to(root))
           for path, _ in dataset.samples]
    labels = [int(label) for _, label in dataset.samples]
    return {
        'ids': ids,
        'labels': labels,
        'sample_id_hash_sha256': sha256_lines(ids),
        'label_hash_sha256': sha256_lines(labels),
        'num_samples': len(ids),
        'classes': list(dataset.classes),
        'class_to_idx': dict(dataset.class_to_idx),
    }


STREAM_ORDER_ALGORITHM = (
    'torch_randperm_after_dataloader_base_seed_legacy_compatible'
)


def fixed_stream_indices(num_samples, order='seeded_random', seed=0):
    """Return one explicit test-stream order.

    The seeded-random branch reproduces the first permutation produced by the
    historical ``DataLoader(shuffle=True, generator=manual_seed(seed))``.  A
    DataLoader consumes one generator draw for its worker/base seed before its
    RandomSampler calls ``randperm``, so that draw is reproduced explicitly.
    """
    if num_samples <= 0:
        raise ValueError('Cannot build a stream for an empty dataset')
    if order == 'class_order':
        return list(range(num_samples))
    if order != 'seeded_random':
        raise ValueError(f'Unknown stream order: {order}')
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.randperm(num_samples, generator=generator).tolist()


def identity_in_stream_order(identity, indices):
    """Reorder IDs and labels using a validated, explicit permutation."""
    expected = list(range(identity['num_samples']))
    if len(indices) != len(expected) or sorted(indices) != expected:
        raise RuntimeError('Stream indices are not a complete permutation')
    ordered_ids = [identity['ids'][index] for index in indices]
    ordered_labels = [identity['labels'][index] for index in indices]
    return {
        **identity,
        'ids': ordered_ids,
        'labels': ordered_labels,
        'sample_id_hash_sha256': sha256_lines(ordered_ids),
        'label_hash_sha256': sha256_lines(ordered_labels),
        'canonical_sample_id_hash_sha256': identity['sample_id_hash_sha256'],
        'canonical_label_hash_sha256': identity['label_hash_sha256'],
        'stream_order_index_hash_sha256': sha256_lines(indices),
    }


def assert_paired_datasets(clean_identity, corrupt_identity, context):
    """Fail rather than truncate or silently compare different samples."""
    checks = {
        'N': clean_identity['num_samples'] == corrupt_identity['num_samples'],
        'IDs': clean_identity['ids'] == corrupt_identity['ids'],
        'labels': clean_identity['labels'] == corrupt_identity['labels'],
        'classes': clean_identity['classes'] == corrupt_identity['classes'],
        'class_to_idx': (
            clean_identity['class_to_idx'] == corrupt_identity['class_to_idx']
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f'Paired-data invariant failed for {context}: {failed}; '
            f"clean_N={clean_identity['num_samples']}, "
            f"corrupt_N={corrupt_identity['num_samples']}, "
            f"clean_id_hash={clean_identity['sample_id_hash_sha256']}, "
            f"corrupt_id_hash={corrupt_identity['sample_id_hash_sha256']}"
        )


def prototype_activations_from_distances(min_distances):
    return torch.log((min_distances + 1.0) / (min_distances + 1e-4))

def set_random_seed(seed):
    """Set all process-level random seeds used by this evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def setup_proto_samplewise_adaptive(
        model, geo_filter_threshold=0.8, delta0=0.25, top_k=3,
        component_gradient_normalization=False,
        adaptive_controller='absolute_distance'):
    """Set up label-free samplewise adaptive ProtoTTA+."""
    return proto_entropy_enhanced.setup_proto_entropy_enhanced(
        model,
        lr=cfg_optim.LR,
        use_sam=False,
        alpha_proto=0.7,
        alpha_softmax=0.3,
        use_entropy_filter=True,
        entropy_margin_scale=0.4,
        use_geometric_filter=True,
        geo_filter_threshold=geo_filter_threshold,
        adaptation_mode='batchnorm_addon',
        steps=cfg_optim.STEPS,
        samplewise_adaptive_lambda=True,
        adaptive_delta0=delta0,
        adaptive_top_k=top_k,
        component_gradient_normalization=component_gradient_normalization,
        adaptive_controller=adaptive_controller,
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
                   verbose=True, efficiency_tracker=None,
                   require_prototype_activations=False):
    """
    Run one online trajectory and capture everything from its returned forwards.

    Args:
        efficiency_tracker: Optional EfficiencyTracker to record per-batch timing

    Returns:
        A dictionary containing accuracy, labels, predictions, logits, and the
        prototype activations returned by those exact same forwards.
    """
    if verbose:
        print(f'\n{description}...')

    model.eval()
    n_examples = 0
    n_correct = 0
    all_labels = []
    all_predictions = []
    all_logits = []
    all_activations = []

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

        all_labels.append(labels.detach().cpu())
        all_predictions.append(predicted.detach().cpu())
        all_logits.append(logits.detach().cpu())
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            all_activations.append(
                prototype_activations_from_distances(outputs[1].detach()).cpu()
            )
        elif require_prototype_activations:
            raise RuntimeError(
                'Online forward did not return prototype distances; refusing '
                'to replay the final adapted model for prototype metrics'
            )

    accuracy = n_correct / n_examples

    if verbose:
        print(f'Accuracy: {accuracy*100:.2f}%')

    trajectory = {
        'accuracy': float(accuracy),
        'labels': torch.cat(all_labels),
        'predictions': torch.cat(all_predictions),
        'logits': torch.cat(all_logits),
        'prototype_activations': (
            torch.cat(all_activations) if all_activations else None
        ),
        'num_samples': n_examples,
    }
    for name in ('labels', 'predictions', 'logits'):
        if len(trajectory[name]) != n_examples:
            raise RuntimeError(
                f'Online trajectory {name} length mismatch: '
                f'{len(trajectory[name])} != {n_examples}'
            )
    if (trajectory['prototype_activations'] is not None and
            len(trajectory['prototype_activations']) != n_examples):
        raise RuntimeError('Prototype activation trajectory length mismatch')
    return trajectory


def strict_same_length(clean_tensor, corrupt_tensor, name):
    if len(clean_tensor) != len(corrupt_tensor):
        raise RuntimeError(
            f'Paired {name} length mismatch: clean={len(clean_tensor)}, '
            f'corrupt={len(corrupt_tensor)}'
        )


def compute_online_metrics(metrics_evaluator, clean_reference, trajectory,
                           top_k=10):
    """Compute paired metrics solely from the captured online trajectory."""
    clean_labels = clean_reference['labels']
    labels = trajectory['labels']
    clean_predictions = clean_reference['predictions']
    predictions = trajectory['predictions']
    clean_logits = clean_reference['logits']
    logits = trajectory['logits']
    clean_activations = clean_reference['prototype_activations']
    activations = trajectory['prototype_activations']

    for name, clean_tensor, corrupt_tensor in (
            ('labels', clean_labels, labels),
            ('predictions', clean_predictions, predictions),
            ('logits', clean_logits, logits),
            ('prototype activations', clean_activations, activations)):
        if clean_tensor is None or corrupt_tensor is None:
            raise RuntimeError(f'Missing paired {name}')
        strict_same_length(clean_tensor, corrupt_tensor, name)
    if not torch.equal(clean_labels, labels):
        raise RuntimeError('Clean/corrupted online label tensors differ')

    pac = metrics_evaluator.compute_prototype_activation_consistency(
        activations, clean_activations=clean_activations, method='cosine'
    )
    pca = metrics_evaluator.compute_prototype_class_alignment(
        activations, labels, top_k=top_k, weight_by_activation=True
    )
    pca_weighted = metrics_evaluator.compute_pca_weighted_by_importance(
        activations, labels, top_k=top_k
    )
    sparsity = metrics_evaluator.compute_prototype_activation_sparsity(
        activations, threshold=0.1
    )
    contribution = metrics_evaluator.compute_class_contribution_change(
        clean_activations, activations, labels
    )
    agreement = float((predictions == clean_predictions).float().mean().item())
    logit_corr = float(F.cosine_similarity(
        logits.float(), clean_logits.float(), dim=1
    ).mean().item())
    return {
        **pac,
        **pca,
        **pca_weighted,
        **sparsity,
        **contribution,
        'calibration_agreement': agreement,
        'calibration_logit_corr': logit_corr,
    }


def evaluate_single_combination(model_path, corruption_type, severity,
                                data_dir, clean_data_dir, mode_name, mode_config,
                                device, batch_size, clean_reference=None,
                                clean_identity=None, provenance=None,
                                compute_proto_metrics=False, track_efficiency=False,
                                seed=0, stream_order='seeded_random',
                                stream_order_seed=0):
    """
    Evaluate a single combination of corruption type and TTA method.

    Returns:
        dict with accuracy, paired metrics, and provenance; failures propagate
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
                raise FileNotFoundError(
                    f"Corrupted dataset not found: {corrupted_path}"
                )
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corrupted_path), transform)

        raw_test_identity = dataset_identity(test_dataset)
        stream_indices = fixed_stream_indices(
            raw_test_identity['num_samples'], stream_order,
            stream_order_seed
        )
        test_identity = identity_in_stream_order(
            raw_test_identity, stream_indices
        )
        if clean_identity is not None:
            assert_paired_datasets(
                clean_identity, test_identity,
                f'{mode_name}/{corruption_type}/severity-{severity}/seed-{seed}'
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=stream_indices,
            num_workers=4,
            pin_memory=True,
            generator=torch.Generator().manual_seed(stream_order_seed),
        )

        # Load model
        base_model = load_model(model_path, device)
        base_model.eval()
        metrics_evaluator = (
            EnhancedPrototypeMetrics(base_model, device=device)
            if compute_proto_metrics else None
        )

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
        elif mode_name in (
                'ProtoSampleAdaptive',
                'ProtoSampleAdaptiveGradNorm',
                'ProtoSampleAdaptiveRelative',
                'ProtoSampleAdaptiveRelativeGradNorm',
                'ProtoSampleAdaptiveTeacherMedian',
                'ProtoSampleAdaptiveTeacherBatchMedian',
                'ProtoSampleAdaptiveCoverageRouter',
                'ProtoAuditForcedRelative',
                'ProtoAuditForcedRelativeGradNorm',
                'ProtoAuditFixedLambda0.2GradNorm',
                'ProtoAuditForcedNativeGradNorm',
                'ProtoAbsoluteConsistencyCoverageRouter',
                'ProtoRatioOnlyRouter',
                'ProtoAbsoluteOnlyRouter'):
            eval_model = setup_proto_samplewise_adaptive(
                base_model,
                geo_filter_threshold=mode_config.get(
                    'geo_filter_threshold', 0.8
                ),
                delta0=mode_config.get('delta0', 0.25),
                top_k=mode_config.get('top_k', 3),
                component_gradient_normalization=mode_config.get(
                    'component_gradient_normalization', False
                ),
                adaptive_controller=mode_config.get(
                    'adaptive_controller', 'absolute_distance'
                ),
            )
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
        trajectory = evaluate_model(
            eval_model, test_loader, device,
            description=f"{mode_name} on {corruption_type}-{severity}",
            verbose=False,
            efficiency_tracker=efficiency_tracker,
            require_prototype_activations=compute_proto_metrics,
        )
        if trajectory['num_samples'] != test_identity['num_samples']:
            raise RuntimeError(
                f"Online N={trajectory['num_samples']} does not match dataset "
                f"N={test_identity['num_samples']}"
            )
        expected_labels = torch.tensor(test_identity['labels'], dtype=torch.long)
        if not torch.equal(trajectory['labels'], expected_labels):
            raise RuntimeError('Online labels/order differ from ImageFolder identity')

        # Record adaptation steps AFTER evaluation
        if efficiency_tracker and mode_name != 'Normal':
            num_batches = len(test_loader)
            efficiency_tracker.record_adaptation_step(num_batches * cfg_optim.STEPS)

        result = {
            'accuracy': trajectory['accuracy'],
            'PAC_mean': None,
            'PCA_weighted_mean': None,
            'calibration_agreement': None,
            'efficiency': None,
            'selection_rate': 0.0 if mode_name == 'Normal' else None,
            'paired_num_samples': trajectory['num_samples'],
            'clean_accuracy_reference': None,
            'stability_lower_bound': None,
            'stability_upper_bound': None,
            'stability_bounds_passed': None,
            'seed': seed,
            'method': mode_name,
            'method_config': dict(mode_config),
            'sample_id_hash_sha256': test_identity['sample_id_hash_sha256'],
            'label_hash_sha256': test_identity['label_hash_sha256'],
            'canonical_sample_id_hash_sha256': test_identity[
                'canonical_sample_id_hash_sha256'
            ],
            'stream_order': stream_order,
            'stream_order_seed': stream_order_seed,
            'stream_order_algorithm': STREAM_ORDER_ALGORITHM,
            'stream_order_index_hash_sha256': test_identity[
                'stream_order_index_hash_sha256'
            ],
            'online_prediction_hash_sha256': sha256_lines(
                trajectory['predictions'].tolist()
            ),
            'online_logits_hash_sha256': sha256_tensor(
                trajectory['logits']
            ),
            'online_prototype_activations_hash_sha256': (
                sha256_tensor(trajectory['prototype_activations'])
                if trajectory['prototype_activations'] is not None else None
            ),
            'online_predictions': trajectory['predictions'].tolist(),
            'online_labels': trajectory['labels'].tolist(),
            'data_path': str(Path(test_dataset.root).resolve()),
            'trajectory_semantics': (
                'pre-update logits and prototype distances returned by the same '
                'online wrapper forward; no final-model replay'
            ),
            **(provenance or evaluator_provenance(model_path)),
        }

        if clean_reference is not None:
            strict_same_length(
                clean_reference['predictions'], trajectory['predictions'],
                'predictions'
            )
            if not torch.equal(clean_reference['labels'], trajectory['labels']):
                raise RuntimeError('Clean/corrupted trajectory labels differ')
            clean_accuracy = float(clean_reference['accuracy'])
            corrupt_accuracy = float(trajectory['accuracy'])
            agreement = float((
                clean_reference['predictions'] == trajectory['predictions']
            ).float().mean().item())
            lower = max(0.0, clean_accuracy + corrupt_accuracy - 1.0)
            upper = 1.0 - abs(clean_accuracy - corrupt_accuracy)
            passed = (lower - 1e-12 <= agreement <= upper + 1e-12)
            result.update({
                'calibration_agreement': agreement,
                'clean_accuracy_reference': clean_accuracy,
                'stability_lower_bound': lower,
                'stability_upper_bound': upper,
                'stability_bounds_passed': passed,
                'clean_prediction_hash_sha256': sha256_lines(
                    clean_reference['predictions'].tolist()
                ),
            })
            if not passed:
                raise RuntimeError(
                    f'Stability bound failed for {mode_name}/{corruption_type}/'
                    f'seed-{seed}: {lower} <= {agreement} <= {upper}'
                )

        if compute_proto_metrics:
            if clean_reference is None:
                raise RuntimeError('Prototype metrics require a clean reference')
            proto_metrics = compute_online_metrics(
                metrics_evaluator, clean_reference, trajectory, top_k=10
            )
            result.update(proto_metrics)

        # Add efficiency metrics if tracking
        if efficiency_tracker:
            efficiency_metrics = efficiency_tracker.get_metrics()
            result['efficiency'] = efficiency_metrics

        # Selection/adaptation fields are provenance, not optional timing data.
        if hasattr(eval_model, 'get_stats'):
            result['adaptation_stats'] = eval_model.get_stats()
        elif hasattr(eval_model, 'adaptation_stats'):
            result['adaptation_stats'] = eval_model.adaptation_stats.copy()
        else:
            result['adaptation_stats'] = {
                'total_samples': trajectory['num_samples'],
                'adapted_samples': 0,
                'total_updates': 0,
            }
        adapted_samples = result['adaptation_stats'].get('adapted_samples')
        if adapted_samples is not None:
            result['selection_rate'] = (
                float(adapted_samples) / trajectory['num_samples']
            )

        return result

    except Exception as e:
        logger.error(f"Failed to evaluate {mode_name} on {corruption_type}-{severity}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
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


def load_or_collect_clean_reference(args, device, clean_dataset,
                                    clean_identity, provenance):
    """Load a verified cache or collect one full unadapted clean trajectory."""
    cache_path = (
        Path(args.clean_reference_cache).resolve()
        if args.clean_reference_cache else None
    )
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        expected = {
            'seed': args.seed,
            'model_checksum_sha256': provenance['model_checksum_sha256'],
            'sample_id_hash_sha256': clean_identity['sample_id_hash_sha256'],
            'label_hash_sha256': clean_identity['label_hash_sha256'],
            'num_samples': clean_identity['num_samples'],
            'batch_size': args.batch_size,
            'shuffle': args.stream_order == 'seeded_random',
            'stream_order': args.stream_order,
            'stream_order_seed': args.stream_order_seed,
            'stream_order_algorithm': STREAM_ORDER_ALGORITHM,
            'stream_order_index_hash_sha256': clean_identity[
                'stream_order_index_hash_sha256'
            ],
        }
        actual = payload.get('metadata', {})
        mismatches = {
            key: (actual.get(key), value) for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f'Clean-reference cache provenance mismatch at {cache_path}: '
                f'{mismatches}'
            )
        reference = payload['trajectory']
        expected_labels = torch.tensor(clean_identity['labels'], dtype=torch.long)
        if (reference['num_samples'] != clean_identity['num_samples'] or
                not torch.equal(reference['labels'], expected_labels)):
            raise RuntimeError('Clean-reference cache labels/order/N mismatch')
        if args.prototype_metrics and reference['prototype_activations'] is None:
            raise RuntimeError('Clean cache lacks required prototype activations')
        logger.info('Using verified clean-reference cache: %s', cache_path)
        return reference

    stream_indices = fixed_stream_indices(
        clean_identity['num_samples'], args.stream_order,
        args.stream_order_seed
    )
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=args.batch_size, sampler=stream_indices,
        num_workers=4, pin_memory=False,
        generator=torch.Generator().manual_seed(args.stream_order_seed),
    )
    base_model = load_model(args.model, device)
    base_model.eval()
    reference = evaluate_model(
        base_model, clean_loader, device,
        description='Clean data evaluation', verbose=True,
        require_prototype_activations=args.prototype_metrics,
    )
    del base_model
    torch.cuda.empty_cache()

    expected_labels = torch.tensor(clean_identity['labels'], dtype=torch.long)
    if (reference['num_samples'] != clean_identity['num_samples'] or
            not torch.equal(reference['labels'], expected_labels)):
        raise RuntimeError('Collected clean trajectory labels/order/N mismatch')

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            raise FileExistsError(
                f'Refusing to overwrite clean-reference cache: {cache_path}'
            )
        payload = {
            'metadata': {
                'seed': args.seed,
                'model_checksum_sha256': provenance['model_checksum_sha256'],
                'sample_id_hash_sha256': clean_identity['sample_id_hash_sha256'],
                'label_hash_sha256': clean_identity['label_hash_sha256'],
                'num_samples': clean_identity['num_samples'],
                'batch_size': args.batch_size,
                'shuffle': args.stream_order == 'seeded_random',
                'stream_order': args.stream_order,
                'stream_order_seed': args.stream_order_seed,
                'stream_order_algorithm': STREAM_ORDER_ALGORITHM,
                'stream_order_index_hash_sha256': clean_identity[
                    'stream_order_index_hash_sha256'
                ],
                'clean_data_path': str(Path(args.clean_data_dir).resolve()),
            },
            'trajectory': reference,
        }
        temporary = cache_path.with_name(
            f'.{cache_path.name}.tmp-{os.getpid()}'
        )
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
        logger.info('Saved clean-reference cache: %s', cache_path)
    return reference


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
        default='./datasets/SICAPv2_c',
        help='Path to corrupted dataset directory'
    )

    parser.add_argument(
        '--clean_data_dir',
        type=str,
        default='./datasets/SICAPv2_cropped/test_cropped',
        help='Path to clean test dataset'
    )

    parser.add_argument(
        '--skip-clean-evaluation',
        action='store_true',
        help='Run corruptions without loading any clean/source images'
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
        '--seed',
        type=int,
        default=0,
        help='Random seed for process RNG and deterministic data ordering'
    )

    parser.add_argument(
        '--lambda-proto', '--lambda',
        dest='lambda_proto',
        type=float,
        default=0.7,
        help='ProtoHybrid prototype-loss weight; output-entropy weight is 1-lambda'
    )

    parser.add_argument(
        '--samplewise-adaptive-lambda',
        action='store_true',
        help='Evaluate samplewise adaptive lambda instead of legacy default modes'
    )

    parser.add_argument(
        '--component-gradient-normalization',
        action='store_true',
        help='Use component gradient normalization with samplewise lambda'
    )

    parser.add_argument(
        '--adaptive-delta0',
        type=float,
        default=0.25,
        help='Label-free samplewise lambda controller scale'
    )

    parser.add_argument(
        '--adaptive-top-k',
        type=int,
        default=3,
        help='Number of predicted-class prototype activations for lambda'
    )

    parser.add_argument(
        '--adaptive-controller',
        choices=[
            'absolute_distance',
            'relative_margin',
            'teacher_median',
            'teacher_batch_median',
            'coverage_router',
            'absolute_consistency_router',
            'ratio_only_router',
            'absolute_only_router',
            'forced_relative',
            'forced_native',
            'fixed_lambda_0.2',
        ],
        default='absolute_distance',
        help='Source-free samplewise lambda controller'
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
        default=None,
        help='Deprecated: paired metrics always use the complete cohort'
    )

    parser.add_argument(
        '--clean-reference-cache',
        type=str,
        default=None,
        help='Optional run-specific .pt cache for the complete clean trajectory'
    )

    parser.add_argument(
        '--stream-order',
        choices=['seeded_random', 'class_order'],
        default='seeded_random',
        help=(
            'Online sample order. seeded_random reproduces the historical '
            'fixed random SICAPv2-C stream; class_order is the separate '
            'label-correlated stress-test protocol.'
        )
    )

    parser.add_argument(
        '--stream-order-seed',
        type=int,
        default=0,
        help='Seed for the fixed label-independent online permutation'
    )

    parser.add_argument(
        '--controlled-cohort-name',
        type=str,
        default=None,
        help='Record and enforce the controlled SICAPv2-C cohort identity'
    )

    parser.add_argument(
        '--refuse-existing-output',
        action='store_true',
        help='Fail if --output already exists instead of resuming it'
    )

    parser.add_argument(
        '--clean-reference-only',
        action='store_true',
        help='Build/validate --clean-reference-cache and exit'
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

    if not 0.0 <= args.lambda_proto <= 1.0:
        parser.error('--lambda-proto must be in [0, 1]')
    if args.adaptive_delta0 <= 0:
        parser.error('--adaptive-delta0 must be positive')
    if args.adaptive_top_k <= 0:
        parser.error('--adaptive-top-k must be positive')
    if args.refuse_existing_output and Path(args.output).exists():
        parser.error(f'refusing to overwrite existing output: {args.output}')

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    set_random_seed(args.seed)
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

    if args.controlled_cohort_name:
        if args.severity != 5:
            parser.error('controlled SICAPv2-C jobs require severity 5')
        if set(corruption_types) != set(HISTOPATHOLOGY_CORRUPTIONS):
            parser.error(
                'controlled SICAPv2-C jobs require exactly the 13 registered '
                'corruptions'
            )

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
            'alpha_proto': args.lambda_proto,
            'alpha_softmax': 1.0 - args.lambda_proto
        },
        'ProtoSampleAdaptive': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': False,
            'adaptive_controller': 'absolute_distance',
        },
        'ProtoSampleAdaptiveGradNorm': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'absolute_distance',
        },
        'ProtoSampleAdaptiveRelative': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': False,
            'adaptive_controller': 'relative_margin',
        },
        'ProtoSampleAdaptiveRelativeGradNorm': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'relative_margin',
        },
        'ProtoSampleAdaptiveTeacherMedian': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': False,
            'adaptive_controller': 'teacher_median',
        },
        'ProtoSampleAdaptiveTeacherBatchMedian': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': False,
            'adaptive_controller': 'teacher_batch_median',
        },
        'ProtoSampleAdaptiveCoverageRouter': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'coverage_router',
        },
        'ProtoAuditForcedRelative': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': False,
            'adaptive_controller': 'forced_relative',
        },
        'ProtoAuditForcedRelativeGradNorm': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'forced_relative',
        },
        'ProtoAuditFixedLambda0.2GradNorm': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'fixed_lambda_0.2',
        },
        'ProtoAuditForcedNativeGradNorm': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'forced_native',
        },
        'ProtoAbsoluteConsistencyCoverageRouter': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'absolute_consistency_router',
        },
        'ProtoRatioOnlyRouter': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'ratio_only_router',
        },
        'ProtoAbsoluteOnlyRouter': {
            'geo_filter_threshold': 0.8,
            'delta0': args.adaptive_delta0,
            'top_k': args.adaptive_top_k,
            'component_gradient_normalization': True,
            'adaptive_controller': 'absolute_only_router',
        },
        'SAR': {},
        'MEMO': {},
    }

    if args.modes:
        modes = {k: v for k, v in all_modes.items() if k in args.modes}
        missing_modes = sorted(set(args.modes) - set(modes))
        if missing_modes:
            parser.error(f'unknown modes: {missing_modes}')
    elif args.samplewise_adaptive_lambda:
        if args.adaptive_controller == 'coverage_router':
            adaptive_mode = 'ProtoSampleAdaptiveCoverageRouter'
        elif args.adaptive_controller == 'teacher_batch_median':
            adaptive_mode = 'ProtoSampleAdaptiveTeacherBatchMedian'
        elif args.adaptive_controller == 'teacher_median':
            adaptive_mode = 'ProtoSampleAdaptiveTeacherMedian'
        elif args.adaptive_controller == 'relative_margin':
            adaptive_mode = (
                'ProtoSampleAdaptiveRelativeGradNorm'
                if args.component_gradient_normalization
                else 'ProtoSampleAdaptiveRelative'
            )
        else:
            adaptive_mode = (
                'ProtoSampleAdaptiveGradNorm'
                if args.component_gradient_normalization
                else 'ProtoSampleAdaptive'
            )
        modes = {adaptive_mode: all_modes[adaptive_mode]}
    else:
        legacy_mode_names = [
            'Normal', 'Tent', 'EATA', 'ProtoEntropy',
            'ProtoEntropy-BN-0.98', 'ProtoHybrid', 'SAR', 'MEMO',
        ]
        modes = {name: all_modes[name] for name in legacy_mode_names}

    severity = args.severity
    severity_key = str(severity)  # Use "5" not "severity_5" to match ProtoViT format
    batch_size = args.batch_size

    provenance = evaluator_provenance(args.model)
    if args.skip_clean_evaluation:
        if args.prototype_metrics:
            raise ValueError(
                '--prototype-metrics requires clean data; disable it for a '
                'strictly source-free run'
            )
        clean_reference = None
        clean_identity = None
        clean_accuracy = None
        print("\nSkipping clean-data evaluation (source-free protocol).")
    else:
        print("\n" + "="*80)
        print("UNADAPTED MODEL PERFORMANCE ON CLEAN DATA")
        print("="*80)
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(
            args.clean_data_dir, transform_clean
        )
        raw_clean_identity = dataset_identity(clean_dataset)
        clean_stream_indices = fixed_stream_indices(
            raw_clean_identity['num_samples'], args.stream_order,
            args.stream_order_seed
        )
        clean_identity = identity_in_stream_order(
            raw_clean_identity, clean_stream_indices
        )
        if (args.proto_baseline_samples is not None and
                args.proto_baseline_samples != clean_identity['num_samples']):
            raise ValueError(
                '--proto-baseline-samples cannot subset paired metrics: '
                f"requested {args.proto_baseline_samples}, controlled N is "
                f"{clean_identity['num_samples']}"
            )
        clean_reference = load_or_collect_clean_reference(
            args, device, clean_dataset, clean_identity, provenance
        )
        clean_accuracy = float(clean_reference['accuracy'])
        print(f"Clean accuracy (unadapted): {clean_accuracy*100:.2f}%")
        print(f"Paired samples: {clean_identity['num_samples']}")
        print(
            'Sample-ID SHA-256: '
            f"{clean_identity['sample_id_hash_sha256']}"
        )
        print("="*80 + "\n")

    if args.clean_reference_only:
        if args.skip_clean_evaluation or not args.clean_reference_cache:
            parser.error(
                '--clean-reference-only requires --clean-reference-cache and '
                'clean evaluation'
            )
        print('Clean-reference-only job complete.')
        return

    # Load existing results for resumability
    results_dict, _ = load_existing_results(args.output)

    # Initialize result structure
    for mode_name in modes:
        if mode_name not in results_dict:
            results_dict[mode_name] = {}

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
                clean_reference=clean_reference,
                clean_identity=clean_identity,
                provenance=provenance,
                compute_proto_metrics=args.prototype_metrics,
                track_efficiency=args.track_efficiency,
                seed=args.seed,
                stream_order=args.stream_order,
                stream_order_seed=args.stream_order_seed,
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
                'seed': args.seed,
                'corruption_types': corruption_types,
                'modes': list(modes.keys()),
                'mode_configs': modes,  # Save full mode configurations
                'loss_definition': {
                    'formula': 'lambda * L_proto + (1 - lambda) * L_out',
                    'lambda_proto': args.lambda_proto,
                    'lambda_output': 1.0 - args.lambda_proto,
                    'shared_reliable_sample_filter': True,
                    'confidence_weighting': False,
                },
                'samplewise_adaptive_lambda': {
                    'enabled': any(
                        config.get('adaptive_controller') is not None
                        for config in modes.values()
                    ),
                    'formula': (
                        'absolute_distance: clip(mean(abs(top_k_target_i'
                        '-0.5))/delta0,0,1); relative_margin: positive '
                        'prototype class margin and output probability margin '
                        'are normalized by their winning scores, then lambda '
                        'is their relative evidence ratio; teacher_median: '
                        'relative-margin lambda with frozen-teacher stability '
                        'and cumulative reliable-batch median fallback; '
                        'teacher_batch_median: use that adaptive median for '
                        'all reliable samples in the batch; coverage_router: '
                        'non-EMA batch choice between native activation and '
                        'relative-evidence controllers using gradient '
                        'consistency and reliable-coverage guard'
                    ),
                    'delta0': args.adaptive_delta0,
                    'top_k': args.adaptive_top_k,
                    'controllers': sorted({
                        config.get('adaptive_controller')
                        for config in modes.values()
                        if config.get('adaptive_controller') is not None
                    }),
                    'lambda_detached': True,
                    'component_gradient_normalization': any(
                        config.get('component_gradient_normalization', False)
                        for config in modes.values()
                    ),
                },
                'optimizer': {
                    'name': 'Adam',
                    'learning_rate': cfg_optim.LR,
                    'betas': [cfg_optim.BETA, 0.999],
                    'weight_decay': cfg_optim.WD,
                    'adaptation_steps': cfg_optim.STEPS,
                },
                'protocol': {
                    'episodic': False,
                    'continual_within_corruption': True,
                    'reset_boundary': 'fresh checkpoint per corruption and method',
                    'shuffle': args.stream_order == 'seeded_random',
                    'stream_order': args.stream_order,
                    'stream_order_seed': args.stream_order_seed,
                    'stream_order_algorithm': STREAM_ORDER_ALGORITHM,
                    'stream_order_index_hash_sha256': clean_identity[
                        'stream_order_index_hash_sha256'
                    ] if clean_identity else None,
                    'num_workers': 4,
                    'adaptation_mode': 'batchnorm_addon',
                    'initial_model_mode': 'eval',
                    'source_free': args.skip_clean_evaluation,
                    'clean_data_loaded': not args.skip_clean_evaluation,
                    'controlled_cohort_name': args.controlled_cohort_name,
                },
                'runtime': {
                    'command': [sys.executable, *sys.argv],
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'cuda_available': torch.cuda.is_available(),
                    'device': str(device),
                    'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                },
                'clean_accuracy_unadapted': clean_accuracy,
                'paired_num_samples': (
                    clean_identity['num_samples'] if clean_identity else None
                ),
                'sample_id_hash_sha256': (
                    clean_identity['sample_id_hash_sha256']
                    if clean_identity else None
                ),
                'canonical_sample_id_hash_sha256': (
                    clean_identity['canonical_sample_id_hash_sha256']
                    if clean_identity else None
                ),
                'label_hash_sha256': (
                    clean_identity['label_hash_sha256']
                    if clean_identity else None
                ),
                'model_checksum_sha256': provenance['model_checksum_sha256'],
                'git_commit': provenance['git_commit'],
                'evaluator_path': provenance['evaluator_path'],
                'evaluator_checksum_sha256': (
                    provenance['evaluator_checksum_sha256']
                ),
                'evaluator_diff_sha256': provenance['evaluator_diff_sha256'],
                'clean_reference_cache': args.clean_reference_cache,
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
