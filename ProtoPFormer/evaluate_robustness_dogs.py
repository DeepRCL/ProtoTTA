#!/usr/bin/env python3
"""Comprehensive robustness evaluation for ProtoPFormer on Stanford Dogs-C."""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import protopformer
from enhanced_prototype_metrics import EnhancedPrototypeMetrics
from memo_adapt import setup_memo
from noise_utils import CORRUPTION_TYPES, get_corrupted_transform
from proto_tta import compute_fishers, setup_eata, setup_proto_tta, setup_tent
from prototype_tta_metrics import PrototypeMetricsEvaluator
from sar_adapt import setup_sar

sys.path.insert(0, str(ROOT.parent))
from tta_baselines import CoTTA, CoTTAImageTransform
import proto_tta as proto_tta_module

_efficiency_spec = importlib.util.spec_from_file_location(
    "protovit_efficiency_metrics",
    ROOT.parent / 'ProtoViT' / 'efficiency_metrics.py'
)
_efficiency_module = importlib.util.module_from_spec(_efficiency_spec)
assert _efficiency_spec.loader is not None
_efficiency_spec.loader.exec_module(_efficiency_module)
EfficiencyTracker = _efficiency_module.EfficiencyTracker
compare_efficiency_metrics = _efficiency_module.compare_efficiency_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def seed_everything(seed):
    """Reset all RNGs before every method/corruption trajectory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_clean_root(clean_dir):
    path = Path(clean_dir)
    if (path / "Images").is_dir():
        return path / "Images"
    return path


def _build_test_subset_dataset(clean_dir, transform):
    """Return Stanford Dogs test subset in ImageFolder order when possible."""
    image_root = _resolve_clean_root(clean_dir)
    dataset = datasets.ImageFolder(str(image_root), transform)

    test_list_mat = image_root.parent / "test_list.mat"
    if not test_list_mat.exists():
        return dataset

    import scipy.io

    mat = scipy.io.loadmat(str(test_list_mat))
    file_list = mat["file_list"].squeeze()
    test_rel_paths = {str(f[0]).replace("\\", "/") for f in file_list}

    indices = [
        idx for idx, (sample_path, _) in enumerate(dataset.samples)
        if Path(sample_path).relative_to(image_root).as_posix() in test_rel_paths
    ]
    return torch.utils.data.Subset(dataset, indices)


def _loader_sample_ids(loader):
    """Return class-relative file names in exact DataLoader iteration order."""
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        indices = dataset.indices
    else:
        base = dataset
        indices = range(len(dataset))
    if not hasattr(base, 'samples') or not hasattr(base, 'root'):
        raise RuntimeError("Paired metrics require a file-backed dataset with stable sample IDs")
    root = Path(base.root)
    return [Path(base.samples[index][0]).relative_to(root).as_posix() for index in indices]


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError(f"Unexpected checkpoint format in {model_path}")

    args = ckpt['args']
    model = protopformer.construct_PPNet(
        base_architecture=args.base_architecture,
        pretrained=False,
        img_size=args.img_size,
        prototype_shape=args.prototype_shape,
        num_classes=args.nb_classes,
        reserve_layers=args.reserve_layers,
        reserve_token_nums=args.reserve_token_nums,
        use_global=args.use_global,
        use_ppc_loss=args.use_ppc_loss,
        ppc_cov_thresh=args.ppc_cov_thresh,
        ppc_mean_thresh=args.ppc_mean_thresh,
        global_coe=args.global_coe,
        global_proto_per_class=args.global_proto_per_class,
        prototype_activation_function=args.prototype_activation_function,
        add_on_layers_type=args.add_on_layers_type,
    )
    model.load_state_dict(ckpt['model'])
    logger.info("Loaded checkpoint from epoch %s", ckpt.get('epoch', '?'))
    return model.to(device).eval()


def build_clean_loader(clean_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = _build_test_subset_dataset(clean_dir, transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )


def load_corrupted_dataset(data_dir, corruption_type, severity, batch_size, num_workers=4):
    path = Path(data_dir) / corruption_type / str(severity)
    if not path.exists():
        raise FileNotFoundError(f"Corrupted dataset not found: {path}")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(str(path), transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )


def load_on_the_fly(clean_dir, corruption_type, severity, batch_size, num_workers=4):
    transform = get_corrupted_transform(
        IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, corruption_type, severity
    )
    dataset = _build_test_subset_dataset(clean_dir, transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )


def _output_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _find_matching_raw_output(raw_outputs, logits):
    for raw in reversed(raw_outputs):
        raw_logits = _output_logits(raw)
        if not isinstance(raw_logits, torch.Tensor):
            continue
        if raw_logits is logits:
            return raw
        if (raw_logits.shape == logits.shape and raw_logits.device == logits.device
                and raw_logits.data_ptr() == logits.data_ptr()):
            return raw
    raise RuntimeError(
        "Could not match returned predictions to a backbone forward; refusing to "
        "compute Table-3 metrics from a different checkpoint/pass"
    )


def evaluate_model(model, loader, device, description='Eval', verbose=True,
                   efficiency_tracker=None, proto_evaluator=None,
                   compute_proto_metrics=False):
    if verbose:
        print(f'\n{description}...')

    n_correct = 0
    n_total = 0
    actual_model = (
        model.metric_model if hasattr(model, 'metric_model')
        else model.model if hasattr(model, 'model') else model
    )
    raw_outputs = []
    hook = None
    if compute_proto_metrics and proto_evaluator is not None:
        hook = actual_model.register_forward_hook(
            lambda _module, _inputs, output: raw_outputs.append(output)
        )
    all_activations, all_logits, all_predictions, all_labels = [], [], [], []
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)
            raw_outputs.clear()

            context = efficiency_tracker.track_inference(batch_size) if efficiency_tracker else nullcontext()
            with context:
                outputs = model(images)
                logits = _output_logits(outputs)
                preds = logits.argmax(dim=1)

            n_correct += preds.eq(labels).sum().item()
            n_total += batch_size
            if compute_proto_metrics and proto_evaluator is not None:
                raw = getattr(model, 'last_metric_output', None)
                if raw is None:
                    raw = _find_matching_raw_output(raw_outputs, logits)
                elif _output_logits(raw).data_ptr() != logits.data_ptr():
                    raise RuntimeError('CoTTA metric output does not match returned logits')
                activations = proto_evaluator._extract_prototype_activations(model, raw)
                all_activations.append(activations.detach().cpu())
                all_logits.append(logits.detach().cpu())
                all_predictions.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
    finally:
        if hook is not None:
            hook.remove()

    accuracy = n_correct / max(n_total, 1)
    if verbose:
        print(f'Accuracy: {accuracy*100:.2f}%')
    proto_metrics = {}
    if compute_proto_metrics and proto_evaluator is not None:
        activations = torch.cat(all_activations)
        labels = torch.cat(all_labels)
        if isinstance(proto_evaluator, EnhancedPrototypeMetrics):
            proto_metrics = proto_evaluator.evaluate_collected_outputs(
                activations, torch.cat(all_logits),
                torch.cat(all_predictions), labels, top_k=10
            )
        else:
            proto_metrics.update(proto_evaluator.compute_prototype_activation_consistency(activations))
            proto_metrics.update(proto_evaluator.compute_prototype_class_alignment(activations, labels, top_k=10))
            proto_metrics.update(proto_evaluator.compute_prototype_activation_sparsity(activations))
    return accuracy, proto_metrics


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


def setup_method(model, mode_name, mode_config, device, model_path, loader, fishers=None):
    if mode_name == 'normal':
        return model
    if mode_name == 'memo':
        return setup_memo(
            model,
            lr=mode_config.get('lr', 0.00025),
            batch_size=mode_config.get('batch_size', 16),
            steps=mode_config.get('steps', 1),
        )
    if mode_name == 'tent':
        return setup_tent(
            model, lr=mode_config.get('lr', 1e-3), steps=mode_config.get('steps', 1),
            model_mode=mode_config.get('model_mode', 'train'),
        )
    if mode_name == 'cotta':
        adaptation_mode = mode_config.get('adaptation_mode', 'layernorm_attn_bias')
        configured = proto_tta_module.configure_model(
            model, adaptation_mode=adaptation_mode,
            model_mode=mode_config.get('model_mode', 'train'),
        )
        params, _ = proto_tta_module.collect_params(configured, adaptation_mode)
        optimizer = torch.optim.Adam(params, lr=mode_config.get('lr', 1e-3))
        return CoTTA(
            configured, optimizer,
            CoTTAImageTransform(IMAGENET_MEAN, IMAGENET_STD, image_size=IMG_SIZE),
            steps=mode_config.get('steps', 1), mt_alpha=0.999,
            rst_m=0.001, ap=0.1, n_augmentations=32,
        )
    if mode_name == 'eata':
        current_fishers = fishers
        if current_fishers is None:
            fisher_model = load_model(model_path, device)
            current_fishers = compute_fishers(fisher_model, loader, device, num_samples=500)
            del fisher_model
            torch.cuda.empty_cache()
        return setup_eata(
            model, fishers=current_fishers, lr=mode_config.get('lr', 1e-3),
            steps=mode_config.get('steps', 1),
            model_mode=mode_config.get('model_mode', 'train'),
        )
    if mode_name == 'sar':
        return setup_sar(
            model,
            lr=mode_config.get('lr', 1e-4),
            steps=mode_config.get('steps', 1),
            margin_e0=mode_config.get('margin_e0'),
            reset_constant_em=mode_config.get('reset_constant_em', 0.2),
            rho=mode_config.get('rho', 0.05),
            model_mode=mode_config.get('model_mode', 'train'),
        )
    if mode_name.startswith('proto_tta'):
        return setup_proto_tta(
            model,
            lr=mode_config.get('lr', 1e-3),
            steps=mode_config.get('steps', 1),
            episodic=mode_config.get('episodic', False),
            use_importance=mode_config.get('use_importance', True),
            use_confidence=mode_config.get('use_confidence', True),
            adapt_all_prototypes=mode_config.get('adapt_all_prototypes', False),
            use_geometric_filter=mode_config.get('use_geometric_filter', True),
            geo_filter_threshold=mode_config.get('geo_filter_threshold', 0.30),
            consensus_strategy=mode_config.get('consensus_strategy', 'max'),
            consensus_ratio=mode_config.get('consensus_ratio', 0.5),
            adaptation_mode=mode_config.get('adaptation_mode', 'layernorm_attn_bias'),
            use_branch_agreement=mode_config.get('use_branch_agreement', False),
            prototype_branch=mode_config.get('prototype_branch', 'both'),
            similarity_mapping=mode_config.get('similarity_mapping', 'sigmoid'),
            sigmoid_center=mode_config.get('sigmoid_center', 1.0),
            sigmoid_temp=mode_config.get('sigmoid_temp', 1.0),
            proto_weight=mode_config.get('proto_weight', 1.0),
            logit_weight=mode_config.get('logit_weight', 0.0),
            shared_confidence_weighting=mode_config.get('shared_confidence_weighting', False),
            gradient_normalize=mode_config.get('gradient_normalize', False),
            adaptive_lambda=mode_config.get('adaptive_lambda', False),
            adaptive_lambda_strategy=mode_config.get('adaptive_lambda_strategy', 'relative_reliability'),
            adaptive_delta0=mode_config.get('adaptive_delta0', 0.25),
            adaptive_topk=mode_config.get('adaptive_topk', 3),
            router_min_consistency=mode_config.get('router_min_consistency', 0.25),
            lambda_ema_momentum=mode_config.get('lambda_ema_momentum', 0.9),
            lambda_min=mode_config.get('lambda_min', 0.05),
            lambda_max=mode_config.get('lambda_max', 0.95),
            record_diagnostics=mode_config.get('record_diagnostics', False),
            lambda_search=mode_config.get('lambda_search', False),
            lambda_search_radius=mode_config.get('lambda_search_radius', 0.1),
            lambda_search_teacher_temp=mode_config.get('lambda_search_teacher_temp', 0.5),
            lambda_search_min_improvement=mode_config.get('lambda_search_min_improvement', 0.0),
            samplewise_lambda=mode_config.get('samplewise_lambda', False),
            adaptive_branch_weighting=mode_config.get('adaptive_branch_weighting', False),
            branch_weight_floor=mode_config.get('branch_weight_floor', 0.1),
            model_mode=mode_config.get('model_mode', 'train'),
            reset_mode=mode_config.get('reset_mode', None),
            reset_frequency=mode_config.get('reset_frequency', 10),
            confidence_threshold=mode_config.get('confidence_threshold', 0.7),
            ema_alpha=mode_config.get('ema_alpha', 0.999),
        )
    raise ValueError(f'Unknown mode: {mode_name}')


def evaluate_single_combination(
    model_path,
    corruption_type,
    severity,
    data_dir,
    clean_dir,
    on_the_fly,
    mode_name,
    mode_config,
    device,
    batch_size,
    num_workers,
    fishers=None,
    proto_evaluator=None,
    compute_proto_metrics=False,
    track_efficiency=False,
    seed=0,
):
    efficiency_tracker = EfficiencyTracker(mode_name, device=str(device)) if track_efficiency else None

    try:
        seed_everything(seed)
        # MEMO is episodic and adapts one test image at a time. Its independent
        # ``batch_size`` setting is the number of augmented views per image.
        loader_batch_size = 1 if mode_name == 'memo' else batch_size
        loader = load_on_the_fly(
            clean_dir, corruption_type, severity, loader_batch_size, num_workers
        ) if on_the_fly else load_corrupted_dataset(
            data_dir, corruption_type, severity, loader_batch_size, num_workers
        )
        if compute_proto_metrics and proto_evaluator is not None:
            clean_ids = getattr(proto_evaluator, 'clean_sample_ids', None)
            corrupt_ids = _loader_sample_ids(loader)
            if clean_ids is None:
                raise RuntimeError("Clean sample IDs were not recorded for paired metrics")
            if clean_ids != corrupt_ids:
                mismatch = next(
                    (i for i, pair in enumerate(zip(clean_ids, corrupt_ids)) if pair[0] != pair[1]),
                    min(len(clean_ids), len(corrupt_ids)),
                )
                raise RuntimeError(
                    "Clean/corrupted sample IDs are not paired: "
                    f"clean_n={len(clean_ids)}, corrupt_n={len(corrupt_ids)}, "
                    f"first_mismatch={mismatch}"
                )

        base_model = load_model(model_path, device)
        eval_model = setup_method(base_model, mode_name, mode_config, device, model_path, loader, fishers=fishers)

        if efficiency_tracker:
            actual_model = eval_model.model if hasattr(eval_model, 'model') else eval_model
            adapted_params = [] if mode_name == 'normal' else [p for p in actual_model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(actual_model, adapted_params)

        accuracy, proto_metrics = evaluate_model(
            eval_model, loader, device, description=f'{mode_name} / {corruption_type}-{severity}',
            verbose=False, efficiency_tracker=efficiency_tracker,
            proto_evaluator=proto_evaluator,
            compute_proto_metrics=compute_proto_metrics,
        )

        if efficiency_tracker and mode_name != 'normal':
            efficiency_tracker.record_adaptation_step(len(loader) * mode_config.get('steps', 1))

        result = {'accuracy': float(accuracy)}
        if compute_proto_metrics and proto_evaluator is not None:
            keys = [
                'PAC_mean', 'PAC_std', 'PCA_mean', 'PCA_std',
                'sparsity_gini_mean', 'sparsity_active_mean',
                'PCA_weighted_mean', 'PCA_weighted_std',
                'calibration_agreement', 'calibration_logit_corr',
                'gt_class_contrib_improvement', 'gt_class_contrib_change_mean',
                'adaptation_rate', 'avg_updates_per_sample',
                'paired_num_samples', 'clean_accuracy_reference',
                'stability_bound_lower', 'stability_bound_upper',
                'stability_bounds_passed',
            ]
            for key in keys:
                if key in proto_metrics:
                    result[key] = proto_metrics[key]

        if hasattr(eval_model, 'adaptation_stats'):
            result['adaptation_stats'] = eval_model.adaptation_stats.copy()

        if efficiency_tracker:
            result['efficiency'] = efficiency_tracker.get_metrics()

        return result
    except Exception as exc:
        logger.error("FAILED %s / %s-%s: %s", mode_name, corruption_type, severity, exc)
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        torch.cuda.empty_cache()


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, data, metadata=None):
    obj = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'results': data,
    }
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def build_run_config(args, selected_modes, corruptions, clean_dir):
    """Return the accuracy-affecting configuration used for safe resumption."""
    return {
        'model': os.path.abspath(args.model),
        'data_dir': os.path.abspath(args.data_dir),
        'clean_dir': os.path.abspath(clean_dir),
        'on_the_fly': args.on_the_fly,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'adapt_model_mode': args.adapt_model_mode,
        'num_workers': args.num_workers,
        'severity': args.severity,
        'modes': selected_modes,
        'corruptions': corruptions,
        'lr': args.lr,
        'steps': args.steps,
        'use_global_fisher': args.use_global_fisher,
        'sar_lr': args.sar_lr,
        'sar_margin': args.sar_margin,
        'sar_reset': args.sar_reset,
        'sar_rho': args.sar_rho,
        'memo_lr': args.memo_lr,
        'memo_views': args.memo_views,
        'memo_steps': args.memo_steps,
        'proto_threshold': args.proto_threshold,
        'proto_mapping': args.proto_mapping,
        'proto_sigmoid_center': args.proto_sigmoid_center,
        'proto_sigmoid_temp': args.proto_sigmoid_temp,
        'proto_branch': args.proto_branch,
        'proto_use_importance': not args.proto_no_importance,
        'proto_branch_agreement': args.proto_branch_agreement,
        'proto_all_prototypes': args.proto_all_prototypes,
        'proto_lambda': args.proto_lambda,
        'proto_shared_confidence_weighting': args.proto_shared_confidence_weighting,
        'proto_gradient_normalize': args.proto_gradient_normalize,
        'proto_lambda_ema_momentum': args.proto_lambda_ema_momentum,
        'proto_adaptive_strategy': args.proto_adaptive_strategy,
        'proto_adaptive_delta0': args.proto_adaptive_delta0,
        'proto_adaptive_topk': args.proto_adaptive_topk,
        'proto_lambda_min': args.proto_lambda_min,
        'proto_lambda_max': args.proto_lambda_max,
        'proto_record_diagnostics': args.proto_record_diagnostics,
        'proto_lambda_search_radius': args.proto_lambda_search_radius,
        'proto_lambda_search_teacher_temp': args.proto_lambda_search_teacher_temp,
        'proto_lambda_search_min_improvement': args.proto_lambda_search_min_improvement,
        'proto_branch_weight_floor': args.proto_branch_weight_floor,
        'prototype_metrics': args.prototype_metrics,
        'proto_baseline_samples': args.proto_baseline_samples,
        'use_enhanced_metrics': args.use_enhanced_metrics,
        'track_efficiency': args.track_efficiency,
    }


def validate_resume(existing, run_config, output_path, overwrite=False):
    """Resume only when the existing JSON was produced by the same config."""
    if not existing or overwrite:
        return {}

    previous_config = existing.get('metadata', {}).get('run_config')
    if previous_config is None:
        raise ValueError(
            f"Refusing to reuse {output_path}: it has no run_config provenance. "
            "Use a new output path or pass --overwrite intentionally."
        )

    if previous_config != run_config:
        changed = sorted(
            key for key in set(previous_config) | set(run_config)
            if previous_config.get(key) != run_config.get(key)
        )
        details = ', '.join(
            f"{key}: {previous_config.get(key)!r} -> {run_config.get(key)!r}"
            for key in changed
        )
        raise ValueError(
            f"Refusing to reuse {output_path} with a different configuration "
            f"({details}). Use a new output path or pass --overwrite intentionally."
        )

    return existing.get('results', {})


def summarize_metric(results, modes, corruptions, severity_key, metric):
    values = {mode: [] for mode in modes}
    for mode in modes:
        for corruption in corruptions:
            entry = results.get(mode, {}).get(corruption, {}).get(severity_key)
            if isinstance(entry, dict) and metric in entry and entry[metric] is not None:
                values[mode].append(entry[metric])
    return {mode: float(np.mean(vals)) for mode, vals in values.items() if vals}


def main():
    parser = argparse.ArgumentParser(description='Comprehensive robustness evaluation on Stanford Dogs-C')
    parser.add_argument('--model', required=True, help='Path to trained ProtoPFormer checkpoint')
    parser.add_argument('--data_dir', default='datasets/stanford_dogs_c', help='Pre-generated Dogs-C directory')
    parser.add_argument('--clean_dir', default=None, help='Clean Dogs test directory')
    parser.add_argument('--on_the_fly', action='store_true', help='Generate corruptions on the fly')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adapt_model_mode', choices=['train', 'eval'], default='train',
                        help='train matches upstream Tent and historical paper runs; '
                             'eval disables DropPath/dropout as a deterministic ablation')
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--output', default='robustness_results_dogs.json', help='Output JSON path')
    parser.add_argument('--overwrite', action='store_true',
                        help='Discard an existing output JSON instead of resuming it')
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--use_global_fisher', action='store_true', help='Use one clean Fisher for all EATA runs')
    parser.add_argument('--prototype-metrics', action='store_true', default=False,
                        help='Compute prototype-based metrics (PAC, PCA, Sparsity). Requires clean baseline.')
    parser.add_argument('--proto-baseline-samples', type=int, default=1000,
                        help='Number of clean samples to use for prototype baseline')
    parser.add_argument('--track-efficiency', action='store_true', default=False,
                        help='Track and report computational efficiency metrics (timing, adapted parameters, etc.)')
    parser.add_argument('--use-enhanced-metrics', action='store_true', default=False,
                        help='Use enhanced prototype metrics (PCA-Weighted, Calibration, GT Class Contribution).')
    parser.add_argument('--modes', nargs='+', default=[
        'normal', 'tent', 'eata', 'sar', 'cotta',
        'proto_tta', 'proto_tta_plus_7030', 'proto_tta_plus_7525', 'proto_tta_plus_8020',
    ])
    parser.add_argument('--corruptions', nargs='+', default=['all'], help='"all" or specific corruption names')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument(
        '--sar-lr', type=float, default=1e-4,
        help='Learning rate for SAR only (SGD inside SAM). Default 1e-4 is gentler on ViT LayerNorms than 1e-3.',
    )
    parser.add_argument(
        '--sar-margin', type=float, default=None,
        help='SAR reliable-sample entropy threshold; default is 0.4*log(num_classes).',
    )
    parser.add_argument('--sar-reset', type=float, default=0.2, help='EMA threshold for SAR model recovery reset.')
    parser.add_argument('--sar-rho', type=float, default=0.05, help='SAM perturbation radius rho for SAR.')
    parser.add_argument('--memo-lr', type=float, default=0.00025,
                        help='MEMO SGD learning rate (ProtoViT baseline: 2.5e-4)')
    parser.add_argument('--memo-views', type=int, default=16,
                        help='Number of AugMix views per MEMO test sample')
    parser.add_argument('--memo-steps', type=int, default=1,
                        help='Episodic MEMO update steps per test sample')
    parser.add_argument('--proto_threshold', type=float, default=0.62,
                        help='Geometric threshold for all ProtoTTA variants')
    parser.add_argument('--proto_mapping', type=str, default='sigmoid', choices=['sigmoid', 'linear'],
                        help='Similarity mapping for all ProtoTTA variants')
    parser.add_argument('--proto_sigmoid_center', type=float, default=1.0,
                        help='Sigmoid center for ProtoTTA similarity mapping')
    parser.add_argument('--proto_sigmoid_temp', type=float, default=1.0,
                        help='Sigmoid temperature for ProtoTTA similarity mapping')
    parser.add_argument('--proto_branch', type=str, default='both', choices=['local', 'global', 'both'],
                        help='Prototype branch used by all ProtoTTA variants')
    parser.add_argument('--proto_no_importance', action='store_true', default=False,
                        help='Disable prototype importance weighting for all ProtoTTA variants')
    parser.add_argument('--proto_branch_agreement', action='store_true', default=False,
                        help='Require branch agreement for all ProtoTTA variants')
    parser.add_argument('--proto_all_prototypes', action='store_true', default=False,
                        help='Adapt all prototypes for all ProtoTTA variants')
    parser.add_argument('--proto_lambda', type=float, default=1.0,
                        help='Unified ProtoTTA interpolation λ ∈ [0,1]: '
                             '1.0 = pure prototype entropy (ProtoTTA), '
                             '0.0 = pure logit entropy (Tent-style), '
                             '0.7 = ProtoTTA+ default. '
                             'Loss = λ*proto_loss + (1-λ)*logit_entropy. (default: 1.0)')
    parser.add_argument('--proto_shared_confidence_weighting', action='store_true',
                        help='Apply the same confidence weight to prototype and output losses')
    parser.add_argument('--proto_gradient_normalize', action='store_true',
                        help='Normalize each loss by its gradient norm before interpolation')
    parser.add_argument('--proto_lambda_ema_momentum', type=float, default=0.9)
    parser.add_argument('--proto_adaptive_strategy', default='relative_reliability',
                        choices=['relative_reliability', 'activation_margin',
                                 'relative_evidence', 'gradient_consistency',
                                 'source_free_router',
                                 'source_free_router_absolute',
                                 'source_free_router_evidence',
                                 'source_free_router_coverage',
                                 'source_free_router_coverage_absolute',
                                 'source_free_router_coverage_ema'])
    parser.add_argument('--proto_adaptive_delta0', type=float, default=0.25)
    parser.add_argument('--proto_adaptive_topk', type=int, default=3)
    parser.add_argument('--proto_router_min_consistency', type=float, default=0.25,
                        help='Absolute [0,1] prototype-gradient consistency floor for the absolute router')
    parser.add_argument('--proto_lambda_min', type=float, default=0.05)
    parser.add_argument('--proto_lambda_max', type=float, default=0.95)
    parser.add_argument('--proto_record_diagnostics', action='store_true',
                        help='Save per-batch component losses, gradient norms, reliabilities, and lambda')
    parser.add_argument('--proto_lambda_search_radius', type=float, default=0.1,
                        help='Radius for adaptive candidates lambda_hat +/- radius')
    parser.add_argument('--proto_lambda_search_teacher_temp', type=float, default=0.5,
                        help='Temperature used to sharpen the frozen teacher in label-free search')
    parser.add_argument('--proto_lambda_search_min_improvement', type=float, default=0.0,
                        help='Minimum held-out consistency improvement required to commit an update')
    parser.add_argument('--proto_branch_weight_floor', type=float, default=0.1,
                        help='Minimum local/global weight in adaptive branch experiments')
    args = parser.parse_args()

    if not 0.0 <= args.proto_lambda <= 1.0:
        parser.error('--proto_lambda must be between 0.0 and 1.0')
    if not 0.0 <= args.proto_lambda_min <= args.proto_lambda_max <= 1.0:
        parser.error('Require 0 <= --proto_lambda_min <= --proto_lambda_max <= 1')
    if args.proto_adaptive_delta0 <= 0 or args.proto_adaptive_topk < 1:
        parser.error('--proto_adaptive_delta0 must be > 0 and --proto_adaptive_topk >= 1')
    if not 0.0 <= args.proto_router_min_consistency <= 1.0:
        parser.error('--proto_router_min_consistency must be in [0,1]')
    if args.proto_lambda_search_radius < 0:
        parser.error('--proto_lambda_search_radius must be non-negative')
    if args.proto_lambda_search_teacher_temp <= 0:
        parser.error('--proto_lambda_search_teacher_temp must be positive')
    if args.proto_lambda_search_min_improvement < 0:
        parser.error('--proto_lambda_search_min_improvement must be non-negative')
    if not 0.0 <= args.proto_branch_weight_floor <= 0.5:
        parser.error('--proto_branch_weight_floor must be in [0, 0.5]')
    if args.memo_lr <= 0 or args.memo_views < 1 or args.memo_steps < 1:
        parser.error('MEMO requires positive lr, views, and steps')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    clean_dir = args.clean_dir or str(Path(args.data_dir).parent / 'stanford_dogs' / 'Images')
    corruptions = CORRUPTION_TYPES if 'all' in args.corruptions else args.corruptions
    severity_key = str(args.severity)

    proto_common = {
        'lr': args.lr, 'steps': args.steps,
        'use_importance': not args.proto_no_importance, 'use_confidence': True,
        'adapt_all_prototypes': args.proto_all_prototypes,
        'use_geometric_filter': True, 'geo_filter_threshold': args.proto_threshold,
        'consensus_strategy': 'max', 'consensus_ratio': 0.5,
        'adaptation_mode': 'layernorm_attn_bias',
        'use_branch_agreement': args.proto_branch_agreement,
        'prototype_branch': args.proto_branch, 'similarity_mapping': args.proto_mapping,
        'sigmoid_center': args.proto_sigmoid_center, 'sigmoid_temp': args.proto_sigmoid_temp,
        'shared_confidence_weighting': args.proto_shared_confidence_weighting,
        'gradient_normalize': args.proto_gradient_normalize,
        'lambda_ema_momentum': args.proto_lambda_ema_momentum,
        'adaptive_lambda_strategy': args.proto_adaptive_strategy,
        'adaptive_delta0': args.proto_adaptive_delta0,
        'adaptive_topk': args.proto_adaptive_topk,
        'router_min_consistency': args.proto_router_min_consistency,
        'lambda_min': args.proto_lambda_min, 'lambda_max': args.proto_lambda_max,
        'record_diagnostics': args.proto_record_diagnostics,
        'lambda_search_radius': args.proto_lambda_search_radius,
        'lambda_search_teacher_temp': args.proto_lambda_search_teacher_temp,
        'lambda_search_min_improvement': args.proto_lambda_search_min_improvement,
        'branch_weight_floor': args.proto_branch_weight_floor,
        'model_mode': args.adapt_model_mode,
    }
    modes = {
        'normal': {},
        'memo': {
            'lr': args.memo_lr,
            'batch_size': args.memo_views,
            'steps': args.memo_steps,
        },
        'tent': {'lr': args.lr, 'steps': args.steps, 'model_mode': args.adapt_model_mode},
        'cotta': {
            'lr': args.lr, 'steps': args.steps,
            'model_mode': args.adapt_model_mode,
            'adaptation_mode': 'layernorm_attn_bias',
        },
        'eata': {'lr': args.lr, 'steps': args.steps, 'model_mode': args.adapt_model_mode},
        'sar': {
            'lr': args.sar_lr,
            'steps': args.steps,
            'margin_e0': args.sar_margin,
            'reset_constant_em': args.sar_reset,
            'rho': args.sar_rho,
            'model_mode': args.adapt_model_mode,
        },
        'proto_tta': {
            **proto_common,
            'proto_weight': args.proto_lambda, 'logit_weight': 1.0 - args.proto_lambda,
        },
        'proto_tta_plus_7030': {
            **proto_common,
            # Express this identically to --proto_lambda 0.7 so the two aliases
            # are bit-for-bit the same Python configuration as well as the
            # same mathematical objective.
            'proto_weight': 0.7, 'logit_weight': 1.0 - 0.7,
        },
        'proto_tta_plus_7525': {
            **proto_common,
            'proto_weight': 0.75, 'logit_weight': 0.25,
        },
        'proto_tta_plus_8020': {
            **proto_common,
            'proto_weight': 0.8, 'logit_weight': 0.2,
        },
        'proto_tta_adaptive': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
        },
        'proto_tta_adaptive_search': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'lambda_search': True,
        },
        'proto_tta_adaptive_samplewise': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'samplewise_lambda': True,
        },
        'proto_tta_adaptive_relative_evidence': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'relative_evidence',
            'samplewise_lambda': True,
        },
        'proto_tta_adaptive_gradient_consistency': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'gradient_consistency',
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router_absolute': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router_absolute',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router_evidence': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router_evidence',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router_coverage': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router_coverage',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router_coverage_absolute': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router_coverage_absolute',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_source_free_router_coverage_ema': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_lambda_strategy': 'source_free_router_coverage_ema',
            'samplewise_lambda': True,
            'gradient_normalize': True,
        },
        'proto_tta_adaptive_branch': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'adaptive_branch_weighting': True,
        },
        'proto_tta_adaptive_samplewise_branch': {
            **proto_common,
            'proto_weight': 0.5, 'logit_weight': 0.5,
            'adaptive_lambda': True,
            'samplewise_lambda': True,
            'adaptive_branch_weighting': True,
        },
    }

    if abs(args.proto_lambda - 0.7) < 1e-12:
        if modes['proto_tta'] != modes['proto_tta_plus_7030']:
            raise AssertionError('proto_tta at lambda=0.7 is not identical to ProtoTTA+ 70/30')
        logger.info('Verified: proto_tta(lambda=0.7) and proto_tta_plus_7030 configs are identical')

    unknown_modes = [m for m in args.modes if m not in modes]
    if unknown_modes:
        parser.error(f"Unknown modes: {', '.join(unknown_modes)}")
    selected_modes = list(args.modes)
    run_config = build_run_config(args, selected_modes, corruptions, clean_dir)
    existing = load_json(args.output)
    results = validate_resume(
        existing, run_config, args.output, overwrite=args.overwrite
    )

    for mode in selected_modes:
        results.setdefault(mode, {})
        for corruption in corruptions:
            results[mode].setdefault(corruption, {})
            results[mode][corruption].setdefault(severity_key, None)

    proto_evaluator = None
    if args.prototype_metrics:
        clean_loader = build_clean_loader(clean_dir, args.batch_size, args.num_workers)
        base_model = load_model(args.model, device)
        proto_evaluator = EnhancedPrototypeMetrics(base_model, device=str(device)) \
            if args.use_enhanced_metrics else PrototypeMetricsEvaluator(base_model, device=str(device))
        if args.use_enhanced_metrics:
            proto_evaluator.collect_clean_baseline_enhanced(clean_loader, max_samples=args.proto_baseline_samples, verbose=True)
        else:
            proto_evaluator.collect_clean_baseline(clean_loader, max_samples=args.proto_baseline_samples, verbose=True)
        proto_evaluator.clean_sample_ids = _loader_sample_ids(clean_loader)
        del base_model
        torch.cuda.empty_cache()

    global_fishers = None
    if 'eata' in selected_modes and args.use_global_fisher:
        base_model = load_model(args.model, device)
        clean_loader = build_clean_loader(clean_dir, 32, args.num_workers)
        global_fishers = compute_fishers(base_model, clean_loader, device, num_samples=500)
        del base_model
        torch.cuda.empty_cache()

    pending = [
        (mode, corruption)
        for mode in selected_modes for corruption in corruptions
        if results[mode][corruption].get(severity_key) is None
    ]

    print("=" * 80)
    print("ProtoPFormer Robustness Evaluation — Stanford Dogs-C")
    print("=" * 80)
    print(f"Model      : {args.model}")
    print(f"Severity   : {args.severity}")
    print(f"Corruptions: {len(corruptions)}")
    print(f"Methods    : {selected_modes}")
    print(f"Output     : {args.output}")
    print("=" * 80)

    start = time.time()
    for mode_name, corruption_type in tqdm(pending, desc='Evaluating', unit='combo'):
        result = evaluate_single_combination(
            model_path=args.model,
            corruption_type=corruption_type,
            severity=args.severity,
            data_dir=args.data_dir,
            clean_dir=clean_dir,
            on_the_fly=args.on_the_fly,
            mode_name=mode_name,
            mode_config=modes[mode_name],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fishers=global_fishers if mode_name == 'eata' else None,
            proto_evaluator=proto_evaluator,
            compute_proto_metrics=args.prototype_metrics,
            track_efficiency=args.track_efficiency,
            seed=args.seed,
        )
        results[mode_name][corruption_type][severity_key] = result
        save_json(args.output, results, metadata={
            'model': args.model,
            'severity': args.severity,
            'modes': selected_modes,
            'corruptions': corruptions,
            'track_efficiency': args.track_efficiency,
            'use_enhanced_metrics': args.use_enhanced_metrics,
            'prototype_metrics': args.prototype_metrics,
            'run_config': run_config,
        })

    print(f"\nTotal evaluation time: {(time.time() - start)/60:.1f} min")
    acc_summary = summarize_metric(results, selected_modes, corruptions, severity_key, 'accuracy')
    print("\nAccuracy summary:")
    for mode, value in acc_summary.items():
        print(f"  {mode:<22} {value*100:.2f}%")

    if args.prototype_metrics:
        pac_summary = summarize_metric(results, selected_modes, corruptions, severity_key, 'PAC_mean')
        pca_summary = summarize_metric(results, selected_modes, corruptions, severity_key, 'PCA_mean')
        if pac_summary:
            print("\nPrototype metrics summary:")
            for mode in selected_modes:
                pac = pac_summary.get(mode)
                pca = pca_summary.get(mode)
                if pac is not None or pca is not None:
                    print(f"  {mode:<22} PAC={pac*100:.2f}%  PCA={pca*100:.2f}%")

    if args.track_efficiency:
        trackers = {}
        adaptation_stats = {}
        for mode in selected_modes:
            mode_entries = [
                results[mode][corruption][severity_key]
                for corruption in corruptions
                if isinstance(results[mode][corruption][severity_key], dict)
            ]
            if not mode_entries:
                continue
            eff = mode_entries[0].get('efficiency')
            if eff:
                tracker = EfficiencyTracker(mode, device=str(device))
                tracker.total_time = eff.get('total_time_sec', 0.0)
                tracker.num_samples = eff.get('num_samples', 0)
                tracker.batch_times = [eff.get('avg_batch_time_ms', 0.0) / 1000.0]
                tracker.num_adapted_params = eff.get('num_adapted_params', 0)
                tracker.total_params = eff.get('total_params', 0)
                tracker.num_adaptation_steps = eff.get('num_adaptation_steps', 0)
                tracker.total_optimizer_steps = eff.get('total_optimizer_steps', 0)
                trackers[mode] = tracker
            if 'adaptation_stats' in mode_entries[0]:
                adaptation_stats[mode] = mode_entries[0]['adaptation_stats']
        if trackers:
            comparison = compare_efficiency_metrics(trackers, baseline_method='normal')
            print("\nEfficiency summary:")
            for mode, metrics in comparison.items():
                print(f"  {mode:<22} {metrics['time_per_sample_ms']:.2f} ms/sample")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
