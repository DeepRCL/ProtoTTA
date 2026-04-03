#!/usr/bin/env python3
"""Comprehensive robustness evaluation for ProtoPFormer on Stanford Dogs-C."""

import argparse
import importlib.util
import json
import logging
import os
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
from noise_utils import CORRUPTION_TYPES, get_corrupted_transform
from proto_tta import compute_fishers, setup_eata, setup_proto_tta, setup_tent
from prototype_tta_metrics import PrototypeMetricsEvaluator
from sar_adapt import setup_sar

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

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


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


def evaluate_model(model, loader, device, description='Eval', verbose=True, efficiency_tracker=None):
    if verbose:
        print(f'\n{description}...')

    n_correct = 0
    n_total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        context = efficiency_tracker.track_inference(batch_size) if efficiency_tracker else nullcontext()
        with context:
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)

        n_correct += preds.eq(labels).sum().item()
        n_total += batch_size

    accuracy = n_correct / max(n_total, 1)
    if verbose:
        print(f'Accuracy: {accuracy*100:.2f}%')
    return accuracy


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


def setup_method(model, mode_name, mode_config, device, model_path, loader, fishers=None):
    if mode_name == 'normal':
        return model
    if mode_name == 'tent':
        return setup_tent(model, lr=mode_config.get('lr', 1e-3), steps=mode_config.get('steps', 1))
    if mode_name == 'eata':
        current_fishers = fishers
        if current_fishers is None:
            fisher_model = load_model(model_path, device)
            current_fishers = compute_fishers(fisher_model, loader, device, num_samples=500)
            del fisher_model
            torch.cuda.empty_cache()
        return setup_eata(model, fishers=current_fishers, lr=mode_config.get('lr', 1e-3), steps=mode_config.get('steps', 1))
    if mode_name == 'sar':
        return setup_sar(
            model,
            lr=mode_config.get('lr', 1e-4),
            steps=mode_config.get('steps', 1),
            margin_e0=mode_config.get('margin_e0'),
            reset_constant_em=mode_config.get('reset_constant_em', 0.2),
            rho=mode_config.get('rho', 0.05),
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
):
    efficiency_tracker = EfficiencyTracker(mode_name, device=str(device)) if track_efficiency else None

    try:
        loader = load_on_the_fly(clean_dir, corruption_type, severity, batch_size, num_workers) \
            if on_the_fly else load_corrupted_dataset(data_dir, corruption_type, severity, batch_size, num_workers)

        base_model = load_model(model_path, device)
        eval_model = setup_method(base_model, mode_name, mode_config, device, model_path, loader, fishers=fishers)

        if efficiency_tracker:
            actual_model = eval_model.model if hasattr(eval_model, 'model') else eval_model
            adapted_params = [] if mode_name == 'normal' else [p for p in actual_model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(actual_model, adapted_params)

        accuracy = evaluate_model(
            eval_model, loader, device, description=f'{mode_name} / {corruption_type}-{severity}',
            verbose=False, efficiency_tracker=efficiency_tracker
        )

        if efficiency_tracker and mode_name != 'normal':
            efficiency_tracker.record_adaptation_step(len(loader) * mode_config.get('steps', 1))

        result = {'accuracy': float(accuracy)}
        if compute_proto_metrics and proto_evaluator is not None:
            if isinstance(proto_evaluator, EnhancedPrototypeMetrics):
                proto_metrics = proto_evaluator.evaluate_tta_method_enhanced(
                    eval_model, loader, top_k=10, max_samples=None, verbose=False, track_adaptation_rate=True
                )
            else:
                proto_metrics = proto_evaluator.evaluate_tta_method(
                    eval_model, loader, top_k=10, max_samples=None, verbose=False
                )
            keys = [
                'PAC_mean', 'PAC_std', 'PCA_mean', 'PCA_std',
                'sparsity_gini_mean', 'sparsity_active_mean',
                'PCA_weighted_mean', 'PCA_weighted_std',
                'calibration_agreement', 'calibration_logit_corr',
                'gt_class_contrib_improvement', 'gt_class_contrib_change_mean',
                'adaptation_rate', 'avg_updates_per_sample',
            ]
            for key in keys:
                if key in proto_metrics:
                    result[key] = proto_metrics[key]

        if efficiency_tracker:
            result['efficiency'] = efficiency_tracker.get_metrics()
            if hasattr(eval_model, 'adaptation_stats'):
                result['adaptation_stats'] = eval_model.adaptation_stats.copy()

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
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--output', default='robustness_results_dogs.json', help='Output JSON path')
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
        'normal', 'tent', 'eata', 'sar',
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
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    clean_dir = args.clean_dir or str(Path(args.data_dir).parent / 'stanford_dogs' / 'Images')
    corruptions = CORRUPTION_TYPES if 'all' in args.corruptions else args.corruptions
    severity_key = str(args.severity)

    modes = {
        'normal': {},
        'tent': {'lr': args.lr, 'steps': args.steps},
        'eata': {'lr': args.lr, 'steps': args.steps},
        'sar': {
            'lr': args.sar_lr,
            'steps': args.steps,
            'margin_e0': args.sar_margin,
            'reset_constant_em': args.sar_reset,
            'rho': args.sar_rho,
        },
        'proto_tta': {
            'lr': args.lr, 'steps': args.steps, 'use_importance': not args.proto_no_importance, 'use_confidence': True,
            'adapt_all_prototypes': args.proto_all_prototypes,
            'use_geometric_filter': True, 'geo_filter_threshold': args.proto_threshold,
            'consensus_strategy': 'max', 'consensus_ratio': 0.5,
            'adaptation_mode': 'layernorm_attn_bias', 'use_branch_agreement': args.proto_branch_agreement,
            'prototype_branch': args.proto_branch, 'similarity_mapping': args.proto_mapping,
            'sigmoid_center': args.proto_sigmoid_center, 'sigmoid_temp': args.proto_sigmoid_temp,
            'proto_weight': 1.0, 'logit_weight': 0.0,
        },
        'proto_tta_plus_7030': {
            'lr': args.lr, 'steps': args.steps, 'use_importance': not args.proto_no_importance, 'use_confidence': True,
            'adapt_all_prototypes': args.proto_all_prototypes,
            'use_geometric_filter': True, 'geo_filter_threshold': args.proto_threshold,
            'consensus_strategy': 'max', 'consensus_ratio': 0.5,
            'adaptation_mode': 'layernorm_attn_bias', 'use_branch_agreement': args.proto_branch_agreement,
            'prototype_branch': args.proto_branch, 'similarity_mapping': args.proto_mapping,
            'sigmoid_center': args.proto_sigmoid_center, 'sigmoid_temp': args.proto_sigmoid_temp,
            'proto_weight': 0.7, 'logit_weight': 0.3,
        },
        'proto_tta_plus_7525': {
            'lr': args.lr, 'steps': args.steps, 'use_importance': not args.proto_no_importance, 'use_confidence': True,
            'adapt_all_prototypes': args.proto_all_prototypes,
            'use_geometric_filter': True, 'geo_filter_threshold': args.proto_threshold,
            'consensus_strategy': 'max', 'consensus_ratio': 0.5,
            'adaptation_mode': 'layernorm_attn_bias', 'use_branch_agreement': args.proto_branch_agreement,
            'prototype_branch': args.proto_branch, 'similarity_mapping': args.proto_mapping,
            'sigmoid_center': args.proto_sigmoid_center, 'sigmoid_temp': args.proto_sigmoid_temp,
            'proto_weight': 0.75, 'logit_weight': 0.25,
        },
        'proto_tta_plus_8020': {
            'lr': args.lr, 'steps': args.steps, 'use_importance': not args.proto_no_importance, 'use_confidence': True,
            'adapt_all_prototypes': args.proto_all_prototypes,
            'use_geometric_filter': True, 'geo_filter_threshold': args.proto_threshold,
            'consensus_strategy': 'max', 'consensus_ratio': 0.5,
            'adaptation_mode': 'layernorm_attn_bias', 'use_branch_agreement': args.proto_branch_agreement,
            'prototype_branch': args.proto_branch, 'similarity_mapping': args.proto_mapping,
            'sigmoid_center': args.proto_sigmoid_center, 'sigmoid_temp': args.proto_sigmoid_temp,
            'proto_weight': 0.8, 'logit_weight': 0.2,
        },
    }

    selected_modes = [m for m in args.modes if m in modes]
    existing = load_json(args.output)
    results = existing.get('results', {}) if existing else {}

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
