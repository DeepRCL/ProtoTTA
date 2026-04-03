#!/usr/bin/env python3
"""
Comprehensive visualization script for robustness evaluation results.
Handles both accuracy metrics and prototype-based TTA metrics.

Creates:
- Bar plots comparing methods across corruptions
- Category-wise analysis
- Prototype metrics visualizations (PAC, PCA, Sparsity, etc.)
- Radar charts for multi-metric comparison
- Summary tables

Usage:
    python visualize_robustness_results.py \
        --input robustness_results_sev5_metrics.json \
        --output_dir ./plots/robustness_analysis \
        --severity 5
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from math import pi

try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define corruption categories (ImageNet-C standard categories)
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'Blur': ['gaussian_blur', 'defocus_blur'],
    'Weather': ['fog', 'frost', 'brightness'],
    'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform', 'saturate', 'spatter']
}

# Method display names
METHOD_DISPLAY_NAMES = {
    'normal': 'Normal',
    'tent': 'Tent',
    'eata': 'EATA',
    'sar': 'SAR',
    'loss': 'LossAdapt',
    'proto_imp_conf_v1': 'ProtoTTA-v1',
    'proto_imp_conf_v2': 'ProtoTTA-v2',
    'proto_imp_conf_v3': 'ProtoTTA-v3',
}


def extract_accuracy(result):
    """Extract accuracy from result (handles both dict and float formats)."""
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get('accuracy')
    return result


def extract_metric(result, metric_name, default=None):
    """Extract a specific metric from result dict."""
    if result is None or not isinstance(result, dict):
        return default
    return result.get(metric_name, default)


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def get_method_averages(results_dict, severity='5', exclude_list=None):
    """Calculate overall average accuracy for each method.
    
    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude
    """
    if exclude_list is None:
        exclude_list = []
    
    method_averages = {}
    
    for method_name, corruptions in results_dict.items():
        accuracies = []
        for corruption_type, severities in corruptions.items():
            # Skip excluded corruptions
            if corruption_type in exclude_list:
                continue
            if severity in severities and severities[severity] is not None:
                acc = extract_accuracy(severities[severity])
                if acc is not None:
                    accuracies.append(acc)
        
        if accuracies:
            method_averages[method_name] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'count': len(accuracies)
            }
        else:
            method_averages[method_name] = None
    
    return method_averages


def get_category_averages(results_dict, severity='5', exclude_list=None):
    """Calculate average accuracy per category for each method."""
    if exclude_list is None:
        exclude_list = []
    
    category_averages = defaultdict(dict)
    
    for method_name, corruptions in results_dict.items():
        for category, corruption_list in CORRUPTION_CATEGORIES.items():
            accuracies = []
            for corruption_type in corruption_list:
                if corruption_type in exclude_list:
                    continue
                if corruption_type in corruptions:
                    if severity in corruptions[corruption_type]:
                        acc = extract_accuracy(corruptions[corruption_type][severity])
                        if acc is not None:
                            accuracies.append(acc)
            
            if accuracies:
                category_averages[method_name][category] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'count': len(accuracies)
                }
            else:
                category_averages[method_name][category] = None
    
    return category_averages


def get_prototype_metrics_averages(results_dict, severity='5', exclude_list=None):
    """Calculate average prototype metrics for each method."""
    if exclude_list is None:
        exclude_list = []
    
    metrics_averages = {}
    metric_names = ['PAC_mean', 'PCA_mean', 'sparsity_gini_mean', 
                    'PCA_weighted_mean', 'calibration_agreement', 
                    'gt_class_contrib_improvement']
    
    for method_name, corruptions in results_dict.items():
        method_metrics = {metric: [] for metric in metric_names}
        
        for corruption_type, severities in corruptions.items():
            if corruption_type in exclude_list:
                continue
            if severity in severities and severities[severity] is not None:
                result = severities[severity]
                if isinstance(result, dict):
                    for metric in metric_names:
                        val = result.get(metric)
                        if val is not None:
                            method_metrics[metric].append(val)
        
        # Compute averages
        metrics_averages[method_name] = {}
        for metric in metric_names:
            if method_metrics[metric]:
                metrics_averages[method_name][metric] = {
                    'mean': np.mean(method_metrics[metric]),
                    'std': np.std(method_metrics[metric]),
                    'count': len(method_metrics[metric])
                }
            else:
                metrics_averages[method_name][metric] = None
    
    return metrics_averages


def plot_overall_comparison(results_dict, severity, output_dir, exclude_list=None):
    """Create bar plot comparing overall accuracy across methods."""
    method_averages = get_method_averages(results_dict, severity, exclude_list)
    
    if not method_averages:
        print("No data available for overall comparison")
        return
    
    # Sort methods by mean accuracy
    sorted_methods = sorted(method_averages.items(), 
                           key=lambda x: x[1]['mean'] if x[1] else 0, 
                           reverse=True)
    
    method_names = [METHOD_DISPLAY_NAMES.get(m[0], m[0]) for m in sorted_methods]
    means = [m[1]['mean'] * 100 if m[1] else 0 for m in sorted_methods]
    stds = [m[1]['std'] * 100 if m[1] else 0 for m in sorted_methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(method_names))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                   color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Robustness Comparison (Severity {severity})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: overall_comparison.png")


def plot_category_comparison(results_dict, severity, output_dir, exclude_list=None):
    """Create grouped bar plot comparing methods across corruption categories."""
    category_averages = get_category_averages(results_dict, severity, exclude_list)
    
    if not category_averages:
        print("No data available for category comparison")
        return
    
    categories = list(CORRUPTION_CATEGORIES.keys())
    methods = list(category_averages.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    
    for i, method_name in enumerate(methods):
        means = []
        stds = []
        for category in categories:
            if category in category_averages[method_name] and category_averages[method_name][category]:
                means.append(category_averages[method_name][category]['mean'] * 100)
                stds.append(category_averages[method_name][category]['std'] * 100)
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(methods)/2 + 0.5) * width
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        ax.bar(x + offset, means, width, label=display_name, 
               yerr=stds, capsize=3, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Category Robustness Comparison (Severity {severity})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: category_comparison.png")


def plot_per_corruption_heatmap(results_dict, severity, output_dir, exclude_list=None):
    """Create heatmap showing accuracy for each method on each corruption."""
    if exclude_list is None:
        exclude_list = []
    
    # Get all corruptions
    all_corruptions = set()
    for corruptions in results_dict.values():
        all_corruptions.update(corruptions.keys())
    all_corruptions = sorted([c for c in all_corruptions if c not in exclude_list])
    
    methods = list(results_dict.keys())
    
    # Build matrix
    matrix = np.zeros((len(methods), len(all_corruptions)))
    for i, method_name in enumerate(methods):
        for j, corruption in enumerate(all_corruptions):
            if corruption in results_dict[method_name]:
                if severity in results_dict[method_name][corruption]:
                    acc = extract_accuracy(results_dict[method_name][corruption][severity])
                    if acc is not None:
                        matrix[i, j] = acc * 100
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_xticks(np.arange(len(all_corruptions)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(all_corruptions, rotation=45, ha='right')
    ax.set_yticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods])
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(all_corruptions)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Accuracy Heatmap (Severity {severity})', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corruption_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: corruption_heatmap.png")


def plot_prototype_metrics_comparison(results_dict, severity, output_dir, exclude_list=None):
    """Create bar charts comparing prototype metrics across methods."""
    metrics_averages = get_prototype_metrics_averages(results_dict, severity, exclude_list)
    
    if not metrics_averages:
        print("No prototype metrics available")
        return
    
    # Filter out methods without any metrics
    metrics_averages = {k: v for k, v in metrics_averages.items() 
                        if any(v[metric] is not None for metric in v)}
    
    if not metrics_averages:
        print("No prototype metrics data available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = list(metrics_averages.keys())
    x = np.arange(len(methods))
    width = 0.6
    
    metric_configs = [
        ('PAC_mean', 'PAC (Consistency)', 100, 'coral', axes[0, 0]),
        ('PCA_mean', 'PCA (Alignment)', 100, 'mediumseagreen', axes[0, 1]),
        ('sparsity_gini_mean', 'Sparsity (Gini)', 1, 'mediumpurple', axes[0, 2]),
        ('PCA_weighted_mean', 'PCA-Weighted', 1, 'goldenrod', axes[1, 0]),
        ('calibration_agreement', 'Calibration Agreement', 100, 'teal', axes[1, 1]),
        ('gt_class_contrib_improvement', 'GT Contrib Δ', 1, 'crimson', axes[1, 2]),
    ]
    
    for metric_key, title, scale, color, ax in metric_configs:
        means = []
        stds = []
        for method in methods:
            if metrics_averages[method][metric_key] is not None:
                means.append(metrics_averages[method][metric_key]['mean'] * scale)
                stds.append(metrics_averages[method][metric_key]['std'] * scale)
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                      color=color, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            label_format = '.1f' if scale == 100 else '.2f'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:{label_format}}', ha='center', va='bottom', fontsize=9)
        
        ylabel = f'{title} (%)' if scale == 100 else title
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                          rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Prototype-Based TTA Metrics Comparison (Severity {severity})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prototype_metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: prototype_metrics_comparison.png")


def plot_radar_chart(results_dict, severity, output_dir, exclude_list=None):
    """Create radar chart for multi-metric comparison."""
    metrics_averages = get_prototype_metrics_averages(results_dict, severity, exclude_list)
    method_averages = get_method_averages(results_dict, severity, exclude_list)
    
    if not metrics_averages or not method_averages:
        print("Insufficient data for radar chart")
        return
    
    # Filter methods with complete data
    complete_methods = {}
    for method in metrics_averages:
        if method_averages.get(method) and metrics_averages[method]['PAC_mean']:
            complete_methods[method] = {
                'accuracy': method_averages[method]['mean'] * 100,
                'pac': metrics_averages[method]['PAC_mean']['mean'] * 100,
                'pca': metrics_averages[method]['PCA_mean']['mean'] * 100,
                'sparsity': metrics_averages[method]['sparsity_gini_mean']['mean'] * 100,
                'pca_weighted': metrics_averages[method]['PCA_weighted_mean']['mean'] * 100 if metrics_averages[method]['PCA_weighted_mean'] else 0,
                'calib': metrics_averages[method]['calibration_agreement']['mean'] * 100 if metrics_averages[method]['calibration_agreement'] else 0,
            }
    
    if not complete_methods:
        print("No complete data for radar chart")
        return
    
    categories = ['Accuracy', 'PAC', 'PCA', 'Sparsity', 'PCA-Weighted', 'Calibration']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(complete_methods)))
    
    for idx, (method_name, method_metrics) in enumerate(complete_methods.items()):
        values = [
            method_metrics['accuracy'],
            method_metrics['pac'],
            method_metrics['pca'],
            method_metrics['sparsity'],
            method_metrics['pca_weighted'],
            method_metrics['calib']
        ]
        values += values[:1]
        
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        ax.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title(f'Multi-Metric Comparison (Severity {severity})', 
                 size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: radar_comparison.png")


def generate_summary_tables(results_dict, severity, output_dir, exclude_list=None):
    """Generate markdown tables summarizing results."""
    method_averages = get_method_averages(results_dict, severity, exclude_list)
    category_averages = get_category_averages(results_dict, severity, exclude_list)
    metrics_averages = get_prototype_metrics_averages(results_dict, severity, exclude_list)
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severity, exclude_list)
    
    output_file = os.path.join(output_dir, 'summary_tables.md')
    
    with open(output_file, 'w') as f:
        f.write(f"# Robustness Evaluation Summary (Severity {severity})\n\n")
        
        # Overall accuracy
        f.write("## Overall Accuracy\n\n")
        f.write("| Method | Mean Accuracy | Std Dev | Min | Max | # Corruptions |\n")
        f.write("|--------|---------------|---------|-----|-----|---------------|\n")
        
        sorted_methods = sorted(method_averages.items(), 
                               key=lambda x: x[1]['mean'] if x[1] else 0, 
                               reverse=True)
        
        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                f.write(f"| {display_name} | {stats['mean']*100:.2f}% | {stats['std']*100:.2f}% | "
                       f"{stats['min']*100:.2f}% | {stats['max']*100:.2f}% | {stats['count']} |\n")
        
        # Category-wise accuracy
        f.write("\n## Category-wise Accuracy\n\n")
        f.write("| Method | Noise | Blur | Weather | Digital |\n")
        f.write("|--------|-------|------|---------|----------|\n")
        
        for method_name in method_averages:
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            row = [display_name]
            for category in ['Noise', 'Blur', 'Weather', 'Digital']:
                if (method_name in category_averages and 
                    category in category_averages[method_name] and
                    category_averages[method_name][category]):
                    mean = category_averages[method_name][category]['mean'] * 100
                    row.append(f"{mean:.2f}%")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        # Prototype metrics
        if any(metrics_averages.values()):
            f.write("\n## Prototype-Based TTA Metrics\n\n")
            f.write("| Method | PAC | PCA | Sparsity | PCA-Weighted | Calibration | GT Δ |\n")
            f.write("|--------|-----|-----|----------|--------------|-------------|------|\n")
            
            for method_name in method_averages:
                if method_name in metrics_averages:
                    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                    row = [display_name]
                    
                    for metric in ['PAC_mean', 'PCA_mean', 'sparsity_gini_mean', 
                                  'PCA_weighted_mean', 'calibration_agreement', 
                                  'gt_class_contrib_improvement']:
                        if metrics_averages[method_name][metric]:
                            mean = metrics_averages[method_name][metric]['mean']
                            if metric in ['PAC_mean', 'PCA_mean', 'calibration_agreement']:
                                row.append(f"{mean*100:.1f}%")
                            else:
                                row.append(f"{mean:.2f}")
                        else:
                            row.append("N/A")
                    
                    f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n**Metrics Legend:**\n")
            f.write("- **PAC**: Prototype Activation Consistency (higher = better preservation)\n")
            f.write("- **PCA**: Prototype Class Alignment (higher = better semantic alignment)\n")
            f.write("- **Sparsity**: Gini coefficient (higher = more sparse/selective)\n")
            f.write("- **PCA-Weighted**: Weighted prototype alignment\n")
            f.write("- **Calibration**: Top prototype matches prediction\n")
            f.write("- **GT Δ**: Ground truth class contribution improvement\n")
        
        # Efficiency metrics
        if any(efficiency_averages.values()):
            f.write("\n## Computational Efficiency\n\n")
            f.write("| Method | Time/Sample (ms) | Throughput (samp/s) | Adapted Params | Param % | Update % | Steps/Sample | Memory (MB) |\n")
            f.write("|--------|------------------|---------------------|----------------|---------|----------|--------------|-------------|\n")
            
            for method_name in method_averages:
                if method_name in efficiency_averages:
                    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                    row = [display_name]
                    
                    # Time per sample
                    if efficiency_averages[method_name]['time_per_sample_ms']:
                        val = efficiency_averages[method_name]['time_per_sample_ms']['mean']
                        row.append(f"{val:.2f}")
                    else:
                        row.append("N/A")
                    
                    # Throughput
                    if efficiency_averages[method_name]['throughput_samples_per_sec']:
                        val = efficiency_averages[method_name]['throughput_samples_per_sec']['mean']
                        row.append(f"{val:.1f}")
                    else:
                        row.append("N/A")
                    
                    # Adapted params
                    if efficiency_averages[method_name]['num_adapted_params']:
                        val = efficiency_averages[method_name]['num_adapted_params']['mean']
                        row.append(f"{int(val):,}")
                    else:
                        row.append("N/A")
                    
                    # Adaptation ratio (% of params)
                    if efficiency_averages[method_name]['adaptation_ratio']:
                        val = efficiency_averages[method_name]['adaptation_ratio']['mean'] * 100
                        row.append(f"{val:.2f}%")
                    else:
                        row.append("N/A")
                    
                    # Adaptation rate (% of samples updated)
                    if efficiency_averages[method_name]['adaptation_rate']:
                        val = efficiency_averages[method_name]['adaptation_rate']['mean'] * 100
                        row.append(f"{val:.1f}%")
                    else:
                        row.append("N/A")
                    
                    # Steps per sample
                    if efficiency_averages[method_name]['steps_per_sample']:
                        val = efficiency_averages[method_name]['steps_per_sample']['mean']
                        row.append(f"{val:.3f}")
                    else:
                        row.append("N/A")
                    
                    # Memory
                    if efficiency_averages[method_name]['peak_memory_mb']:
                        val = efficiency_averages[method_name]['peak_memory_mb']['mean']
                        row.append(f"{int(val):,}")
                    else:
                        row.append("N/A")
                    
                    f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n**Efficiency Metrics:**\n")
            f.write("- **Time/Sample**: Average inference time per sample (lower = faster)\n")
            f.write("- **Throughput**: Samples processed per second (higher = faster)\n")
            f.write("- **Adapted Params**: Number of parameters adapted during TTA\n")
            f.write("- **Param %**: Percentage of total parameters adapted\n")
            f.write("- **Update %**: Percentage of samples that triggered an update (lower = more selective)\n")
            f.write("- **Steps/Sample**: Average optimizer steps per sample\n")
            f.write("- **Memory**: Peak GPU memory usage\n")
    
    print(f"✓ Saved: summary_tables.md")


def get_efficiency_metrics_averages(results_dict, severity='5', exclude_list=None):
    """Calculate average efficiency metrics for each method."""
    if exclude_list is None:
        exclude_list = []
    
    efficiency_averages = {}
    efficiency_keys = ['time_per_sample_ms', 'throughput_samples_per_sec', 
                       'num_adapted_params', 'adaptation_ratio', 
                       'steps_per_sample', 'peak_memory_mb']
    
    # Also extract adaptation_rate from prototype metrics (outside efficiency dict)
    
    for method_name, corruptions in results_dict.items():
        method_efficiency = {key: [] for key in efficiency_keys}
        method_efficiency['adaptation_rate'] = []  # Add this separately
        
        for corruption_type, severities in corruptions.items():
            if corruption_type in exclude_list:
                continue
            if severity in severities and severities[severity] is not None:
                result = severities[severity]
                if isinstance(result, dict):
                    # Extract efficiency metrics from 'efficiency' sub-dict
                    if 'efficiency' in result:
                        eff = result['efficiency']
                        for key in efficiency_keys:
                            val = eff.get(key)
                            if val is not None:
                                method_efficiency[key].append(val)
                    
                    # Extract adaptation_rate from main result dict (prototype metrics)
                    adapt_rate = result.get('adaptation_rate')
                    if adapt_rate is not None:
                        method_efficiency['adaptation_rate'].append(adapt_rate)
        
        # Compute averages
        efficiency_averages[method_name] = {}
        for key in efficiency_keys + ['adaptation_rate']:
            if method_efficiency[key]:
                efficiency_averages[method_name][key] = {
                    'mean': np.mean(method_efficiency[key]),
                    'std': np.std(method_efficiency[key]),
                    'count': len(method_efficiency[key])
                }
            else:
                efficiency_averages[method_name][key] = None
    
    return efficiency_averages


def plot_efficiency_comparison(results_dict, severity, output_dir, exclude_list=None):
    """Create plots comparing efficiency metrics across methods."""
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severity, exclude_list)
    
    if not efficiency_averages:
        print("No efficiency metrics available")
        return
    
    # Filter out methods without any efficiency data
    efficiency_averages = {k: v for k, v in efficiency_averages.items() 
                          if any(v[metric] is not None for metric in v)}
    
    if not efficiency_averages:
        print("No efficiency data available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = list(efficiency_averages.keys())
    x = np.arange(len(methods))
    width = 0.6
    
    # Time per sample
    ax = axes[0, 0]
    means = []
    stds = []
    for method in methods:
        if efficiency_averages[method]['time_per_sample_ms'] is not None:
            means.append(efficiency_averages[method]['time_per_sample_ms']['mean'])
            stds.append(efficiency_averages[method]['time_per_sample_ms']['std'])
        else:
            means.append(0)
            stds.append(0)
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color='steelblue', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Time per Sample', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Throughput
    ax = axes[0, 1]
    means = []
    stds = []
    for method in methods:
        if efficiency_averages[method]['throughput_samples_per_sec'] is not None:
            means.append(efficiency_averages[method]['throughput_samples_per_sec']['mean'])
            stds.append(efficiency_averages[method]['throughput_samples_per_sec']['std'])
        else:
            means.append(0)
            stds.append(0)
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color='mediumseagreen', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Samples/sec', fontweight='bold')
    ax.set_title('Throughput', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Adapted parameters
    ax = axes[0, 2]
    means = []
    for method in methods:
        if efficiency_averages[method]['num_adapted_params'] is not None:
            means.append(efficiency_averages[method]['num_adapted_params']['mean'])
        else:
            means.append(0)
    
    bars = ax.bar(x, means, width, color='coral', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Number of Parameters', fontweight='bold')
    ax.set_title('Adapted Parameters', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(mean):,}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Adaptation rate (% samples updated)
    ax = axes[1, 0]
    means = []
    for method in methods:
        if efficiency_averages[method]['adaptation_rate'] is not None:
            means.append(efficiency_averages[method]['adaptation_rate']['mean'] * 100)
        else:
            means.append(0)
    
    bars = ax.bar(x, means, width, color='mediumpurple', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Adaptation Rate (%)', fontweight='bold')
    ax.set_title('% of Samples Updated', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Steps per sample
    ax = axes[1, 1]
    means = []
    for method in methods:
        if efficiency_averages[method]['steps_per_sample'] is not None:
            means.append(efficiency_averages[method]['steps_per_sample']['mean'])
        else:
            means.append(0)
    
    bars = ax.bar(x, means, width, color='goldenrod', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Steps/Sample', fontweight='bold')
    ax.set_title('Adaptation Steps per Sample', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Peak memory
    ax = axes[1, 2]
    means = []
    stds = []
    for method in methods:
        if efficiency_averages[method]['peak_memory_mb'] is not None:
            means.append(efficiency_averages[method]['peak_memory_mb']['mean'])
            stds.append(efficiency_averages[method]['peak_memory_mb']['std'])
        else:
            means.append(0)
            stds.append(0)
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color='teal', edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_ylabel('Memory (MB)', fontweight='bold')
    ax.set_title('Peak GPU Memory', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(mean):,}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Computational Efficiency Comparison (Severity {severity})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: efficiency_comparison.png")


def filter_and_rename_methods(results_dict, include_methods=None, rename_map=None):
    """Filter results to only include specified methods and apply renaming.
    
    Args:
        results_dict: Original results dictionary
        include_methods: List of method names to include (None = include all)
        rename_map: Dict mapping old names to new names
    
    Returns:
        Filtered and renamed results dictionary
    """
    if include_methods is None:
        filtered_results = results_dict.copy()
    else:
        filtered_results = {k: v for k, v in results_dict.items() if k in include_methods}
    
    if rename_map:
        renamed_results = {}
        for old_name, data in filtered_results.items():
            new_name = rename_map.get(old_name, old_name)
            renamed_results[new_name] = data
        
        # Update METHOD_DISPLAY_NAMES - key fix: set display name to the NEW name, not old
        global METHOD_DISPLAY_NAMES
        for old_name, new_name in rename_map.items():
            # Remove old entry if exists
            if old_name in METHOD_DISPLAY_NAMES:
                del METHOD_DISPLAY_NAMES[old_name]
            # Add new entry with the renamed display name
            METHOD_DISPLAY_NAMES[new_name] = new_name
        
        return renamed_results
    
    return filtered_results


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization of robustness evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_robustness_results.py --input results.json
  
  # Select specific methods and rename v3 to ProtoTTA
  python visualize_robustness_results.py --input results.json \\
      --methods normal tent eata proto_imp_conf_v3 \\
      --rename proto_imp_conf_v3=ProtoTTA
  
  # Exclude corruptions
  python visualize_robustness_results.py --input results.json \\
      --exclude saturate spatter
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output_dir', type=str, default='./plots/robustness_analysis',
                       help='Output directory for plots and tables')
    parser.add_argument('--severity', type=str, default='5',
                       help='Severity level to analyze (default: 5)')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Corruption types to exclude from analysis')
    parser.add_argument('--methods', nargs='*', default=None,
                       help='Specific methods to include (default: all). '
                            'Example: --methods normal tent eata proto_imp_conf_v3')
    parser.add_argument('--rename', nargs='*', default=[],
                       help='Rename methods using old=new format. '
                            'Example: --rename proto_imp_conf_v3=ProtoTTA proto_imp_conf_v1=ProtoTTA-v1')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results_dict = data.get('results', {})
    
    if not results_dict:
        print("ERROR: No results found in JSON file")
        return
    
    # Parse rename arguments
    rename_map = {}
    if args.rename:
        for rename_arg in args.rename:
            if '=' in rename_arg:
                old_name, new_name = rename_arg.split('=', 1)
                rename_map[old_name.strip()] = new_name.strip()
    
    # Filter and rename methods
    if args.methods or rename_map:
        results_dict = filter_and_rename_methods(results_dict, args.methods, rename_map)
        
        if args.methods:
            print(f"Including methods: {', '.join(args.methods)}")
        if rename_map:
            print(f"Renamed methods: {', '.join(f'{k}→{v}' for k, v in rename_map.items())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    if args.exclude:
        print(f"Excluding corruptions: {', '.join(args.exclude)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_overall_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_category_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_per_corruption_heatmap(results_dict, args.severity, args.output_dir, args.exclude)
    plot_prototype_metrics_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_efficiency_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_radar_chart(results_dict, args.severity, args.output_dir, args.exclude)
    generate_summary_tables(results_dict, args.severity, args.output_dir, args.exclude)
    
    print(f"\n{'='*80}")
    print("✓ All visualizations and tables generated successfully!")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
