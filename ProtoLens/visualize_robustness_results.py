#!/usr/bin/env python3
"""
Comprehensive visualization script for ProtoLens robustness evaluation results.
Handles both accuracy metrics and prototype-based TTA metrics.

Creates:
- Bar plots comparing methods across corruptions
- Category-wise analysis (by corruption type: qwerty, aggressive, etc.)
- Prototype metrics visualizations (PAC, PCA, Sparsity, etc.)
- Radar charts for multi-metric comparison
- Summary tables with overall performance

Usage:
    python visualize_robustness_results.py \
        --input Datasets/Amazon-C/results/robustness_results_main.json \
        --output_dir ./plots/robustness_analysis \
        --severities 20 40 60 80
    
    # Default: analyze all available severities
    python visualize_robustness_results.py --input results.json
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

# Define corruption categories (ProtoLens text corruption categories)
CORRUPTION_CATEGORIES = {
    'Keyboard': ['qwerty'],
    'Character': ['swap', 'remove_char'],
    'Combined': ['mixed', 'aggressive']
}

# All corruption types
ALL_CORRUPTIONS = ['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive']

# Method display names
METHOD_DISPLAY_NAMES = {
    'baseline': 'Baseline',
    'tent': 'Tent',
    'eata': 'EATA',
    'sar': 'SAR',
    'prototta': 'ProtoTTA',
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


def get_all_severities(results_dict):
    """Get all available severities from the results."""
    all_severities = set()
    for method_data in results_dict.values():
        for corruption_data in method_data.values():
            all_severities.update(corruption_data.keys())
    return sorted([s for s in all_severities], key=lambda x: int(x))


def get_method_averages(results_dict, severities=None, exclude_list=None):
    """Calculate overall average accuracy for each method across specified severities.
    
    Args:
        results_dict: Dictionary of results
        severities: List of severity levels to analyze (as strings). None = all
        exclude_list: List of corruption types to exclude
    """
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    method_averages = {}
    
    for method_name, corruptions in results_dict.items():
        accuracies = []
        for corruption_type, severity_data in corruptions.items():
            if corruption_type in exclude_list:
                continue
            for severity in severities:
                if severity in severity_data and severity_data[severity] is not None:
                    acc = extract_accuracy(severity_data[severity])
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


def get_per_corruption_averages(results_dict, severities=None, exclude_list=None):
    """Calculate average accuracy per corruption type for each method (across all severities).
    
    Args:
        results_dict: Dictionary of results
        severities: List of severity levels to include
        exclude_list: List of corruption types to exclude
    """
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    corruption_averages = defaultdict(dict)
    
    for method_name, corruptions in results_dict.items():
        for corruption_type, severity_data in corruptions.items():
            if corruption_type in exclude_list:
                continue
            accuracies = []
            for severity in severities:
                if severity in severity_data and severity_data[severity] is not None:
                    acc = extract_accuracy(severity_data[severity])
                    if acc is not None:
                        accuracies.append(acc)
            
            if accuracies:
                corruption_averages[method_name][corruption_type] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'count': len(accuracies)
                }
            else:
                corruption_averages[method_name][corruption_type] = None
    
    return corruption_averages


def get_category_averages(results_dict, severities=None, exclude_list=None):
    """Calculate average accuracy per category for each method."""
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    category_averages = defaultdict(dict)
    
    for method_name, corruptions in results_dict.items():
        for category, corruption_list in CORRUPTION_CATEGORIES.items():
            accuracies = []
            for corruption_type in corruption_list:
                if corruption_type in exclude_list:
                    continue
                if corruption_type in corruptions:
                    for severity in severities:
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


def get_prototype_metrics_averages(results_dict, severities=None, exclude_list=None):
    """Calculate average prototype metrics for each method."""
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    metrics_averages = {}
    metric_names = ['PAC_mean', 'PCA_mean', 'sparsity_gini_mean', 
                    'PCA_weighted_mean', 'calibration_agreement', 
                    'gt_class_contrib_improvement']
    
    for method_name, corruptions in results_dict.items():
        method_metrics = {metric: [] for metric in metric_names}
        
        for corruption_type, severity_data in corruptions.items():
            if corruption_type in exclude_list:
                continue
            for severity in severities:
                if severity in severity_data and severity_data[severity] is not None:
                    result = severity_data[severity]
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


def get_efficiency_metrics_averages(results_dict, severities=None, exclude_list=None):
    """Calculate average efficiency metrics for each method."""
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    efficiency_averages = {}
    efficiency_keys = ['time_per_sample_ms', 'throughput_samples_per_sec', 
                       'num_adapted_params', 'adaptation_ratio', 
                       'steps_per_sample', 'peak_memory_mb']
    
    for method_name, corruptions in results_dict.items():
        method_efficiency = {key: [] for key in efficiency_keys}
        method_efficiency['adaptation_rate'] = []
        
        for corruption_type, severity_data in corruptions.items():
            if corruption_type in exclude_list:
                continue
            for severity in severities:
                if severity in severity_data and severity_data[severity] is not None:
                    result = severity_data[severity]
                    if isinstance(result, dict):
                        if 'efficiency' in result:
                            eff = result['efficiency']
                            for key in efficiency_keys:
                                val = eff.get(key)
                                if val is not None:
                                    method_efficiency[key].append(val)
                        
                        adapt_rate = result.get('adaptation_rate')
                        if adapt_rate is not None:
                            method_efficiency['adaptation_rate'].append(adapt_rate)
        
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


def plot_overall_comparison(results_dict, severities, output_dir, exclude_list=None):
    """Create bar plot comparing overall accuracy across methods."""
    method_averages = get_method_averages(results_dict, severities, exclude_list)
    
    if not method_averages:
        print("No data available for overall comparison")
        return
    
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
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    sev_str = ', '.join(severities) if severities else 'All'
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Robustness Comparison (Severities: {sev_str})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: overall_comparison.png")


def plot_per_corruption_comparison(results_dict, severities, output_dir, exclude_list=None):
    """Create grouped bar plot comparing methods for each corruption type."""
    corruption_averages = get_per_corruption_averages(results_dict, severities, exclude_list)
    
    if not corruption_averages:
        print("No data available for per-corruption comparison")
        return
    
    corruptions = [c for c in ALL_CORRUPTIONS if c not in (exclude_list or [])]
    methods = list(corruption_averages.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(corruptions))
    width = 0.8 / len(methods)
    
    for i, method_name in enumerate(methods):
        means = []
        stds = []
        for corruption in corruptions:
            if corruption in corruption_averages[method_name] and corruption_averages[method_name][corruption]:
                means.append(corruption_averages[method_name][corruption]['mean'] * 100)
                stds.append(corruption_averages[method_name][corruption]['std'] * 100)
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(methods)/2 + 0.5) * width
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        ax.bar(x + offset, means, width, label=display_name, 
               yerr=stds, capsize=3, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    sev_str = ', '.join(severities) if severities else 'All'
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Corruption Robustness Comparison (Severities: {sev_str})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_corruption_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: per_corruption_comparison.png")


def plot_category_comparison(results_dict, severities, output_dir, exclude_list=None):
    """Create grouped bar plot comparing methods across corruption categories."""
    category_averages = get_category_averages(results_dict, severities, exclude_list)
    
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
    
    sev_str = ', '.join(severities) if severities else 'All'
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Category Robustness Comparison (Severities: {sev_str})', 
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


def plot_per_corruption_heatmap(results_dict, severities, output_dir, exclude_list=None):
    """Create heatmap showing accuracy for each method on each corruption."""
    if exclude_list is None:
        exclude_list = []
    if severities is None:
        severities = get_all_severities(results_dict)
    
    all_corruptions = set()
    for corruptions in results_dict.values():
        all_corruptions.update(corruptions.keys())
    all_corruptions = sorted([c for c in all_corruptions if c not in exclude_list])
    
    methods = list(results_dict.keys())
    
    matrix = np.zeros((len(methods), len(all_corruptions)))
    for i, method_name in enumerate(methods):
        for j, corruption in enumerate(all_corruptions):
            accs = []
            if corruption in results_dict[method_name]:
                for severity in severities:
                    if severity in results_dict[method_name][corruption]:
                        acc = extract_accuracy(results_dict[method_name][corruption][severity])
                        if acc is not None:
                            accs.append(acc)
            if accs:
                matrix[i, j] = np.mean(accs) * 100
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(all_corruptions)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(all_corruptions, rotation=45, ha='right')
    ax.set_yticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods])
    
    for i in range(len(methods)):
        for j in range(len(all_corruptions)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    sev_str = ', '.join(severities) if severities else 'All'
    ax.set_title(f'Accuracy Heatmap (Severities: {sev_str})', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corruption_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: corruption_heatmap.png")


def plot_severity_comparison(results_dict, output_dir, exclude_list=None):
    """Create line plot showing accuracy across different severity levels."""
    if exclude_list is None:
        exclude_list = []
    
    all_severities = set()
    for method_data in results_dict.values():
        for corruption_data in method_data.values():
            all_severities.update(corruption_data.keys())
    severities = sorted([int(s) for s in all_severities])
    
    if len(severities) < 2:
        print("Not enough severity levels for severity comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for idx, (method_name, method_data) in enumerate(results_dict.items()):
        means = []
        stds = []
        for sev in severities:
            sev_str = str(sev)
            accs = []
            for corruption_type, corruption_data in method_data.items():
                if corruption_type in exclude_list:
                    continue
                if sev_str in corruption_data:
                    acc = extract_accuracy(corruption_data[sev_str])
                    if acc is not None:
                        accs.append(acc)
            if accs:
                means.append(np.mean(accs) * 100)
                stds.append(np.std(accs) * 100)
            else:
                means.append(None)
                stds.append(None)
        
        valid_severities = [s for s, m in zip(severities, means) if m is not None]
        valid_means = [m for m in means if m is not None]
        valid_stds = [s for s, m in zip(stds, means) if m is not None]
        
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        ax.errorbar(valid_severities, valid_means, yerr=valid_stds, 
                   marker='o', linewidth=2, markersize=8, capsize=4,
                   label=display_name, color=colors[idx])
    
    ax.set_xlabel('Severity Level (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Corruption Severity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(severities)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'severity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: severity_comparison.png")


def plot_prototype_metrics_comparison(results_dict, severities, output_dir, exclude_list=None):
    """Create bar charts comparing prototype metrics across methods."""
    metrics_averages = get_prototype_metrics_averages(results_dict, severities, exclude_list)
    
    if not metrics_averages:
        print("No prototype metrics available")
        return
    
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
    
    sev_str = ', '.join(severities) if severities else 'All'
    plt.suptitle(f'Prototype-Based TTA Metrics Comparison (Severities: {sev_str})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prototype_metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: prototype_metrics_comparison.png")


def plot_efficiency_comparison(results_dict, severities, output_dir, exclude_list=None):
    """Create plots comparing efficiency metrics across methods."""
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severities, exclude_list)
    
    if not efficiency_averages:
        print("No efficiency metrics available")
        return
    
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
    
    # Adaptation rate
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
    
    sev_str = ', '.join(severities) if severities else 'All'
    plt.suptitle(f'Computational Efficiency Comparison (Severities: {sev_str})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: efficiency_comparison.png")


def plot_radar_chart(results_dict, severities, output_dir, exclude_list=None):
    """Create radar chart for multi-metric comparison."""
    metrics_averages = get_prototype_metrics_averages(results_dict, severities, exclude_list)
    method_averages = get_method_averages(results_dict, severities, exclude_list)
    
    if not metrics_averages or not method_averages:
        print("Insufficient data for radar chart")
        return
    
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
    sev_str = ', '.join(severities) if severities else 'All'
    ax.set_title(f'Multi-Metric Comparison (Severities: {sev_str})', 
                 size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: radar_comparison.png")


def generate_summary_tables(results_dict, severities, output_dir, exclude_list=None):
    """Generate comprehensive markdown tables summarizing results."""
    if severities is None:
        severities = get_all_severities(results_dict)
    
    method_averages = get_method_averages(results_dict, severities, exclude_list)
    category_averages = get_category_averages(results_dict, severities, exclude_list)
    corruption_averages = get_per_corruption_averages(results_dict, severities, exclude_list)
    metrics_averages = get_prototype_metrics_averages(results_dict, severities, exclude_list)
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severities, exclude_list)
    
    output_file = os.path.join(output_dir, 'summary_tables.md')
    sev_str = ', '.join(severities) if severities else 'All'
    
    with open(output_file, 'w') as f:
        f.write(f"# ProtoLens Robustness Evaluation Summary\n\n")
        f.write(f"**Severities Analyzed:** {sev_str}\n\n")
        
        # Overall accuracy (across all corruptions and severities)
        f.write("## Overall Model Performance (All Corruptions)\n\n")
        f.write("| Method | Mean Accuracy | Std Dev | Min | Max | # Samples |\n")
        f.write("|--------|---------------|---------|-----|-----|----------|\n")
        
        sorted_methods = sorted(method_averages.items(), 
                               key=lambda x: x[1]['mean'] if x[1] else 0, 
                               reverse=True)
        
        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                f.write(f"| {display_name} | {stats['mean']*100:.2f}% | {stats['std']*100:.2f}% | "
                       f"{stats['min']*100:.2f}% | {stats['max']*100:.2f}% | {stats['count']} |\n")
        
        # Per-corruption accuracy table
        f.write("\n## Performance by Corruption Type\n\n")
        corruptions = [c for c in ALL_CORRUPTIONS if c not in (exclude_list or [])]
        f.write("| Method | " + " | ".join(corruptions) + " | **Overall** |\n")
        f.write("|--------" + "|:------:" * len(corruptions) + "|:------:|\n")
        
        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                row = [display_name]
                for corruption in corruptions:
                    if method_name in corruption_averages and corruption in corruption_averages[method_name]:
                        if corruption_averages[method_name][corruption]:
                            mean = corruption_averages[method_name][corruption]['mean'] * 100
                            row.append(f"{mean:.1f}%")
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")
                # Add overall
                row.append(f"**{stats['mean']*100:.1f}%**")
                f.write("| " + " | ".join(row) + " |\n")
        
        # Category-wise accuracy
        f.write("\n## Performance by Corruption Category\n\n")
        f.write("| Method | Keyboard | Character | Combined |\n")
        f.write("|--------|:--------:|:---------:|:--------:|\n")
        
        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                row = [display_name]
                for category in ['Keyboard', 'Character', 'Combined']:
                    if (method_name in category_averages and 
                        category in category_averages[method_name] and
                        category_averages[method_name][category]):
                        mean = category_averages[method_name][category]['mean'] * 100
                        row.append(f"{mean:.2f}%")
                    else:
                        row.append("N/A")
                f.write("| " + " | ".join(row) + " |\n")
        
        # Detailed per-severity breakdown
        f.write("\n## Performance by Severity Level\n\n")
        f.write("| Method | " + " | ".join([f"Sev {s}" for s in severities]) + " |\n")
        f.write("|--------" + "|:------:" * len(severities) + "|\n")
        
        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                row = [display_name]
                for severity in severities:
                    accs = []
                    for corruption in results_dict.get(method_name, {}).values():
                        if severity in corruption:
                            acc = extract_accuracy(corruption[severity])
                            if acc is not None:
                                accs.append(acc)
                    if accs:
                        row.append(f"{np.mean(accs)*100:.1f}%")
                    else:
                        row.append("N/A")
                f.write("| " + " | ".join(row) + " |\n")
        
        # Prototype metrics
        if any(metrics_averages.values()):
            f.write("\n## Prototype-Based TTA Metrics\n\n")
            f.write("| Method | PAC | PCA | Sparsity | PCA-Weighted | Calibration | GT Δ |\n")
            f.write("|--------|:---:|:---:|:--------:|:------------:|:-----------:|:----:|\n")
            
            for method_name, stats in sorted_methods:
                if method_name in metrics_averages and stats:
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
            f.write("|--------|:----------------:|:-------------------:|:--------------:|:-------:|:--------:|:------------:|:-----------:|\n")
            
            for method_name, stats in sorted_methods:
                if method_name in efficiency_averages and stats:
                    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                    row = [display_name]
                    
                    if efficiency_averages[method_name]['time_per_sample_ms']:
                        val = efficiency_averages[method_name]['time_per_sample_ms']['mean']
                        row.append(f"{val:.2f}")
                    else:
                        row.append("N/A")
                    
                    if efficiency_averages[method_name]['throughput_samples_per_sec']:
                        val = efficiency_averages[method_name]['throughput_samples_per_sec']['mean']
                        row.append(f"{val:.1f}")
                    else:
                        row.append("N/A")
                    
                    if efficiency_averages[method_name]['num_adapted_params']:
                        val = efficiency_averages[method_name]['num_adapted_params']['mean']
                        row.append(f"{int(val):,}")
                    else:
                        row.append("N/A")
                    
                    if efficiency_averages[method_name]['adaptation_ratio']:
                        val = efficiency_averages[method_name]['adaptation_ratio']['mean'] * 100
                        row.append(f"{val:.4f}%")
                    else:
                        row.append("N/A")
                    
                    if efficiency_averages[method_name]['adaptation_rate']:
                        val = efficiency_averages[method_name]['adaptation_rate']['mean'] * 100
                        row.append(f"{val:.1f}%")
                    else:
                        row.append("N/A")
                    
                    if efficiency_averages[method_name]['steps_per_sample']:
                        val = efficiency_averages[method_name]['steps_per_sample']['mean']
                        row.append(f"{val:.4f}")
                    else:
                        row.append("N/A")
                    
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


def filter_and_rename_methods(results_dict, include_methods=None, rename_map=None):
    """Filter results to only include specified methods and apply renaming."""
    if include_methods is None and rename_map is None:
        return results_dict
    
    filtered = {}
    
    for method_name, method_data in results_dict.items():
        if include_methods and method_name not in include_methods:
            continue
        
        new_name = rename_map.get(method_name, method_name) if rename_map else method_name
        filtered[new_name] = method_data
    
    return filtered


def print_json_structure(results_dict, severities):
    """Print a summary of what's in the JSON to help diagnose issues."""
    print("\n" + "="*60)
    print("JSON Structure Summary")
    print("="*60)
    
    for method_name, corruptions in results_dict.items():
        print(f"\nMethod: {method_name}")
        for corruption_type, severity_data in corruptions.items():
            print(f"  Corruption: {corruption_type}")
            for sev, result in severity_data.items():
                if severities and sev not in severities:
                    continue
                if isinstance(result, dict):
                    acc = result.get('accuracy')
                    print(f"    Sev {sev}: acc={acc:.4f if acc else 'N/A'}, keys={list(result.keys())[:5]}...")
                else:
                    print(f"    Sev {sev}: {result}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization of ProtoLens robustness evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all severities (default)
  python visualize_robustness_results.py --input results.json
  
  # Analyze specific severities
  python visualize_robustness_results.py --input results.json --severities 20 40 60 80
  
  # Analyze only high severities
  python visualize_robustness_results.py --input results.json --severities 60 80
  
  # Select specific methods and rename
  python visualize_robustness_results.py --input results.json \\
      --methods baseline tent eata prototta \\
      --rename prototta=ProtoTTA
  
  # Exclude corruptions
  python visualize_robustness_results.py --input results.json \\
      --exclude mixed
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output_dir', type=str, default='./plots/robustness_analysis',
                       help='Output directory for plots and tables')
    parser.add_argument('--severities', nargs='*', default=None,
                       help='Severity levels to analyze (default: all). Example: --severities 20 40 60 80')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Corruption types to exclude from analysis')
    parser.add_argument('--methods', nargs='*', default=None,
                       help='Specific methods to include (default: all). '
                            'Example: --methods baseline tent eata prototta')
    parser.add_argument('--rename', nargs='*', default=[],
                       help='Rename methods using old=new format. '
                            'Example: --rename prototta=ProtoTTA')
    parser.add_argument('--debug', action='store_true',
                       help='Print JSON structure for debugging')
    
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
    
    # Determine severities
    if args.severities is None:
        severities = get_all_severities(results_dict)
        print(f"Analyzing all available severities: {', '.join(severities)}")
    else:
        severities = args.severities
        print(f"Analyzing specified severities: {', '.join(severities)}")
    
    # Debug output
    if args.debug:
        print_json_structure(results_dict, severities)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    if args.exclude:
        print(f"Excluding corruptions: {', '.join(args.exclude)}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    plot_overall_comparison(results_dict, severities, args.output_dir, args.exclude)
    plot_per_corruption_comparison(results_dict, severities, args.output_dir, args.exclude)
    plot_category_comparison(results_dict, severities, args.output_dir, args.exclude)
    plot_per_corruption_heatmap(results_dict, severities, args.output_dir, args.exclude)
    plot_severity_comparison(results_dict, args.output_dir, args.exclude)
    plot_prototype_metrics_comparison(results_dict, severities, args.output_dir, args.exclude)
    plot_efficiency_comparison(results_dict, severities, args.output_dir, args.exclude)
    plot_radar_chart(results_dict, severities, args.output_dir, args.exclude)
    generate_summary_tables(results_dict, severities, args.output_dir, args.exclude)
    
    print(f"\n{'='*80}")
    print("✓ All visualizations and tables generated successfully!")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
