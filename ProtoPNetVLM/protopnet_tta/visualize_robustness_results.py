#!/usr/bin/env python3
"""
Comprehensive visualization script for ProtoPNet robustness evaluation results.
Handles both accuracy metrics and prototype-based TTA metrics.

Creates:
- Bar plots comparing methods across corruptions
- Category-wise analysis
- Prototype metrics visualizations (PAC, PCA-W, Calibration, etc.)
- Efficiency comparison
- Summary tables (in plain text and markdown)

Usage:
    python -m protopnet_tta.visualize_robustness_results \
        --input results.json \
        --output_dir ./plots/robustness_analysis_v2
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

# Define corruption categories (histopathology-specific)
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'Blur': ['gaussian_blur', 'defocus_blur'],
    'Weather': ['fog', 'frost', 'brightness'],
    'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform']
}

# Method display names
METHOD_DISPLAY_NAMES = {
    'Normal': 'Unadapted',
    'Tent': 'Tent',
    'EATA': 'EATA',
    'SAR': 'SAR',
    'MEMO': 'MEMO',
    'ProtoEntropy': 'ProtoTTA',
    'ProtoEntropy-BN': 'ProtoTTA-BN',
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
        exclude_list: List of corruption types to exclude (including 'clean')
    """
    if exclude_list is None:
        exclude_list = []

    method_averages = {}

    for method_name, corruptions in results_dict.items():
        accuracies = []
        for corruption_type, severities in corruptions.items():
            # Skip excluded corruptions and 'clean' data
            if corruption_type in exclude_list or corruption_type == 'clean':
                continue

            # Handle nested severity structure
            if isinstance(severities, dict):
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
                    severities = corruptions[corruption_type]
                    if isinstance(severities, dict) and severity in severities:
                        acc = extract_accuracy(severities[severity])
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
    metric_names = ['PAC_mean', 'PAC_std', 'PCA_mean', 'PCA_std',
                    'sparsity_gini_mean', 'PCA_weighted_mean', 'PCA_weighted_std',
                    'calibration_agreement']

    for method_name, corruptions in results_dict.items():
        method_metrics = {metric: [] for metric in metric_names}

        for corruption_type, severities in corruptions.items():
            if corruption_type in exclude_list or corruption_type == 'clean':
                continue

            if isinstance(severities, dict) and severity in severities:
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


def get_efficiency_metrics_averages(results_dict, severity='5', exclude_list=None):
    """Calculate average efficiency metrics for each method."""
    if exclude_list is None:
        exclude_list = []

    efficiency_averages = {}
    efficiency_keys = ['time_per_sample_ms', 'throughput_samples_per_sec',
                       'num_adapted_params', 'adaptation_ratio',
                       'steps_per_sample', 'peak_memory_mb']

    for method_name, corruptions in results_dict.items():
        method_efficiency = {key: [] for key in efficiency_keys}
        method_efficiency['adaptation_rate'] = []  # % of samples updated

        for corruption_type, severities in corruptions.items():
            if corruption_type in exclude_list or corruption_type == 'clean':
                continue

            if isinstance(severities, dict) and severity in severities:
                result = severities[severity]
                if isinstance(result, dict):
                    # Extract efficiency metrics from 'efficiency' sub-dict
                    if 'efficiency' in result:
                        eff = result['efficiency']
                        for key in efficiency_keys:
                            val = eff.get(key)
                            if val is not None:
                                method_efficiency[key].append(val)

                    # Extract adaptation_rate from adaptation_stats
                    if 'adaptation_stats' in result:
                        stats = result['adaptation_stats']
                        total = stats.get('total_samples', 0)
                        adapted = stats.get('adapted_samples', 0)
                        if total > 0:
                            method_efficiency['adaptation_rate'].append(adapted / total)

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
    ax.set_ylim(0, max(means) * 1.2)

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: category_comparison.png")


def plot_per_corruption_heatmap(results_dict, severity, output_dir, exclude_list=None):
    """Create heatmap showing accuracy for each method on each corruption."""
    if exclude_list is None:
        exclude_list = []

    # Get all corruptions (excluding 'clean')
    all_corruptions = set()
    for corruptions in results_dict.values():
        all_corruptions.update(corruptions.keys())
    all_corruptions = sorted([c for c in all_corruptions
                             if c not in exclude_list and c != 'clean'])

    methods = list(results_dict.keys())

    # Build matrix
    matrix = np.zeros((len(methods), len(all_corruptions)))
    for i, method_name in enumerate(methods):
        for j, corruption in enumerate(all_corruptions):
            if corruption in results_dict[method_name]:
                severities = results_dict[method_name][corruption]
                if isinstance(severities, dict) and severity in severities:
                    acc = extract_accuracy(severities[severity])
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


def generate_summary_tables(results_dict, severity, output_dir, exclude_list=None):
    """Generate markdown and plain text tables summarizing results."""
    method_averages = get_method_averages(results_dict, severity, exclude_list)
    category_averages = get_category_averages(results_dict, severity, exclude_list)
    metrics_averages = get_prototype_metrics_averages(results_dict, severity, exclude_list)
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severity, exclude_list)

    # Sort methods by accuracy
    sorted_methods = sorted(method_averages.items(),
                           key=lambda x: x[1]['mean'] if x[1] else 0,
                           reverse=True)

    # ========== MARKDOWN OUTPUT ==========
    output_file_md = os.path.join(output_dir, 'summary_tables.md')

    with open(output_file_md, 'w') as f:
        f.write(f"# ProtoPNet Robustness Evaluation Summary (Severity {severity})\n\n")

        # Overall accuracy
        f.write("## Overall Accuracy\n\n")
        f.write("| Method | Mean Accuracy | Std Dev | Min | Max | # Corruptions |\n")
        f.write("|--------|---------------|---------|-----|-----|---------------|\n")

        for method_name, stats in sorted_methods:
            if stats:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                f.write(
                    f"| {display_name} | {stats['mean']*100:.2f}% | {stats['std']*100:.2f}% | "
                    f"{stats['min']*100:.2f}% | {stats['max']*100:.2f}% | {stats['count']} |\n"
                )

        # Category-wise accuracy
        f.write("\n## Category-wise Accuracy\n\n")
        f.write("| Method | Noise | Blur | Weather | Digital |\n")
        f.write("|--------|-------|------|---------|---------|\n")

        for method_name, _ in sorted_methods:
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

        # Prototype metrics (if available)
        if any(metrics_averages.values()):
            f.write("\n## Interpretability Metrics\n\n")
            f.write("| Method | PAC (Mean ± Std) | PCA-W (Mean ± Std) | Calibration |\n")
            f.write("|--------|------------------|-------------------|-------------|\n")

            for method_name, _ in sorted_methods:
                if method_name in metrics_averages:
                    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                    row = [display_name]

                    # PAC
                    if metrics_averages[method_name]['PAC_mean']:
                        pac_mean = metrics_averages[method_name]['PAC_mean']['mean'] * 100
                        pac_std = metrics_averages[method_name]['PAC_std']['mean'] * 100
                        row.append(f"{pac_mean:.1f} ± {pac_std:.1f}")
                    else:
                        row.append("N/A")

                    # PCA-Weighted
                    if metrics_averages[method_name]['PCA_weighted_mean']:
                        pca_mean = metrics_averages[method_name]['PCA_weighted_mean']['mean'] * 100
                        pca_std = metrics_averages[method_name]['PCA_weighted_std']['mean'] * 100
                        row.append(f"{pca_mean:.1f} ± {pca_std:.1f}")
                    else:
                        row.append("N/A")

                    # Calibration
                    if metrics_averages[method_name]['calibration_agreement']:
                        calib = metrics_averages[method_name]['calibration_agreement']['mean'] * 100
                        row.append(f"{calib:.1f}%")
                    else:
                        row.append("N/A")

                    f.write("| " + " | ".join(row) + " |\n")

        # Efficiency metrics (if available)
        if any(efficiency_averages.values()):
            f.write("\n## Computational Efficiency\n\n")
            f.write("| Method | Selection Rate | Rel. Speed | Adapted Params |\n")
            f.write("|--------|----------------|-----------|----------------|\n")

            # Find baseline throughput (Normal/Unadapted)
            baseline_throughput = None
            if 'Normal' in efficiency_averages:
                baseline_data = efficiency_averages['Normal'].get('throughput_samples_per_sec')
                if baseline_data:
                    baseline_throughput = baseline_data['mean']

            for method_name, _ in sorted_methods:
                if method_name in efficiency_averages:
                    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                    row = [display_name]

                    # Selection rate (% of samples updated)
                    if efficiency_averages[method_name]['adaptation_rate']:
                        rate = efficiency_averages[method_name]['adaptation_rate']['mean'] * 100
                        row.append(f"{rate:.1f}%")
                    else:
                        row.append("0.0%")

                    # Relative speed
                    if efficiency_averages[method_name]['throughput_samples_per_sec'] and baseline_throughput:
                        throughput = efficiency_averages[method_name]['throughput_samples_per_sec']['mean']
                        rel_speed = (throughput / baseline_throughput) * 100
                        row.append(f"{rel_speed:.1f}%")
                    else:
                        row.append("N/A")

                    # Adapted params
                    if efficiency_averages[method_name]['num_adapted_params']:
                        params = efficiency_averages[method_name]['num_adapted_params']['mean']
                        row.append(f"{int(params):,}")
                    else:
                        row.append("0")

                    f.write("| " + " | ".join(row) + " |\n")

    print("✓ Saved: summary_tables.md")

    # ========== PLAIN TEXT OUTPUT (similar to LaTeX table format) ==========
    output_file_txt = os.path.join(output_dir, 'summary_tables.txt')

    with open(output_file_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"PROTOPNET ROBUSTNESS EVALUATION SUMMARY (Severity {severity})\n")
        f.write("="*80 + "\n\n")

        # Table 1: Overall Accuracy by Category
        f.write("TABLE 1: Category-wise Robustness\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<20} {'Noise':>12} {'Blur':>12} {'Weather':>12} {'Digital':>12}\n")
        f.write("-"*80 + "\n")

        for method_name, _ in sorted_methods:
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            row = [f"{display_name:<20}"]
            for category in ['Noise', 'Blur', 'Weather', 'Digital']:
                if (method_name in category_averages and
                    category in category_averages[method_name] and
                    category_averages[method_name][category]):
                    mean = category_averages[method_name][category]['mean'] * 100
                    row.append(f"{mean:>11.1f}%")
                else:
                    row.append(f"{'N/A':>12}")
            f.write("".join(row) + "\n")

        f.write("-"*80 + "\n\n")

        # Table 2: Interpretability & Efficiency (LaTeX-style)
        if any(metrics_averages.values()) and any(efficiency_averages.values()):
            f.write("TABLE 2: Efficiency and Interpretability Analysis\n")
            f.write("-"*120 + "\n")
            f.write(f"{'Method':<20} {'PAC':>18} {'PCA-W':>18} {'Calib.':>10} {'Select.':>10} {'Rel.Speed':>12}\n")
            f.write("-"*120 + "\n")

            # Find baseline throughput
            baseline_throughput = None
            if 'Normal' in efficiency_averages:
                baseline_data = efficiency_averages['Normal'].get('throughput_samples_per_sec')
                if baseline_data:
                    baseline_throughput = baseline_data['mean']

            for method_name, _ in sorted_methods:
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                row = [f"{display_name:<20}"]

                # PAC (Mean ± Std)
                if (method_name in metrics_averages and
                    metrics_averages[method_name]['PAC_mean']):
                    pac_mean = metrics_averages[method_name]['PAC_mean']['mean'] * 100
                    pac_std = metrics_averages[method_name]['PAC_std']['mean'] * 100
                    row.append(f"{pac_mean:>5.1f} ± {pac_std:<5.1f}")
                else:
                    row.append(f"{'N/A':>18}")

                # PCA-Weighted (Mean ± Std)
                if (method_name in metrics_averages and
                    metrics_averages[method_name]['PCA_weighted_mean']):
                    pca_mean = metrics_averages[method_name]['PCA_weighted_mean']['mean'] * 100
                    pca_std = metrics_averages[method_name]['PCA_weighted_std']['mean'] * 100
                    row.append(f"{pca_mean:>5.1f} ± {pca_std:<5.1f}")
                else:
                    row.append(f"{'N/A':>18}")

                # Calibration
                if (method_name in metrics_averages and
                    metrics_averages[method_name]['calibration_agreement']):
                    calib = metrics_averages[method_name]['calibration_agreement']['mean'] * 100
                    row.append(f"{calib:>9.1f}%")
                else:
                    row.append(f"{'N/A':>10}")

                # Selection Rate
                if (method_name in efficiency_averages and
                    efficiency_averages[method_name]['adaptation_rate']):
                    rate = efficiency_averages[method_name]['adaptation_rate']['mean'] * 100
                    row.append(f"{rate:>9.1f}%")
                else:
                    row.append(f"{0.0:>9.1f}%")

                # Relative Speed
                if (method_name in efficiency_averages and
                    efficiency_averages[method_name]['throughput_samples_per_sec'] and
                    baseline_throughput):
                    throughput = efficiency_averages[method_name]['throughput_samples_per_sec']['mean']
                    rel_speed = (throughput / baseline_throughput) * 100
                    row.append(f"{rel_speed:>11.1f}%")
                else:
                    row.append(f"{'N/A':>12}")

                f.write("".join(row) + "\n")

            f.write("-"*120 + "\n")

            # Add baseline throughput as footnote
            if baseline_throughput:
                f.write(
                    "\nNote: Relative Speed compares throughput against Unadapted baseline "
                )
                f.write(f"(100% ≈ {baseline_throughput:.1f} samples/sec)\n")

            f.write("\nMetrics Legend:\n")
            f.write("  PAC: Prototype Activation Consistency (higher = better)\n")
            f.write("  PCA-W: Weighted Prototype Class Alignment (higher = better)\n")
            f.write("  Calib.: Calibration Agreement (higher = better)\n")
            f.write("  Select.: Selection Rate - % of samples triggering updates (lower = more selective)\n")
            f.write("  Rel.Speed: Relative throughput compared to baseline (higher = faster)\n")

    print("✓ Saved: summary_tables.txt")


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

        # Update METHOD_DISPLAY_NAMES
        global METHOD_DISPLAY_NAMES
        for old_name, new_name in rename_map.items():
            if old_name in METHOD_DISPLAY_NAMES:
                del METHOD_DISPLAY_NAMES[old_name]
            METHOD_DISPLAY_NAMES[new_name] = new_name

        return renamed_results

    return filtered_results


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization of ProtoPNet robustness evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m protopnet_tta.visualize_robustness_results --input results.json

  # Select specific methods and rename
  python -m protopnet_tta.visualize_robustness_results --input results.json \\
      --methods Normal Tent EATA ProtoEntropy-BN \\
      --rename ProtoEntropy-BN=ProtoTTA

  # Exclude corruptions
  python -m protopnet_tta.visualize_robustness_results --input results.json \\
      --exclude elastic_transform
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
                            'Example: --methods Normal Tent EATA ProtoEntropy-BN')
    parser.add_argument('--rename', nargs='*', default=[],
                       help='Rename methods using old=new format. '
                            'Example: --rename ProtoEntropy-BN=ProtoTTA')

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
    print("\\nGenerating visualizations...")

    plot_overall_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_category_comparison(results_dict, args.severity, args.output_dir, args.exclude)
    plot_per_corruption_heatmap(results_dict, args.severity, args.output_dir, args.exclude)
    generate_summary_tables(results_dict, args.severity, args.output_dir, args.exclude)

    print(f"\\n{'='*80}")
    print("✓ All visualizations and tables generated successfully!")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
