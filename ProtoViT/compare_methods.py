#!/usr/bin/env python3
"""
Compare performance differences between adaptation methods across corruption types.
Shows how much better/worse one method is compared to another for each corruption.

Usage:
    # Compare EATA vs ProtoEntropy
    python compare_methods.py --input robustness_results_sev5.json \
        --method1 eata --method2 proto_imp_conf_v1 --output_dir ./plots/comparison
    
    # Compare multiple methods against baseline
    python compare_methods.py --input robustness_results_sev5.json \
        --baseline normal --methods tent eata proto_imp_conf_v1 --output_dir ./plots/comparison
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Method display names
METHOD_DISPLAY_NAMES = {
    'normal': 'Normal',
    'tent': 'Tent',
    'proto_imp_conf_v1': 'Proto-Imp+Conf (v1)',
    'proto_imp_conf_v2': 'Proto-Imp+Conf (v2)',
    'proto_imp_conf_v3': 'Proto-Imp+Conf (v3)',
    'loss': 'LossAdapt',
    'eata': 'EATA',
    'sar': 'SAR'
}

# Color palette
COLORS = {
    'positive': '#2ca02c',  # Green for improvements
    'negative': '#d62728',   # Red for declines
    'neutral': '#7f7f7f'    # Gray for no change
}


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def calculate_differences(results_dict, method1, method2, severity='5', exclude_list=None):
    """Calculate performance differences between two methods for each corruption.
    
    Returns:
        Dictionary mapping corruption_type -> {
            'method1_acc': float,
            'method2_acc': float,
            'difference': float (method2 - method1, in percentage points),
            'relative_improvement': float (percentage improvement)
        }
    """
    if exclude_list is None:
        exclude_list = []
    
    differences = {}
    
    # Get all corruption types that both methods have
    corruptions1 = set(results_dict.get(method1, {}).keys())
    corruptions2 = set(results_dict.get(method2, {}).keys())
    all_corruptions = corruptions1 & corruptions2
    
    # Filter excluded
    all_corruptions = [c for c in all_corruptions if c not in exclude_list]
    
    for corruption_type in all_corruptions:
        acc1 = None
        acc2 = None
        
        if (corruption_type in results_dict.get(method1, {}) and
            severity in results_dict[method1][corruption_type]):
            acc1 = results_dict[method1][corruption_type][severity]
        
        if (corruption_type in results_dict.get(method2, {}) and
            severity in results_dict[method2][corruption_type]):
            acc2 = results_dict[method2][corruption_type][severity]
        
        if acc1 is not None and acc2 is not None:
            diff = (acc2 - acc1) * 100  # Convert to percentage points
            relative_improvement = ((acc2 - acc1) / acc1 * 100) if acc1 > 0 else 0
            
            differences[corruption_type] = {
                'method1_acc': acc1 * 100,
                'method2_acc': acc2 * 100,
                'difference': diff,
                'relative_improvement': relative_improvement
            }
    
    return differences


def compare_multiple_to_baseline(results_dict, baseline, methods, severity='5', exclude_list=None):
    """Compare multiple methods against a baseline.
    
    Returns:
        Dictionary mapping method_name -> {
            corruption_type -> {
                'baseline_acc': float,
                'method_acc': float,
                'difference': float,
                'relative_improvement': float
            }
        }
    """
    if exclude_list is None:
        exclude_list = []
    
    comparisons = {}
    
    for method in methods:
        if method == baseline:
            continue
        
        differences = calculate_differences(
            results_dict, baseline, method, severity, exclude_list
        )
        
        # Rename keys to be more descriptive
        method_comparison = {}
        for corruption_type, diff_data in differences.items():
            method_comparison[corruption_type] = {
                'baseline_acc': diff_data['method1_acc'],
                'method_acc': diff_data['method2_acc'],
                'difference': diff_data['difference'],
                'relative_improvement': diff_data['relative_improvement']
            }
        
        comparisons[method] = method_comparison
    
    return comparisons


def plot_difference_bar(differences, method1_name, method2_name, output_dir='./plots'):
    """Create bar plot showing performance differences."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not differences:
        print("No differences to plot.")
        return
    
    corruptions = sorted(differences.keys())
    diffs = [differences[c]['difference'] for c in corruptions]
    
    # Color bars based on positive/negative
    colors = [COLORS['positive'] if d >= 0 else COLORS['negative'] for d in diffs]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(corruptions))
    bars = ax.bar(x, diffs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        height = bar.get_height()
        label_y = height + (1 if height >= 0 else -3)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{diff:+.2f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Difference (Percentage Points)', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Difference: {METHOD_DISPLAY_NAMES.get(method2_name, method2_name)} vs {METHOD_DISPLAY_NAMES.get(method1_name, method1_name)}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in corruptions],
                       rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['positive'], label='Improvement'),
        Patch(facecolor=COLORS['negative'], label='Decline')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'difference_{method1_name}_vs_{method2_name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved difference bar plot to: {output_path}")


def plot_multiple_comparisons(comparisons, baseline_name, output_dir='./plots'):
    """Create grouped bar plot comparing multiple methods against baseline."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not comparisons:
        print("No comparisons to plot.")
        return
    
    # Get all corruption types
    all_corruptions = set()
    for method_comparison in comparisons.values():
        all_corruptions.update(method_comparison.keys())
    all_corruptions = sorted(list(all_corruptions))
    
    methods = list(comparisons.keys())
    x = np.arange(len(all_corruptions))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Color palette for methods
    method_colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method_name in enumerate(methods):
        diffs = []
        for corruption in all_corruptions:
            if corruption in comparisons[method_name]:
                diffs.append(comparisons[method_name][corruption]['difference'])
            else:
                diffs.append(0)
        
        offset = (i - len(methods)/2 + 0.5) * width
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        
        bars = ax.bar(x + offset, diffs, width, 
                     label=display_name, color=method_colors[i], alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Difference vs Baseline (Percentage Points)', 
                 fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison: All Methods vs {METHOD_DISPLAY_NAMES.get(baseline_name, baseline_name)}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in all_corruptions],
                       rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'multiple_comparisons_vs_{baseline_name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved multiple comparisons plot to: {output_path}")


def plot_difference_heatmap(comparisons, baseline_name, output_dir='./plots'):
    """Create heatmap showing differences for all methods vs baseline."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not comparisons:
        print("No comparisons to plot.")
        return
    
    # Get all corruption types
    all_corruptions = set()
    for method_comparison in comparisons.values():
        all_corruptions.update(method_comparison.keys())
    all_corruptions = sorted(list(all_corruptions))
    
    methods = list(comparisons.keys())
    
    # Prepare data matrix
    data_matrix = []
    method_labels = []
    
    for method_name in methods:
        row = []
        for corruption in all_corruptions:
            if corruption in comparisons[method_name]:
                row.append(comparisons[method_name][corruption]['difference'])
            else:
                row.append(np.nan)
        data_matrix.append(row)
        method_labels.append(METHOD_DISPLAY_NAMES.get(method_name, method_name))
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use diverging colormap (red-white-green)
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-max(abs(data_matrix[~np.isnan(data_matrix)])) if not np.all(np.isnan(data_matrix)) else -10,
                   vmax=max(abs(data_matrix[~np.isnan(data_matrix)])) if not np.all(np.isnan(data_matrix)) else 10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(all_corruptions)))
    ax.set_yticks(np.arange(len(method_labels)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in all_corruptions])
    ax.set_yticklabels(method_labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(method_labels)):
        for j in range(len(all_corruptions)):
            if not np.isnan(data_matrix[i, j]):
                text = ax.text(j, i, f'{data_matrix[i, j]:+.1f}',
                             ha="center", va="center", color="black", 
                             fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Difference (Percentage Points)', 
                   rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title(f'Performance Difference Heatmap: All Methods vs {METHOD_DISPLAY_NAMES.get(baseline_name, baseline_name)}',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'difference_heatmap_vs_{baseline_name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved difference heatmap to: {output_path}")


def print_difference_table(differences, method1_name, method2_name):
    """Print formatted table of differences."""
    if not differences:
        print("No differences to display.")
        return
    
    print("\n" + "="*100)
    print(f"PERFORMANCE DIFFERENCE: {METHOD_DISPLAY_NAMES.get(method2_name, method2_name)} vs {METHOD_DISPLAY_NAMES.get(method1_name, method1_name)}")
    print("="*100)
    print(f"{'Corruption':<25s} {'Method1 Acc':>12s} {'Method2 Acc':>12s} {'Difference':>12s} {'Rel. Impr.':>12s}")
    print("-"*100)
    
    # Sort by difference (descending)
    sorted_corruptions = sorted(differences.items(), 
                              key=lambda x: x[1]['difference'], 
                              reverse=True)
    
    for corruption_type, diff_data in sorted_corruptions:
        print(f"{corruption_type.replace('_', ' ').title():<25s} "
              f"{diff_data['method1_acc']:11.2f}% "
              f"{diff_data['method2_acc']:11.2f}% "
              f"{diff_data['difference']:+11.2f}% "
              f"{diff_data['relative_improvement']:+10.2f}%")
    
    # Summary statistics
    all_diffs = [d['difference'] for d in differences.values()]
    print("-"*100)
    print(f"{'Average Difference':<25s} {np.mean(all_diffs):+11.2f}%")
    print(f"{'Median Difference':<25s} {np.median(all_diffs):+11.2f}%")
    print(f"{'Max Improvement':<25s} {np.max(all_diffs):+11.2f}%")
    print(f"{'Max Decline':<25s} {np.min(all_diffs):+11.2f}%")
    print(f"{'Corruptions Improved':<25s} {sum(1 for d in all_diffs if d > 0):>11d}")
    print(f"{'Corruptions Declined':<25s} {sum(1 for d in all_diffs if d < 0):>11d}")
    print("="*100)


def print_multiple_comparison_table(comparisons, baseline_name):
    """Print formatted table comparing multiple methods against baseline."""
    if not comparisons:
        print("No comparisons to display.")
        return
    
    print("\n" + "="*100)
    print(f"MULTIPLE METHODS COMPARISON vs {METHOD_DISPLAY_NAMES.get(baseline_name, baseline_name)}")
    print("="*100)
    
    # Get all corruption types
    all_corruptions = set()
    for method_comparison in comparisons.values():
        all_corruptions.update(method_comparison.keys())
    all_corruptions = sorted(list(all_corruptions))
    
    # Header
    header = f"{'Corruption':<25s}"
    for method_name in comparisons.keys():
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        header += f" {display_name[:15]:>15s}"
    print(header)
    print("-"*100)
    
    # Data rows
    for corruption in all_corruptions:
        row = f"{corruption.replace('_', ' ').title():<25s}"
        for method_name in comparisons.keys():
            if corruption in comparisons[method_name]:
                diff = comparisons[method_name][corruption]['difference']
                row += f" {diff:+14.2f}%"
            else:
                row += f" {'N/A':>15s}"
        print(row)
    
    # Summary row
    print("-"*100)
    row = f"{'Average Difference':<25s}"
    for method_name in comparisons.keys():
        diffs = [c['difference'] for c in comparisons[method_name].values()]
        if diffs:
            row += f" {np.mean(diffs):+14.2f}%"
        else:
            row += f" {'N/A':>15s}"
    print(row)
    print("="*100)


def save_differences_json(differences, method1_name, method2_name, output_file):
    """Save differences to JSON file."""
    output_data = {
        'method1': method1_name,
        'method2': method2_name,
        'method1_display': METHOD_DISPLAY_NAMES.get(method1_name, method1_name),
        'method2_display': METHOD_DISPLAY_NAMES.get(method2_name, method2_name),
        'differences': {}
    }
    
    for corruption_type, diff_data in differences.items():
        output_data['differences'][corruption_type] = {
            'method1_acc': float(diff_data['method1_acc']),
            'method2_acc': float(diff_data['method2_acc']),
            'difference': float(diff_data['difference']),
            'relative_improvement': float(diff_data['relative_improvement'])
        }
    
    # Add summary statistics
    all_diffs = [d['difference'] for d in differences.values()]
    output_data['summary'] = {
        'average_difference': float(np.mean(all_diffs)),
        'median_difference': float(np.median(all_diffs)),
        'max_improvement': float(np.max(all_diffs)),
        'max_decline': float(np.min(all_diffs)),
        'corruptions_improved': int(sum(1 for d in all_diffs if d > 0)),
        'corruptions_declined': int(sum(1 for d in all_diffs if d < 0)),
        'total_corruptions': len(all_diffs)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved differences to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare performance differences between adaptation methods',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input JSON results file')
    parser.add_argument('--output_dir', type=str, default='./plots/comparison',
                       help='Directory to save output plots')
    
    # Two comparison modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--method1', type=str,
                      help='First method to compare (for pairwise comparison)')
    group.add_argument('--baseline', type=str,
                      help='Baseline method (for multiple comparisons)')
    
    parser.add_argument('--method2', type=str,
                       help='Second method to compare (required if --method1 is used)')
    parser.add_argument('--methods', nargs='+',
                       help='Methods to compare against baseline (required if --baseline is used)')
    
    parser.add_argument('--severity', type=str, default='5',
                       help='Severity level to analyze (default: 5)')
    parser.add_argument('--exclude', nargs='+', default=[],
                       help='Corruption types to exclude from analysis')
    parser.add_argument('--save_json', type=str, default=None,
                       help='Path to save differences JSON (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method1 and not args.method2:
        parser.error("--method2 is required when using --method1")
    if args.baseline and not args.methods:
        parser.error("--methods is required when using --baseline")
    
    # Process exclude list
    exclude_list = [c.strip().lower() for c in args.exclude] if args.exclude else []
    
    if exclude_list:
        print(f"Excluding corruptions: {', '.join(exclude_list)}")
    
    # Load results
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results_dict = data['results']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform comparison
    if args.method1:
        # Pairwise comparison
        print(f"\nComparing {args.method1} vs {args.method2}...")
        differences = calculate_differences(
            results_dict, args.method1, args.method2, 
            severity=args.severity, exclude_list=exclude_list
        )
        
        if not differences:
            print("No valid comparisons found. Check method names and data.")
            return
        
        # Print table
        print_difference_table(differences, args.method1, args.method2)
        
        # Generate plot
        print("\nGenerating plots...")
        plot_difference_bar(differences, args.method1, args.method2, output_dir=args.output_dir)
        
        # Save JSON if requested
        if args.save_json:
            save_differences_json(differences, args.method1, args.method2, args.save_json)
    
    else:
        # Multiple comparisons against baseline
        print(f"\nComparing methods against baseline {args.baseline}...")
        comparisons = compare_multiple_to_baseline(
            results_dict, args.baseline, args.methods,
            severity=args.severity, exclude_list=exclude_list
        )
        
        if not comparisons:
            print("No valid comparisons found. Check method names and data.")
            return
        
        # Print table
        print_multiple_comparison_table(comparisons, args.baseline)
        
        # Generate plots
        print("\nGenerating plots...")
        plot_multiple_comparisons(comparisons, args.baseline, output_dir=args.output_dir)
        plot_difference_heatmap(comparisons, args.baseline, output_dir=args.output_dir)
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

