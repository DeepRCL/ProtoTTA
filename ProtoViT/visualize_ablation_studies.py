#!/usr/bin/env python3
"""
Visualize and summarize ablation study results.
Creates tables, bar plots, and heatmaps to understand which configurations work better.

Usage:
    python visualize_ablation_studies.py --input_dir ./ablation_studies --output_dir ./ablation_studies/visualizations
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'seaborn-v0_8-whitegrid') else 'default')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Experiment names and their display names
EXPERIMENT_NAMES = {
    'exp1_consensus': 'Experiment 1: Consensus Strategy',
    'exp2_adaptation': 'Experiment 2: Adaptation Mode',
    'exp3_target_prototypes': 'Experiment 3: Target vs All Prototypes',
    'exp4_geometric_filter': 'Experiment 4: Geometric Filter',
    'exp5_weighting': 'Experiment 5: Weighting Strategies'
}

# Corruption categories for grouping
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'Blur': ['gaussian_blur', 'defocus_blur'],
    'Weather': ['fog', 'frost', 'brightness'],
    'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform']
}


def load_experiment_results(input_dir):
    """Load all experiment results from JSON files."""
    results = {}
    input_path = Path(input_dir)
    
    for exp_name in EXPERIMENT_NAMES.keys():
        json_file = input_path / f"{exp_name}_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[exp_name] = {
                    'metadata': data.get('metadata', {}),
                    'results': data.get('results', {})
                }
        else:
            print(f"Warning: {json_file} not found")
    
    return results


def calculate_statistics(results_dict):
    """Calculate mean, std, min, max for each configuration."""
    stats = {}
    for config_name, corruptions in results_dict.items():
        accuracies = [acc for acc in corruptions.values() if acc is not None]
        if accuracies:
            stats[config_name] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'count': len(accuracies)
            }
        else:
            stats[config_name] = None
    return stats


def print_experiment_summary(exp_name, results_dict, output_file=None):
    """Print summary table for an experiment."""
    stats = calculate_statistics(results_dict)
    
    print(f"\n{'='*80}")
    print(f"{EXPERIMENT_NAMES.get(exp_name, exp_name)}")
    print(f"{'='*80}")
    print(f"{'Configuration':<30s} {'Mean Acc (%)':>12s} {'Std (%)':>10s} {'Min (%)':>10s} {'Max (%)':>10s} {'Count':>8s}")
    print("-"*80)
    
    # Sort by mean accuracy (descending)
    sorted_configs = sorted(stats.items(), 
                           key=lambda x: x[1]['mean'] if x[1] else 0, 
                           reverse=True)
    
    for config_name, stat in sorted_configs:
        if stat is not None:
            print(f"{config_name:<30s} {stat['mean']*100:11.2f}% "
                  f"{stat['std']*100:9.2f}% {stat['min']*100:9.2f}% "
                  f"{stat['max']*100:9.2f}% {stat['count']:7d}")
        else:
            print(f"{config_name:<30s} {'N/A':>12s}")
    
    print("="*80)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{EXPERIMENT_NAMES.get(exp_name, exp_name)}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Configuration':<30s} {'Mean Acc (%)':>12s} {'Std (%)':>10s} {'Min (%)':>10s} {'Max (%)':>10s} {'Count':>8s}\n")
            f.write("-"*80 + "\n")
            for config_name, stat in sorted_configs:
                if stat is not None:
                    f.write(f"{config_name:<30s} {stat['mean']*100:11.2f}% "
                           f"{stat['std']*100:9.2f}% {stat['min']*100:9.2f}% "
                           f"{stat['max']*100:9.2f}% {stat['count']:7d}\n")
                else:
                    f.write(f"{config_name:<30s} {'N/A':>12s}\n")
            f.write("="*80 + "\n")


def plot_experiment_bar(exp_name, results_dict, output_dir):
    """Create bar plot comparing configurations in an experiment."""
    stats = calculate_statistics(results_dict)
    
    # Filter out None stats
    valid_stats = {k: v for k, v in stats.items() if v is not None}
    if not valid_stats:
        print(f"No valid results for {exp_name}")
        return
    
    # Sort by mean accuracy
    sorted_configs = sorted(valid_stats.items(), 
                           key=lambda x: x[1]['mean'], 
                           reverse=True)
    
    config_names = [name for name, _ in sorted_configs]
    means = [stat['mean']*100 for _, stat in sorted_configs]
    stds = [stat['std']*100 for _, stat in sorted_configs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(config_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                  alpha=0.8, edgecolor='black', linewidth=1.2,
                  color=plt.cm.viridis(np.linspace(0, 1, len(config_names))))
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                f'{mean_val:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(EXPERIMENT_NAMES.get(exp_name, exp_name), 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(means) + max(stds) + 5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{exp_name}_barplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved bar plot to: {output_path}")


def plot_experiment_heatmap(exp_name, results_dict, output_dir):
    """Create heatmap showing performance per corruption type."""
    # Get all corruption types
    all_corruptions = set()
    for corruptions in results_dict.values():
        all_corruptions.update(corruptions.keys())
    all_corruptions = sorted([c for c in all_corruptions])
    
    if not all_corruptions:
        print(f"No corruption data for {exp_name}")
        return
    
    # Build data matrix
    config_names = sorted(results_dict.keys())
    data_matrix = []
    
    for config_name in config_names:
        row = []
        for corruption in all_corruptions:
            acc = results_dict[config_name].get(corruption)
            if acc is not None:
                row.append(acc * 100)
            else:
                row.append(np.nan)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(14, len(all_corruptions)*0.8), max(6, len(config_names)*0.5)))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(all_corruptions)))
    ax.set_yticks(np.arange(len(config_names)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in all_corruptions], 
                      rotation=45, ha='right')
    ax.set_yticklabels(config_names)
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(len(all_corruptions)):
            if not np.isnan(data_matrix[i, j]):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title(f'{EXPERIMENT_NAMES.get(exp_name, exp_name)}\nPerformance by Corruption Type', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{exp_name}_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def plot_category_comparison(exp_name, results_dict, output_dir):
    """Create grouped bar plot by corruption category."""
    stats = calculate_statistics(results_dict)
    valid_stats = {k: v for k, v in stats.items() if v is not None}
    if not valid_stats:
        return
    
    # Calculate category averages
    category_stats = defaultdict(dict)
    for config_name, corruptions in results_dict.items():
        if config_name not in valid_stats:
            continue
        for category, corruption_list in CORRUPTION_CATEGORIES.items():
            accuracies = []
            for corruption in corruption_list:
                if corruption in corruptions and corruptions[corruption] is not None:
                    accuracies.append(corruptions[corruption] * 100)
            if accuracies:
                category_stats[config_name][category] = np.mean(accuracies)
    
    if not category_stats:
        return
    
    categories = list(CORRUPTION_CATEGORIES.keys())
    config_names = sorted(valid_stats.keys(), 
                         key=lambda x: valid_stats[x]['mean'], 
                         reverse=True)
    
    x = np.arange(len(categories))
    width = 0.8 / len(config_names)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    for i, config_name in enumerate(config_names):
        means = [category_stats[config_name].get(cat, np.nan) for cat in categories]
        offset = (i - len(config_names)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=config_name, 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Corruption Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{EXPERIMENT_NAMES.get(exp_name, exp_name)}\nPerformance by Category', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{exp_name}_category_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved category comparison to: {output_path}")


def create_overall_summary(all_results, output_dir):
    """Create overall summary table comparing best configs from each experiment."""
    print(f"\n{'='*80}")
    print("OVERALL ABLATION SUMMARY")
    print(f"{'='*80}")
    
    summary_data = []
    
    for exp_name, exp_data in all_results.items():
        results_dict = exp_data.get('results', {})
        stats = calculate_statistics(results_dict)
        
        # Find best configuration
        valid_stats = {k: v for k, v in stats.items() if v is not None}
        if valid_stats:
            best_config = max(valid_stats.items(), key=lambda x: x[1]['mean'])
            config_name, stat = best_config
            summary_data.append({
                'experiment': EXPERIMENT_NAMES.get(exp_name, exp_name),
                'best_config': config_name,
                'mean_acc': stat['mean'] * 100,
                'std': stat['std'] * 100
            })
    
    # Print summary table
    print(f"{'Experiment':<40s} {'Best Config':<30s} {'Mean Acc (%)':>12s} {'Std (%)':>10s}")
    print("-"*80)
    for row in summary_data:
        print(f"{row['experiment']:<40s} {row['best_config']:<30s} "
              f"{row['mean_acc']:11.2f}% {row['std']:9.2f}%")
    print("="*80)
    
    # Save to file
    summary_file = os.path.join(output_dir, 'overall_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("OVERALL ABLATION SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Experiment':<40s} {'Best Config':<30s} {'Mean Acc (%)':>12s} {'Std (%)':>10s}\n")
        f.write("-"*80 + "\n")
        for row in summary_data:
            f.write(f"{row['experiment']:<40s} {row['best_config']:<30s} "
                   f"{row['mean_acc']:11.2f}% {row['std']:9.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"\nSaved overall summary to: {summary_file}")
    
    # Create bar plot of best configs
    if summary_data:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        experiments = [row['experiment'] for row in summary_data]
        means = [row['mean_acc'] for row in summary_data]
        stds = [row['std'] for row in summary_data]
        configs = [row['best_config'] for row in summary_data]
        
        x = np.arange(len(experiments))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                     alpha=0.8, edgecolor='black', linewidth=1.2,
                     color=plt.cm.viridis(np.linspace(0, 1, len(experiments))))
        
        # Add value labels
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                    f'{mean_val:.2f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Best Configuration Performance by Experiment', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([e.split(':')[0] for e in experiments], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(means) + max(stds) + 5])
        
        # Add legend with config names
        legend_labels = [f"{exp.split(':')[0]}: {config}" 
                        for exp, config in zip(experiments, configs)]
        ax.legend(bars, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'overall_summary_barplot.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overall summary plot to: {output_path}")


def save_detailed_csv(all_results, output_dir):
    """Save detailed results to CSV format."""
    import csv
    
    csv_file = os.path.join(output_dir, 'detailed_results.csv')
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Experiment', 'Configuration', 'Corruption', 'Accuracy'])
        
        # Data rows
        for exp_name, exp_data in all_results.items():
            results_dict = exp_data.get('results', {})
            for config_name, corruptions in results_dict.items():
                for corruption, acc in corruptions.items():
                    if acc is not None:
                        writer.writerow([
                            EXPERIMENT_NAMES.get(exp_name, exp_name),
                            config_name,
                            corruption,
                            f"{acc*100:.4f}"
                        ])
    
    print(f"Saved detailed CSV to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and summarize ablation study results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, default='./ablation_studies',
                       help='Directory containing ablation study JSON files')
    parser.add_argument('--output_dir', type=str, default='./ablation_studies/visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all results
    print("Loading ablation study results...")
    all_results = load_experiment_results(args.input_dir)
    
    if not all_results:
        print("No results found! Make sure JSON files exist in the input directory.")
        return
    
    print(f"Loaded {len(all_results)} experiments")
    
    # Process each experiment
    for exp_name, exp_data in all_results.items():
        results_dict = exp_data.get('results', {})
        if not results_dict:
            print(f"Skipping {exp_name} (no results)")
            continue
        
        print(f"\nProcessing {exp_name}...")
        
        # Print summary table
        summary_file = os.path.join(args.output_dir, f'{exp_name}_summary.txt')
        print_experiment_summary(exp_name, results_dict, summary_file)
        
        # Create visualizations
        plot_experiment_bar(exp_name, results_dict, args.output_dir)
        plot_experiment_heatmap(exp_name, results_dict, args.output_dir)
        plot_category_comparison(exp_name, results_dict, args.output_dir)
    
    # Create overall summary
    create_overall_summary(all_results, args.output_dir)
    
    # Save detailed CSV
    save_detailed_csv(all_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

