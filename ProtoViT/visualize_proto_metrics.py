#!/usr/bin/env python3
"""
Visualize prototype-based TTA metrics results.

This script creates visualizations comparing different TTA methods using:
- Accuracy vs. PAC (Prototype Activation Consistency)
- Accuracy vs. PCA (Prototype Class Alignment)
- Radar plots for multi-metric comparison
- Bar charts for method comparison

Usage:
    python visualize_proto_metrics.py \
        --input proto_metrics_results.json \
        --output_dir ./plots/proto_metrics
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_accuracy_vs_pac(results, output_dir):
    """Plot Accuracy vs. Prototype Activation Consistency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    method_colors = {}
    
    for corruption_idx, (corruption_type, corruption_results) in enumerate(results.items()):
        for method_idx, (method_name, metrics) in enumerate(corruption_results.items()):
            acc = metrics.get('accuracy', 0) * 100
            pac = metrics.get('PAC_mean', 0) * 100
            
            # Assign consistent colors to methods
            if method_name not in method_colors:
                method_colors[method_name] = colors[len(method_colors) % len(colors)]
            
            ax.scatter(pac, acc, s=150, alpha=0.7, 
                      color=method_colors[method_name],
                      label=method_name if corruption_idx == 0 else "",
                      edgecolors='black', linewidth=1)
            
            # Add method label
            ax.annotate(method_name, (pac, acc), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('PAC - Prototype Activation Consistency (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs. Prototype Activation Consistency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_pac.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: accuracy_vs_pac.png")


def plot_accuracy_vs_pca(results, output_dir):
    """Plot Accuracy vs. Prototype Class Alignment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    method_colors = {}
    
    for corruption_idx, (corruption_type, corruption_results) in enumerate(results.items()):
        for method_idx, (method_name, metrics) in enumerate(corruption_results.items()):
            acc = metrics.get('accuracy', 0) * 100
            pca = metrics.get('PCA_mean', 0) * 100
            
            # Assign consistent colors to methods
            if method_name not in method_colors:
                method_colors[method_name] = colors[len(method_colors) % len(colors)]
            
            ax.scatter(pca, acc, s=150, alpha=0.7,
                      color=method_colors[method_name],
                      label=method_name if corruption_idx == 0 else "",
                      edgecolors='black', linewidth=1)
            
            # Add method label
            ax.annotate(method_name, (pca, acc),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('PCA - Prototype Class Alignment (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs. Prototype Class Alignment', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: accuracy_vs_pca.png")


def plot_method_comparison_bars(results, output_dir):
    """Create grouped bar chart comparing methods across metrics."""
    # Aggregate results across corruptions
    methods = {}
    
    for corruption_type, corruption_results in results.items():
        for method_name, metrics in corruption_results.items():
            if method_name not in methods:
                methods[method_name] = {
                    'acc': [], 'pac': [], 'pca': [], 'sparsity': [],
                    'pca_weighted': [], 'calib': []
                }
            
            methods[method_name]['acc'].append(metrics.get('accuracy', 0) * 100)
            methods[method_name]['pac'].append(metrics.get('PAC_mean', 0) * 100)
            methods[method_name]['pca'].append(metrics.get('PCA_mean', 0) * 100)
            methods[method_name]['sparsity'].append(metrics.get('sparsity_gini_mean', 0))
            methods[method_name]['pca_weighted'].append(metrics.get('PCA_weighted_mean', 0))
            methods[method_name]['calib'].append(metrics.get('calibration_agreement', 0) * 100)
    
    # Average across corruptions
    for method_name in methods:
        for metric in methods[method_name]:
            methods[method_name][metric] = np.mean(methods[method_name][metric])
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    width = 0.6
    
    # Accuracy
    ax = axes[0, 0]
    acc_vals = [methods[m]['acc'] for m in method_names]
    bars = ax.bar(x, acc_vals, width, color='steelblue', edgecolor='black', linewidth=1)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Classification Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # PAC
    ax = axes[0, 1]
    pac_vals = [methods[m]['pac'] for m in method_names]
    bars = ax.bar(x, pac_vals, width, color='coral', edgecolor='black', linewidth=1)
    ax.set_ylabel('PAC (%)', fontweight='bold')
    ax.set_title('Prototype Activation Consistency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # PCA
    ax = axes[1, 0]
    pca_vals = [methods[m]['pca'] for m in method_names]
    bars = ax.bar(x, pca_vals, width, color='mediumseagreen', edgecolor='black', linewidth=1)
    ax.set_ylabel('PCA (%)', fontweight='bold')
    ax.set_title('Prototype Class Alignment', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Sparsity
    ax = axes[1, 1]
    # PCA-Weighted
    ax = axes[0, 2]
    pca_w_vals = [methods[m]['pca_weighted'] for m in method_names]
    bars = ax.bar(x, pca_w_vals, width, color='goldenrod', edgecolor='black', linewidth=1)
    ax.set_ylabel('PCA-Weighted', fontweight='bold')
    ax.set_title('PCA-Weighted', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Calibration
    ax = axes[1, 2]
    calib_vals = [methods[m]['calib'] for m in method_names]
    bars = ax.bar(x, calib_vals, width, color='teal', edgecolor='black', linewidth=1)
    ax.set_ylabel('Calibration (%)', fontweight='bold')
    ax.set_title('Calibration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    sparsity_vals = [methods[m]['sparsity'] for m in method_names]
    bars = ax.bar(x, sparsity_vals, width, color='mediumpurple', edgecolor='black', linewidth=1)
    ax.set_ylabel('Gini Coefficient', fontweight='bold')
    ax.set_title('Prototype Activation Sparsity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('TTA Methods Comparison - Prototype-Based Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: method_comparison_bars.png")


def plot_radar_chart(results, output_dir):
    """Create radar chart for multi-metric comparison."""
    from math import pi
    
    # Aggregate results
    methods = {}
    
    for corruption_type, corruption_results in results.items():
        for method_name, metrics in corruption_results.items():
            if method_name not in methods:
                methods[method_name] = {
                    'acc': [], 'pac': [], 'pca': [], 'sparsity': [],
                    'pca_weighted': [], 'calib': []
                }
            
            methods[method_name]['acc'].append(metrics.get('accuracy', 0) * 100)
            methods[method_name]['pac'].append(metrics.get('PAC_mean', 0) * 100)
            methods[method_name]['pca'].append(metrics.get('PCA_mean', 0) * 100)
            # Normalize sparsity to 0-100 scale
            methods[method_name]['sparsity'].append(metrics.get('sparsity_gini_mean', 0) * 100)
            methods[method_name]['pca_weighted'].append(metrics.get('PCA_weighted_mean', 0) * 100)
            methods[method_name]['calib'].append(metrics.get('calibration_agreement', 0) * 100)
    
    # Average
    for method_name in methods:
        for metric in methods[method_name]:
            methods[method_name][metric] = np.mean(methods[method_name][metric])
    
    # Create radar chart
    categories = ['Accuracy', 'PAC', 'PCA', 'Sparsity', 'PCA-Weighted', 'Calibration']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, (method_name, method_metrics) in enumerate(methods.items()):
        values = [
            method_metrics['acc'],
            method_metrics['pac'],
            method_metrics['pca'],
            method_metrics['sparsity'],
            method_metrics['pca_weighted'],
            method_metrics['calib']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Metric Comparison of TTA Methods', 
                 size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: radar_comparison.png")


def generate_summary_table(results, output_dir):
    """Generate a markdown table summarizing results."""
    # Aggregate results
    methods = {}
    
    for corruption_type, corruption_results in results.items():
        for method_name, metrics in corruption_results.items():
            if method_name not in methods:
                methods[method_name] = {
                    'acc': [], 'pac': [], 'pca': [], 
                    'sparsity': [], 'pac_std': [], 'pca_std': []
                }
            
            methods[method_name]['acc'].append(metrics.get('accuracy', 0) * 100)
            methods[method_name]['pac'].append(metrics.get('PAC_mean', 0) * 100)
            methods[method_name]['pca'].append(metrics.get('PCA_mean', 0) * 100)
            methods[method_name]['sparsity'].append(metrics.get('sparsity_gini_mean', 0))
            methods[method_name]['pac_std'].append(metrics.get('PAC_std', 0) * 100)
            methods[method_name]['pca_std'].append(metrics.get('PCA_std', 0) * 100)
    
    # Compute statistics
    summary = []
    for method_name in methods:
        summary.append({
            'Method': method_name,
            'Accuracy': f"{np.mean(methods[method_name]['acc']):.2f}%",
            'PAC': f"{np.mean(methods[method_name]['pac']):.2f}% ± {np.mean(methods[method_name]['pac_std']):.2f}",
            'PCA': f"{np.mean(methods[method_name]['pca']):.2f}% ± {np.mean(methods[method_name]['pca_std']):.2f}",
            'Sparsity': f"{np.mean(methods[method_name]['sparsity']):.3f}",
        })
    
    # Write markdown table
    output_file = os.path.join(output_dir, 'summary_table.md')
    with open(output_file, 'w') as f:
        f.write("# Prototype-Based TTA Metrics Summary\n\n")
        f.write("| Method | Accuracy | PAC (Consistency) | PCA (Alignment) | Sparsity (Gini) |\n")
        f.write("|--------|----------|-------------------|-----------------|----------------|\n")
        
        for row in summary:
            f.write(f"| {row['Method']} | {row['Accuracy']} | {row['PAC']} | {row['PCA']} | {row['Sparsity']} |\n")
        
        f.write("\n**Metrics:**\n")
        f.write("- **Accuracy**: Classification accuracy on corrupted images\n")
        f.write("- **PAC (Prototype Activation Consistency)**: Similarity between clean and adapted prototype activations (higher = better preservation)\n")
        f.write("- **PCA (Prototype Class Alignment)**: Proportion of top-k prototypes matching true class (higher = better semantic alignment)\n")
        f.write("- **Sparsity (Gini)**: Gini coefficient of prototype activations (higher = more sparse/selective)\n")
    
    print(f"✓ Saved: summary_table.md")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize prototype-based TTA metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output_dir', type=str, default='./plots/proto_metrics',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results = data.get('results', {})
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_vs_pac(results, args.output_dir)
    plot_accuracy_vs_pca(results, args.output_dir)
    plot_method_comparison_bars(results, args.output_dir)
    plot_radar_chart(results, args.output_dir)
    generate_summary_table(results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("✓ All visualizations generated successfully!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
