#!/usr/bin/env python3
"""
Visualization script for TTA method comparison including efficiency metrics.

Generates comprehensive plots showing:
1. Accuracy comparison across methods and corruptions
2. Prototype metrics (PAC, PCA, Sparsity) comparison
3. Efficiency metrics (time, adapted parameters)
4. Trade-off analyses (accuracy vs efficiency)

Usage:
    python visualize_tta_comparison.py --input results.json --output ./figures/
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(input_file: str) -> Dict:
    """Load results from JSON file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data


def extract_metrics_dataframe(results: Dict) -> pd.DataFrame:
    """Extract all metrics into a pandas DataFrame for easy plotting."""
    rows = []
    
    for corruption_key, corruption_data in results['results'].items():
        if '-' in corruption_key:
            corruption, severity = corruption_key.rsplit('-', 1)
        else:
            corruption = corruption_key
            severity = 'unknown'
        
        for method, metrics in corruption_data.items():
            if metrics is None:
                continue
            
            row = {
                'corruption': corruption,
                'severity': severity,
                'method': method,
                'accuracy': metrics.get('accuracy', 0) * 100,  # Convert to percentage
            }
            
            # Add prototype metrics if available
            if 'PAC_mean' in metrics:
                row['PAC'] = metrics['PAC_mean']
            if 'PCA_mean' in metrics:
                row['PCA'] = metrics['PCA_mean']
            if 'sparsity_gini_mean' in metrics:
                row['Sparsity'] = metrics['sparsity_gini_mean']
            if 'PCA_weighted_mean' in metrics:
                row['PCA_Weighted'] = metrics['PCA_weighted_mean']
            if 'calibration_agreement' in metrics:
                row['Calibration'] = metrics['calibration_agreement']
            if 'gt_class_contrib_improvement' in metrics:
                row['GT_Delta'] = metrics['gt_class_contrib_improvement']
            
            # Add efficiency metrics if available
            if 'efficiency' in metrics:
                eff = metrics['efficiency']
                row['time_per_sample_ms'] = eff.get('time_per_sample_ms', 0)
                row['num_adapted_params'] = eff.get('num_adapted_params', 0)
                row['adaptation_ratio'] = eff.get('adaptation_ratio', 0) * 100  # Percentage
                row['throughput'] = eff.get('throughput_samples_per_sec', 0)
            
            # Add adaptation stats if available
            if 'adaptation_stats' in metrics:
                stats = metrics['adaptation_stats']
                total_samples = max(stats.get('total_samples', 0), 1)
                row['Adapt_Rate'] = (stats.get('adapted_samples', 0) / total_samples) * 100
                row['Updates_Per_Sample'] = stats.get('total_updates', 0) / total_samples
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_accuracy_by_corruption(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy comparison across corruptions."""
    methods = df['method'].unique()
    corruptions = df['corruption'].unique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plot
    x = np.arange(len(corruptions))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        accuracies = [
            method_data[method_data['corruption'] == corr]['accuracy'].mean()
            for corr in corruptions
        ]
        ax.bar(x + i * width, accuracies, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Corruption Types')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_corruption.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: accuracy_by_corruption.png")


def plot_average_accuracy(df: pd.DataFrame, output_dir: Path):
    """Plot average accuracy across all corruptions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    avg_acc = df.groupby('method')['accuracy'].mean().sort_values(ascending=True)
    
    colors = sns.color_palette("RdYlGn", len(avg_acc))
    avg_acc.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Average Accuracy (%)')
    ax.set_ylabel('Method')
    ax.set_title('Average Accuracy Across All Corruptions')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (method, acc) in enumerate(avg_acc.items()):
        ax.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'average_accuracy.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: average_accuracy.png")


def plot_prototype_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot prototype-based metrics comparison."""
    if 'PAC' not in df.columns:
        print("⚠ No prototype metrics found, skipping prototype metric plots")
        return
    
    methods = df['method'].unique()
    metrics_to_plot = ['PAC', 'PCA', 'Sparsity', 'PCA_Weighted', 'Calibration', 'GT_Delta']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not available_metrics:
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Average across corruptions
        avg_metric = df.groupby('method')[metric].mean().sort_values(ascending=False)
        
        colors = sns.color_palette("viridis", len(avg_metric))
        avg_metric.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(metric)
        ax.set_xlabel('Method')
        ax.set_title(f'{metric} (Higher is Better)' if metric != 'Sparsity' else f'{metric} (Gini Coefficient)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (method, val) in enumerate(avg_metric.items()):
            ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prototype_metrics.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: prototype_metrics.png")


def plot_efficiency_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot efficiency metrics comparison."""
    if 'time_per_sample_ms' not in df.columns:
        print("⚠ No efficiency metrics found, skipping efficiency plots")
        return
    
    # Average across corruptions
    agg_cols = {
        'time_per_sample_ms': 'mean',
        'adaptation_ratio': 'mean',
        'throughput': 'mean'
    }
    if 'Adapt_Rate' in df.columns:
        agg_cols['Adapt_Rate'] = 'mean'
    if 'Updates_Per_Sample' in df.columns:
        agg_cols['Updates_Per_Sample'] = 'mean'
    
    avg_df = df.groupby('method').agg(agg_cols).reset_index()
    
    # Sort by time
    avg_df = avg_df.sort_values('time_per_sample_ms')
    
    # Build dynamic list of plots
    plots = [
        ('time_per_sample_ms', 'Time per Sample (ms)', 'Inference Time Comparison'),
        ('adaptation_ratio', 'Adapted Parameters (%)', 'Parameter Adaptation Footprint')
    ]
    if 'Adapt_Rate' in avg_df.columns:
        plots.append(('Adapt_Rate', 'Adapt %', 'Adaptation Rate'))
    if 'Updates_Per_Sample' in avg_df.columns:
        plots.append(('Updates_Per_Sample', 'Updates/Sample', 'Updates Per Sample'))
    
    fig, axes = plt.subplots(1, len(plots), figsize=(7 * len(plots), 5))
    if len(plots) == 1:
        axes = [axes]
    
    # Plot 1: Time per sample
    ax1 = axes[0]
    colors = sns.color_palette("coolwarm_r", len(avg_df))
    bars = ax1.barh(avg_df['method'], avg_df['time_per_sample_ms'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Time per Sample (ms)')
    ax1.set_ylabel('Method')
    ax1.set_title('Inference Time Comparison')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (method, time) in enumerate(zip(avg_df['method'], avg_df['time_per_sample_ms'])):
        ax1.text(time + 0.2, i, f'{time:.2f} ms', va='center', fontsize=8)
    
    # Plot 2: Adapted parameters ratio
    ax2 = axes[1]
    sorted_df = avg_df.sort_values('adaptation_ratio')
    colors2 = sns.color_palette("YlOrRd", len(sorted_df))
    bars2 = ax2.barh(sorted_df['method'], sorted_df['adaptation_ratio'], color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Adapted Parameters (%)')
    ax2.set_ylabel('Method')
    ax2.set_title('Parameter Adaptation Footprint')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (method, ratio) in enumerate(zip(sorted_df['method'], sorted_df['adaptation_ratio'])):
        ax2.text(ratio + 0.05, i, f'{ratio:.2f}%', va='center', fontsize=8)
    
    plot_idx = 2
    if 'Adapt_Rate' in avg_df.columns:
        ax = axes[plot_idx]
        sorted_df = avg_df.sort_values('Adapt_Rate')
        colors3 = sns.color_palette("Greens", len(sorted_df))
        ax.barh(sorted_df['method'], sorted_df['Adapt_Rate'], color=colors3, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Adapt %')
        ax.set_ylabel('Method')
        ax.set_title('Adaptation Rate')
        ax.grid(axis='x', alpha=0.3)
        for i, (method, val) in enumerate(zip(sorted_df['method'], sorted_df['Adapt_Rate'])):
            ax.text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=8)
        plot_idx += 1
    
    if 'Updates_Per_Sample' in avg_df.columns:
        ax = axes[plot_idx]
        sorted_df = avg_df.sort_values('Updates_Per_Sample')
        colors4 = sns.color_palette("Blues", len(sorted_df))
        ax.barh(sorted_df['method'], sorted_df['Updates_Per_Sample'], color=colors4, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Updates/Sample')
        ax.set_ylabel('Method')
        ax.set_title('Updates Per Sample')
        ax.grid(axis='x', alpha=0.3)
        for i, (method, val) in enumerate(zip(sorted_df['method'], sorted_df['Updates_Per_Sample'])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_metrics.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: efficiency_metrics.png")


def plot_accuracy_vs_efficiency_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs efficiency trade-off scatter plot."""
    if 'time_per_sample_ms' not in df.columns:
        print("⚠ No efficiency metrics found, skipping trade-off plot")
        return
    
    # Average across corruptions
    avg_df = df.groupby('method').agg({
        'accuracy': 'mean',
        'time_per_sample_ms': 'mean',
        'adaptation_ratio': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy vs Time
    ax1 = axes[0]
    scatter1 = ax1.scatter(avg_df['time_per_sample_ms'], avg_df['accuracy'], 
                          s=150, alpha=0.6, c=range(len(avg_df)), cmap='viridis', 
                          edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for i, row in avg_df.iterrows():
        ax1.annotate(row['method'], 
                    (row['time_per_sample_ms'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax1.set_xlabel('Time per Sample (ms)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs. Speed Trade-off')
    ax1.grid(alpha=0.3)
    
    # Add pareto frontier
    sorted_by_time = avg_df.sort_values('time_per_sample_ms')
    pareto_front = []
    max_acc = -np.inf
    for _, row in sorted_by_time.iterrows():
        if row['accuracy'] > max_acc:
            pareto_front.append(row)
            max_acc = row['accuracy']
    if pareto_front:
        pareto_df = pd.DataFrame(pareto_front)
        ax1.plot(pareto_df['time_per_sample_ms'], pareto_df['accuracy'], 
                'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
        ax1.legend()
    
    # Plot 2: Accuracy vs Adapted Parameters
    ax2 = axes[1]
    scatter2 = ax2.scatter(avg_df['adaptation_ratio'], avg_df['accuracy'], 
                          s=150, alpha=0.6, c=range(len(avg_df)), cmap='plasma',
                          edgecolors='black', linewidth=1)
    
    # Add labels
    for i, row in avg_df.iterrows():
        ax2.annotate(row['method'], 
                    (row['adaptation_ratio'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Adapted Parameters (%)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs. Parameter Efficiency')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_efficiency_tradeoff.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: accuracy_efficiency_tradeoff.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of accuracy across methods and corruptions."""
    # Pivot data
    pivot = df.pivot_table(values='accuracy', index='method', columns='corruption', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=pivot.values.mean(),
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Method')
    ax.set_title('Accuracy Heatmap: Methods vs. Corruptions')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_heatmap.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: accuracy_heatmap.png")


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary table with all metrics."""
    # Average across corruptions
    summary = df.groupby('method').agg({
        'accuracy': ['mean', 'std'],
    }).round(2)
    
    # Add prototype metrics if available
    if 'PAC' in df.columns:
        summary[('PAC', 'mean')] = df.groupby('method')['PAC'].mean().round(4)
    if 'PCA' in df.columns:
        summary[('PCA', 'mean')] = df.groupby('method')['PCA'].mean().round(4)
    if 'Sparsity' in df.columns:
        summary[('Sparsity', 'mean')] = df.groupby('method')['Sparsity'].mean().round(4)
    
    # Add efficiency metrics if available
    if 'time_per_sample_ms' in df.columns:
        summary[('Time (ms)', 'mean')] = df.groupby('method')['time_per_sample_ms'].mean().round(2)
    if 'adaptation_ratio' in df.columns:
        summary[('Adapted %', 'mean')] = df.groupby('method')['adaptation_ratio'].mean().round(2)
    
    # Save as CSV
    summary.to_csv(output_dir / 'summary_table.csv')
    
    # Save as LaTeX
    latex_table = summary.to_latex(float_format='%.2f')
    with open(output_dir / 'summary_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary.to_string())
    print("="*80)
    print(f"\n✓ Saved: summary_table.csv and summary_table.tex")


def main():
    parser = argparse.ArgumentParser(description='Visualize TTA method comparison results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON results file from evaluate_robustness.py')
    parser.add_argument('--output', type=str, default='./figures/',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TTA Method Comparison Visualization")
    print(f"{'='*80}\n")
    print(f"Input file: {args.input}")
    print(f"Output directory: {output_dir}\n")
    
    # Load results
    print("Loading results...")
    results = load_results(args.input)
    
    # Extract metrics into DataFrame
    print("Extracting metrics...")
    df = extract_metrics_dataframe(results)
    print(f"✓ Extracted data for {len(df['method'].unique())} methods across {len(df['corruption'].unique())} corruptions\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    plot_average_accuracy(df, output_dir)
    plot_accuracy_by_corruption(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_prototype_metrics(df, output_dir)
    plot_efficiency_metrics(df, output_dir)
    plot_accuracy_vs_efficiency_tradeoff(df, output_dir)
    
    # Generate summary table
    print("\nGenerating summary table...")
    generate_summary_table(df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
