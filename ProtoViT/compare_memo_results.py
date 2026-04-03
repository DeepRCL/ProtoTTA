#!/usr/bin/env python3
"""
Quick script to compare MEMO performance with other methods.

Usage:
    python compare_memo_results.py --input robustness_results_sev5_with_memo.json
"""

import argparse
import json
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Compare MEMO with other TTA methods')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to results JSON with MEMO')
    args = parser.parse_args()
    
    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Get all methods and corruptions
    methods = list(results.keys())
    corruptions = []
    for method in methods:
        if method in results and isinstance(results[method], dict):
            corruptions = list(results[method].keys())
            break
    
    severity = '5'
    
    # Build comparison table
    comparison_data = []
    
    for corruption in corruptions:
        row = {'Corruption': corruption}
        for method in methods:
            if method in results and corruption in results[method]:
                if severity in results[method][corruption]:
                    result = results[method][corruption][severity]
                    if result is not None:
                        if isinstance(result, dict):
                            acc = result.get('accuracy', 0)
                        else:
                            acc = result
                        row[method] = acc * 100
                    else:
                        row[method] = np.nan
                else:
                    row[method] = np.nan
            else:
                row[method] = np.nan
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Calculate mean performance
    mean_row = {'Corruption': 'MEAN'}
    for method in methods:
        if method in df.columns:
            mean_row[method] = df[method].mean()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Print table
    print("\n" + "="*100)
    print("ROBUSTNESS COMPARISON (Accuracy % on CUB-200-C, Severity 5)")
    print("="*100)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*100)
    
    # Print MEMO vs others comparison
    if 'memo' in df.columns and 'normal' in df.columns:
        print("\n" + "="*100)
        print("MEMO vs NORMAL (Improvement in %)")
        print("="*100)
        
        memo_vs_normal = []
        for idx, row in df.iterrows():
            if row['Corruption'] != 'MEAN':
                corruption = row['Corruption']
                memo_acc = row.get('memo', np.nan)
                normal_acc = row.get('normal', np.nan)
                
                if not np.isnan(memo_acc) and not np.isnan(normal_acc):
                    improvement = memo_acc - normal_acc
                    memo_vs_normal.append({
                        'Corruption': corruption,
                        'Normal': f'{normal_acc:.2f}',
                        'MEMO': f'{memo_acc:.2f}',
                        'Δ': f'{improvement:+.2f}'
                    })
        
        comparison_df = pd.DataFrame(memo_vs_normal)
        print(comparison_df.to_string(index=False))
        print("="*100)
    
    # Rank methods by mean performance
    print("\n" + "="*100)
    print("METHOD RANKING (by Mean Accuracy)")
    print("="*100)
    
    mean_perf = {}
    for method in methods:
        if method in df.columns:
            # Get mean (last row)
            mean_perf[method] = df[df['Corruption'] == 'MEAN'][method].values[0]
    
    # Sort by performance
    sorted_methods = sorted(mean_perf.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (method, acc) in enumerate(sorted_methods, 1):
        print(f"{rank}. {method:<20s} {acc:>6.2f}%")
    
    print("="*100)
    
    # Statistical summary for MEMO
    if 'memo' in df.columns:
        print("\n" + "="*100)
        print("MEMO STATISTICAL SUMMARY")
        print("="*100)
        
        memo_accs = df[df['Corruption'] != 'MEAN']['memo'].dropna()
        
        print(f"Mean Accuracy:    {memo_accs.mean():.2f}%")
        print(f"Std Dev:          {memo_accs.std():.2f}%")
        print(f"Min Accuracy:     {memo_accs.min():.2f}% ({df[df['memo'] == memo_accs.min()]['Corruption'].values[0]})")
        print(f"Max Accuracy:     {memo_accs.max():.2f}% ({df[df['memo'] == memo_accs.max()]['Corruption'].values[0]})")
        
        print("="*100)


if __name__ == '__main__':
    main()
