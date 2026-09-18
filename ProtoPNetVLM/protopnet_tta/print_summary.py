#!/usr/bin/env python3
"""Simple summary generator without complex dependencies."""

import json
import sys

def main():
    # Load results
    with open('kan-head/results.json', 'r') as f:
        data = json.load(f)

    results = data['results']
    severity = '5'

    # Calculate category averages
    categories = {
        'Noise': ['gaussian_noise', 'shot_noise'],
        'Blur': [],
        'Weather': [],
        'Digital': []
    }

    methods = list(results.keys())

    print("="*80)
    print("PROTOPNET ROBUSTNESS SUMMARY (Severity 5)")
    print("="*80)
    print()
    print(f"{'Method':<20} {'Noise':>12} {'PAC':>20} {'PCA-W':>20} {'Calib':>10} {'Select%':>10} {'Speed%':>10}")
    print("-"*110)

    # Get baseline throughput
    baseline_tp = results['Normal']['gaussian_noise'][severity]['efficiency']['throughput_samples_per_sec']

    for method in methods:
        # Get Noise average (gaussian_noise and shot_noise)
        acc_list = []
        for corruption in ['gaussian_noise', 'shot_noise']:
            if corruption in results[method] and severity in results[method][corruption]:
                acc = results[method][corruption][severity]['accuracy']
                acc_list.append(acc * 100)
        noise_avg = sum(acc_list) / len(acc_list) if acc_list else 0

        # Get metrics from gaussian_noise
        res =results[method]['gaussian_noise'][severity] if 'gaussian_noise' in results[method] else {}

        pac_mean = res.get('PAC_mean', 0) * 100 if res.get('PAC_mean') else 0
        pac_std = res.get('PAC_std', 0) * 100 if res.get('PAC_std') else 0
        pca_mean = res.get('PCA_weighted_mean', 0) * 100 if res.get('PCA_weighted_mean') else 0
        pca_std = res.get('PCA_weighted_std', 0) * 100 if res.get('PCA_weighted_std') else 0
        calib = res.get('calibration_agreement', 0) * 100 if res.get('calibration_agreement') else 0

        # Efficiency
        eff = res.get('efficiency', {})
        adapt_stats = res.get('adaptation_stats', {})
        select_rate = 0
        if adapt_stats:
            total = adapt_stats.get('total_samples', 1)
            adapted = adapt_stats.get('adapted_samples', 0)
            select_rate = (adapted / total) * 100

        tp = eff.get('throughput_samples_per_sec', 0)
        rel_speed = (tp / baseline_tp) * 100 if baseline_tp else 0

        # Print row
        pac_str = f"{pac_mean:.1f}±{pac_std:.1f}"
        pca_str = f"{pca_mean:.1f}±{pca_std:.1f}"

        display_name = {'Normal': 'Unadapted', 'ProtoEntropy': 'ProtoTTA', 'ProtoEntropy-BN': 'ProtoTTA-BN'}.get(method, method)
        print(f"{display_name:<20} {noise_avg:>11.1f}% {pac_str:>19} {pca_str:>19} {calib:>9.1f}% {select_rate:>9.1f}% {rel_speed:>9.1f}%")

    print("-"*110)
    print()
    print(f"Note: Baseline speed = {baseline_tp:.1f} samples/sec")
    print("="*80)

if __name__ == '__main__':
    main()
