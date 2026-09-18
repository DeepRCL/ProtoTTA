#!/usr/bin/env python3
"""Generate comprehensive summary tables for ProtoPNet robustness results."""

import json

def main():
    # Load results
    with open('results.json', 'r') as f:
        data = json.load(f)

    results = data['results']
    severity = '5'
    methods = ['Normal', 'Tent', 'EATA', 'SAR', 'ProtoEntropy', 'ProtoEntropy-BN']
    method_names = {'Normal': 'Unadapted', 'ProtoEntropy': 'ProtoTTA', 'ProtoEntropy-BN': 'ProtoTTA-BN'}

    corruptions_noise = ['gaussian_noise', 'shot_noise']

    output = []
    output.append("="*120)
    output.append("PROTOPNET ROBUSTNESS EVALUATION RESULTS (Severity 5 - SICAPv2)")
    output.append("="*120)
    output.append("")

    # ========== TABLE 1: Main Results ==========
    output.append("TABLE 1: Efficiency and Interpretability Analysis on SICAPv2-C")
    output.append("Semantic Consistency (PAC) and Prototype Alignment (PCA-W) reported as Mean ± Std")
    output.append("Selection Rate = % of samples triggering updates")
    output.append("Relative Speed = throughput vs Unadapted baseline (100% ≈ {:.1f} samples/sec)".format(
        results['Normal']['gaussian_noise'][severity]['efficiency']['throughput_samples_per_sec']
    ))
    output.append("")
    output.append(f"{'Method':<20} {'Acc(Noise)':>12} {'PAC':>20} {'PCA-W':>20} {'Calib':>10} {'Select%':>10} {'Speed%':>10}")
    output.append("-"*120)

    baseline_tp = results['Normal']['gaussian_noise'][severity]['efficiency']['throughput_samples_per_sec']

    for method in methods:
        # Noise average (gaussian + shot noise)
        acc_list = []
        for corruption in corruptions_noise:
            if corruption in results[method] and severity in results[method][corruption]:
                acc = results[method][corruption][severity]['accuracy']
                acc_list.append(acc * 100)
        noise_avg = sum(acc_list) / len(acc_list) if acc_list else 0

        # Get metrics from gaussian_noise
        res = results[method]['gaussian_noise'][severity] if 'gaussian_noise' in results[method] else {}

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

        # Format row
        pac_str = f"{pac_mean:.1f} ± {pac_std:.1f}"
        pca_str = f"{pca_mean:.1f} ± {pca_std:.1f}"

        display_name = method_names.get(method, method)
        output.append(f"{display_name:<20} {noise_avg:>11.1f}% {pac_str:>19} {pca_str:>19} {calib:>9.1f}% {select_rate:>9.1f}% {rel_speed:>9.1f}%")

    output.append("-"*120)
    output.append("")
    output.append("Column Descriptions:")
    output.append("  - Acc(Noise): Classification accuracy averaged over Noise corruptions (gaussian_noise, shot_noise)")
    output.append("  - PAC: Prototype Activation Consistency - measures semantic consistency (higher = better)")
    output.append("  - PCA-W: Weighted Prototype Class Alignment - measures prototype-class alignment (higher = better)")
    output.append("  - Calib: Calibration Agreement - top prototype matches prediction (higher = better)")
    output.append("  - Select%: Selection Rate - % samples triggering updates (lower = more selective)")
    output.append("  - Speed%: Relative computational speed vs baseline (higher = faster)")
    output.append("")
    output.append("="*120)

    # Print to stdout
    for line in output:
        print(line)

    # Save to file
    with open('summary_table_detailed.txt', 'w') as f:
        for line in output:
            f.write(line + '\n')

if __name__ == '__main__':
    main()
