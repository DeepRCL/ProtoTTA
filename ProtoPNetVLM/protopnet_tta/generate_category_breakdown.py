#!/usr/bin/env python3
"""
Generate category-wise summary (like ProtoViT's category tables).
Shows performance breakdown by corruption categories: Noise, Blur, Weather, Digital.
"""

import json

def main():
    # Load results
    with open('results.json', 'r') as f:
        data = json.load(f)

    results = data['results']
    severity = '5'

    # Define categories
    categories = {
        'Noise': ['gaussian_noise', 'shot_noise'],  # Only have these two in current results
        # 'Blur': ['gaussian_blur', 'defocus_blur'],
        # 'Weather': ['fog', 'frost', 'brightness'],
        # 'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform']
    }

    methods = ['Normal', 'Tent', 'EATA', 'SAR', 'ProtoEntropy', 'ProtoEntropy-BN']
    method_names = {
        'Normal': 'Unadapted',
        'ProtoEntropy': 'ProtoTTA',
        'ProtoEntropy-BN': 'ProtoTTA-BN'
    }

    print("="*100)
    print("CATEGORY-WISE ROBUSTNESS BREAKDOWN (Severity 5)")
    print("="*100)
    print()

    for category, corruption_list in categories.items():
        print(f"\n### {category} Corruptions ###\n")

        # Header
        header = f"{'Method':<20}"
        for corruption in corruption_list:
            header += f" {corruption.replace('_', ' ').title():>18}"
        header += f" {'Average':>12}"
        print(header)
        print("-" * len(header))

        # Data rows
        for method in methods:
            display_name = method_names.get(method, method)
            row = f"{display_name:<20}"

            accuracies = []
            for corruption in corruption_list:
                if corruption in results[method] and severity in results[method][corruption]:
                    acc = results[method][corruption][severity]['accuracy'] * 100
                    accuracies.append(acc)
                    row += f" {acc:>17.1f}%"
                else:
                    row += f" {'N/A':>18}"

            # Average
            if accuracies:
                avg = sum(accuracies) / len(accuracies)
                row += f" {avg:>11.1f}%"
            else:
                row += f" {'N/A':>12}"

            print(row)

        print()

    print("="*100)
    print("\nKEY FINDINGS:")
    print("-"*100)

    # Calculate improvements
    if 'gaussian_noise' in results['Normal'] and 'gaussian_noise' in results['ProtoEntropy-BN']:
        baseline = results['Normal']['gaussian_noise'][severity]['accuracy'] * 100
        prototta_bn = results['ProtoEntropy-BN']['gaussian_noise'][severity]['accuracy'] * 100
        improvement = prototta_bn - baseline
        rel_improvement = (improvement / baseline) * 100

        print(f"1. ProtoTTA-BN achieves {prototta_bn:.1f}% accuracy on gaussian_noise")
        print(f"   - Absolute improvement over Unadapted: +{improvement:.1f}%")
        print(f"   - Relative improvement: +{rel_improvement:.1f}%")

    if 'gaussian_noise' in results['ProtoEntropy'] and 'gaussian_noise' in results['ProtoEntropy-BN']:
        prototta = results['ProtoEntropy']['gaussian_noise'][severity]['accuracy'] * 100
        prototta_bn = results['ProtoEntropy-BN']['gaussian_noise'][severity]['accuracy'] * 100
        diff = prototta_bn - prototta

        print(f"\n2. ProtoTTA-BN vs ProtoTTA on gaussian_noise:")
        print(f"   - ProtoTTA-BN: {prototta_bn:.1f}%")
        print(f"   - ProtoTTA: {prototta:.1f}%")
        print(f"   - Difference: +{diff:.1f}% (BN-only adaptation is more selective)")

    # Compare with other methods
    if 'gaussian_noise' in results['EATA'] and 'gaussian_noise' in results['ProtoEntropy-BN']:
        eata = results['EATA']['gaussian_noise'][severity]['accuracy'] * 100
        prototta_bn = results['ProtoEntropy-BN']['gaussian_noise'][severity]['accuracy'] * 100
        diff = prototta_bn - eata

        print(f"\n3. ProtoTTA-BN vs EATA on gaussian_noise:")
        print(f"   - ProtoTTA-BN: {prototta_bn:.1f}%")
        print(f"   - EATA: {eata:.1f}%")
        print(f"   - Difference: +{diff:.1f}%")

    print("\n" + "="*100)

    # Save to file
    # (Redirect stdout to capture the output if needed)

if __name__ == '__main__':
    main()
