#!/usr/bin/env python3
"""
Generate summary tables for ProtoPNet robustness evaluation results.
This is a lightweight version that only requires json and basic Python libraries.

Usage:
    python generate_summary_tables.py --input results.json --output_file summary.txt
"""

import json
import argparse
from collections import defaultdict


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


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def get_method_averages(results_dict, severity='5', exclude_list=None):
    """Calculate overall average accuracy for each method."""
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
            mean_val = sum(accuracies) / len(accuracies)
            std_val = (sum((x - mean_val) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
            method_averages[method_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': min(accuracies),
                'max': max(accuracies),
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
                mean_val = sum(accuracies) / len(accuracies)
                std_val = (sum((x - mean_val) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
                category_averages[method_name][category] = {
                    'mean': mean_val,
                    'std': std_val,
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
                mean_val = sum(method_metrics[metric]) / len(method_metrics[metric])
                std_val = (sum((x - mean_val) ** 2 for x in method_metrics[metric]) / len(method_metrics[metric])) ** 0.5
                metrics_averages[method_name][metric] = {
                    'mean': mean_val,
                    'std': std_val,
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
                       'num_adapted_params', 'adaptation_ratio']

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
                mean_val = sum(method_efficiency[key]) / len(method_efficiency[key])
                efficiency_averages[method_name][key] = {
                    'mean': mean_val,
                    'count': len(method_efficiency[key])
                }
            else:
                efficiency_averages[method_name][key] = None

    return efficiency_averages


def generate_summary_tables(results_dict, severity, output_file, exclude_list=None):
    """Generate plain text tables summarizing results."""
    method_averages = get_method_averages(results_dict, severity, exclude_list)
    category_averages = get_category_averages(results_dict, severity, exclude_list)
    metrics_averages = get_prototype_metrics_averages(results_dict, severity, exclude_list)
    efficiency_averages = get_efficiency_metrics_averages(results_dict, severity, exclude_list)

    # Sort methods by accuracy
    sorted_methods = sorted(method_averages.items(),
                           key=lambda x: x[1]['mean'] if x[1] else 0,
                           reverse=True)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"PROTOPNET ROBUSTNESS EVALUATION SUMMARY (Severity {severity})\n")
        f.write("="*80 + "\n\n")

        # ===== TABLE 1: Category-wise Robustness =====
        f.write("TABLE 1: Category-wise Robustness\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Method':<20} {'Noise':>12} {'Blur':>12} "
            f"{'Weather':>12} {'Digital':>12}\n"
        )
        f.write("-" * 80 + "\n")

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

        f.write("-" * 80 + "\n\n")

        # ===== TABLE 2: Interpretability & Efficiency =====
        if any(metrics_averages.values()) and any(efficiency_averages.values()):
            f.write("TABLE 2: Efficiency and Interpretability Analysis\n")
            f.write(
                "(Semantic Consistency (PAC) and Prototype Alignment (PCA-W) "
                "reported as Mean ± Std)\n"
            )
            f.write(
                "(Selection Rate = % samples triggering updates; Relative Speed "
                "vs Unadapted baseline)\n"
            )
            f.write("-" * 120 + "\n")
            f.write(
                f"{'Method':<20} {'PAC':>18} {'PCA-W':>18} "
                f"{'Calib.':>10} {'Select.':>10} {'Rel.Speed':>12}\n"
            )
            f.write("-" * 120 + "\n")

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

            f.write("-" * 120 + "\n")

            # Add baseline throughput as footnote
            if baseline_throughput:
                f.write(
                    "\nNote: Relative Speed compares throughput against "
                    "Unadapted baseline "
                )
                f.write(f"(100% ≈ {baseline_throughput:.1f} samples/sec)\n")

            f.write("\nColumn Descriptions:\n")
            f.write(
                "  - PAC (Semantic Consistency): Prototype Activation Consistency "
                "(higher = better)\n"
            )
            f.write(
                "  - PCA-W (Prototype Alignment): Weighted Prototype Class "
                "Alignment (higher = better)\n"
            )
            f.write(
                "  - Calibration: Agreement between prototype and prediction "
                "(higher = better)\n"
            )
            f.write(
                "  - Selection Rate: % of samples triggering updates "
                "(lower = more selective)\n"
            )
            f.write(
                "  - Relative Speed: Throughput relative to baseline "
                "(higher = faster)\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary tables for ProtoPNet robustness evaluation results'
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output_file', type=str, default='summary_tables.txt',
                       help='Output file for summary tables')
    parser.add_argument('--severity', type=str, default='5',
                       help='Severity level to analyze (default: 5)')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Corruption types to exclude from analysis')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results_dict = data.get('results', {})

    if not results_dict:
        print("ERROR: No results found in JSON file")
        return

    print(f"Found {len(results_dict)} methods in results")
    print(f"Methods: {', '.join(results_dict.keys())}")

    if args.exclude:
        print(f"Excluding corruptions: {', '.join(args.exclude)}")

    # Generate summary tables
    print("\nGenerating summary tables...")
    generate_summary_tables(results_dict, args.severity, args.output_file, args.exclude)

    print(f"\n{'=' * 80}")
    print("✓ Summary tables generated successfully!")
    print(f"  Output file: {args.output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
