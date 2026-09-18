#!/usr/bin/env python3
"""
Simple script to generate a markdown table of method accuracies on different corruptions.

Usage:
    python -m protopnet_tta.generate_accuracy_table --input results.json --output table.md
"""

import argparse
import json


# Define corruption categories for grouping
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'Blur': ['gaussian_blur', 'defocus_blur'],
    'Weather': ['fog', 'frost', 'brightness'],
    'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform']
}

# Method display names for cleaner output
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


def generate_markdown_table(results_dict, severity='5', exclude_list=None):
    """Generate markdown table of accuracies.

    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude

    Returns:
        Markdown table as a string
    """
    if exclude_list is None:
        exclude_list = []

    # Get all methods and corruptions
    methods = list(results_dict.keys())

    # Collect all corruptions from all methods (excluding clean and exclude_list)
    all_corruptions = set()
    for method_data in results_dict.values():
        all_corruptions.update(method_data.keys())
    all_corruptions = sorted([c for c in all_corruptions if c != 'clean' and c not in exclude_list])

    # Build the table
    lines = []
    lines.append(f"# Accuracy Table (Severity {severity})\n")

    # Header row
    header = "| Method |"
    separator = "|--------|"
    for corr in all_corruptions:
        header += f" {corr.replace('_', ' ').title()} |"
        separator += "------------:|"
    header += " **Mean** |"
    separator += "--------:|"

    lines.append(header)
    lines.append(separator)

    # Data rows
    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row = f"| {display_name} |"
        accs = []

        for corr in all_corruptions:
            if corr in results_dict[method]:
                severities = results_dict[method][corr]
                if isinstance(severities, dict) and severity in severities:
                    acc = extract_accuracy(severities[severity])
                    if acc is not None:
                        row += f" {acc*100:.1f}% |"
                        accs.append(acc)
                    else:
                        row += " - |"
                else:
                    row += " - |"
            else:
                row += " - |"

        # Compute mean
        if accs:
            mean_acc = sum(accs) / len(accs)
            row += f" **{mean_acc*100:.1f}%** |"
        else:
            row += " - |"

        lines.append(row)

    return "\n".join(lines)


def generate_category_table(results_dict, severity='5', exclude_list=None):
    """Generate markdown table with category-wise averages.

    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude

    Returns:
        Markdown table as a string
    """
    if exclude_list is None:
        exclude_list = []

    methods = list(results_dict.keys())

    lines = []
    lines.append("\n## Category-wise Average Accuracy\n")

    # Header
    header = "| Method | Noise | Blur | Weather | Digital | **Overall** |"
    separator = "|--------|------:|-----:|--------:|--------:|------------:|"
    lines.append(header)
    lines.append(separator)

    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row = f"| {display_name} |"
        all_accs = []

        for category in ['Noise', 'Blur', 'Weather', 'Digital']:
            cat_accs = []
            for corr in CORRUPTION_CATEGORIES.get(category, []):
                if corr in exclude_list:
                    continue
                if corr in results_dict[method]:
                    severities = results_dict[method][corr]
                    if isinstance(severities, dict) and severity in severities:
                        acc = extract_accuracy(severities[severity])
                        if acc is not None:
                            cat_accs.append(acc)
                            all_accs.append(acc)

            if cat_accs:
                mean_acc = sum(cat_accs) / len(cat_accs)
                row += f" {mean_acc*100:.1f}% |"
            else:
                row += " - |"

        # Overall mean
        if all_accs:
            overall = sum(all_accs) / len(all_accs)
            row += f" **{overall*100:.1f}%** |"
        else:
            row += " - |"

        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate markdown table of method accuracies on corruptions'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output markdown file (default: print to stdout)')
    parser.add_argument('--severity', type=str, default='5',
                       help='Severity level to analyze (default: 5)')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Corruption types to exclude from analysis')
    parser.add_argument('--methods', nargs='*', default=None,
                       help='Specific methods to include (default: all)')
    parser.add_argument('--compact', action='store_true',
                       help='Only show category-wise table (compact view)')

    args = parser.parse_args()

    # Load results
    data = load_results(args.input)
    results_dict = data.get('results', {})

    if not results_dict:
        print("ERROR: No results found in JSON file")
        return

    # Filter methods if specified
    if args.methods:
        results_dict = {k: v for k, v in results_dict.items() if k in args.methods}

    # Generate tables
    output_lines = []

    if not args.compact:
        output_lines.append(generate_markdown_table(results_dict, args.severity, args.exclude))

    output_lines.append(generate_category_table(results_dict, args.severity, args.exclude))

    output_content = "\n".join(output_lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_content)
        print(f"✓ Table saved to: {args.output}")
    else:
        print(output_content)


if __name__ == '__main__':
    main()
