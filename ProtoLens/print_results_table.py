#!/usr/bin/env python3
"""
Quick script to print a markdown table of all models on all corruption-severity settings.
"""

import json
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Print markdown table of robustness results')
    parser.add_argument('--input', type=str, 
                       default='Datasets/Amazon-C/results/robustness_results_main.json',
                       help='Input JSON file')
    args = parser.parse_args()
    
    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    metadata = data.get('metadata', {})
    
    # Get all methods, corruptions, and severities
    methods = list(results.keys())
    corruption_types = metadata.get('corruption_types', [])
    severities = metadata.get('severities', [])
    
    # If not in metadata, infer from results
    if not corruption_types or not severities:
        all_corruptions = set()
        all_severities = set()
        for method_data in results.values():
            for corruption in method_data.keys():
                all_corruptions.add(corruption)
                for severity in method_data[corruption].keys():
                    all_severities.add(int(severity))
        corruption_types = sorted(list(all_corruptions))
        severities = sorted(list(all_severities))
    
    # Build column headers: corruption_severity
    columns = []
    for corruption in corruption_types:
        for severity in severities:
            columns.append(f"{corruption}_s{severity}")
    
    # Build table data
    table_data = []
    for method in methods:
        row = [method]
        for corruption in corruption_types:
            for severity in severities:
                sev_str = str(severity)
                accuracy = None
                if method in results:
                    if corruption in results[method]:
                        if sev_str in results[method][corruption]:
                            result = results[method][corruption][sev_str]
                            if result and 'accuracy' in result:
                                accuracy = result['accuracy']
                row.append(f"{accuracy:.4f}" if accuracy is not None else "N/A")
        table_data.append(row)
    
    # Print markdown table
    header = ["Method"] + columns
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    
    for row in table_data:
        print("| " + " | ".join(row) + " |")

if __name__ == '__main__':
    main()
