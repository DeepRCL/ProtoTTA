# Robustness Results Visualization Guide

## Overview

The `visualize_robustness_results.py` script is a comprehensive tool for analyzing and visualizing robustness evaluation results with prototype-based TTA metrics **and computational efficiency metrics**.

## Features

### 1. Accuracy Visualizations
- **Overall Comparison**: Bar chart showing mean accuracy across all corruptions
- **Category Comparison**: Grouped bars comparing methods across Noise/Blur/Weather/Digital categories
- **Corruption Heatmap**: Detailed heatmap of accuracy for each method×corruption combination

### 2. Prototype Metrics Visualizations
- **Prototype Metrics Comparison**: Six-panel comparison of:
  - PAC (Prototype Activation Consistency)
  - PCA (Prototype Class Alignment)
  - Sparsity (Gini coefficient)
  - PCA-Weighted
  - Calibration Agreement
  - GT Class Contribution Δ
- **Radar Chart**: Multi-metric comparison showing all metrics on a single plot

### 3. Efficiency Metrics Visualizations ⭐ NEW
- **Efficiency Comparison**: Six-panel comparison of:
  - Time per Sample (ms)
  - Throughput (samples/sec)
  - Number of Adapted Parameters
  - Adaptation Ratio (% of params)
  - Steps per Sample
  - Peak GPU Memory (MB)

### 4. Summary Tables
- Overall accuracy statistics (mean, std, min, max)
- Category-wise breakdown
- Complete prototype metrics table with legend
- **Computational efficiency table** ⭐ NEW

## Usage

### Basic Usage
```bash
python visualize_robustness_results.py \
    --input robustness_results_sev5_metrics_with_memo.json \
    --output_dir ./plots/robustness_analysis \
    --severity 5
```

### Select Specific Methods and Rename (Recommended)
```bash
# Show only best methods and rename v3 to ProtoTTA
python visualize_robustness_results.py \
    --input robustness_results_sev5_with_memo.json \
    --output_dir ./plots/paper_results_v2 \
    --severity 5 \
    --methods normal tent eata sar memo proto_imp_conf_v3 \
    --rename proto_imp_conf_v3=ProtoTTA \
    --exclude saturate spatter
```

### Multiple Renames
```bash
# Rename all variants for paper
python visualize_robustness_results.py \
    --input robustness_results_sev5_metrics.json \
    --output_dir ./plots/ablation_study \
    --methods normal proto_imp_conf_v1 proto_imp_conf_v2 proto_imp_conf_v3 \
    --rename proto_imp_conf_v1=ProtoTTA-Full \
           proto_imp_conf_v2=ProtoTTA-LayerNorm \
           proto_imp_conf_v3=ProtoTTA
```

## Output Files

The script generates **7 files**:
- `overall_comparison.png` - Bar chart of mean accuracy
- `category_comparison.png` - Category-wise grouped bars
- `corruption_heatmap.png` - Method×corruption heatmap
- `prototype_metrics_comparison.png` - 6-panel prototype metrics
- `efficiency_comparison.png` - **6-panel efficiency metrics** ⭐ NEW
- `radar_comparison.png` - Multi-metric radar chart
- `summary_tables.md` - Comprehensive markdown tables

## Key Results Interpretation

### Accuracy Metrics
- **Mean Accuracy**: Overall robustness across all corruptions
- **Std Dev**: Consistency across different corruption types (lower = more consistent)
- **Category Performance**: Shows which corruption categories are most challenging

### Prototype Metrics
- **PAC (88-92%)**: Higher = model preserves prototype structure better during adaptation
- **PCA (27-38%)**: Higher = prototypes align better with true class
- **Sparsity (~0.46-0.47)**: Moderate sparsity indicates balanced prototype activation
- **PCA-Weighted (0.56-0.83)**: Importance-weighted alignment (higher = better)
- **Calibration (45-68%)**: How often top prototype matches prediction
- **GT Δ (-4 to -10)**: Contribution improvement (less negative = better)

### Efficiency Metrics ⭐ NEW
- **Time/Sample (4-9 ms)**: Lower = faster inference
- **Throughput (113-232 samp/s)**: Higher = better performance
- **Adapted Params (0-37K)**: Number of parameters updated during TTA
- **Param % (0-0.13%)**: Fraction of total model parameters adapted
- **Update % (58-100%)**: **Percentage of samples that triggered an update** - **Lower = more selective** ⭐ KEY METRIC
- **Steps/Sample (0-0.008)**: Optimizer steps per sample
- **Memory (4.6-10.2 GB)**: Peak GPU memory usage

## Example Results (from your data)

### Best Method: ProtoTTA (v3)
**Accuracy:**
- Mean: 60.06% (+8.17% vs Normal)
- Noise: 55.74% (+15.2% vs Normal)
- Blur: 44.62% (+3.7% vs Normal)

**Prototype Quality:**
- PAC: 91.9% (excellent preservation)
- PCA: 38.2% (best alignment)
- Calibration: 68.7% (strong agreement)
- GT Δ: -4.27 (best improvement)

**Efficiency:**
- Time/Sample: 4.49 ms (only 4% slower than baseline)
- Throughput: 222.8 samp/s
- Adapted Params: 37,632 (0.13% of model)
- **Update %: 58.0%** ⭐ **Only updates 58% of samples (vs 100% for Tent)** - More selective!
- Memory: 10.2 GB (vs 4.6 GB baseline)

### Key Insights
1. **ProtoTTA is fast**: Only 0.18 ms overhead vs Normal (4%)
2. **ProtoTTA is efficient**: Adapts only 37K params (0.13% of 28M total)
3. **ProtoTTA is selective**: Updates only 58% of samples vs 100% for Tent (geometric filtering working!)
4. **ProtoTTA is consistent**: Low std dev (10.64%) vs baseline (13.03%)
5. **SAR is slow**: 2x slower than ProtoTTA (8.84 ms vs 4.49 ms)
6. **All TTA methods use ~10GB memory** vs 4.6GB for Normal

## Command-Line Options

### `--methods`
Filter which methods to include in visualizations:
```bash
--methods normal tent eata proto_imp_conf_v3
```

### `--rename`
Rename methods for publication-ready plots:
```bash
--rename proto_imp_conf_v3=ProtoTTA proto_imp_conf_v1=ProtoTTA-Ablation1
```

### `--exclude`
Exclude specific corruption types:
```bash
--exclude saturate spatter
```

### `--severity`
Specify severity level (default: 5):
```bash
--severity 5
```

## Tips for Paper Figures

1. **Main results**: Use `--methods normal tent eata sar proto_imp_conf_v3 --rename proto_imp_conf_v3=ProtoTTA`
2. **Ablation study**: Use all v1/v2/v3 with descriptive renames
3. **Efficiency comparison**: The efficiency_comparison.png shows ProtoTTA's computational overhead is minimal
4. **Category analysis**: category_comparison.png shows ProtoTTA excels on Noise corruptions
