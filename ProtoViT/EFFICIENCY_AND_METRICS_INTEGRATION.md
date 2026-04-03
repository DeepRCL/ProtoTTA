# Efficiency Metrics and Prototype Metrics Integration

## Summary of Changes

This document outlines the integration of computational efficiency tracking and prototype-based metrics into the ProtoTTA codebase.

## 1. Changed Default for `--proto-baseline-samples`

**Previous**: Default was 500 samples
**Current**: Default is `None` (uses all available samples)

**Reason**: You requested to evaluate metrics over the full test dataset, not just a subset.

**Usage**:
```bash
# Use all samples (default)
python run_inference.py --prototype-metrics

# Limit to specific number
python run_inference.py --prototype-metrics --proto-baseline-samples 500
```

## 2. New Efficiency Metrics Module (`efficiency_metrics.py`)

Created a comprehensive module to track computational costs:

### Metrics Tracked:
- **Timing**:
  - Total inference time
  - Time per sample (ms)
  - Throughput (samples/sec)
  - Per-batch timing statistics

- **Parameter Adaptation**:
  - Total parameters in model
  - Number of adapted parameters
  - Adaptation ratio (percentage)
  - Names of adapted parameters

- **Adaptation Steps**:
  - Number of adaptation steps
  - Total optimizer steps

- **Memory** (optional, CUDA only):
  - Peak memory usage

### Key Classes:

#### `EfficiencyTracker`
Tracks metrics for a single method:
```python
tracker = EfficiencyTracker('ProtoTTA', device='cuda')
tracker.count_adapted_parameters(model, adapted_params)

with tracker.track_inference(batch_size=32):
    # Your inference code here
    pass

tracker.record_adaptation_step(num_steps=1)
metrics = tracker.get_metrics()
```

#### Helper Functions:
- `compare_efficiency_metrics()`: Compare multiple methods
- `print_efficiency_comparison()`: Print formatted comparison table

### Output Example:
```
====================================================================================================
COMPUTATIONAL EFFICIENCY COMPARISON
====================================================================================================
Method                    Time/Sample     Overhead        Adapted Params       Steps     
                          (ms)            (vs baseline)   (count / %)                    
----------------------------------------------------------------------------------------------------
Normal                    15.23           -               0 (0.00%)            0          
ProtoTTA                  18.45           +3.22 ms (+21.1%)   245,760 (1.12%)      1          
EATA                      22.10           +6.87 ms (+45.1%)   1,048,576 (4.78%)    1          
Tent                      19.03           +3.80 ms (+25.0%)   524,288 (2.39%)      1          
====================================================================================================
```

## 3. Integration into `run_inference.py`

### New Flag:
```bash
--track-efficiency
```

### Changes Made:
1. **Import**: Added `EfficiencyTracker` and `print_efficiency_comparison`
2. **Function Signature**: Added `track_efficiency` parameter
3. **Tracker Initialization**: Created `efficiency_trackers` dict
4. **Evaluation Wrapper**: Modified `evaluate_with_optional_metrics()` to accept `efficiency_tracker`
5. **Method Integration**: Added tracker initialization for each TTA method
6. **Results Display**: Added efficiency comparison at the end

### Usage:
```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode proto_importance_confidence,eata,tent,normal \
    --prototype-metrics \
    --track-efficiency
```

### Output:
- Shows prototype metrics (PAC, PCA, Sparsity) alongside accuracy
- Displays comprehensive efficiency comparison table
- Reports time overhead vs. baseline
- Shows adapted parameter counts

## 4. Integration into `evaluate_robustness.py`

### New Flag:
```bash
--track-efficiency
```

### Status:
- Import added
- Argument added
- **TODO**: Need to integrate tracker into `evaluate_single_combination()` function
- **TODO**: Save efficiency metrics to JSON output
- **TODO**: Create visualization plots for efficiency metrics

## 5. Visualization (To Be Implemented)

### Recommended Plots:
1. **Time Overhead Bar Chart**: Compare time per sample across methods
2. **Adapted Parameters Bar Chart**: Show adaptation footprint
3. **Accuracy vs. Efficiency Scatter**: Trade-off plot
4. **Time vs. Corruption Type**: Heatmap showing which methods are faster on which corruptions
5. **Memory Usage Comparison**: If CUDA is available

### Suggested Script: `visualize_efficiency.py`
- Load JSON results from `evaluate_robustness.py`
- Generate publication-quality plots
- Create summary tables

## 6. Benefits for Your Paper

### Claims You Can Now Make:
1. **Computational Efficiency**: "ProtoTTA adapts only X% of parameters, achieving Y% of the accuracy improvement with Z% less overhead compared to baseline TTA methods."

2. **Scalability**: "Our method processes N samples/sec, making it suitable for real-time applications."

3. **Parameter Efficiency**: "By leveraging prototype information, we adapt X times fewer parameters than EATA while maintaining comparable accuracy."

4. **Trade-off Analysis**: "Figure X shows the accuracy-efficiency trade-off, demonstrating that ProtoTTA achieves the best balance."

### Suggested Paper Sections:
- **Section 4.X: Computational Efficiency Analysis**
  - Table comparing all methods
  - Bar chart of adapted parameters
  - Scatter plot of accuracy vs. time

- **Supplementary Material**:
  - Per-corruption efficiency breakdown
  - Memory usage analysis
  - Detailed timing statistics

## 7. Next Steps

### Immediate:
1. ✅ Modified `--proto-baseline-samples` default to `None`
2. ✅ Created `efficiency_metrics.py`
3. ✅ Integrated into `run_inference.py`
4. ⚠️  Partially integrated into `evaluate_robustness.py` (argument added, needs full integration)

### TODO:
1. Complete integration into `evaluate_robustness.py`:
   - Modify `evaluate_single_combination()` to use `EfficiencyTracker`
   - Save efficiency metrics to JSON output
   - Update result format to include efficiency data

2. Create `visualize_efficiency.py`:
   - Load results JSON
   - Generate plots
   - Create LaTeX tables for paper

3. Add efficiency metrics to all TTA methods in `run_inference.py`:
   - Tent
   - EATA
   - SAR
   - MEMO
   - All ProtoTTA variants

4. Document findings in paper draft

## 8. Usage Examples

### Quick Test (Single Corruption):
```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode proto_importance_confidence,eata,tent,normal \
    --prototype-metrics \
    --track-efficiency
```

### Full Evaluation (All Corruptions):
```bash
python evaluate_robustness.py \
    --model ./saved_models/best_model.pth \
    --data_dir ./datasets/cub200_c/ \
    --output ./results_with_metrics.json \
    --prototype-metrics \
    --track-efficiency
```

### Analysis and Visualization:
```bash
# To be implemented
python visualize_efficiency.py \
    --input ./results_with_metrics.json \
    --output ./figures/
```

## 9. Expected Output Format

### JSON Structure (evaluate_robustness.py):
```json
{
  "timestamp": "2026-01-25T...",
  "metadata": {...},
  "results": {
    "gaussian_noise-5": {
      "normal": {
        "accuracy": 0.7543,
        "efficiency": {
          "time_per_sample_ms": 15.23,
          "num_adapted_params": 0,
          "adaptation_ratio": 0.0,
          ...
        }
      },
      "proto_imp_conf_v1": {
        "accuracy": 0.8234,
        "PAC_mean": 0.7654,
        "PCA_mean": 0.8123,
        "Sparsity_mean": 0.2345,
        "efficiency": {
          "time_per_sample_ms": 18.45,
          "num_adapted_params": 245760,
          "adaptation_ratio": 0.0112,
          "time_overhead_vs_baseline_ms": 3.22,
          ...
        }
      },
      ...
    }
  }
}
```

## 10. Paper Impact

This integration allows you to:
1. Show ProtoTTA is **computationally efficient**
2. Demonstrate the **interpretability** advantage through prototype metrics
3. Provide **comprehensive comparison** against other TTA methods
4. Support claims about **parameter efficiency**
5. Enable **trade-off analysis** (accuracy vs. speed vs. adaptation size)

This strengthens your paper by addressing a common concern about TTA methods: computational overhead.
