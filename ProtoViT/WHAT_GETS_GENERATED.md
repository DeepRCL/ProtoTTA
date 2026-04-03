# Output from run_inference.py

## Command:
```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode proto_importance_confidence,eata,tent,normal \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias \
    --prototype-metrics \
    --track-efficiency
```

## What Gets Generated:

### 1. Console Output ✅
Displays in terminal:
- Progress for each method (Normal, Tent, EATA, ProtoEntropy-Imp+Conf)
- Accuracy for each method
- **NEW**: Prototype metrics (PAC, PCA, Sparsity) for each method
- **NEW**: Efficiency comparison table showing:
  - Time per sample (ms)
  - Overhead vs baseline
  - Adapted parameters count & percentage
  - Adaptation steps

Example output:
```
==================================================
FINAL RESULTS SUMMARY
==================================================
Normal                    Acc: 37.25%  |  PAC: 0.8413  |  PCA: 0.1984  |  Sparsity: 0.4560
Tent                      Acc: 45.81%  |  PAC: 0.7583  |  PCA: 0.2248  |  Sparsity: 0.4503
ProtoEntropy-Imp+Conf     Acc: 51.81%  |  PAC: 0.8234  |  PCA: 0.2876  |  Sparsity: 0.4633
EATA                      Acc: 50.07%  |  PAC: 0.8961  |  PCA: 0.3226  |  Sparsity: 0.4712
==================================================

====================================================================================================
COMPUTATIONAL EFFICIENCY COMPARISON
====================================================================================================
Method                    Time/Sample     Overhead        Adapted Params       Steps     
                          (ms)            (vs baseline)   (count / %)                    
----------------------------------------------------------------------------------------------------
Normal                    31.73           -               0 (0.00%)            0         
ProtoEntropy-Imp+Conf     33.45           +1.72 ms (+5.4%) 245,760 (1.12%)      1         
EATA                      34.10           +2.36 ms (+7.5%) 19,200 (0.07%)       1         
Tent                      34.86           +3.12 ms (+9.8%) 19,200 (0.07%)       1         
====================================================================================================
```

### 2. NO Plots Generated ❌
`run_inference.py` does NOT create plots automatically!

It only:
- Prints results to console
- Optionally saves per-batch accuracy data internally (not to file)

### 3. NO JSON File Generated ❌
Results are NOT saved to disk by default.

---

## To Get Plots & Saved Results:

### Option 1: Use evaluate_robustness.py (Recommended for Paper)

```bash
python evaluate_robustness.py \
    --model ./saved_models/best_model.pth \
    --data_dir ./datasets/cub200_c/ \
    --output ./results_complete.json \
    --prototype-metrics \
    --track-efficiency
```

**This will:**
- ✅ Evaluate ALL methods on ALL corruptions
- ✅ Save comprehensive JSON with all metrics
- ✅ Can be used to generate plots later

### Option 2: Generate Plots from Saved JSON

```bash
python visualize_tta_comparison.py \
    --input ./results_complete.json \
    --output ./figures/
```

**This creates:**
- ✅ `average_accuracy.png`
- ✅ `accuracy_by_corruption.png`
- ✅ `accuracy_heatmap.png`
- ✅ `prototype_metrics.png`
- ✅ `efficiency_metrics.png`
- ✅ `accuracy_efficiency_tradeoff.png` (with Pareto frontier)
- ✅ `summary_table.csv`
- ✅ `summary_table.tex` (for LaTeX paper)

---

## Summary:

**`run_inference.py`** = Quick test with console output only (no plots, no saved files)

**`evaluate_robustness.py`** = Full evaluation with JSON output

**`visualize_tta_comparison.py`** = Generate plots from JSON

### Recommended Workflow:
1. Quick test: `run_inference.py` (what you just did)
2. Full run: `evaluate_robustness.py --output results.json`
3. Make plots: `visualize_tta_comparison.py --input results.json`
