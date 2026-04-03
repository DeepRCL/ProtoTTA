#!/bin/bash
# Example commands for generating paper-ready visualizations

# 1. Main results (for paper) - Show best method renamed as "ProtoTTA"
echo "=== Generating main results for paper ==="
python visualize_robustness_results.py \
    --input robustness_results_sev5_metrics.json \
    --output_dir ./plots/paper_main_results \
    --severity 5 \
    --methods normal tent eata sar proto_imp_conf_v3 \
    --rename proto_imp_conf_v3=ProtoTTA \
    --exclude saturate spatter

# 2. Ablation study - Compare v1, v2, v3
echo ""
echo "=== Generating ablation study ==="
python visualize_robustness_results.py \
    --input robustness_results_sev5_metrics.json \
    --output_dir ./plots/ablation_study \
    --severity 5 \
    --methods normal proto_imp_conf_v1 proto_imp_conf_v2 proto_imp_conf_v3 \
    --rename proto_imp_conf_v1="ProtoTTA-Full" \
            proto_imp_conf_v2="ProtoTTA-LayerNorm" \
            proto_imp_conf_v3="ProtoTTA"

# 3. All methods comparison (for supplementary)
echo ""
echo "=== Generating complete comparison ==="
python visualize_robustness_results.py \
    --input robustness_results_sev5_metrics.json \
    --output_dir ./plots/complete_comparison \
    --severity 5

echo ""
echo "=== All visualizations generated! ==="
echo "Main results: ./plots/paper_main_results/"
echo "Ablation study: ./plots/ablation_study/"
echo "Complete comparison: ./plots/complete_comparison/"
