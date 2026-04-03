#!/bin/bash
# Quick script to evaluate TTA methods with prototype-based metrics

echo "=========================================="
echo "Prototype-Based TTA Metrics Evaluation"
echo "=========================================="
echo ""

# Configuration
MODEL="./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
CORRUPTIONS="gaussian_noise fog brightness contrast defocus_blur"
SEVERITY=5
OUTPUT="./proto_metrics_results.json"
PLOTS_DIR="./plots/proto_metrics"
GPUID="0"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    echo "Please update the MODEL variable in this script."
    exit 1
fi

# Step 1: Evaluate
echo "Step 1/2: Evaluating TTA methods..."
echo "Corruptions: $CORRUPTIONS"
echo "Severity: $SEVERITY"
echo ""

python evaluate_with_proto_metrics.py \
    --model "$MODEL" \
    --corruption $CORRUPTIONS \
    --severity $SEVERITY \
    --output "$OUTPUT" \
    --gpuid "$GPUID"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Evaluation failed!"
    exit 1
fi

echo ""
echo "✓ Evaluation completed!"
echo ""

# Step 2: Visualize
echo "Step 2/2: Generating visualizations..."
echo "Output directory: $PLOTS_DIR"
echo ""

python visualize_proto_metrics.py \
    --input "$OUTPUT" \
    --output_dir "$PLOTS_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Visualization failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All done!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - JSON:  $OUTPUT"
echo "  - Plots: $PLOTS_DIR/"
echo ""
echo "Generated files:"
echo "  - accuracy_vs_pac.png"
echo "  - accuracy_vs_pca.png"
echo "  - method_comparison_bars.png"
echo "  - radar_comparison.png"
echo "  - summary_table.md"
echo ""
echo "Next: Check $PLOTS_DIR/ for visualizations!"
echo ""
