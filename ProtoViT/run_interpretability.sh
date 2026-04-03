#!/bin/bash

# Comprehensive Interpretability Analysis Script
# This script runs the interpretability analysis for Normal, EATA, and ProtoEntropy-Imp+Conf
# with the specified settings.

echo "=========================================="
echo "ProtoViT Interpretability Analysis"
echo "=========================================="
echo ""

# Default values (can be overridden with command-line arguments)
CORRUPTION=${1:-"gaussian_noise"}
SEVERITY=${2:-5}
NUM_IMAGES=${3:-3}
MODEL_PATH=${4:-"./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"}

echo "Configuration:"
echo "  Corruption: $CORRUPTION"
echo "  Severity: $SEVERITY"
echo "  Number of images: $NUM_IMAGES"
echo "  Model: $MODEL_PATH"
echo ""

# Run the inference with interpretability enabled
python run_inference.py \
    -model "$MODEL_PATH" \
    -corruption "$CORRUPTION" \
    -severity "$SEVERITY" \
    -mode normal,eata,proto_importance_confidence \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias \
    --interpretability-num-images "$NUM_IMAGES" \
    --output-dir ./plots

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the output in: ./plots/interpretability_comprehensive/"

