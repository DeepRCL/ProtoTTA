#!/bin/bash

# ProtoLens Interpretability Analysis Script
# This script runs interpretability analysis to visualize how different TTA methods
# handle corrupted text samples.

echo "=========================================="
echo "ProtoLens Interpretability Analysis"
echo "=========================================="
echo ""

# Default values (can be overridden with command-line arguments)
CORRUPTION=${1:-"qwerty"}
SEVERITY=${2:-40}
NUM_SAMPLES=${3:-5}
MODEL_PATH=${4:-"./log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth"}

# Data paths
CLEAN_DATA=${5:-"./Datasets/Amazon/test.csv"}
CORRUPTED_DATA=${6:-"./Datasets/Amazon-C/amazon_c_${CORRUPTION}_s${SEVERITY}.csv"}

echo "Configuration:"
echo "  Corruption: $CORRUPTION"
echo "  Severity: $SEVERITY"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Model: $MODEL_PATH"
echo "  Clean data: $CLEAN_DATA"
echo "  Corrupted data: $CORRUPTED_DATA"
echo ""

# Run the interpretability analysis
python run_interpretability_analysis.py \
    --model_path "$MODEL_PATH" \
    --clean_data "$CLEAN_DATA" \
    --corrupted_data "$CORRUPTED_DATA" \
    --corruption_name "$CORRUPTION" \
    --severity "$SEVERITY" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir ./plots \
    --learning_rate 0.000005 \
    --adaptation_mode layernorm_attn_bias \
    --geo_filter \
    --geo_threshold 0.1 \
    --e_margin 0.6 \
    --d_margin 0.05 \
    --sigmoid_temperature 5.0

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the output in: ./plots/interpretability_comprehensive/${CORRUPTION}_sev${SEVERITY}/"
