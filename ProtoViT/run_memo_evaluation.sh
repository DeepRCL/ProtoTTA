#!/bin/bash
# Quick start guide for running MEMO evaluation

echo "=========================================="
echo "ProtoViT MEMO Evaluation - Quick Start"
echo "=========================================="
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "protovit"; then
    echo "ERROR: conda environment 'protovit' not found"
    echo "Please create it first: conda create -n protovit python=3.8"
    exit 1
fi

# Activate environment
echo "1. Activating conda environment..."
conda activate protovit

# Change to project directory
echo "2. Changing to ProtoViT directory..."
cd /home/mahdi.abootorabi/protovit/ProtoViT/ || exit 1

# Create logs directory
echo "3. Creating logs directory..."
mkdir -p logs

# Check if model exists
MODEL_PATH="./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "=========================================="
echo "Choose evaluation mode:"
echo "=========================================="
echo "1. Quick test (100 samples, ~2 minutes)"
echo "2. Add MEMO to existing results (recommended, ~13-26 hours)"
echo "3. Full evaluation with all methods (~30-50 hours)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        python test_memo_quick.py \
            --model "$MODEL_PATH" \
            --num_samples 100 \
            --gpuid 0
        ;;
    2)
        echo ""
        echo "Submitting MEMO-only SLURM job..."
        sbatch run_memo_only.slurm
        echo ""
        echo "Job submitted! Monitor with:"
        echo "  squeue -u $USER"
        echo "  tail -f logs/memo_eval_*.out"
        ;;
    3)
        echo ""
        echo "Submitting full evaluation SLURM job..."
        sbatch run_robustness_eval.slurm
        echo ""
        echo "Job submitted! Monitor with:"
        echo "  squeue -u $USER"
        echo "  tail -f logs/robustness_eval_*.out"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
