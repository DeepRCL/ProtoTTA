#!/bin/bash
#SBATCH --job-name=protolens_train
#SBATCH --output=/home/mahdi.abootorabi/protovit/ProtoLens/logs/training_%j.out
#SBATCH --error=/home/mahdi.abootorabi/protovit/ProtoLens/logs/training_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --partition=main

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Print GPU information
echo ""
echo "GPU Information:"
echo "=========================================="
nvidia-smi
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protovit

# Change to project directory
cd /home/mahdi.abootorabi/protovit/ProtoLens/

# Create logs directory if it doesn't exist
mkdir -p logs

# Print training configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Dataset: ${DATASET:-Yelp}"
echo "Prototypes: ${NUM_PROTOTYPES:-50}"
echo "Epochs: ${EPOCHS:-15}"
echo "Batch Size: ${BATCH_SIZE:-128}"
echo "Learning Rate: ${LEARNING_RATE:-0.0005}"
echo "GPU ID: ${GPU_ID:-0}"
echo "Window Size: ${WINDOW_SIZE:-5}"
echo "=========================================="
echo ""

# Run ProtoLens training
# You can override parameters via environment variables or modify defaults here
python experiment.py \
    -d ${DATASET:-Yelp} \
    -pn ${NUM_PROTOTYPES:-50} \
    -e ${EPOCHS:-15} \
    -bs ${BATCH_SIZE:-128} \
    -lr ${LEARNING_RATE:-0.0005} \
    -i ${GPU_ID:-0} \
    -ws ${WINDOW_SIZE:-5}

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: $EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi
