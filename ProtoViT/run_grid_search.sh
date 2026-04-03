#!/bin/bash

# Script to run ProtoEntropy grid search for loss weights
# Make sure to activate your conda/virtualenv before running this script

cd /home/mahdi.abootorabi/protovit/ProtoViT

echo "Running ProtoEntropy Grid Search for Loss Weights"
echo "=================================================="
echo ""

python grid_search_proto_weights.py \
    -model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    -corruption gaussian_noise \
    -severity 4 \
    -gpuid 0

echo ""
echo "Grid search complete!"

