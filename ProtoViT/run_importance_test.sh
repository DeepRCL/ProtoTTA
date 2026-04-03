#!/bin/bash

# Script to test Prototype Importance Weighting feature
# Make sure to activate your conda/virtualenv before running this script

cd /home/mahdi.abootorabi/protovit/ProtoViT

echo "Testing Prototype Importance Weighting for ProtoEntropy"
echo "========================================================"
echo ""

python test_prototype_importance.py \
    -model ./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth \
    -corruption gaussian_noise \
    -severity 4 \
    -gpuid 0

echo ""
echo "Test complete!"

