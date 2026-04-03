#!/bin/bash
# Verify ProtoLens setup is ready for training

echo "================================================================================"
echo "ProtoLens Setup Verification"
echo "================================================================================"
echo ""

cd /home/mahdi.abootorabi/protovit/ProtoLens

# 1. Check prototype files
echo "1. Checking prototype files..."
if ls Datasets/Yelp/all-mpnet-base-v2/Yelp_cluster_50_*.* 1> /dev/null 2>&1; then
    echo "   ✓ Prototype files exist:"
    ls -lh Datasets/Yelp/all-mpnet-base-v2/Yelp_cluster_50_*.*
else
    echo "   ✗ Prototype files missing! Run: python generate_prototypes.py ..."
    exit 1
fi

# 2. Check data files
echo ""
echo "2. Checking data files..."
ALL_GOOD=true
for f in train.csv test.csv; do
    if [ -f "Datasets/Yelp/$f" ]; then
        lines=$(wc -l < "Datasets/Yelp/$f")
        echo "   ✓ $f exists ($lines lines)"
    else
        echo "   ✗ $f missing!"
        ALL_GOOD=false
    fi
done

if [ -f "Datasets/Yelp/val.csv" ]; then
    lines=$(wc -l < "Datasets/Yelp/val.csv")
    echo "   ✓ val.csv exists ($lines lines)"
else
    echo "   ⚠ val.csv missing - will be created from train.csv"
    echo "     Run: python split_yelp_data.py"
fi

# 3. Check Python environment
echo ""
echo "3. Checking Python environment..."
python -c "
import sys
try:
    import torch
    import transformers
    import sentence_transformers
    import sklearn
    import pandas as pd
    import numpy as np
    print('   ✓ All required packages installed')
    print(f'     - PyTorch: {torch.__version__}')
    print(f'     - Transformers: {transformers.__version__}')
    print(f'     - Sentence-Transformers: {sentence_transformers.__version__}')
except ImportError as e:
    print(f'   ✗ Missing package: {e}')
    sys.exit(1)
" || exit 1

# 4. Test model imports
echo ""
echo "4. Testing ProtoLens imports..."
python -c "
from PLens import BERTClassifier
from utils import get_data_loader
from experiment import evaluate
print('   ✓ ProtoLens modules import successfully')
" 2>&1 | grep -v "Warning" || echo "   ✗ Import errors detected!"

# 5. Check GPU availability
echo ""
echo "5. Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'     - CUDA version: {torch.version.cuda}')
    print(f'     - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('   ⚠ No GPU detected - training will be slow on CPU')
"

# Summary
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
echo "If all checks passed, you're ready to:"
echo ""
echo "  1. Split data (if val.csv missing):"
echo "     python split_yelp_data.py"
echo ""
echo "  2. Start training:"
echo "     python experiment.py -d Yelp -pn 50 -e 25 -bs 16 -lr 0.0005 -i 0"
echo ""
echo "  3. Monitor progress:"
echo "     tail -f log_folder/Yelp/*/log.txt"
echo ""
echo "================================================================================"
