# Complete ProtoLens Training & Evaluation Workflow

## ✅ What's Been Fixed

1. **Hardcoded paths removed** from `experiment.py` and `utils.py`
2. **Proper data splitting** setup (train/val/test)
3. **Evaluation scripts** created for proper testing

---

## 📊 Data Structure

### Current Setup (After Running Step 1):

```
Datasets/Yelp/
├── train.csv              # 381,229 samples (80% of original, for training)
├── val.csv                # 95,308 samples (20% of original, for validation)
├── test.csv               # 30,000 samples (held-out, for final evaluation)
└── train_original_backup.csv  # 476,537 samples (backup)
```

### Why This Split?

- **Train (80%)**: Used to train the model
- **Validation (20%)**: Used during training to select best model & prevent overfitting
- **Test (separate)**: Final evaluation on completely unseen data

This is the **proper ML workflow** for getting the best model performance!

---

## 🚀 Complete Step-by-Step Workflow

### **STEP 0: Verify Prototype Files Exist**

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Check prototype files
ls -lh Datasets/Yelp/all-mpnet-base-v2/
# Should show:
# - Yelp_cluster_50_centers.npy (151K)
# - Yelp_cluster_50_to_sub_sentence.csv (47K)
```

✅ **Already done!**

---

### **STEP 1: Split Data into Train/Val/Test**

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

python split_yelp_data.py
```

**What this does:**
- Splits original `train.csv` into `train.csv` (80%) + `val.csv` (20%)
- Keeps `test.csv` unchanged as final test set
- Maintains class balance (stratified split)

**Expected output:**
```
✓ Saved train set: train.csv (381,229 samples)
✓ Saved validation set: val.csv (95,308 samples)
✓ Test set unchanged: test.csv (30,000 samples)
```

**Verify:**
```bash
wc -l Datasets/Yelp/*.csv
```

---

### **STEP 2: Train Model**

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Train with 50 prototypes (matching what we generated)
python experiment.py \
    -d Yelp \
    -pn 50 \
    -e 25 \
    -bs 16 \
    -lr 0.0005 \
    -i 0
```

**Training parameters:**
- `-d Yelp`: Dataset
- `-pn 50`: 50 prototypes (matching generated files)
- `-e 25`: 25 epochs
- `-bs 16`: Batch size 16
- `-lr 0.0005`: Learning rate
- `-i 0`: GPU 0

**What happens during training:**
1. Loads `train.csv` for training
2. Loads `val.csv` for validation (every batch/epoch)
3. Saves **best model** based on validation accuracy
4. `test.csv` is **NOT USED** (kept for final evaluation)

**Monitor progress:**
```bash
# Watch training log
tail -f log_folder/Yelp/*/log.txt

# Or in another terminal
watch -n 5 'tail -20 log_folder/Yelp/*/log.txt'
```

**Expected training output:**
```
Epoch 1/25
Validation Accuracy: 0.75XX
...
Epoch 10/25
Validation Accuracy: 0.85XX
...
Epoch 25/25
Validation Accuracy: 0.88XX
```

**Training time:** ~2-4 hours (depends on GPU)

**Model saved to:**
```
log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_25_pNum_50_lr0.0005/
└── model.pth  <- Best model based on validation accuracy
```

---

### **STEP 3: Evaluate on Validation Set**

This happens **automatically during training**, but you can also run it manually:

```bash
# Check validation results in training log
cat log_folder/Yelp/*/log.txt | grep "Validation Accuracy"
```

**Expected validation accuracy:** ~85-90%

---

### **STEP 4: Evaluate on Final Test Set (Unseen Data)**

After training completes, evaluate on the held-out test set:

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Find your trained model
MODEL_PATH=$(find log_folder/Yelp -name "model.pth" | head -1)
echo "Using model: $MODEL_PATH"

# Evaluate on test set
python evaluate_test.py \
    --model_path "$MODEL_PATH" \
    --dataset Yelp \
    --batch_size 16
```

**Expected output:**
```
================================================================================
FINAL TEST RESULTS on Yelp
================================================================================

Test Accuracy: 0.XXXX (XX.XX%)

Classification Report:
              precision    recall  f1-score   support

           0       0.XX      0.XX      0.XX     XXXXX
           1       0.XX      0.XX      0.XX     XXXXX

    accuracy                           0.XX     30000
   macro avg       0.XX      0.XX      0.XX     30000
weighted avg       0.XX      0.XX      0.XX     30000

✓ Results saved to: log_folder/Yelp/.../test_results_Yelp.txt
```

**Expected test accuracy:** ~85-90% (similar to validation)

---

## 📈 Performance Expectations

### Source Domain (Yelp → Yelp)

| Split | Expected Accuracy | Purpose |
|-------|-------------------|---------|
| **Training** | ~90-95% | Model is learning from this |
| **Validation** | ~85-90% | Used to select best model |
| **Test** | ~85-90% | Final performance metric |

### Target Domain (Yelp → Hotel) - For Later TTA

| Method | Expected Accuracy | Notes |
|--------|-------------------|-------|
| **Source-only** | ~70-75% | No adaptation (domain shift) |
| **TENT** | ~73-78% | Basic TTA |
| **NOTE** | ~74-79% | Norm + Entropy TTA |
| **MEMO** | ~75-80% | Augmentation-based TTA |
| **Your ProtoTTA** | **~78-85%** | **Novel prototype-aware TTA** |

---

## 🔍 Verify Everything is Ready

Run these checks before training:

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

# 1. Check prototype files
echo "1. Checking prototype files..."
ls -lh Datasets/Yelp/all-mpnet-base-v2/Yelp_cluster_50_*.* 2>/dev/null && echo "✓ Prototypes OK" || echo "✗ Prototypes missing!"

# 2. Check data files
echo -e "\n2. Checking data files..."
for f in train.csv val.csv test.csv; do
    if [ -f "Datasets/Yelp/$f" ]; then
        echo "✓ $f exists ($(wc -l < Datasets/Yelp/$f) lines)"
    else
        echo "✗ $f missing!"
    fi
done

# 3. Check for hardcoded paths (should show no results or only commented lines)
echo -e "\n3. Checking for hardcoded paths..."
grep -n "/home/bwei2" experiment.py utils.py 2>/dev/null | grep -v "^#" | grep -v "    #" && echo "⚠ Found uncommented hardcoded paths!" || echo "✓ No hardcoded paths"

# 4. Test imports
echo -e "\n4. Testing Python imports..."
python -c "from PLens import BERTClassifier; from utils import get_data_loader; print('✓ Imports OK')" 2>/dev/null || echo "✗ Import errors!"

echo -e "\n✅ All checks complete!"
```

---

## 📋 Quick Command Reference

```bash
# Navigate to directory
cd /home/mahdi.abootorabi/protovit/ProtoLens

# STEP 1: Split data (one-time)
python split_yelp_data.py

# STEP 2: Train model (2-4 hours)
python experiment.py -d Yelp -pn 50 -e 25 -bs 16 -lr 0.0005 -i 0

# STEP 3: Monitor training
tail -f log_folder/Yelp/*/log.txt

# STEP 4: Evaluate on test set
python evaluate_test.py --model_path log_folder/Yelp/*/model.pth --dataset Yelp
```

---

## 🎯 What You Get After Training

1. **Trained model** saved in `log_folder/Yelp/.../model.pth`
2. **Training log** with validation accuracies
3. **Test results** showing final performance on unseen data
4. **Best model** selected based on validation performance

This gives you:
- ✅ **In-domain performance** (Yelp → Yelp test)
- ✅ **Baseline for TTA** (to later test Yelp → Hotel)
- ✅ **Proper evaluation** (no data leakage!)

---

## 🚨 Important Notes

### During Training:
- Model is trained on `train.csv` only
- Validated on `val.csv` to select best checkpoint
- `test.csv` is **never seen** during training
- Best model = highest validation accuracy

### For TTA (Later):
- Use trained model from `model.pth`
- Apply TTA methods on Hotel/Amazon target domain
- Compare: Source-only vs TENT vs NOTE vs MEMO vs Your ProtoTTA
- Show adaptation improves performance!

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: val.csv"
**Solution:** Run `python split_yelp_data.py` first

### Issue: "Prototype files not found"
**Solution:** Check `Datasets/Yelp/all-mpnet-base-v2/` for `Yelp_cluster_50_*.npy|csv`

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size: `-bs 8` instead of `-bs 16`

### Issue: "Hardcoded path error"
**Solution:** Check that the path fixes were applied (should show "Using validation file" in output)

---

## ✅ Summary

**Before training:**
1. ✓ Prototypes generated (50 prototypes)
2. ✓ Hardcoded paths fixed
3. ✓ Data split created (train/val/test)

**To train:**
```bash
python split_yelp_data.py  # One-time
python experiment.py -d Yelp -pn 50 -e 25 -bs 16 -lr 0.0005 -i 0
```

**To evaluate:**
```bash
python evaluate_test.py --model_path log_folder/Yelp/*/model.pth --dataset Yelp
```

**Ready to start training? Run the commands above!** 🚀
