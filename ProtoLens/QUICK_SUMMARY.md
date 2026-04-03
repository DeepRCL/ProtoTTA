# ProtoLens Setup Summary

## Quick Answers to Your Questions

### 1. **Are prototypes class-specific?**
**NO** - Prototypes are **shared across all classes** in ProtoLens.

- Architecture: 20 prototypes → Linear layer → 2 classes
- Each prototype represents a **semantic concept** (e.g., "service quality", "food taste")
- Final layer learns which prototype activations indicate which class
- Prototypes can contribute **+ or −** to each class

**Example from paper:**
- Prototype 6: "amazing game" → +0.935 contribution to positive
- Prototype 8: "worst game" → -0.806 contribution to positive

### 2. **Are all datasets binary classification?**
**YES** - All datasets are binary sentiment:
- Yelp: 1-2 stars (negative) vs 4-5 stars (positive), **excluding 3-star**
- Amazon: Negative vs Positive  
- IMDB: Negative vs Positive
- Steam: Negative vs Positive
- Hotel: Negative vs Positive

### 3. **Does TTA make sense without class-specific prototypes?**
**ABSOLUTELY YES!** It's actually **more interesting**:

✅ **Why it works better:**
- Prototypes encode **domain-dependent semantics**
- Example: "service" in restaurant vs hotel reviews has different meanings
- Your TTA must adapt:
  1. Prototype embeddings (semantic concepts)
  2. FC layer weights (concept-to-class mapping)
- Makes the problem more challenging = more impressive if solved!

### 4. **Should prototype files be generated from data?**
**YES, CORRECT!**

Files needed (per dataset):
```
Datasets/{dataset}/all-mpnet-base-v2/
├── {dataset}_cluster_20_centers.npy          # [20, 768] prototype embeddings
└── {dataset}_cluster_20_to_sub_sentence.csv  # [20, 40] representative sentences
```

**Generation process:**
1. Extract n-grams (5-word windows) from training texts
2. Compute sentence embeddings with `all-mpnet-base-v2`
3. K-means clustering → get 20 cluster centers
4. Find top-40 representative sentences per cluster

**Status:** ✅ Script created and running now!

### 5. **Yelp Dataset Status**

✅ **Downloaded and Processed:**
- **Raw data**: 6.99M reviews
- **Filtered**: Removed 3-star reviews (neutral)
- **Binary mapping**: 
  - 1-2 stars → 0 (negative)
  - 4-5 stars → 1 (positive)
- **Final size**:
  - Train: 476,537 samples (57% pos, 43% neg)
  - Test: 30,000 samples (57% pos, 43% neg)
- **Location**: `Datasets/Yelp/train.csv` and `test.csv`

---

## Why ProtoLens is Perfect for Your TTA Project

### ✅ Strong Arguments

1. **Cross-Modal Generalization**
   - ProtoViT: Computer Vision (ImageNet)
   - ProtoLens: NLP (Text Sentiment)
   - Shows your TTA method works across modalities!

2. **Real-World Domain Shift**
   - **Recommended**: Yelp (restaurants) → Hotel (lodging)
   - Similar task but different contexts
   - Vocabulary overlap but domain-specific semantics
   - Very practical and meaningful

3. **Interpretability Preserved**
   - Can visualize which prototypes adapt during TTA
   - Show extracted text spans before/after adaptation
   - Strong qualitative analysis possible

4. **Built-in Adaptation Mechanisms**
   - Prototype alignment (line 299-338 in PLens.py) already adapts!
   - Can leverage this or modify for TTA
   - Novel contribution: prototype-aware TTA

---

## Files Created

1. ✅ `prepare_yelp_data.py` - Processes raw Yelp JSON → train/test CSV
2. ✅ `generate_prototypes.py` - Creates prototype initialization files
3. ✅ `PROTOLENS_TTA_GUIDE.md` - Complete 12-section guide (everything you need!)
4. ⏳ Prototype generation running (ETA: 10-15 minutes)

---

## Current Status

### Completed ✅
- [x] Downloaded Yelp dataset (6.99M reviews)
- [x] Processed into binary classification (476K train, 30K test)
- [x] Created data preparation script
- [x] Created prototype generation script
- [x] Created comprehensive TTA guide
- [x] Started prototype generation (running now)

### Next Steps 🔄
1. ⏳ Wait for prototype generation to complete (~10-15 min)
2. 📝 Fix hardcoded paths in experiment.py
3. 🚀 Train baseline model on Yelp
4. 📊 Obtain Hotel dataset (or use Amazon as fallback)
5. 🔬 Implement TTA evaluation pipeline

### Prototype Generation Progress
```bash
# Check progress:
tail -f /home/mahdi.abootorabi/protovit/ProtoLens/prototype_gen_run2.log

# Current status: K-means clustering in progress (step 81/2000)
```

---

## Expected Workflow

### 1. Train Baseline (After Prototypes Ready)
```bash
python experiment.py \
    -d Yelp \
    -pn 20 \
    -e 25 \
    -bs 16 \
    -lr 0.0005 \
    -i 0
```

### 2. Evaluate Without Adaptation
```bash
python evaluate.py \
    --source Yelp \
    --target Hotel \
    --model_path log_folder/Yelp/.../model.pth
```
**Expected**: ~10-15% accuracy drop

### 3. Apply TTA Methods
- TENT: Entropy minimization
- NOTE: Normalize + Entropy
- MEMO: Marginal entropy with augmentations
- **Your ProtoTTA**: Novel prototype-aware adaptation

### 4. Compare Results
| Method | Yelp (source) | Hotel (no adapt) | Hotel (TTA) | Gain |
|--------|---------------|------------------|-------------|------|
| Baseline | ~88% | ~75% | - | - |
| TENT | - | - | ~78% | +3% |
| **ProtoTTA** | - | - | **~82%** | **+7%** |

---

## Key Insights for Your Project

### Why Non-Class-Specific is Better for TTA

**Advantages:**
1. **Semantic Flexibility**: Prototypes represent concepts that transcend classes
2. **Domain Adaptation**: Must adapt meaning, not just features
3. **More Challenging**: Makes solving it more impressive
4. **Interpretable**: Can show how "service" concept changes Yelp→Hotel

**Example Adaptation:**
```
Source (Yelp):
  Prototype 5: "excellent service" → +0.8 for positive

Target (Hotel) - Before TTA:
  Prototype 5: misaligned, predicts incorrectly

Target (Hotel) - After Your TTA:
  Prototype 5: "friendly staff" → +0.82 for positive
  (Adapted to hotel-specific service language)
```

### TTA Opportunities in ProtoLens

**What to Adapt:**
1. **Prototype embeddings** (`model.prototype_vectors`)
2. **Classification head** (`model.fc`)
3. **Normalization layers** (if using BatchNorm)
4. **Alignment mechanism** (novel contribution!)

**Your Novel Contribution:**
- Adapt prototype alignment dynamically during test time
- Use target domain samples to refine representative sentences
- Prototype-aware consistency loss across augmentations

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Data preparation | 1 day | ✅ Done |
| Prototype generation | 1 day | ⏳ In progress |
| Baseline training | 1-2 days | 📝 Next |
| TTA implementation | 3-4 days | 📝 Upcoming |
| Experiments | 2-3 days | 📝 Upcoming |
| Analysis & writing | 2-3 days | 📝 Upcoming |
| **Total** | **~2 weeks** | |

---

## Conclusion

**ProtoLens is an EXCELLENT choice for your TTA project!**

✅ **Strengths:**
- Different domain (Vision → NLP) shows generalization
- Real-world shift (Yelp → Hotel) is practical
- Interpretable results (qualitative analysis)
- Built-in mechanisms to leverage
- Shared prototypes make adaptation more interesting

✅ **Your Contributions:**
1. First cross-modal prototype TTA (Vision + NLP)
2. Novel prototype-aware adaptation strategy
3. Interpretability preservation during adaptation
4. Real-world domain shift evaluation

**You're on the right track! This will strengthen your ProtoTTA project significantly.**

---

## Quick Reference Commands

```bash
# Check prototype generation progress
tail -f /home/mahdi.abootorabi/protovit/ProtoLens/prototype_gen_run2.log

# After completion, verify files
ls -lh /home/mahdi.abootorabi/protovit/ProtoLens/Datasets/Yelp/all-mpnet-base-v2/

# Train baseline
cd /home/mahdi.abootorabi/protovit/ProtoLens
python experiment.py -d Yelp -pn 20 -e 25 -bs 16 -lr 0.0005

# Monitor training
tail -f log_folder/Yelp/*/log.txt
```

---

**All questions answered! Ready to proceed when prototype generation completes.**
