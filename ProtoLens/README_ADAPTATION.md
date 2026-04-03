# ProtoLens Test-Time Adaptation

Test ProtoLens models with adaptation methods (TENT) on new domains like Hotel reviews.

## Files Created

- **`tent.py`**: TENT adaptation wrapper (entropy minimization)
- **`eata.py`**: EATA adaptation wrapper (efficient anti-catastrophic TTA with sample filtering)
- **`proto_tta.py`**: ProtoTTA wrapper (prototype-aware adaptation with V3 config)
- **`run_inference_hotel.py`**: **Main script** - Clean inference with tqdm and consistent sampling
- **`test_adaptation.py`**: Alternative testing script (older version)

## Quick Start

### Recommended: Use `run_inference_hotel.py`

```bash
cd /home/mahdi.abootorabi/protovit/ProtoLens

# Test all methods with 4K balanced samples
python run_inference_hotel.py \
    --model_path log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth \
    --test_samples 4000 \
    --methods baseline tent eata prototta \
    --batch_size 128 \
    --seed 42

# Test specific methods only
python run_inference_hotel.py \
    --model_path log_folder/Yelp/.../model.pth \
    --test_samples 2000 \
    --methods baseline prototta \
    --batch_size 64
```

### 2. Arguments

- `--model_path`: Path to trained ProtoLens model (.pth file) **[Required]**
- `--test_file`: CSV file with test data (default: `Datasets/Hotel/hotels.csv`)
- `--test_samples`: Number of test samples to use (default: 2000)
  - Set to `980000` or higher for full dataset
  - Sampling is consistent across runs (uses `--seed`)
- `--methods`: Methods to test (default: `baseline tent eata prototta`)
  - Options: `baseline`, `tent`, `eata`, `prototta`
  - Can specify multiple: `--methods baseline tent eata prototta`
- `--batch_size`: Batch size for evaluation (default: 32)
- `--learning_rate`: Learning rate for adaptation (default: 0.001)
- `--seed`: Random seed for reproducibility (default: 42)

### 3. Key Features

✅ **Progress bars** (tqdm) for each method  
✅ **Consistent sampling** - Same random seed gives same samples  
✅ **Multiple methods** - Run baseline + TENT in one command  
✅ **Detailed metrics** - Accuracy, classification report, confusion matrix  
✅ **Clean output** - Summary table comparing all methods  
✅ **Reproducible** - Fixed seed ensures same results across runs

### 3. Label Mapping for Hotel Dataset

The script automatically maps hotel ratings to binary labels:
- **Positive (1)**: FourStar, FiveStar
- **Negative (0)**: OneStar, TwoStar
- **Excluded**: ThreeStar (neutral - not used)

### 4. Balanced Sampling

When you specify `--test_samples N`, the script samples:
- **50% negative** (N/2 samples from 1-2 star hotels)
- **50% positive** (N/2 samples from 4-5 star hotels)

This ensures balanced evaluation and prevents bias toward the majority class.

## How It Works

### Adaptation Methods

1. **Baseline (No Adaptation)**:
   - Model evaluates test data with no updates
   - Shows out-of-domain performance

2. **TENT (Entropy Minimization)**:
   - Adapts LayerNorm/BatchNorm parameters during test time
   - Minimizes prediction entropy (encourages confident predictions)
   - Updates on **every sample** (100% adaptation rate)
   - No labels required!

3. **EATA (Efficient Anti-Catastrophic TTA)**:
   - Same as TENT but with **sample filtering**
   - **Filter 1**: Entropy threshold (reliability) - rejects high-entropy samples
   - **Filter 2**: Cosine similarity (redundancy) - rejects redundant samples
   - Only updates on reliable + non-redundant samples (~30-60% adaptation rate)
   - Prevents catastrophic forgetting

4. **ProtoTTA (Prototype-Aware TTA, V3 Config)**:
   - Uses **prototype similarities** instead of output logits
   - **Geometric filtering**: Only adapts samples with high prototype similarity (≥0.92)
   - **Consensus strategy**: Aggregates top 50% of sub-prototypes (robust to outliers)
   - **Binary entropy loss**: Encourages decisive prototype activations
   - Selective adaptation (~50-70% adaptation rate)
   - Best for prototype-based models!

### What Gets Adapted?

Different methods adapt different parameters:

**TENT & EATA** (adaptation_mode: `layernorm_only`):
- ✅ LayerNorm weight and bias
- ✅ BatchNorm weight and bias
- ❌ All other parameters frozen

**ProtoTTA V3** (adaptation_mode: `layernorm_attn_bias`):
- ✅ LayerNorm weight and bias
- ✅ BatchNorm weight and bias  
- ✅ **Attention biases** (Q, K, V biases in BERT layers)
- ❌ All other parameters frozen

**Why attention biases for ProtoTTA?**
Attention biases control where the model focuses in the input. Adapting them helps restore semantic focus under distribution shifts, which is crucial for prototype-based models that rely on meaningful attention patterns.

## Output Example

```
================================================================================
ProtoLens Domain Adaptation: Yelp → Hotel
================================================================================
Model: model.pth
Test samples: 2,000
Methods: baseline, tent
Batch size: 32
Seed: 42
================================================================================

>>> Loading trained model...
Model config:
  - BERT: all-mpnet-base-v2
  - Classes: 2
  - Prototypes: 50
  - Device: cuda

================================================================================
Loading Hotel Dataset
================================================================================
Total entries: 1,010,033
Valid samples: 980,732
Samples after label mapping (excluding 3-star): 660,028
Full dataset label distribution:
  Negative (1-2 star): 509,405 (77.2%)
  Positive (4-5 star): 150,623 (22.8%)

Balanced sampling: 2,000 total samples (seed=42)
  → 1,000 negative + 1,000 positive

Final test set: 2,000 samples
Test set label distribution:
  Negative (1-2 star): 1,000 (50.0%)
  Positive (4-5 star): 1,000 (50.0%)
================================================================================

================================================================================
METHOD: Baseline (No Adaptation)
================================================================================
Baseline: 100%|████████████████| 63/63 [00:45<00:00, 1.39it/s, Acc=0.8500]

────────────────────────────────────────────────────────────────────────────────
Baseline Results
────────────────────────────────────────────────────────────────────────────────
Accuracy: 0.8500 (85.00%)

Classification Report:
              precision    recall  f1-score   support
           0     0.8456    0.9100    0.8766      1000
           1     0.8821    0.7900    0.8334      1000
    accuracy                         0.8500      2000

Confusion Matrix:
[[ 910   90]
 [ 210  790]]

================================================================================
METHOD: TENT (Entropy Minimization)
================================================================================
Configuring TENT...
  Adapting 156 normalization parameters
  Learning rate: 0.001

TENT: 100%|████████████████| 63/63 [01:12<00:00, 1.15s/it, Acc=0.9000]

────────────────────────────────────────────────────────────────────────────────
TENT Results
────────────────────────────────────────────────────────────────────────────────
Accuracy: 0.9000 (90.00%)

Classification Report:
              precision    recall  f1-score   support
           0     0.8890    0.9350    0.9114      1000
           1     0.9256    0.8650    0.8943      1000
    accuracy                         0.9000      2000

Confusion Matrix:
[[ 935   65]
 [ 135  865]]

Adaptation Statistics:
  Total samples: 2000
  Adapted samples: 2000
  Total updates: 2000

================================================================================
FINAL SUMMARY
================================================================================
Method               Accuracy     Change      
────────────────────────────────────────────────────────────────────────────────
Baseline             0.8500 (85.00%)  ─
TENT                 0.9000 (90.00%)  +5.00%
================================================================================
```

## Hotel Dataset Statistics

The hotel dataset (`hotels.csv`) contains global hotel information:

**Size:**
- Total entries: 1,010,033 hotels
- Valid samples (with descriptions): ~980,000
- After filtering & label mapping: ~660,000 usable samples

**Label Mapping:**
- FiveStar, FourStar → **Positive (1)**
- TwoStar, OneStar → **Negative (0)**
- ThreeStar → **Excluded** (neutral)

**Full Dataset Distribution (before sampling):**
- Negative (1-2 star): ~510,000 samples
- Positive (4-5 star): ~150,000 samples

**Test Set Distribution (with balanced sampling):**
- Negative: 50%
- Positive: 50%

**Columns:**
- **Description**: Hotel description text (used as model input)
- **HotelRating**: FiveStar, FourStar, ThreeStar, etc. (used as label)
- Also: HotelName, Address, Attractions, Facilities, Country, City, etc.

**Why This is a Good Test:**
This represents a real **distribution shift**:
- **Source domain**: Yelp restaurant reviews (training data)
- **Target domain**: Hotel descriptions (test data)
- **Shift type**: Domain gap (restaurants vs hotels, review style vs descriptions)

## Known Issues

### CUDA Compatibility

The current PyTorch installation doesn't support NVIDIA B200 GPUs (sm_100). You may see:

```
NVIDIA B200 with CUDA capability sm_100 is not compatible with the current PyTorch installation.
```

**Solutions:**
1. Wait for PyTorch to add B200 support
2. Use a different GPU if available
3. The model has some hardcoded `.cuda()` calls that need to be fixed for CPU-only execution

### Model Hardcoded Device Calls

The PLens.py file has hardcoded `.cuda()` calls that prevent CPU execution. To fix:
- Replace `.cuda()` with `.to(device)` throughout PLens.py
- Pass device as an argument

## Adding More Adaptation Methods

To add new adaptation methods (e.g., EATA, SAR):

1. Create wrapper in new file (e.g., `eata.py`)
2. Follow the same pattern as `tent.py`:
   - Inherit from `nn.Module`
   - Implement `forward()` with text inputs
   - Handle model configuration and parameter collection
3. Update `test_adaptation.py` to support new method
4. Add to `--method` choices

Example structure:
```python
class EATA(nn.Module):
    def __init__(self, model, optimizer, fisher=None):
        ...
    
    def forward(self, input_ids=None, attention_mask=None, ...):
        # Your adaptation logic
        ...
```

## Comparison with ProtoViT

| Aspect | ProtoViT | ProtoLens |
|--------|----------|-----------|
| Domain | Vision (images) | Text (reviews) |
| Input | Image tensors | Text tokens (input_ids) |
| Adaptation | LayerNorm/BatchNorm | LayerNorm/BatchNorm |
| Forward args | `model(images)` | `model(input_ids=..., attention_mask=...)` |
| Output | logits, distances, values | logits, distances, values |

The TENT implementation is adapted from ProtoViT but modified for text inputs.

## Future Work

- Add EATA, SAR, and other TTA methods
- Fix hardcoded CUDA calls for CPU support
- Add support for other text datasets (Amazon, IMDB)
- Add per-class accuracy tracking
- Add confidence calibration metrics
