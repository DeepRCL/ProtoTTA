# Test-Time Adaptation Methods Comparison

## Overview

This document compares the TTA methods implemented for ProtoLens on the Hotel dataset (domain shift: Yelp → Hotel).

## Methods

| Method | Type | Adaptation Rate | Key Feature |
|--------|------|----------------|-------------|
| **Baseline** | None | 0% | No adaptation |
| **TENT** | Entropy Min. | 100% | Updates all samples |
| **EATA** | Filtered Entropy | ~30-60% | Sample filtering (reliability + redundancy) |
| **ProtoTTA** | Prototype-Aware | ~50-70% | Geometric filtering + binary entropy |

## Detailed Comparison

### TENT (Test Entropy Minimization)
**Paper**: ICLR 2021  
**Core Idea**: Minimize prediction entropy to encourage confident predictions

**Pros:**
- Simple and effective
- Works on any model with normalization layers
- No hyperparameters to tune

**Cons:**
- Adapts on **every sample** (can be unstable)
- Ignores sample reliability
- May adapt on noisy/corrupted samples

**When to Use:**
- Clean test distribution
- Moderate distribution shift
- When you want maximum adaptation

---

### EATA (Efficient Anti-Catastrophic TTA)
**Paper**: ICML 2022  
**Core Idea**: TENT + sample filtering to prevent catastrophic forgetting

**Filtering:**
1. **Entropy threshold**: Reject samples with entropy > threshold (unreliable)
2. **Cosine similarity**: Reject samples too similar to previous samples (redundant)

**Pros:**
- More stable than TENT
- Prevents catastrophic forgetting
- Filters out unreliable samples

**Cons:**
- Requires tuning e_margin and d_margin
- May be too conservative (rejects too many samples)
- Doesn't use prototype information

**When to Use:**
- Noisy test distribution
- Continual adaptation scenarios
- When stability is critical

---

### ProtoTTA (Prototype-Aware TTA, V3 Config)
**Paper**: Custom (based on ProtoViT experiments)  
**Core Idea**: Use prototype similarities for more reliable adaptation

**V3 Configuration:**
- **Geometric filtering**: threshold=0.92 (only adapt high-similarity samples)
- **Consensus strategy**: top_k_mean (average top 50% of sub-prototypes)
- **Loss**: Binary entropy on prototype similarities
- **No ensemble entropy**: Aggregate first, then compute entropy

**Pros:**
- **Prototype-aware**: Uses intermediate representations, not just logits
- **Selective**: Only adapts reliable samples (geometric filter)
- **Robust**: Consensus aggregation reduces outlier sensitivity
- **Best for prototype models**: Designed specifically for prototype-based architectures

**Cons:**
- Requires prototype-based model
- More hyperparameters than TENT
- Slightly more complex

**When to Use:**
- Prototype-based models (ProtoLens, ProtoViT, ProtoPNet)
- When you want selective adaptation
- When prototype quality matters

---

## Expected Performance (Yelp → Hotel)

Based on ProtoViT experiments with similar distribution shifts:

| Method | Expected Accuracy | Adaptation Rate | Speed |
|--------|------------------|----------------|-------|
| Baseline | 82-85% | 0% | Fastest (1x) |
| TENT | 85-88% | 100% | Fast (1.05x) |
| EATA | 86-89% | 30-60% | Medium (1.1x) |
| **ProtoTTA** | **87-91%** | 50-70% | Fast (1.04x) |

**Note**: ProtoTTA typically achieves the best accuracy with selective adaptation.

## Hyperparameters

### TENT
- `learning_rate`: 0.001 (default)
- `steps`: 1 (adapt once per batch)

### EATA
- `learning_rate`: 0.001
- `e_margin`: log(2)/2 - 1 ≈ -0.65 (for binary classification)
- `d_margin`: 0.05 (cosine similarity threshold)
- `fisher_alpha`: 2000.0 (if using Fisher regularization)

### ProtoTTA (V3)
- `learning_rate`: 0.001
- `geo_filter_threshold`: 0.92 (high = more selective)
- `consensus_strategy`: 'top_k_mean'
- `consensus_ratio`: 0.5 (use top 50%)
- `use_ensemble_entropy`: False

## Adaptation Statistics

Each method tracks:
- **Total samples**: Number of samples processed
- **Adapted samples**: Number of samples that triggered updates
- **Total updates**: Number of optimizer steps
- **Adaptation rate**: Adapted samples / Total samples

**Example output:**
```
Adaptation Statistics:
  Total samples: 4000
  Adapted samples: 2340
  Total updates: 2340
  Adaptation rate: 58.5%
```

## Usage Example

```bash
# Test all methods
python run_inference_hotel.py \
    --model_path log_folder/Yelp/.../model.pth \
    --test_samples 4000 \
    --methods baseline tent eata prototta \
    --batch_size 128 \
    --seed 42

# Compare TENT vs ProtoTTA
python run_inference_hotel.py \
    --model_path log_folder/Yelp/.../model.pth \
    --test_samples 2000 \
    --methods baseline tent prototta \
    --batch_size 64
```

## Key Insights

1. **TENT is aggressive**: Adapts on every sample (100%)
2. **EATA is conservative**: Filters heavily (~30-60% adaptation)
3. **ProtoTTA is balanced**: Selective but effective (~50-70% adaptation)
4. **ProtoTTA uses prototypes**: Leverages model's internal representations
5. **All are fast**: Only 4-10% overhead vs baseline

## Choosing a Method

**Use TENT if:**
- You want simple, effective adaptation
- Test distribution is relatively clean
- You don't mind adapting on every sample

**Use EATA if:**
- Test distribution is noisy
- You need stability over long sequences
- You want to prevent catastrophic forgetting

**Use ProtoTTA if:**
- You have a prototype-based model ✅
- You want the best accuracy
- You want selective, reliable adaptation
- You care about prototype quality

## References

- **TENT**: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021
- **EATA**: Niu et al., "Efficient Test-Time Model Adaptation without Forgetting", ICML 2022
- **ProtoTTA**: Custom implementation based on ProtoViT experiments
