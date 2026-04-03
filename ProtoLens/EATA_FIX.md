# EATA Filtering Issue - Fixed

## Problem

EATA was showing **0% adaptation rate** - all samples were being filtered out.

## Root Cause

The entropy threshold (`e_margin`) was set incorrectly:

```python
# WRONG (before):
e_margin = math.log(2) / 2 - 1  # = -0.65 (NEGATIVE!)
```

**Problem**: Entropy is always **positive** (ranges from 0 to log(num_classes)). 
- For binary classification: entropy ∈ [0, 0.69]
- With threshold = -0.65, the filter `entropys < e_margin` filters out **ALL** samples

## Fix

Changed to a reasonable positive threshold:

```python
# CORRECT (after):
e_margin = 0.4  # Keeps samples with entropy < 0.4 (confident predictions)
```

## EATA Filtering Logic

EATA has **two filters**:

1. **Entropy Filter** (Reliability):
   - Keeps samples where `entropy < e_margin`
   - Filters out uncertain/unreliable predictions
   - For binary classification: keeps samples with entropy < 0.4

2. **Cosine Similarity Filter** (Redundancy):
   - Keeps samples where `|cosine_similarity| < d_margin` (default: 0.05)
   - Filters out samples too similar to previously seen samples
   - Prevents redundant updates

## Expected Behavior After Fix

- **Entropy filter**: Should keep ~60-80% of samples (confident predictions)
- **Redundancy filter**: Further reduces to ~30-60% of original samples
- **Final adaptation rate**: ~30-60% (varies by dataset)

## Comparison with Other Methods

| Method | Filtering | Adaptation Rate |
|--------|-----------|----------------|
| **TENT** | None | 100% (all samples) |
| **EATA** | Entropy + Redundancy | ~30-60% |
| **ProtoTTA** | Geometric (similarity) | ~50-70% |

## Verification

After the fix, EATA should show:
```
Adaptation Statistics:
  Total samples: 4000
  Reliable samples (entropy filter): ~2400-3200 (60-80%)
  Adapted samples (both filters): ~1200-2400 (30-60%)
  Adaptation rate: ~30-60%
```

If adaptation rate is still 0%, check:
1. Model predictions are confident (entropy < 0.4)
2. Samples are diverse (not all identical)
3. Model is in train mode for adaptation
