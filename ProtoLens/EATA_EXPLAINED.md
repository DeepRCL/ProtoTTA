# How EATA Works: Efficient Anti-Catastrophic Test-Time Adaptation

## Overview

**EATA** (Efficient Anti-Catastrophic Test-Time Adaptation) is an improved version of TENT that prevents **catastrophic forgetting** during test-time adaptation. It was introduced in ICML 2022.

**Key Idea**: Only adapt on **reliable and non-redundant** samples to prevent the model from forgetting what it learned during training.

## The Problem EATA Solves

### TENT's Limitation
- TENT adapts on **every sample** (100% adaptation rate)
- Problem: Some samples are:
  - **Unreliable**: High entropy (uncertain predictions) → noisy gradients
  - **Redundant**: Too similar to previous samples → redundant updates
- Result: Model can "forget" original knowledge → **catastrophic forgetting**

### EATA's Solution
- **Filter unreliable samples**: Only adapt on confident predictions
- **Filter redundant samples**: Skip samples too similar to what we've already seen
- Result: More stable adaptation, prevents forgetting

## How EATA Works (Step by Step)

### Step 1: Forward Pass
```python
outputs, proto_dist, proto_val = model(input_ids, ...)
```

### Step 2: Compute Entropy
```python
entropys = softmax_entropy(outputs)  # [Batch]
# Entropy measures prediction uncertainty
# Low entropy = confident prediction (e.g., [0.95, 0.05])
# High entropy = uncertain prediction (e.g., [0.51, 0.49])
```

**Entropy Formula:**
```
H(p) = -Σ p_i * log(p_i)
```
- Binary classification: entropy ∈ [0, log(2) ≈ 0.69]
- Entropy = 0: perfectly confident (one class = 1.0)
- Entropy = 0.69: completely uncertain (both classes = 0.5)

### Step 3: Filter 1 - Reliability Filter (Entropy Threshold)

```python
# Keep samples with LOW entropy (confident predictions)
filter_ids_1 = torch.where(entropys < e_margin)
# e_margin = 0.4 for binary classification
```

**What this does:**
- ✅ **Keeps**: Samples with entropy < 0.4 (confident predictions)
- ❌ **Filters out**: Samples with entropy ≥ 0.4 (uncertain predictions)

**Why?**
- Uncertain predictions → noisy gradients → bad adaptation
- Confident predictions → clean gradients → good adaptation

**Example:**
```
Sample 1: [0.95, 0.05] → entropy = 0.29 → ✅ KEEP (confident)
Sample 2: [0.51, 0.49] → entropy = 0.69 → ❌ FILTER (uncertain)
```

### Step 4: Filter 2 - Redundancy Filter (Cosine Similarity)

```python
# Compare current predictions with moving average of previous predictions
cosine_similarities = F.cosine_similarity(
    current_model_probs.unsqueeze(0),  # Moving average [1, num_classes]
    outputs[filter_ids_1].softmax(1),   # Current predictions [batch, num_classes]
    dim=1
)

# Keep samples that are DIFFERENT from previous samples
filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
# d_margin = 0.05 (default)
```

**What this does:**
- ✅ **Keeps**: Samples with |cosine_similarity| < 0.05 (different from previous)
- ❌ **Filters out**: Samples with |cosine_similarity| ≥ 0.05 (similar to previous)

**Why?**
- Similar samples → redundant gradients → wasted computation
- Different samples → new information → useful adaptation

**Example:**
```
Previous avg: [0.8, 0.2]
Sample 1: [0.82, 0.18] → cosine = 0.99 → ❌ FILTER (too similar)
Sample 2: [0.3, 0.7] → cosine = 0.2 → ✅ KEEP (different)
```

### Step 5: Reweight Loss

```python
# Samples closer to threshold get higher weight
coeff = 1 / (torch.exp(entropys - e_margin) + 1e-8)
entropys_weighted = entropys * coeff
loss = entropys_weighted.mean()
```

**What this does:**
- Samples with entropy **just below** threshold get **higher weight**
- Samples with entropy **much lower** get **lower weight**
- Encourages adaptation on "borderline confident" samples

**Example:**
```
Sample 1: entropy = 0.35 → coeff ≈ 1.06 → higher weight
Sample 2: entropy = 0.10 → coeff ≈ 1.37 → even higher weight
```

### Step 6: Update Model

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Only updates on samples that passed **both filters**.

### Step 7: Update Moving Average

```python
# Update moving average of predictions
updated_probs = update_model_probs(
    current_model_probs,
    outputs[filter_ids_1][filter_ids_2].softmax(1),
    num_samples_update
)
```

**What this does:**
- Maintains running average of prediction probabilities
- Used for redundancy filtering in next batch

## Complete Flow Diagram

```
Input Batch (N samples)
    ↓
Forward Pass → Get Predictions
    ↓
Compute Entropy for each sample
    ↓
┌─────────────────────────────────┐
│ Filter 1: Entropy Threshold    │
│ Keep: entropy < 0.4            │
│ Result: M samples (M ≤ N)      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Filter 2: Cosine Similarity     │
│ Keep: |cos| < 0.05              │
│ Result: K samples (K ≤ M)      │
└─────────────────────────────────┘
    ↓
Reweight Loss (by entropy)
    ↓
Backward Pass + Update
    ↓
Update Moving Average
```

## Key Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `e_margin` | 0.4 | Entropy threshold (keeps confident predictions) |
| `d_margin` | 0.05 | Cosine similarity threshold (keeps diverse samples) |
| `fisher_alpha` | 2000.0 | Weight for Fisher regularization (if used) |

## Adaptation Statistics

EATA tracks:
- **Total samples**: All samples processed
- **Reliable samples**: Samples passing entropy filter (Filter 1)
- **Adapted samples**: Samples passing both filters (Filter 1 + Filter 2)
- **Adaptation rate**: Adapted samples / Total samples

**Typical rates:**
- Entropy filter: ~60-80% pass
- Redundancy filter: ~50-70% of entropy-filtered pass
- **Final adaptation rate: ~30-60%**

## Comparison with TENT

| Aspect | TENT | EATA |
|--------|------|------|
| **Filtering** | None | Two filters |
| **Adaptation Rate** | 100% | ~30-60% |
| **Stability** | Can forget | Prevents forgetting |
| **Speed** | Faster | Slightly slower (filtering overhead) |
| **Use Case** | Clean test data | Noisy/redundant test data |

## Why It Works

1. **Reliability Filter**: 
   - Only adapts on confident predictions
   - Avoids noisy gradients from uncertain samples
   - More stable updates

2. **Redundancy Filter**:
   - Skips similar samples
   - Focuses on diverse information
   - Prevents overfitting to repeated patterns

3. **Reweighting**:
   - Prioritizes borderline samples
   - Balances adaptation vs. stability

## Example Scenario

**Test Set**: 1000 hotel reviews

**EATA Processing:**
1. Forward pass: 1000 samples
2. Entropy filter: 700 samples pass (confident predictions)
3. Redundancy filter: 400 samples pass (diverse samples)
4. Adaptation: Update model on 400 samples
5. **Adaptation rate: 40%**

**Result**: Model adapts to hotel domain while preserving Yelp knowledge!

## Code Reference

See `ProtoLens/eata.py` for the implementation:
- `forward_and_adapt_eata()`: Main adaptation logic
- `softmax_entropy()`: Entropy computation
- `update_model_probs()`: Moving average update
