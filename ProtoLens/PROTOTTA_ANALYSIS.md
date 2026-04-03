# ProtoTTA Design Analysis for ProtoLens

## Executive Summary

The ProtoTTA implementation has **good foundational ideas** but several **design choices that may not be optimal** for ProtoLens architecture. This document analyzes each component.

---

## ✅ What Makes Sense

### 1. **Using Actual Prototype Similarities**
```python
outputs, loss_mu, augmented_loss, similarities = model(...)
```
- ✅ **Correct**: Uses the actual similarity tensor `[B, P]` from ProtoLens forward pass
- ✅ **Matches ProtoViT approach**: Directly leverages prototype information
- ✅ **Better than logits-only**: Prototype similarities provide richer signal

### 2. **Geometric Filtering Concept**
- ✅ **Reasonable idea**: Filter unreliable samples before adaptation
- ✅ **Prevents negative adaptation**: Avoids adapting on noisy/uncertain samples
- ✅ **Matches ProtoViT philosophy**: Selective adaptation is important

### 3. **Parameter Selection (LayerNorm + Attention Biases)**
- ✅ **Appropriate for TTA**: Small, impactful parameter set (~98 params)
- ✅ **Attention biases make sense**: Control where model focuses (important for text)
- ✅ **Low risk**: Won't cause catastrophic forgetting

### 4. **Safety Measures**
- ✅ **Gradient clipping**: Prevents exploding gradients
- ✅ **NaN/Inf checks**: Robust error handling
- ✅ **Episodic mode support**: Can reset between batches

---

## ⚠️ Issues and Concerns

### 1. **Geometric Filter Threshold is Too Low**

**Problem:**
```python
# Default in run_inference_amazon_c.py
geo_threshold = 0.1  # For cosine similarity in [-1, 1]
```

**Analysis:**
- Cosine similarity range: **[-1, 1]**
- Typical ProtoLens similarities: **0.5-0.9** (from comments)
- Threshold **0.1** filters almost **nothing** (only very dissimilar samples)
- Comment in `proto_tta.py` says "0.3 is reasonable" but default is 0.1

**Impact:**
- Geometric filtering becomes ineffective
- May adapt on unreliable samples
- Wastes computation on filtering that doesn't filter

**Recommendation:**
```python
# Should be closer to actual similarity distribution
geo_threshold = 0.3  # Or even 0.5 for more selective filtering
# Or make it adaptive based on batch statistics
```

### 2. **Binary Entropy Loss May Be Inappropriate**

**Current Implementation:**
```python
# Normalize from [-1, 1] to [0, 1]
sim_normalized = (similarities_filtered + 1.0) / 2.0
# Binary entropy: encourages similarity → 0 or 1
entropy = -(sim_normalized * log(sim_normalized) + (1-sim_normalized) * log(1-sim_normalized))
```

**Problem:**
- **ProtoLens uses shared prototypes** (not class-specific)
- FC layer learns **class-specific weights** for each prototype
- A prototype can contribute **+0.9 to class 0** and **-0.8 to class 1**
- **Forcing similarity → 0 or 1** may hurt this nuanced weighting

**Why it works in ProtoViT:**
- ProtoViT often has class-specific prototypes
- Extreme activations (0 or 1) make sense for class-specific prototypes

**Why it might not work for ProtoLens:**
- Shared prototypes need **moderate activations** to allow FC layer flexibility
- Similarity of 0.6 might be optimal if FC weights are [-0.5, +0.8]
- Forcing to 0 or 1 removes this flexibility

**Alternative Approaches:**
1. **Sparsity loss**: Encourage some prototypes to be inactive (0), but allow others to be moderate
2. **Confidence-based loss**: Only encourage extreme activations for high-confidence samples
3. **Class-conditional loss**: Different objectives for prototypes that contribute to predicted class

### 3. **Prototype Importance Weighting May Be Suboptimal**

**Current Implementation:**
```python
fc_weights = model.fc.weight  # [num_classes, num_prototypes]
prototype_importance = torch.abs(fc_weights).mean(dim=0)  # [num_prototypes]
```

**Issue:**
- Uses **mean of absolute weights** across classes
- Ignores **sign** (positive vs negative contribution)
- A prototype with weights [+0.9, -0.8] gets same importance as [+0.1, +0.1]

**Better Approach:**
```python
# Weight by magnitude of contribution to predicted class
probs = torch.softmax(outputs_filtered, dim=1)
pred_class = probs.argmax(dim=1)  # [num_reliable]
prototype_importance = torch.abs(fc_weights[pred_class])  # [num_reliable, num_prototypes]
```

### 4. **Not Adapting Prototype Vectors**

**Current State:**
- Only adapts LayerNorm + Attention biases
- **Prototype vectors are frozen** (`self.prototype_vectors`)

**Why This Might Be Limiting:**
- Under domain shift (Yelp → Hotel), **prototype semantics may need adjustment**
- Example: "service" prototype trained on restaurant reviews may need to shift for hotel reviews
- LayerNorm/attention adaptation can only do so much

**ProtoViT Comparison:**
- ProtoViT can adapt prototype vectors (in some modes)
- This allows prototypes to shift semantically

**Consideration:**
- Adapting prototypes is **riskier** (more parameters, higher risk of collapse)
- But might be **necessary** for large domain shifts
- Could add as optional mode: `layernorm_attn_bias_proto`

### 5. **Consensus Strategy 'max' is Correct, but 'top_k_mean' Doesn't Make Sense**

**Current Code:**
```python
consensus_strategy='max'  # Used in setup
# But code also supports 'top_k_mean' which requires sub-prototypes
```

**Analysis:**
- ✅ **'max' is correct**: ProtoLens has no sub-prototypes, so max makes sense
- ❌ **'top_k_mean' doesn't apply**: ProtoLens doesn't have sub-prototypes like ProtoViT
- The `compute_consensus_similarity` function supports it, but it's not meaningful here

**Recommendation:**
- Remove or deprecate `top_k_mean` for ProtoLens
- Or document that it's only for ProtoViT compatibility

### 6. **Sample Confidence Weighting May Be Circular**

**Current Implementation:**
```python
probs = torch.softmax(outputs_filtered, dim=1)
sample_confidence = probs.max(dim=1)[0]  # [num_reliable]
weighted_entropy = weighted_entropy * sample_confidence.unsqueeze(1)
```

**Issue:**
- Uses **model's own predictions** to weight the loss
- If model is wrong, it weights wrong predictions more
- Can create **confirmation bias**: model becomes more confident in wrong predictions

**Alternative:**
- Use **prototype similarity** as confidence instead
- Or use **entropy** (low entropy = high confidence, but less circular)

---

## 🔧 Recommended Improvements

### 1. **Fix Geometric Filter Threshold**
```python
# Adaptive threshold based on batch statistics
with torch.no_grad():
    batch_mean_sim = consensus_sims.mean().item()
    batch_std_sim = consensus_sims.std().item()
    # Use percentile-based threshold (e.g., 25th percentile)
    threshold = torch.quantile(consensus_sims, 0.25).item()
```

### 2. **Rethink Binary Entropy Loss**
```python
# Option A: Sparsity loss (encourage some prototypes to be inactive)
sparsity_loss = torch.relu(0.3 - similarities_filtered).mean()  # Encourage < 0.3

# Option B: Confidence-weighted entropy (only for high-confidence samples)
high_conf_mask = sample_confidence > 0.7
if high_conf_mask.sum() > 0:
    entropy_loss = binary_entropy_loss(similarities_filtered[high_conf_mask])
else:
    entropy_loss = torch.tensor(0.0)

# Option C: Class-conditional loss
pred_class = outputs_filtered.argmax(dim=1)
# For prototypes that contribute to predicted class, encourage activation
# For others, encourage deactivation
```

### 3. **Improve Prototype Importance Weighting**
```python
# Weight by contribution to predicted class
pred_class = outputs_filtered.argmax(dim=1)  # [num_reliable]
prototype_importance = torch.abs(fc_weights[pred_class])  # [num_reliable, num_prototypes]
# Or use gradient-based importance
```

### 4. **Consider Adapting Prototypes (Optional)**
```python
# Add to adapt_utils.py
if 'proto' in adaptation_mode:
    if hasattr(model, 'prototype_vectors'):
        params.append(model.prototype_vectors)
        names.append('prototype_vectors')
```

### 5. **Fix Sample Confidence Weighting**
```python
# Use prototype similarity as confidence instead
prototype_confidence = consensus_sims[reliable_mask]  # Already filtered
# Or use entropy (inverted)
entropy = softmax_entropy(outputs_filtered)
confidence = 1.0 - entropy / np.log(num_classes)  # Normalize to [0, 1]
```

---

## 📊 Architecture-Specific Considerations

### ProtoLens vs ProtoViT Differences

| Aspect | ProtoViT | ProtoLens | Impact on ProtoTTA |
|--------|----------|-----------|-------------------|
| **Prototypes** | Often class-specific | **Shared across classes** | Binary entropy less appropriate |
| **Similarity Range** | Distance-based | **Cosine [-1, 1]** | Threshold needs adjustment |
| **Sub-prototypes** | Yes (patch-level) | **No** | `top_k_mean` doesn't apply |
| **FC Layer** | Simple | **Learns class-specific weights** | Importance weighting should consider class |
| **Domain** | Images | **Text** | Attention biases more important |

### Key Insight

**ProtoLens is fundamentally different from ProtoViT:**
- Shared prototypes require **nuanced activation patterns**
- FC layer does the heavy lifting for class discrimination
- ProtoTTA should **preserve this flexibility**, not force extreme activations

---

## 🎯 Summary of Recommendations

### High Priority
1. ✅ **Increase geometric filter threshold** to 0.3-0.5 (or make adaptive)
2. ✅ **Rethink binary entropy loss** - consider sparsity or class-conditional alternatives
3. ✅ **Fix prototype importance weighting** - use class-specific weights

### Medium Priority
4. ⚠️ **Fix sample confidence weighting** - avoid circular dependency
5. ⚠️ **Remove/deprecate `top_k_mean`** for ProtoLens

### Low Priority (Experimental)
6. 💡 **Consider adapting prototype vectors** for large domain shifts
7. 💡 **Add adaptive threshold** based on batch statistics

---

## 📝 Code Changes Needed

See individual recommendations above for specific code changes. The main files to modify:
- `proto_tta.py`: Loss function, threshold logic, weighting
- `adapt_utils.py`: Optional prototype adaptation mode
- `run_inference_amazon_c.py`: Default threshold value
