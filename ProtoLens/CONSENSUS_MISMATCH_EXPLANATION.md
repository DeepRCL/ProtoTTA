# The Consensus Mismatch: ProtoViT vs ProtoLens

## The Confusion

You're absolutely right to be confused! The `consensus` function was **repurposed** from ProtoViT, but it's doing something **completely different** in ProtoLens.

## In ProtoViT (Original Purpose)

**Consensus aggregates SUB-PROTOTYPES within each prototype:**

```
Input:  [Batch, Prototypes, Sub-prototypes]
        e.g., [32, 20, 196]  (20 prototypes, 196 patches per prototype)

Purpose: Aggregate multiple patches → single similarity per prototype

Example:
  Prototype 0: similarities = [0.1, 0.2, 0.9, 0.3, ...]  (196 values)
  → consensus → 0.9 (if max) or 0.375 (if mean)

Output: [Batch, Prototypes]
        e.g., [32, 20]  (one similarity per prototype)
```

**Why it's called "consensus":**
- Multiple patches (sub-prototypes) must "agree" on the similarity
- `max`: Best patch wins (no consensus needed)
- `mean`: All patches must agree (true consensus)
- `top_k_mean`: Top patches agree (soft consensus)

## In ProtoLens (Repurposed)

**Consensus aggregates ACROSS PROTOTYPES to get a single score per sample:**

```
Input:  [Batch, Prototypes]
        e.g., [32, 50]  (50 prototypes, one similarity per prototype)

Purpose: Aggregate multiple prototypes → single "reliability score" per sample

Example:
  Sample 0: similarities = [0.1, 0.2, 0.9, 0.3, ...]  (50 values)
  → consensus → 0.9 (if max) or 0.375 (if mean)

Output: [Batch]
        e.g., [32]  (one score per sample)
```

**Why it's used:**
- NOT for sub-prototype aggregation (ProtoLens has no sub-prototypes!)
- For **geometric filtering**: Get a single "reliability score" to decide if sample should be adapted
- The function name is misleading - it's not really "consensus" anymore

## The Connection to Geometric Filtering

The repurposed consensus function is used like this:

```python
# Step 1: Get similarities per prototype
similarities = model(...)  # [Batch, Prototypes]

# Step 2: Aggregate to single score per sample (misleadingly called "consensus")
consensus_sims = compute_consensus_similarity(similarities, strategy='max')
# [Batch, Prototypes] → [Batch]

# Step 3: Use this score for filtering
reliable_mask = consensus_sims >= geo_filter_threshold  # [Batch]
```

**The question is:** How should we aggregate 50 prototype similarities into a single "reliability score"?

- `max`: Sample is reliable if it matches at least ONE prototype strongly
- `mean`: Sample is reliable if it matches MANY prototypes moderately
- `top_k_mean`: Sample is reliable if it matches TOP K prototypes well

## The Problem

The function name `compute_consensus_similarity` is **misleading** because:
1. It's not aggregating sub-prototypes (ProtoLens has none)
2. It's aggregating across prototypes for filtering
3. The "consensus" concept doesn't really apply here

## Better Names

This function should probably be called:
- `aggregate_prototype_similarities()` 
- `compute_reliability_score()`
- `filter_score_from_similarities()`

Not `compute_consensus_similarity()` which implies sub-prototype aggregation.

## Why It Still Works

Even though the name is misleading, the function still works because:
- It's just an aggregation operation (max/mean/top_k)
- The math is the same (aggregate along one dimension)
- It serves a different purpose (filtering vs sub-proto aggregation) but uses the same mechanism

## Summary

| Aspect | ProtoViT | ProtoLens |
|--------|----------|-----------|
| **Input shape** | `[B, P, K]` (K=sub-prototypes) | `[B, P]` (no sub-prototypes) |
| **Aggregation** | Within each prototype (across sub-prototypes) | Across all prototypes |
| **Output shape** | `[B, P]` (one per prototype) | `[B]` (one per sample) |
| **Purpose** | Get similarity per prototype | Get reliability score per sample |
| **Used for** | Adaptation loss computation | Geometric filtering |
| **Name accuracy** | ✅ "Consensus" makes sense | ❌ "Consensus" is misleading |

## Recommendation

The function is being **repurposed** from ProtoViT code. It works, but the name and purpose are different. For ProtoLens, it's really about:
- **"How reliable is this sample?"** (based on prototype matches)
- Not about consensus among sub-prototypes

The `max` strategy makes sense: "A sample is reliable if it matches at least one prototype strongly."
