# Consensus Strategy Effect in ProtoTTA

## Where It's Used

The `consensus_strategy` is **ONLY used for geometric filtering**, not for the actual adaptation loss.

### Code Flow:

```python
# 1. Get similarities: [Batch, Prototypes]
similarities = model(...)  # [B, P]

# 2. Aggregate to single score per sample: [Batch]
consensus_sims = compute_consensus_similarity(
    similarities, 
    strategy=consensus_strategy,  # <-- Only used here!
    ratio=consensus_ratio
)  # [B]

# 3. Filter samples based on consensus score
reliable_mask = consensus_sims >= geo_filter_threshold  # [B]
similarities_filtered = similarities[reliable_mask]  # [num_reliable, P]

# 4. Loss computation uses FULL similarities (not consensus)
entropy_per_proto = binary_entropy(similarities_filtered)  # Uses all prototypes
```

## Effect

**The consensus strategy affects:**
- ✅ **Which samples are adapted** (geometric filtering)
- ✅ **How many samples pass the filter**

**The consensus strategy does NOT affect:**
- ❌ **How samples are adapted** (loss computation uses full similarity tensor)
- ❌ **Which prototypes matter** (all prototypes are used in loss)

## Strategies Comparison

### For ProtoLens (no sub-prototypes):

**`max` (current default):**
```python
consensus_sims = similarities.max(dim=1)[0]  # [Batch]
```
- Takes the **maximum similarity** across all prototypes
- **Most selective**: Only samples with at least one high similarity pass
- Example: If sample has similarities `[0.1, 0.2, 0.9, 0.3]`, consensus = `0.9`
- Good for: Samples that strongly match at least one prototype

**`mean`:**
```python
consensus_sims = similarities.mean(dim=1)  # [Batch]
```
- Takes the **average similarity** across all prototypes
- **Less selective**: Samples with moderate similarities across many prototypes can pass
- Example: If sample has similarities `[0.1, 0.2, 0.9, 0.3]`, consensus = `0.375`
- Good for: Samples that moderately match multiple prototypes

**`top_k_mean`:**
```python
k = max(1, int(similarities.size(1) * ratio))
topk_values, _ = torch.topk(similarities, k, dim=1)
consensus_sims = topk_values.mean(dim=1)  # [Batch]
```
- Takes the **mean of top k prototypes** (e.g., top 50% if ratio=0.5)
- **Middle ground**: Between max and mean
- Example: If sample has similarities `[0.1, 0.2, 0.9, 0.3]` and k=2, consensus = `mean([0.9, 0.3]) = 0.6`
- Good for: Samples with a few strong matches

## Impact on Adaptation

The consensus strategy has an **indirect effect**:

1. **Different strategies → different samples filtered → different adaptation set**

2. **Example scenario:**
   - Sample A: similarities = `[0.1, 0.2, 0.9, 0.3]`
   - Sample B: similarities = `[0.4, 0.5, 0.4, 0.5]`
   - Threshold = `0.5`

   - **`max` strategy:**
     - Sample A: consensus = `0.9` → **PASSES** ✅
     - Sample B: consensus = `0.5` → **PASSES** ✅ (borderline)
   
   - **`mean` strategy:**
     - Sample A: consensus = `0.375` → **FILTERED OUT** ❌
     - Sample B: consensus = `0.45` → **FILTERED OUT** ❌
   
   - **`top_k_mean` (k=2):**
     - Sample A: consensus = `mean([0.9, 0.3]) = 0.6` → **PASSES** ✅
     - Sample B: consensus = `mean([0.5, 0.5]) = 0.5` → **PASSES** ✅

3. **Result**: Different strategies adapt different sets of samples, which can lead to different final accuracies.

## Recommendation for ProtoLens

Since ProtoLens has **no sub-prototypes**, `max` is the most appropriate:
- ✅ Simple and interpretable
- ✅ Focuses on samples with strong prototype matches
- ✅ Matches the architecture (single similarity per prototype)

However, `mean` or `top_k_mean` could be useful if:
- You want to adapt samples with moderate matches across multiple prototypes
- You find `max` is too selective (filters out too many samples)

## Testing

To test the effect, you can:
1. Run with `--consensus_strategy max` (default)
2. Run with `--consensus_strategy mean`
3. Compare:
   - Number of samples adapted (adaptation rate)
   - Final accuracy
   - Which samples are filtered

The consensus strategy is a **hyperparameter** that controls the selectivity of geometric filtering, not the adaptation mechanism itself.
