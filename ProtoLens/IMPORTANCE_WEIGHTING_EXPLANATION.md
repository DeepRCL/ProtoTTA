# Prototype Importance Weighting: The Issue Explained

## Your Current Implementation

```python
fc_weights = model.fc.weight  # [num_classes, num_prototypes]
prototype_importance = torch.abs(fc_weights).mean(dim=0)  # [num_prototypes]
```

This computes a **global importance** for each prototype by averaging the absolute weights across all classes.

## The Problem with a Concrete Example

Let's say you have 3 prototypes and 2 classes (binary classification):

### FC Layer Weights:
```
Prototype 0: [w_class0=+0.9, w_class1=-0.1]  → contributes strongly to class 0
Prototype 1: [w_class0=-0.1, w_class1=+0.9]  → contributes strongly to class 1  
Prototype 2: [w_class0=+0.3, w_class1=+0.3]  → contributes weakly to both
```

### Your Current Importance Calculation:
```python
prototype_importance[0] = mean(|+0.9|, |-0.1|) = 0.5
prototype_importance[1] = mean(|-0.1|, |+0.9|) = 0.5
prototype_importance[2] = mean(|+0.3|, |+0.3|) = 0.3
```

**Result**: Prototypes 0 and 1 get the same importance (0.5), even though they're class-specific!

### The Issue During Adaptation:

Now consider a sample that the model **predicts as class 0**:

**For this specific sample:**
- **Prototype 0** has weight **+0.9** for class 0 → **very important** for this prediction
- **Prototype 1** has weight **-0.1** for class 0 → **not important** for this prediction
- **Prototype 2** has weight **+0.3** for class 0 → **moderately important**

**But your code weights them as:**
- Prototype 0: 0.5 (should be 0.9!)
- Prototype 1: 0.5 (should be 0.1!)
- Prototype 2: 0.3 (this is okay)

### Why This Matters:

When adapting, you want to:
1. **Focus on prototypes that matter for the current prediction**
2. **Make those prototypes more decisive** (similarity → 0 or 1)

If you use **global importance** (averaged across classes):
- You give equal weight to Prototype 0 and Prototype 1
- But for a class-0 prediction, Prototype 0 is 9× more important!
- You're wasting adaptation effort on Prototype 1 (which doesn't contribute to this prediction)

## The Fix: Class-Specific Importance

Instead of averaging across classes, use the weight for the **predicted class**:

```python
# Get predicted class for each sample
pred_class = outputs_filtered.argmax(dim=1)  # [num_reliable]

# Get importance for the predicted class
prototype_importance = torch.abs(fc_weights[pred_class])  # [num_reliable, num_prototypes]
```

**For the same example (sample predicts class 0):**
- Prototype 0: importance = |+0.9| = **0.9** ✅
- Prototype 1: importance = |-0.1| = **0.1** ✅
- Prototype 2: importance = |+0.3| = **0.3** ✅

Now the adaptation focuses on the right prototypes!

## Visual Comparison

### Current (Global Average):
```
Sample predicts class 0:
  Prototype 0 entropy × 0.5  ← underweighted!
  Prototype 1 entropy × 0.5  ← overweighted!
  Prototype 2 entropy × 0.3
```

### Fixed (Class-Specific):
```
Sample predicts class 0:
  Prototype 0 entropy × 0.9  ← correctly weighted!
  Prototype 1 entropy × 0.1  ← correctly weighted!
  Prototype 2 entropy × 0.3
```

## Code Change

**Current:**
```python
prototype_importance = torch.abs(fc_weights).mean(dim=0)  # [num_prototypes]
weighted_entropy = entropy_per_proto * prototype_importance.unsqueeze(0)
```

**Fixed:**
```python
pred_class = outputs_filtered.argmax(dim=1)  # [num_reliable]
prototype_importance = torch.abs(fc_weights[pred_class])  # [num_reliable, num_prototypes]
weighted_entropy = entropy_per_proto * prototype_importance
```

Note: Now `prototype_importance` is `[num_reliable, num_prototypes]` instead of `[num_prototypes]`, so no need for `unsqueeze(0)`.

## Why Your Current Approach Still Works (But Suboptimally)

Your current approach **does work** because:
- Prototypes with large weights (in any class) are still important overall
- Averaging gives a reasonable approximation

But it's **suboptimal** because:
- It doesn't adapt to the specific prediction
- It wastes effort on irrelevant prototypes
- It underweights important prototypes for specific predictions

## Summary

**The issue**: You're using a **global importance** (averaged across classes) when you should use **class-specific importance** (for the predicted class).

**The impact**: Adaptation effort is distributed suboptimally - some prototypes get too much attention, others too little, relative to what matters for each specific prediction.

**The fix**: Use the FC weight for the predicted class, not the average across all classes.
