# Interpretability Metrics for ProtoTTA

This guide explains how to implement the prototype-level metrics used in this repository.

The metrics are designed to answer a question standard robustness scores cannot answer:

**Did adaptation restore the model's semantic prototype behavior, or did it only improve predictions numerically?**

## 1. Required Inputs

To compute the core metrics, you need:

- a clean baseline loader,
- a shifted or corrupted loader,
- a way to extract prototype activations without adaptation,
- class labels,
- optional classifier weights for importance-aware metrics.

Recommended evaluator hooks:

```python
collect_clean_baseline(clean_loader)
extract_prototype_activations(model, loader)
forward_no_adapt(batch)
```

Good references:

- [`ProtoViT/prototype_tta_metrics.py`](../ProtoViT/prototype_tta_metrics.py)
- [`ProtoViT/enhanced_prototype_metrics.py`](../ProtoViT/enhanced_prototype_metrics.py)
- [`ProtoLens/prototype_metrics.py`](../ProtoLens/prototype_metrics.py)
- [`ProtoPFormer/prototype_tta_metrics.py`](../ProtoPFormer/prototype_tta_metrics.py)
- [`ProtoPFormer/enhanced_prototype_metrics.py`](../ProtoPFormer/enhanced_prototype_metrics.py)
- [`protosvit/evaluate_robustness_cars_c.py`](../protosvit/evaluate_robustness_cars_c.py)

## 2. Standardize Prototype Activations First

Different models emit different prototype outputs:

- `[B, P]` direct prototype scores,
- `[B, P, K]` prototype + sub-prototype scores,
- tuple or dict outputs with local/global prototype branches.

Before computing metrics, convert every sample to a single activation vector:

```python
activations: [N, P_total]
```

Examples:

- aggregate `[B, P, K]` to `[B, P]` using `sum`, `max`, or the model's natural inference rule,
- concatenate local and global branches,
- use the same extraction rule for clean and adapted runs.

## 3. PAC: Prototype Activation Consistency

PAC measures whether prototype behavior under shift remains close to the clean baseline.

### Intuition

If ProtoTTA is doing the right thing, the adapted sample should reactivate prototype patterns that resemble the clean sample more closely than a degraded or misaligned baseline would.

### Implementation

For each aligned clean/adapted sample pair:

```python
pac_i = cosine_similarity(adapted_activation_i, clean_activation_i)
```

Aggregate over all samples:

```python
PAC_mean = mean(pac_i)
PAC_std = std(pac_i)
```

You can also use:

- normalized L2 similarity,
- correlation.

Cosine similarity is the default in this repository.

## 4. PCA: Prototype Class Alignment

PCA measures whether the most active prototypes belong to the correct class.

### Class-specific prototypes

If each prototype belongs to one class:

1. get top-k activated prototypes,
2. check whether each prototype belongs to the true label,
3. compute a weighted or unweighted alignment score.

Typical pattern:

```python
top_vals, top_idx = torch.topk(activations, k=top_k)
correct_mask = (proto_ids[top_idx] == label).float()
weights = softmax(top_vals)
score = (correct_mask * weights).sum()
```

### Shared prototypes

If prototypes are shared across classes, use classifier weights instead of hard ownership.

## 5. PCA-W: Weighted Prototype Class Alignment

PCA-W is one of the most useful metrics when prototypes are not equally important.

### Intuition

A prototype being active is not enough. What matters is whether that active prototype actually contributes to the correct class decision.

### Implementation

For each sample:

1. select top-k activated prototypes,
2. get classifier weights for the ground-truth class,
3. combine activation strength and class importance,
4. measure how much of the contribution comes from semantically correct prototypes.

Typical class-specific form:

```python
importance = abs(classifier_weight[true_class, top_idx])
contrib = top_vals * importance
score = (contrib * correct_mask).sum() / (contrib.sum() + eps)
```

Typical shared-prototype form:

```python
top_vals, top_idx = topk(sample_acts)
class_weights = classifier_weight[true_class, top_idx]
act_normalized = softmax(top_vals)
contributions = act_normalized * class_weights
score = positive_contribution / total_absolute_contribution
```

## 6. Sparsity

Sparser prototype usage is often easier to interpret.

Common summaries:

- Gini coefficient of prototype activations,
- number of active prototypes above a threshold.

Typical outputs:

```python
sparsity_gini_mean
sparsity_active_mean
```

## 7. Calibration / Prediction Stability

These metrics compare the current method's predictions on noisy data to the clean model's baseline predictions.

Useful signals:

- prediction agreement,
- logit cosine similarity,
- confidence correlation.

This captures whether adaptation preserves the model's overall decision geometry, not only prototype activations.

## 8. Ground-Truth Class Contribution Change

This metric asks whether adaptation increases the contribution of prototypes aligned with the true class.

A practical way to compute it:

1. compute clean contribution of the true-class prototypes,
2. compute adapted contribution of the true-class prototypes,
3. report the mean change or improvement rate.

This is especially useful when accuracy alone hides whether the right semantic evidence was restored.

## 9. Adaptation-Process Metrics

These are not semantic metrics, but they matter for explainable TTA.

Useful quantities:

- adaptation rate,
- average updates per sample,
- filtered sample ratio,
- efficiency metrics such as time per sample and adapted parameter ratio.

These rely on `adaptation_stats` stored by the wrapper.

Recommended keys:

```python
{
    "total_samples": ...,
    "adapted_samples": ...,
    "total_updates": ...,
}
```

## 10. Minimal Evaluator Recipe

For a new model, the metric pipeline should look like this:

1. implement `forward_no_adapt`,
2. implement `extract_prototype_activations`,
3. collect clean baseline activations,
4. collect adapted activations on shifted data,
5. compute PAC,
6. compute PCA,
7. compute PCA-W if classifier weights are available,
8. optionally compute calibration, contribution-change, sparsity, and efficiency.

## 11. What To Reuse vs What To Redefine

You can reuse the metric logic almost unchanged if you can provide:

- standardized activations `[N, P]`,
- label-alignment metadata,
- classifier weights.

You must redefine only:

- how activations are extracted from your model output,
- how prototype ownership is determined,
- how multi-branch or sub-prototype activations are merged.

## 12. Recommended Output Schema

For interoperability, save metrics in a dictionary with stable keys:

```python
{
    "accuracy": ...,
    "PAC_mean": ...,
    "PAC_std": ...,
    "PCA_mean": ...,
    "PCA_std": ...,
    "PCA_weighted_mean": ...,
    "PCA_weighted_std": ...,
    "sparsity_gini_mean": ...,
    "calibration_agreement": ...,
    "adaptation_rate": ...,
}
```

This makes plotting, table generation, and VLM correlation analysis much easier.
