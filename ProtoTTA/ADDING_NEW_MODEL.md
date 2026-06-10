# Adding ProtoTTA to a New Model

This guide explains how to add **ProtoTTA** or **ProtoTTA+** to a new prototype-based architecture.

The goal is not to force every model into the same tensor layout. The goal is to expose a small set of **shared semantic hooks**:

- logits,
- prototype activations or similarities,
- a rule for selecting target prototypes,
- a rule for filtering unreliable test samples,
- a set of parameters that are allowed to update at test time.

## 1. Minimum Model Contract

Your model should support these capabilities.

### A. Forward pass must expose prototype evidence

ProtoTTA needs access to per-sample prototype scores, not just final logits.

Recommended output contract:

```python
{
    "logits": logits,                     # [B, C]
    "prototype_scores": prototype_scores, # [B, P] or [B, P, K]
    "filter_scores": filter_scores,       # [B, P] optional, defaults to prototype_scores
}
```

Equivalent tuple-based contracts are also fine, as long as the wrapper knows how to extract:

- `logits`
- `prototype_scores`
- `filter_scores`

### B. The model must expose how prototypes relate to classes

There are two common cases.

#### Class-specific prototypes

Used in ProtoViT, ProtoPNet, and ProtoPFormer.

Recommended hook:

```python
model.prototype_class_identity  # [P, C]
```

In this case, target prototypes for sample `i` are the prototypes belonging to pseudo-label `argmax(logits[i])`.

#### Shared prototypes

Used in ProtoLens-style models, where a prototype may support multiple classes.

Recommended hook:

```python
model.classifier.weight  # [C, P]
```

In this case, the target mask is derived from class-head weights for the pseudo-label.

### C. The model must allow a restricted set of test-time updates

ProtoTTA is most stable when only a small part of the network adapts.

Typical choices:

- `LayerNorm` or `BatchNorm` parameters,
- attention biases in ViTs,
- small prototype-side modules such as projection layers, depthwise convs, add-on layers, or prototype vectors,
- optionally the classification head.

## 2. Choose the Adaptation Scope

Use the smallest scope that still lets the model recover under shift.

### Vision Transformer-like models

Recommended first choice:

- LayerNorm weights and biases,
- attention `qkv/proj` biases,
- small prototype-side heads if they exist.

### Convolutional prototype models

Recommended first choice:

- BatchNorm or LayerNorm parameters,
- add-on `1x1` convolutions,
- optionally prototype vectors,
- optionally the final linear head.

### Shared-prototype text models

Recommended first choice:

- LayerNorm parameters in the encoder,
- attention biases if they are easy to isolate,
- leave the backbone mostly frozen,
- use head weights only for weighting or masking, not necessarily for adaptation.

## 3. Implement `collect_params` and `configure_model`

Every implementation in this repo uses the same pattern:

1. freeze everything,
2. selectively re-enable gradients for the allowed modules,
3. return both the parameters and their names for debugging.

Minimal pattern:

```python
def collect_params(model, mode="layernorm_only"):
    params, names = [], []
    for name, module in model.named_modules():
        ...
    return params, names

def configure_model(model, mode="layernorm_only"):
    model.train()
    model.requires_grad_(False)
    for p in selected_params:
        p.requires_grad_(True)
    return model
```

Good references:

- [`ProtoViT/proto_entropy.py`](../ProtoViT/proto_entropy.py)
- [`ProtoPFormer/proto_tta.py`](../ProtoPFormer/proto_tta.py)
- [`protosvit/run_inference_cars_c.py`](../protosvit/run_inference_cars_c.py)

## 4. Expose a `forward_no_adapt` Path

Metrics and VLM analysis need a forward path that **does not trigger adaptation**.

Recommended convention:

```python
def forward_no_adapt(self, x):
    return self.model(x)
```

If your wrapper adapts inside `forward`, then `forward_no_adapt` is essential for:

- PAC / PCA evaluation,
- calibration-style metrics,
- clean baseline collection,
- precomputing explainability evidence.

## 5. Implement ProtoTTA

The generic ProtoTTA loop is:

```python
1. run forward pass
2. get pseudo-label from logits
3. build target-prototype mask
4. aggregate sub-prototypes if needed
5. compute reliability mask from filter scores
6. map prototype scores to probabilities in [0, 1]
7. minimize binary entropy on target prototypes
8. optionally weight by prototype importance and prediction confidence
9. take one optimizer step
```

## 6. Mapping Prototype Scores to Probability Space

The binary-entropy loss expects prototype probabilities in `(0, 1)`.

How to map depends on the native score type.

### Cosine similarity in `[-1, 1]`

Used by ProtoViT-style or cosine-based models.

Options:

- linear: `(s + 1) / 2`
- sigmoid: `sigmoid(temp * s)`

### Non-negative similarity or activation in `[0, +inf)`

Used by ProtoS-ViT-style pipelines after ReLU / score heads.

Option:

- `sigmoid(s)`

### Distance-based prototype values

Used by ProtoPNet-style models.

Options:

- convert distance to similarity first,
- or normalize the distance-derived activation into a stable probability range before entropy.

The important point is not the exact formula. The important point is:

- ambiguous prototype states should map near `0.5`,
- confident matches should move toward `1`,
- clear non-matches should move toward `0`.

## 7. Target Prototype Selection

This is the only part that is truly architecture-specific.

### Case 1: explicit class ownership

```python
pred_class = logits.argmax(dim=1)
proto_ids = prototype_class_identity.argmax(dim=1)
target_mask = (proto_ids.unsqueeze(0) == pred_class.unsqueeze(1)).float()
```

### Case 2: shared prototypes with signed or weighted class contributions

```python
pred_class = logits.argmax(dim=1)
class_weights = classifier.weight[pred_class]   # [B, P]
target_mask = (class_weights > 0).float()
```

If your model uses more nuanced prototype routing, keep the same idea: ProtoTTA needs a per-sample binary or soft mask indicating which prototypes should be sharpened.

## 8. Geometric Filtering

ProtoTTA should not adapt on every sample blindly.

Use a reliability score such as:

- maximum prototype similarity,
- top-k mean prototype similarity,
- a dedicated filter score from the model,
- a confidence-aware combination of prototype support and prediction entropy.

Generic pattern:

```python
reliable = (filter_scores.max(dim=1)[0] > threshold).float()
```

If your model has sub-prototypes, aggregate them first with a stable rule such as:

- `max`,
- `mean`,
- `median`,
- `top_k_mean`.

## 9. Prototype Importance Weighting

If your classifier head tells you which prototypes matter most for a class, use it.

Typical pattern:

```python
class_weights = classifier.weight[pred_class]     # [B, P]
importance = abs(class_weights) * target_mask
importance = importance / (importance.sum(dim=1, keepdim=True) + eps)
```

This lets ProtoTTA emphasize prototypes that actually contribute to the pseudo-label, instead of treating every target prototype equally.

## 10. Confidence Weighting

A useful extra stabilizer is:

```python
confidence = logits.softmax(dim=1).max(dim=1)[0]
loss = (loss_per_sample * confidence * reliable).sum() / (reliable.sum() + eps)
```

This helps suppress updates from uncertain pseudo-labels.

## 11. ProtoTTA+

ProtoTTA+ keeps the prototype loss as the main driver but adds a second term, usually output entropy.

Generic form:

```python
loss = proto_weight * prototype_binary_entropy
loss += logit_weight * softmax_entropy(logits)
```

ProtoTTA+ is useful when:

- prototype evidence alone is not enough,
- the model head still carries meaningful uncertainty information,
- you want a hybrid method that interpolates between prototype-aware and output-aware adaptation.

Reference:

- [`protosvit/run_inference_cars_c.py`](../protosvit/run_inference_cars_c.py)
- [`protopnet/proto_entropy_enchanced.py`](../protopnet/proto_entropy_enchanced.py)

## 12. CNN vs ViT Notes

### CNN-style prototype models

Usually easier to adapt:

- prototype vectors,
- add-on conv layers,
- normalization layers,
- last layer.

Filtering is often based on:

- max pooled prototype support,
- distance-derived similarity.

### ViT-style prototype models

Usually safer to adapt:

- LayerNorm,
- attention biases,
- lightweight projection heads,
- optional prototype-side modules.

Filtering is often based on:

- aggregated prototype similarities,
- top-k mean over sub-prototypes,
- explicit prototype support heads.

## 13. Practical Checklist

Before saying ProtoTTA is integrated, confirm:

- your model exposes `logits`,
- your model exposes `prototype_scores`,
- you can define `filter_scores`,
- you can define target prototypes per pseudo-label,
- you implemented `collect_params`,
- you implemented `configure_model`,
- you implemented `forward_no_adapt`,
- you can run one adaptation step without shape ambiguities,
- you can collect clean baseline activations for metrics.

## 14. Starting Template

Use:

- [`templates/prototta_adapter.py`](./templates/prototta_adapter.py)

as the clean starting point for a new model.
