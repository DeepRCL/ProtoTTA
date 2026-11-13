# ProtoTTA Integration Kit

This directory contains a model-agnostic implementation of ProtoTTA as
specified in the paper: the prototype objective (Section 3.1), the adaptive
interpolation between prototype and output entropy (Section 3.2), the update
of normalization and structural parameters (Section 3.3), Algorithm 1
(Appendix G), the shared-prototype objective (Appendix H), and the controller
specification (Appendix L). It is intended for applying ProtoTTA to prototype
models other than the four evaluated backbones.

The results reported in the paper were produced with the backbone-specific
implementations listed in [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md).

## Files

- [`adapter.py`](./adapter.py): `ProtoTTAAdapter`, `ProtoTTAConfig`,
  `PrototypeBatch`, and `configure_model`.
- [`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md): integration guide, including
  the settings that correspond to each backbone configuration in Appendix L.
- [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md): entry points of the four
  backbone implementations.
- [`INTERPRETABILITY_METRICS.md`](./INTERPRETABILITY_METRICS.md): the
  interpretability and operational metrics reported in the paper.
- [`templates/prototta_adapter.py`](./templates/prototta_adapter.py): short
  construction template.
- [`examples/medical_model.py`](./examples/medical_model.py): runnable toy
  example with class-specific prototypes.
- [`tests/test_adapter.py`](./tests/test_adapter.py): unit tests.

The package is imported as `ProtoTTA`. Run the commands below from the
repository root, or add the repository root to `PYTHONPATH`.

## What one update does

For every incoming batch, `ProtoTTAAdapter`:

1. runs the model once and takes the pseudo-label `argmax_c p_ic`;
2. aggregates sub-prototype scores (Top-K mean), maps them to `(0, 1)` and
   clamps them to `[eps_H, 1 - eps_H]`;
3. selects the reliable set `R` with the geometric statistic (and optionally
   an output-entropy condition); if `R` is empty, no update is made;
4. computes the per-sample prototype loss (weighted binary entropy of the
   pseudo-label prototypes, or importance-weighted BCE for shared
   prototypes) and the output entropy `H(p_i)`;
5. weights both losses with the same detached confidence `c_i = max_c p_ic`
   (or `c_i = 1` when confidence weighting is off);
6. normalizes each component by its detached gradient norm over the enabled
   parameters;
7. sets `lambda_i` either to a fixed value or, in adaptive mode, to the
   activation-margin coefficient or the relative-evidence coefficient, chosen
   per batch by the half-batch gradient-consistency router and the coverage
   guard; `lambda_i` can take any value in `[0, 1]`;
8. takes one optimizer step on the unified objective (Eq. 3) and returns the
   output of the forward pass.

No target labels are used. Updates accumulate across calls;
`reset()` restores the source model and optimizer state, for example between
target domains.

## Model contract

The default extractor expects the wrapped model to return:

```python
{
    "logits": logits,                       # [B, C]
    "prototype_scores": prototype_scores,   # [B, P] or [B, P, K]
    "filter_scores": filter_scores,         # optional, defaults to prototype_scores
}
```

Tuple outputs are supported through an `output_extractor` that returns a
`PrototypeBatch`. Scores must stay connected to the computation graph.

The adapter also needs one prototype-routing mechanism:

- `prototype_class_identity` with shape `[P, C]` for class-specific
  prototypes (optionally with `classifier_weights` `[C, P]` for importance
  weights, as in ProtoViT);
- `classifier_weights` alone for shared prototypes, which selects the
  shared-prototype objective of Appendix H; or
- a per-batch `target_mask` `[B, P]` in the model output (adaptive mode then
  also needs `class_evidence` `[B, C]`).

## Minimal integration

```python
import torch

from ProtoTTA import ProtoTTAAdapter, ProtoTTAConfig, configure_model

# Freeze the model; enable normalization affine parameters and, for a
# Transformer, the attention biases.
parameters, names = configure_model(
    model, extra_parameter_patterns=("*attn*bias",)
)
optimizer = torch.optim.Adam(
    parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0
)

adapted = ProtoTTAAdapter(
    model,
    optimizer,
    prototype_class_identity=model.prototype_class_identity,
    classifier_weights=model.last_layer.weight.detach().clone(),
    config=ProtoTTAConfig(),  # ProtoViT configuration of Appendix L
)

for inputs, _ in target_loader:  # labels are not passed to the adapter
    logits = adapted(inputs)["logits"]
```

The fixed-lambda variant uses the same objective with one `lambda` for every
sample (`1` is pure prototype entropy, `0` is pure output entropy):

```python
ProtoTTAConfig(interpolation_mode="fixed", fixed_lambda=1.0)
```

## Defaults

`ProtoTTAConfig()` is the ProtoViT configuration of Appendix L:

| Setting | Default | Paper |
|---|---|---|
| `score_mapping` | `"cosine"`: `(clip(u, -1, 1) + 1) / 2` | Eq. 29 |
| `aggregation`, `aggregation_topk` | `"topk_mean"`, `2` | Eq. 28 |
| `geometric_threshold` | `0.92` on the maximum native score | Eq. 36 |
| `use_confidence_weighting` | `True`, shared by both loss terms | Eq. 3, Appendix L |
| `use_importance_weighting` | `True` (normalized classifier weights) | Eq. 40 |
| `controller_topk`, `controller_delta0` | `3`, `0.25` | Eqs. 4, 45 |
| `controller_consistency_ratio`, `controller_min_consistency` | `2.0`, `0.25` | Eq. 54 |
| `controller_output_floor` | `1e-8` | Eq. 54 |
| `controller_min_reliable` | `4` | Eq. 53 |
| `coverage_output_margin`, `coverage_max_fraction` | `0.90`, `0.40` | Eq. 56 |
| `gradient_normalize` | `True` | Eqs. 2, 59-61 |
| `gradient_epsilon` (`eps_q`) | `1e-12` | Eq. 53 |
| `entropy_epsilon` (`eps_H`), `importance_epsilon` (`eps_W`), `eps` | `1e-6`, `1e-6`, `1e-8` | Appendix L |

The settings for ProtoPFormer, ProtoPNet, and ProtoLens are listed in
[`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md#appendix-l-settings-of-the-four-backbones).

## Tests and example

```bash
python -m unittest -v ProtoTTA.tests.test_adapter
python -m ProtoTTA.examples.medical_model
```
