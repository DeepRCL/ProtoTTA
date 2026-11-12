# Adding ProtoTTA to a New Prototype Model

This guide describes how to connect a prototype model to
[`ProtoTTAAdapter`](./adapter.py). Equation numbers refer to the paper.

## 1. Identify the model semantics

| Question | Determines |
|---|---|
| Which tensor holds the class logits? | `logits`, shape `[B, C]` |
| Which tensor holds the prototype responses? | `prototype_scores`, shape `[B, P]` or `[B, P, K]` |
| Is the response a cosine similarity, another similarity, a distance, or a probability? | `score_mapping` or `score_mapper` |
| Are prototypes class-specific or shared across classes? | routing (Section 3) |
| Which normalization and structural parameters exist? | `configure_model` arguments |
| What marks a new target domain? | when to call `reset()` |

## 2. Expose the output contract

Return a dictionary from `forward`:

```python
def forward(self, inputs):
    features = self.encoder(inputs)
    prototype_scores = self.prototype_layer(features)
    logits = self.classifier(prototype_scores)
    return {
        "logits": logits,
        "prototype_scores": prototype_scores,
        "filter_scores": prototype_scores,
    }
```

or keep `forward` unchanged and pass an extractor:

```python
from ProtoTTA import PrototypeBatch

def extract_output(output):
    logits, min_distances, similarities = output
    return PrototypeBatch(logits=logits, prototype_scores=similarities)
```

The extractor must not detach the tensors or run the model under
`torch.no_grad()`. Optional `PrototypeBatch` fields:

| Field | Shape | Purpose |
|---|---:|---|
| `filter_scores` | `[B, P]` or `[B, P, K]` | Native responses for the geometric filter (default: `prototype_scores`) |
| `target_mask` | `[B, P]` | Per-sample pseudo-label prototypes |
| `importance_weights` | `[B, P]` | Per-prototype loss weights before normalization |
| `prototype_targets` | `[B, P]` | Soft targets for the importance-weighted BCE |
| `reliability_mask` | `[B]` | Reliability indicator `m_i` computed by the model |
| `controller_activations` | `[B, P]` | Mapped activations used only by the controller |
| `class_evidence` | `[B, C]` | Class evidence `S_ic` for the relative-evidence coefficient |
| `activation_lambda` | `[B]` | Activation-margin coefficient computed by the model |

## 3. Prototype routing

**Class-specific prototypes.** Pass `prototype_class_identity` with shape
`[P, C]`. The pseudo-label prototypes are those whose identity row has its
maximum at `yhat_i`. To weight the binary entropies by the classifier head
(Eq. 40), also pass the frozen `classifier_weights` with shape `[C, P]`;
without them, or with `use_importance_weighting=False`, the loss is the
uniform mean over the pseudo-label prototypes.

**Shared prototypes.** Pass only `classifier_weights`. The prototype loss is
then the importance-weighted binary cross-entropy over all `P` prototypes with
importance `|W_yhat,p| / (max_r |W_yhat,r| + eps_W)` and soft target
`sigmoid(2 W_yhat,p)` (Eqs. 22-23). The controller uses the prototypes with
positive weight for the pseudo-label (Eq. 25) and, for each class, the
positively weighted prototypes or the three largest weights (Eqs. 26-27).

**Other layouts.** Return `target_mask` (each row non-empty) and, for
adaptive mode, `class_evidence`.

## 4. Map prototype responses to activations

| `score_mapping` | Activation | Paper |
|---|---|---|
| `"cosine"` | `(clip(u, -1, 1) + 1) / 2` | Eq. 29 |
| `"distance"` | `log((d + 1) / (d + 1e-4)) / 9` of the squared distance `d` | Eq. 33 |
| `"similarity"` | `sigmoid(score_temperature * u)` | |
| `"probability"` | `u` | |

Any other transformation is passed as `score_mapper`, a differentiable,
shape-preserving function of the aggregated native responses, for example

```python
score_mapper=lambda u: torch.sigmoid(5.0 * u.clamp(-1.0, 1.0))  # Eq. 35
```

The prototype loss uses the activation clamped to `[eps_H, 1 - eps_H]`. The
controller statistics use the detached activation clamped to `[eps, 1]`.

## 5. Aggregate sub-prototypes

Scores shaped `[B, P, K]` are reduced over `K` before mapping and filtering:
`aggregation="topk_mean"` with `aggregation_topk` (default `2`, Eq. 28),
`"max"`, or `"mean"`. Table 11 of the paper compares the three on ProtoViT.

## 6. Select the adaptation parameters

`configure_model` freezes all parameters, enables the affine parameters of
LayerNorm and BatchNorm modules, and adds parameters whose names match the
glob patterns in `extra_parameter_patterns` (Section 3.3):

```python
# Transformer: LayerNorm affine parameters and attention biases.
parameters, names = configure_model(model, extra_parameter_patterns=("*attn*bias",))

# CNN: BatchNorm affine parameters with current-batch statistics and the
# weights of the add-on 1x1 convolutions.
parameters, names = configure_model(
    model,
    use_batch_statistics=True,
    extra_parameter_patterns=("*add_on_layers*weight",),
)
print(names)  # check the selected parameters
```

`configure_model` puts the model in evaluation mode. With
`use_batch_statistics=True`, the running buffers of the BatchNorm modules are
removed, so they use current-batch statistics in both training and evaluation
mode. The paper optimizes the selected
parameters with Adam, `betas=(0.9, 0.999)` and zero weight decay; the adapter
takes one optimizer step per batch with a non-empty reliable set.

## 7. Reliable set

The reliability indicator `m_i` uses the full prototype set:

- `use_geometric_filter=True` keeps samples whose maximum native response
  exceeds `geometric_threshold`. The native response is the aggregated
  `filter_scores` (default `prototype_scores`) as returned by the model; with
  `score_mapping="distance"` it is the log-inverse-distance response of
  Eq. 33. The threshold is therefore on the scale of the model's own score
  (e.g. raw cosine `0.92` for ProtoViT, Eq. 36).
- `entropy_filter_factor=f` additionally requires `H(p_i) < f ln C`
  (ProtoPNet, Eq. 38). With `use_geometric_filter=False` it is the entropy
  rule used in Table 3 (left).
- Any other rule is computed by the model and returned as
  `reliability_mask`, which replaces both conditions.

The selection rate in percent is
`100 * adapter.stats["adapted_samples"] / adapter.stats["seen_samples"]`.

## 8. Interpolation

Both modes minimize

```
L = 1/(|R| + eps) * sum_{i in R} c_i * ( lambda_i * l_proto_i / (g_proto + eps)
                                        + (1 - lambda_i) * l_out_i / (g_out + eps) )
```

where `g_proto` and `g_out` are the detached gradient norms of the two
confidence-weighted reliable-set components over all enabled parameters
(Eqs. 59-61; `gradient_normalize=False` sets both scales to `1`). Both terms
use the same confidence weight `c_i`.

**Adaptive mode** (`interpolation_mode="adaptive"`) computes, for each
reliable sample:

1. `lambda_act`: the mean distance of the three largest pseudo-label
   activations from `0.5`, divided by `delta0 = 0.25` and clipped to
   `[0, 1]` (Eq. 45);
2. `lambda_rel = r_proto / (r_proto + r_out + eps)` from the normalized margin
   of the Top-3-mean class evidence and the normalized margin of the two
   largest output probabilities (Eqs. 47-51);
3. the consistency `q_proto` and `q_out` of each component's gradients on the
   two interleaved halves of `R` (Eq. 53; `q = 0` with fewer than four reliable
   samples or a zero half-gradient);
4. the batch route `G = [q_proto >= 2 max(q_out, 1e-8) and q_proto >= 0.25]`
   or the coverage guard `[mean r_out over R >= 0.90 and |R|/B <= 0.40]`
   (Eqs. 54-57);
5. `lambda_i = lambda_act` if `G` else `lambda_rel` (Eq. 58).

All controller quantities, the route, and the gradient norms are detached.
The coefficients can reach both `0` and `1`; there is no temporal smoothing.
Setting `use_coverage_guard=False` removes the guard, and
`gradient_normalize=False` removes the gradient normalization (the controller
variants of Table 3, right). An adaptive update uses seven backward passes
(Appendix N).

**Fixed mode** (`interpolation_mode="fixed"`) uses `lambda_i = fixed_lambda`
for every sample.

Multi-branch models may return `activation_lambda` and `class_evidence`;
these replace only the corresponding controller computation.

## Appendix L settings of the four backbones

The adapter settings below correspond to the configurations specified in
Appendix L. The reported results were produced with the backbone
implementations in [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md).

| | ProtoViT | ProtoPFormer | ProtoPNet | ProtoLens |
|---|---|---|---|---|
| `prototype_scores` | sub-prototype cosine similarities `[B, P, 4]` | local (maximum over locations) and global responses `log((d + 1)/(d + 1e-4))`, concatenated | minimum squared distances `[B, P]` | cosine similarities to the 50 shared prototypes |
| Mapping | `score_mapping="cosine"`, Top-2 mean | `score_mapper=lambda u: torch.sigmoid(torch.relu(u) - 1.0)` | `score_mapping="distance"` | `score_mapper=lambda u: torch.sigmoid(5.0 * u.clamp(-1.0, 1.0))` |
| Routing | `prototype_class_identity` and `classifier_weights` | `prototype_class_identity` over the concatenated prototypes; `importance_weights` equal to one over the number of pseudo-label prototypes of the same branch, so each branch has weight 1/2 (Eq. 41) | `prototype_class_identity`, `use_importance_weighting=False` (Eq. 42) | `classifier_weights` only (Eqs. 43-44) |
| Reliable set | `geometric_threshold=0.92` (Eq. 36) | `reliability_mask`: mean of the local and global maximum responses `> 0.55` (Eq. 37) | `geometric_threshold=0.8`, `entropy_filter_factor=0.4` (Eq. 38) | `reliability_mask`: mean of the 25 largest raw cosines `>= 0.1` (Eq. 39) |
| Confidence weighting | on | on | `use_confidence_weighting=False` | `use_confidence_weighting=False` |
| Controller | defaults | `activation_lambda` = mean of the local and global activation-margin coefficients (Eq. 46) | `gradient_epsilon=1e-8`, `controller_output_floor=0.0` | defaults |
| Parameters | LayerNorm, `"*attn*bias"` | LayerNorm, `"*attn*bias"`; model in training mode | BatchNorm with `use_batch_statistics=True`, `"*add_on_layers*weight"` | LayerNorm, attention-projection biases |
| Adam learning rate, batch size | `1e-3`, 128 | `1e-3`, 128 | `1e-3`, 64 | `5e-6`, 128 |

As in Appendix L, call `reset()` between corruption types (for ProtoLens,
between corruption-severity settings).

## 9. Checks

```bash
python -m unittest -v ProtoTTA.tests.test_adapter
python -m ProtoTTA.examples.medical_model
```

On the target model, check on a small batch that:

1. the extracted tensors have the expected shapes and ranges;
2. every sample selects at least one pseudo-label prototype;
3. only the parameters printed by `configure_model` change after a step;
4. a batch rejected by the reliable-set rule makes no optimizer step;
5. `reset()` restores the source predictions;
6. labels are used only for offline evaluation.
