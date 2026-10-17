# Adding ProtoTTA to a New Prototype Model

This guide is written as an implementation checklist. Complete the tensor
contract first; tune thresholds only after the end-to-end update is verified.

## 1. Identify the Model Semantics

Record these facts before editing code:

| Question | Required answer |
|---|---|
| What tensor produces the final class logits? | Shape `[B, C]` |
| What tensor measures prototype evidence? | Shape `[B, P]` or `[B, P, K]` |
| Is the value a cosine similarity, other similarity, distance, or probability? | Determines `score_mapping` |
| Are prototypes class-specific or shared? | Determines target routing |
| Which small parameter subset can adapt safely? | Determines `configure_model` arguments |
| What marks a new independent target domain? | Determines when to call `reset()` |

Do not infer the score type from a variable name. Trace the tensor through the
model and confirm its range on a real batch.

## 2. Expose the Output Contract

The easiest integration is to return a dictionary:

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

If changing `forward` would break existing training code, keep it unchanged
and use an extractor callback:

```python
from ProtoTTA import PrototypeBatch

def extract_output(output):
    logits, min_distances, similarities = output
    return PrototypeBatch(
        logits=logits,
        prototype_scores=min_distances,
        filter_scores=min_distances,
    )
```

The extractor must return tensors connected to the computation graph. Do not
call `.detach()`, convert to NumPy, or run the model inside `torch.no_grad()`.

## 3. Define Prototype-to-Class Routing

### Class-specific prototypes

Provide a one-hot or non-negative identity tensor with shape `[P, C]`:

```python
adapted = ProtoTTAAdapter(
    model,
    optimizer,
    prototype_class_identity=model.prototype_class_identity,
    config=config,
)
```

The adapter predicts a class from the logits and selects the prototypes whose
largest identity entry belongs to that class. This is the usual choice for
ProtoPNet-like medical models.

### Shared prototypes

Provide the frozen classifier weights with shape `[C, P]`:

```python
adapted = ProtoTTAAdapter(
    model,
    optimizer,
    classifier_weights=model.classifier.weight.detach().clone(),
    config=config,
)
```

Positive weights select supporting prototypes. Their absolute values also act
as importance weights. If a predicted class has no positive weights, the
adapter selects the largest-magnitude prototype as a safe fallback.

### Custom routing

For hierarchical, multi-label, or dynamically routed prototypes, return
`target_mask` with shape `[B, P]` from the extractor. Each row must select at
least one prototype. You may also return `importance_weights` with shape
`[B, P]`.

## 4. Map Native Scores to Probabilities

Binary entropy requires probabilities in `(0, 1)`:

```python
ProtoTTAConfig(score_mapping="cosine")      # (score + 1) / 2
ProtoTTAConfig(score_mapping="similarity")  # sigmoid(temp * score)
ProtoTTAConfig(score_mapping="distance")    # ProtoPNet log-inverse distance / 9
ProtoTTAConfig(score_mapping="probability") # already normalized
```

For a model-specific transformation:

```python
def score_mapper(distances):
    similarity = torch.log((distances + 1.0) / (distances + 1e-4))
    return torch.sigmoid(similarity)

adapted = ProtoTTAAdapter(..., score_mapper=score_mapper)
```

Check the mapped distribution. If nearly every value is saturated near zero or
one before adaptation, prototype entropy will provide little useful gradient.

## 5. Aggregate Sub-Prototypes

Scores shaped `[B, P, K]` need a stable reduction over `K`:

- `aggregation="max"`: permissive; one strong sub-prototype is sufficient;
- `aggregation="mean"`: strict consensus;
- `aggregation="topk_mean"`: practical compromise, controlled by
  `aggregation_topk`.

Use the same semantic ordering for `prototype_scores` and `filter_scores`.

## 6. Select the Adaptation Parameters

Begin with normalization affine parameters:

```python
parameters, names = configure_model(model)
optimizer = torch.optim.Adam(parameters, lr=1e-4)
```

The helper freezes every parameter first. Add small architecture-specific
blocks only when needed:

```python
parameters, names = configure_model(
    model,
    extra_parameter_patterns=(
        "*add_on_layers*",
        "*prototype_projection*",
    ),
)
```

Patterns use shell-style matching against `model.named_parameters()`. Always
print and review `names`; a misspelled pattern raises an error if nothing else
is selected.

Recommended starting scopes:

- **Medical CNN / ProtoPNet:** BatchNorm affine parameters and the small add-on
  `1x1` convolution block. Keep running statistics frozen.
- **Vision Transformer:** LayerNorm affine parameters, then optionally attention
  projection biases.
- **Shared-prototype text model:** LayerNorm affine parameters; keep the
  classifier weights frozen and use them only for routing/importance.

Avoid adapting the full encoder on the first implementation. It makes failures
harder to diagnose and increases drift from unreliable pseudo-labels.

## 7. Configure Filtering and Weighting

```python
config = ProtoTTAConfig(
    steps=1,
    use_geometric_filter=True,
    geometric_threshold=0.3,
    use_importance_weighting=True,
    use_confidence_weighting=True,
)
```

The default filter accepts a sample when its maximum mapped filter probability
exceeds the threshold. If the model has a better geometric signal, return it as
`filter_scores` or implement it in the extractor.

Track:

```python
adapted.stats
# seen_samples, adapted_samples, optimizer_steps, last_loss
```

A zero accepted-sample rate usually means the score mapping is wrong or the
threshold is in the wrong scale. Verify both before disabling the filter.

## 8. Pure and Hybrid Objectives

`output_entropy_weight` is the fraction assigned to ordinary output entropy:

```python
ProtoTTAConfig(output_entropy_weight=0.0)  # prototype objective only
ProtoTTAConfig(output_entropy_weight=0.2)  # 80% prototype + 20% output
ProtoTTAConfig(output_entropy_weight=1.0)  # output entropy only
```

Start with the prototype-only objective. Treat the hybrid weight as a model
choice that must be reported, not a value selected using target labels.

## 9. Choose the Reset Protocol

- `episodic=False` performs continual adaptation across batches.
- `episodic=True` restores the source state before each batch.
- `adapted.reset()` manually restores both the model and optimizer.

For medical deployment experiments, reset between independent hospitals,
scanner domains, acquisition protocols, or corruption types unless the study
explicitly evaluates continual domain sequences.

## 10. Verification Tests

Before a full experiment, test all of the following on a tiny batch:

```bash
python -m unittest -v ProtoTTA.tests.test_adapter
python -m ProtoTTA.examples.medical_model
```

For the target model:

1. log the shapes and min/max values of all extracted tensors;
2. assert every target-mask row is non-empty;
3. save parameter tensors before and after one batch;
4. assert only the selected parameter names changed;
5. assert `reset()` restores every changed tensor;
6. test a batch rejected by the filter and confirm no optimizer step occurs;
7. confirm target labels are used only for offline evaluation;
8. compare clean/source performance before and after adding the wrapper.

## 11. Common Failures

### Loss changes but parameters do not

- The selected parameters may not participate in the prototype-score graph.
- The extractor may have detached the prototype tensor.
- The optimizer may have been constructed before parameters were enabled.

### Every batch is rejected

- The filter may be seeing distances while configured as cosine similarities.
- The threshold may have been copied from a different score scale.
- Sub-prototype aggregation may be too strict.

### Accuracy collapses over time

- Reduce the learning rate or adaptation scope.
- Turn on confidence weighting and geometric filtering.
- Reset between domains.
- Confirm that prototype ownership and class indices use the same ordering.

### BatchNorm results are unstable

`configure_model` leaves BatchNorm in evaluation mode and updates only affine
parameters. Do not call `model.train()` afterward unless changing running
statistics is an intentional part of the protocol.

## 12. Handoff Checklist for Another Agent

Give the implementing agent:

- this repository and the exact source-model checkpoint;
- the file and function where logits and prototype evidence are produced;
- the native score type and expected range;
- the prototype-to-class mapping;
- the desired target-domain reset boundary;
- a small label-free target batch for smoke testing;
- the command used for the original model evaluation.

Ask it to modify only the output contract and a thin construction layer first.
The reusable adapter should not need to be copied into or rewritten inside the
new model repository.
