# ProtoTTA Developer Kit

This directory is the supported entry point for adding ProtoTTA to another
prototype-based model. The reusable adapter is independent of the four
paper-specific codebases and is intended to be easy for both developers and
coding agents to inspect, test, and extend.

Install the repository into the target model's environment with
`python -m pip install -e /path/to/ProtoTTA` so the package can be imported
without copying files.

## Files

- [`adapter.py`](./adapter.py): tested, model-agnostic implementation.
- [`examples/medical_model.py`](./examples/medical_model.py): runnable
  class-specific medical image example.
- [`templates/prototta_adapter.py`](./templates/prototta_adapter.py): short
  copy-and-edit integration template.
- [`tests/test_adapter.py`](./tests/test_adapter.py): contract and update tests.
- [`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md): detailed integration guide.
- [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md): paper implementation
  commands and pointers.
- [`INTERPRETABILITY_METRICS.md`](./INTERPRETABILITY_METRICS.md): optional
  prototype metrics.
- [`VLM_EXPLAINABILITY.md`](./VLM_EXPLAINABILITY.md): optional VLM analysis.

## Required Model Contract

The default adapter expects the wrapped model to return:

```python
{
    "logits": logits,                       # [B, C]
    "prototype_scores": prototype_scores,   # [B, P] or [B, P, K]
    "filter_scores": filter_scores,          # optional
}
```

The adapter also needs one prototype-routing mechanism:

- `prototype_class_identity` with shape `[P, C]` for class-specific
  prototypes;
- `classifier_weights` with shape `[C, P]` for shared prototypes; or
- a per-batch `target_mask` with shape `[B, P]` in the model output.

## Minimal Integration

```python
import torch

from ProtoTTA import ProtoTTAAdapter, ProtoTTAConfig, configure_model

# Freeze the model and explicitly select the parameters that may change.
parameters, parameter_names = configure_model(
    model,
    adapt_layer_norm=True,
    adapt_batch_norm=True,
    extra_parameter_patterns=("*add_on_layers*",),
)
optimizer = torch.optim.Adam(parameters, lr=1e-4)

model = ProtoTTAAdapter(
    model,
    optimizer,
    prototype_class_identity=model.prototype_class_identity,
    config=ProtoTTAConfig(
        score_mapping="cosine",
        aggregation="topk_mean",
        aggregation_topk=1,
        use_geometric_filter=True,
        geometric_threshold=0.3,
        use_importance_weighting=True,
        use_confidence_weighting=True,
        output_entropy_weight=0.0,
    ),
)

for inputs, _ in target_loader:
    output = model(inputs)
    prediction = output["logits"].argmax(dim=1)
```

No target label is passed to the adapter. `model.reset()` restores the source
model and optimizer state before evaluation on a new target domain.

## Custom Model Outputs

Tuple outputs do not require changing the source model. Provide an extractor:

```python
from ProtoTTA import PrototypeBatch

def extract_output(output):
    logits, distances, extra = output
    return PrototypeBatch(
        logits=logits,
        prototype_scores=distances,
        filter_scores=distances,
    )

adapted_model = ProtoTTAAdapter(
    model,
    optimizer,
    prototype_class_identity=identity,
    output_extractor=extract_output,
    config=ProtoTTAConfig(score_mapping="distance"),
)
```

If a model uses a special score transform, pass `score_mapper`. It must map a
score tensor to probabilities in `[0, 1]` without detaching gradients.

## Safe Starting Settings

For a first port:

```python
ProtoTTAConfig(
    steps=1,
    episodic=False,
    score_mapping="cosine",       # change to match the model
    aggregation="topk_mean",
    aggregation_topk=1,
    use_geometric_filter=True,
    geometric_threshold=0.3,
    use_confidence_weighting=True,
    use_importance_weighting=True,
    adapt_all_prototypes=False,
    output_entropy_weight=0.0,
)
```

Keep the adaptation scope small, inspect the returned parameter names, and
measure the accepted-sample ratio:

```python
ratio = model.stats["adapted_samples"] / max(model.stats["seen_samples"], 1)
```

If the ratio is near zero, verify the score mapping before lowering the
geometric threshold.

## Validate an Integration

Run the package tests and example first:

```bash
python -m unittest -v ProtoTTA.tests.test_adapter
python -m ProtoTTA.examples.medical_model
```

Then verify on the target model that:

1. logits have shape `[B, C]` and scores have shape `[B, P]` or `[B, P, K]`;
2. every sample routes to at least one target prototype;
3. only the printed adaptation parameters change;
4. no labels reach `ProtoTTAAdapter.forward`;
5. `reset()` exactly restores source predictions;
6. the source checkpoint is reset between independent target domains.

Continue with [`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md) for architecture
choices, common failure modes, and a handoff checklist for another agent.
