# ProtoTTA: Prototype-Guided Test-Time Adaptation

[![OpenReview](https://img.shields.io/badge/Paper-OpenReview-B31B1B.svg)](https://iclr.cc/virtual/2026/10020241)
[![arXiv](https://img.shields.io/badge/arXiv-2604.15494-b31b1b.svg)](https://arxiv.org/abs/2604.15494)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **ProtoTTA: Prototype-Guided Test-Time
Adaptation**. ProtoTTA adapts a prototype-based model on unlabeled test
batches by sharpening reliable prototype evidence while updating only a small,
explicitly selected part of the model.

This repository has two entry paths:

- **Add ProtoTTA to your model:** use the small reusable package in
  [`ProtoTTA/`](./ProtoTTA).
- **Reproduce an existing experiment:** use the backbone-specific code for
  ProtoViT, ProtoPNet, ProtoPFormer, or ProtoLens.

## Add ProtoTTA to a New Model

The reusable adapter has no dependency on the paper datasets or model
repositories. It requires PyTorch and a model that exposes two tensors:

```python
{
    "logits": logits,                       # [batch, classes]
    "prototype_scores": prototype_scores,   # [batch, prototypes]
}
```

Sub-prototypes with shape `[batch, prototypes, sub_prototypes]` are supported.
An optional `filter_scores` tensor can provide a separate geometric reliability
signal.

### 1. Select the parameters that may adapt

```python
from ProtoTTA import configure_model

parameters, names = configure_model(
    model,
    adapt_layer_norm=True,
    adapt_batch_norm=True,
    use_batch_statistics=False,
    # Optional architecture-specific additions:
    extra_parameter_patterns=("*add_on_layers*",),
)
optimizer = torch.optim.Adam(parameters, lr=1e-4)
print("Adapting:", names)
```

All other parameters are frozen. BatchNorm running statistics remain frozen by
default. ProtoPNet-style models can set `use_batch_statistics=True` to use the
current batch without accumulating running statistics, matching the paper.

### 2. Wrap the model

For class-specific prototypes, provide an identity matrix of shape
`[prototypes, classes]`. If available, also pass the frozen classifier matrix
to reproduce classifier-importance weighting:

```python
from ProtoTTA import ProtoTTAAdapter, ProtoTTAConfig

adapted_model = ProtoTTAAdapter(
    model,
    optimizer,
    prototype_class_identity=model.prototype_class_identity,
    classifier_weights=model.classifier.weight.detach(),  # optional [C, P]
    config=ProtoTTAConfig(
        score_mapping="cosine",
        geometric_threshold=0.3,
        interpolation_mode="adaptive",
        gradient_normalize=True,
    ),
)
```

For shared prototypes, omit `prototype_class_identity` and pass the frozen
classifier matrix with shape `[classes, prototypes]` as `classifier_weights`.
This automatically enables the paper's all-prototype, classifier-derived soft
target BCE objective. For tuple/custom model outputs, pass an
`output_extractor` callback.

### 3. Run online adaptation

```python
for images, _ in test_loader:
    output = adapted_model(images)  # predicts, then performs one update
    predictions = output["logits"].argmax(dim=1)

# Restore the source model before a new domain/corruption.
adapted_model.reset()
```

Start from the complete runnable medical example:

```bash
python -m ProtoTTA.examples.medical_model
python -m unittest -v ProtoTTA.tests.test_adapter
```

- Runnable example: [`ProtoTTA/examples/medical_model.py`](./ProtoTTA/examples/medical_model.py)
- API and quick start: [`ProtoTTA/README.md`](./ProtoTTA/README.md)
- Detailed porting guide: [`ProtoTTA/ADDING_NEW_MODEL.md`](./ProtoTTA/ADDING_NEW_MODEL.md)
- Copy-friendly template: [`ProtoTTA/templates/prototta_adapter.py`](./ProtoTTA/templates/prototta_adapter.py)

## Choosing the Score Mapping

ProtoTTA computes binary entropy in probability space. Set `score_mapping` to
match the tensor returned by your model:

| Native prototype value | Configuration | Mapping |
|---|---|---|
| Cosine similarity in `[-1, 1]` | `"cosine"` | `(score + 1) / 2` |
| Unbounded similarity/logit | `"similarity"` | `sigmoid(temperature * score)` |
| ProtoPNet-style squared distance | `"distance"` | `log((d+1)/(d+1e-4)) / 9` |
| Already a probability | `"probability"` | Clamp to `(eps, 1-eps)` |

If your model uses a different activation, provide `score_mapper=lambda scores:
...` to `ProtoTTAAdapter`.

## Medical-Model Recommendations

For a new medical prototype model, begin conservatively:

- adapt normalization affine parameters and, if needed, the small prototype
  projection/add-on block;
- keep the encoder, classifier, and prototype identities frozen initially;
- use one optimizer step per batch and a low learning rate (`1e-5` to `1e-4`);
- reset to the source checkpoint between independent hospitals, scanners, or
  corruption domains;
- validate the geometric threshold using source/validation data, never test
  labels;
- report the accepted-sample rate from `adapted_model.stats` alongside
  accuracy or AUROC.

For distance-based ProtoPNet-style medical models, use
`score_mapping="distance"`. The SICAPv2 integration under
[`ProtoPNetVLM/protopnet_tta/`](./ProtoPNetVLM/protopnet_tta) is the closest
full medical reference.

## Method Summary

For each unlabeled test batch, ProtoTTA:

1. obtains the model pseudo-label and prototype evidence;
2. selects prototypes associated with the pseudo-label;
3. rejects samples without sufficient prototype support;
4. computes prototype entropy (or the shared-prototype soft-target loss) and
   output entropy;
5. weights them by prototype importance and, where configured, prediction
   confidence;
6. interpolates their gradient-normalized contributions with a fixed scalar or
   the adaptive per-sample controller;
7. updates only the parameters selected by `configure_model`.

The submitted method uses `interpolation_mode="adaptive"`. It computes a
sample-wise prototype coefficient from activation margin and relative class
evidence, routes between them using half-batch gradient consistency and a
coverage guard, and normalizes both loss components by their detached gradient
norms.

The fixed-scalar ablation is also supported:

```python
ProtoTTAConfig(
    interpolation_mode="fixed",
    fixed_lambda=0.7,       # 70% prototype, 30% output entropy
    gradient_normalize=True,
)
```

`fixed_lambda` follows the paper: `1` is pure prototype entropy and `0` is
pure output entropy.

The reusable defaults are intentionally architecture-neutral. Exact
paper-reproduction settings differ by backbone, including activation mapping,
filtering, confidence weighting, aggregation, and adaptation scope. Use
`reliability_mask`, `controller_activations`, `class_evidence`, or
`activation_lambda` in a custom output extractor when a backbone's rule cannot
be represented by the generic settings. The existing backbone directories
remain the reference implementations for the reported experiments.

## Paper Implementations

The paper evaluates four prototype-based backbones:

| Backbone | Benchmark | Main directory |
|---|---|---|
| ProtoViT | CUB-200-C | [`ProtoViT/`](./ProtoViT) |
| ProtoPNet | SICAPv2-C | [`ProtoPNetVLM/protopnet_tta/`](./ProtoPNetVLM/protopnet_tta) |
| ProtoPFormer | Stanford Dogs-C | [`ProtoPFormer/`](./ProtoPFormer) |
| ProtoLens | Amazon-C | [`ProtoLens/`](./ProtoLens) |

Commands and implementation pointers are collected in
[`ProtoTTA/EXISTING_BACKBONES.md`](./ProtoTTA/EXISTING_BACKBONES.md).

## Evaluation and Explainability

- [`ProtoTTA/INTERPRETABILITY_METRICS.md`](./ProtoTTA/INTERPRETABILITY_METRICS.md)
  explains PAC, PCA/PCA-W, sparsity, calibration, and prediction stability.
- [`ProtoTTA/VLM_EXPLAINABILITY.md`](./ProtoTTA/VLM_EXPLAINABILITY.md)
  describes the optional VLM evaluation workflow.

## Installation

For the reusable adapter, an existing PyTorch environment is sufficient:

```bash
git clone git@github.com:DeepRCL/ProtoTTA.git
cd ProtoTTA
python -m pip install -e .
python -m unittest -v ProtoTTA.tests.test_adapter
```

To use the adapter from another repository, install this checkout in that
repository's environment with `python -m pip install -e /path/to/ProtoTTA`.

The paper backbones have different upstream dependencies. Use
[`protovit_env.yml`](./protovit_env.yml) as the starting environment and read
the relevant backbone README before running its evaluation script.

## Citation

```bibtex
@inproceedings{abootorabiprototta,
  title={ProtoTTA: Prototype-Guided Test-Time Adaptation},
  author={Abootorabi, Mohammad Mahdi and Mousavi, Parvin and
          Abolmaesumi, Purang and Shelhamer, Evan},
  booktitle={Third Workshop on Test-Time Updates (Main Track)}
}
```

## Contact

For questions, open a GitHub issue or contact `mahdi.abootorabi2@gmail.com`.
