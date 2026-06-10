# ProtoTTA: Prototype-Guided Test-Time Adaptation

[![OpenReview](https://img.shields.io/badge/Paper-PDF-B31B1B.svg)](https://iclr.cc/virtual/2026/10020241)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2604.15494)
[![Poster](https://img.shields.io/badge/Poster-PDF-4c1.svg)](https://github.com/DeepRCL/ProtoTTA/blob/main/Files/ProtoTTA-Poster-v3.pdf)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Official code repository for **ProtoTTA: Prototype-Guided Test-Time Adaptation**.

This repository packages our ProtoTTA implementations across multiple prototype-based backbones, together with robustness evaluation scripts, prototype-level interpretability metrics, and a vision-language model (VLM) analysis pipeline for studying adaptation dynamics.
<img width="100%" alt="VLM2 (1)" src="https://github.com/user-attachments/assets/498165fc-dcfb-454b-86fb-917b30aec308" />

## Start Here
<img width="100%" src="https://github.com/user-attachments/assets/95239449-27fa-4209-816f-e9f8bc5a3d1f">

This repository supports two main use cases.

### Add ProtoTTA or ProtoTTA+ to a new model

Start with the developer kit in [`ProToTTA/README.md`](./ProToTTA/README.md).

- Model integration guide: [`ProToTTA/ADDING_NEW_MODEL.md`](./ProToTTA/ADDING_NEW_MODEL.md)
- Generic implementation skeleton: [`ProToTTA/templates/prototta_adapter.py`](./ProToTTA/templates/prototta_adapter.py)
- Interpretability metrics guide: [`ProToTTA/INTERPRETABILITY_METRICS.md`](./ProToTTA/INTERPRETABILITY_METRICS.md)
- VLM explainability guide: [`ProToTTA/VLM_EXPLAINABILITY.md`](./ProToTTA/VLM_EXPLAINABILITY.md)



### Run the existing paper implementations

Use the backbone-specific command guide in [`ProToTTA/EXISTING_BACKBONES.md`](./ProToTTA/EXISTING_BACKBONES.md).

## Abstract

Deep networks that rely on prototypes, interpretable representations that can be related to the model input, have gained significant attention for balancing high accuracy with inherent interpretability, which makes them suitable for critical domains such as healthcare. However, these models are limited by their reliance on training data, which hampers their robustness to distribution shifts. While test-time adaptation (TTA) improves the robustness of deep networks by updating parameters and statistics, the prototypes of interpretable models have not been explored for this purpose. We introduce **ProtoTTA**, a general framework for prototypical models that leverages intermediate prototype signals rather than relying solely on model outputs. ProtoTTA minimizes the entropy of the prototype-similarity distribution to encourage more confident and prototype-specific activations on shifted data. To maintain stability, we employ geometric filtering to restrict updates to samples with reliable prototype activations, regularized by prototype-importance weights and model-confidence scores. Experiments across diverse prototypical backbones and benchmarks spanning fine-grained vision, histopathology, and NLP demonstrate that ProtoTTA improves robustness over standard output entropy minimization while restoring correct semantic focus in prototype activations. We also introduce novel interpretability metrics and a VLM-based evaluation framework to explain TTA dynamics.

## Repository Scope

This public release focuses on the ProtoTTA code paths that are needed to reproduce and extend the paper:

- [`ProToTTA/`](./ProToTTA): developer-facing guide for adding ProtoTTA / ProtoTTA+ to a new model, implementing metrics, and extending VLM explainability.
- [`ProtoViT/`](./ProtoViT): ProtoViT-based image classification experiments on CUB-200-C, including the main `run_inference.py` and `evaluate_robustness.py` workflows, the prototype metrics, and the VLM evaluation code.
- [`ProtoLens/`](./ProtoLens): ProtoLens-based NLP robustness experiments on Amazon-C.
- [`ProtoPFormer/`](./ProtoPFormer): ProtoPFormer-based image robustness experiments on Stanford Dogs-C.
- [`protosvit/`](./protosvit): ProtoS-ViT-based robustness experiments on Stanford Cars-C, including `ProtoTTA+`.
- [`protopnet/`](./protopnet): ProtoPNet-style prototype adaptation modules, including the base ProtoTTA objective and an enhanced variant.
- [`protovit_env.yml`](./protovit_env.yml): main environment file for the ProtoViT stack.
- [`vlm_environment.yml`](./vlm_environment.yml): environment file for the VLM evaluation pipeline.
- [`flsh_att_environment.yml`](./flsh_att_environment.yml): optional environment for Flash Attention based VLM runs.

## Core Contributions

1. **Prototype-aware test-time adaptation** that operates on intermediate prototype signals instead of only output entropy.
2. **Geometric filtering and stability weighting** to make online adaptation more reliable under heavy corruption.
3. **Cross-backbone implementations** spanning ProtoViT, ProtoPNet, ProtoLens, ProtoPFormer, and ProtoS-ViT.
4. **Interpretability-aware evaluation** through prototype consistency/alignment metrics and VLM-based reasoning analysis.

## Main Entry Points

- [`ProToTTA/README.md`](./ProToTTA/README.md): primary developer documentation for future users extending ProtoTTA.
- [`ProToTTA/ADDING_NEW_MODEL.md`](./ProToTTA/ADDING_NEW_MODEL.md): how to add ProtoTTA / ProtoTTA+ to a new architecture.
- [`ProToTTA/INTERPRETABILITY_METRICS.md`](./ProToTTA/INTERPRETABILITY_METRICS.md): how to implement PAC, PCA, PCA-W, sparsity, calibration, and related metrics.
- [`ProToTTA/VLM_EXPLAINABILITY.md`](./ProToTTA/VLM_EXPLAINABILITY.md): how to adapt the VLM-based explainability pipeline for a new model.
- [`ProToTTA/EXISTING_BACKBONES.md`](./ProToTTA/EXISTING_BACKBONES.md): commands and pointers for ProtoViT, ProtoLens, ProtoPFormer, ProtoS-ViT, and ProtoPNet-style code paths already in this repo.

## Repository Layout

```text
.
├── README.md
├── ProToTTA/
│   ├── README.md
│   ├── ADDING_NEW_MODEL.md
│   ├── INTERPRETABILITY_METRICS.md
│   ├── VLM_EXPLAINABILITY.md
│   ├── EXISTING_BACKBONES.md
│   └── templates/prototta_adapter.py
├── ProtoViT/
│   ├── run_inference.py
│   ├── evaluate_robustness.py
│   ├── prototype_tta_metrics.py
│   ├── enhanced_prototype_metrics.py
│   ├── vlm_eval.py
│   └── VLM/vlm.py
├── ProtoLens/
│   ├── run_inference_amazon_c.py
│   ├── evaluate_robustness_amazonc.py
│   ├── proto_tta.py
│   └── prototype_metrics.py
├── ProtoPFormer/
│   ├── run_inference_dogs.py
│   ├── evaluate_robustness_dogs.py
│   ├── proto_tta.py
│   ├── prototype_tta_metrics.py
│   └── enhanced_prototype_metrics.py
├── protosvit/
│   ├── run_inference_cars_c.py
│   └── evaluate_robustness_cars_c.py
└── protopnet/
    ├── proto_entropy.py
    └── proto_entropy_enchanced.py
```

## Installation

Because this repository combines several prototype-model codebases, dependencies are partially backbone-specific. A practical starting point is:

```bash
conda env create -f protovit_env.yml
conda activate protovit
```

For the VLM analysis pipeline:

```bash
conda env create -f vlm_environment.yml
conda activate echofar
```

If you plan to run the large Qwen-based VLM setup in [`ProtoViT/VLM/vlm.py`](./ProtoViT/VLM/vlm.py), the Flash Attention environment in [`flsh_att_environment.yml`](./flsh_att_environment.yml) is also provided.

## ProtoTTA and ProtoTTA+

**ProtoTTA** is the main method described in the paper. Across the provided backbones, it consistently follows the same high-level recipe:

1. obtain prototype activations or similarities,
2. select reliable samples using geometric filtering,
3. identify target prototypes from the pseudo-label,
4. minimize prototype-level binary entropy,
5. optionally weight updates by prototype importance and prediction confidence.

Key implementation files:

- [`ProtoViT/proto_entropy.py`](./ProtoViT/proto_entropy.py)
- [`ProtoLens/proto_tta.py`](./ProtoLens/proto_tta.py)
- [`ProtoPFormer/proto_tta.py`](./ProtoPFormer/proto_tta.py)
- [`protosvit/run_inference_cars_c.py`](./protosvit/run_inference_cars_c.py)
- [`protopnet/proto_entropy.py`](./protopnet/proto_entropy.py)

**ProtoTTA+** is the enhanced variant included in this release for settings where a hybrid loss is useful. In the current codebase, the clearest public entry points are:

- [`protosvit/run_inference_cars_c.py`](./protosvit/run_inference_cars_c.py): `proto_tta_plus`, which blends prototype entropy with logit entropy.
- [`protosvit/evaluate_robustness_cars_c.py`](./protosvit/evaluate_robustness_cars_c.py): pre-defined `ProtoTTA+` method configurations.
- [`protopnet/proto_entropy_enchanced.py`](./protopnet/proto_entropy_enchanced.py): enhanced ProtoPNet-oriented variant with hybrid loss and extra stability controls.

For a generic implementation guide for new models, start with [`ProToTTA/README.md`](./ProToTTA/README.md), then use [`ProToTTA/ADDING_NEW_MODEL.md`](./ProToTTA/ADDING_NEW_MODEL.md).

## Developer Documentation

The top-level README is intentionally high-level. The detailed developer-facing documentation lives in the `ProToTTA` folder:

- [`ProToTTA/README.md`](./ProToTTA/README.md): overview of the developer kit.
- [`ProToTTA/ADDING_NEW_MODEL.md`](./ProToTTA/ADDING_NEW_MODEL.md): integration workflow for a new model.
- [`ProToTTA/INTERPRETABILITY_METRICS.md`](./ProToTTA/INTERPRETABILITY_METRICS.md): PAC, PCA, PCA-W, sparsity, calibration, and adaptation-process metrics.
- [`ProToTTA/VLM_EXPLAINABILITY.md`](./ProToTTA/VLM_EXPLAINABILITY.md): explainable TTA with VLM evidence and scoring.
- [`ProToTTA/EXISTING_BACKBONES.md`](./ProToTTA/EXISTING_BACKBONES.md): current backbone-specific run commands and file pointers.

## Metrics and Explainability

The repository includes dedicated code for evaluating not just accuracy, but also how adaptation changes prototype behavior:

- [`ProtoViT/prototype_tta_metrics.py`](./ProtoViT/prototype_tta_metrics.py): PAC, PCA, and sparsity-oriented metrics.
- [`ProtoViT/enhanced_prototype_metrics.py`](./ProtoViT/enhanced_prototype_metrics.py): weighted PCA, calibration agreement, class-contribution change, and related diagnostics.
- [`ProtoViT/efficiency_metrics.py`](./ProtoViT/efficiency_metrics.py): runtime and adaptation-efficiency tracking.
- [`ProtoLens/prototype_metrics.py`](./ProtoLens/prototype_metrics.py): NLP-side prototype metrics.
- [`ProtoPFormer/prototype_tta_metrics.py`](./ProtoPFormer/prototype_tta_metrics.py): prototype metrics for ProtoPFormer.
- [`ProtoPFormer/enhanced_prototype_metrics.py`](./ProtoPFormer/enhanced_prototype_metrics.py): extended interpretability metrics for ProtoPFormer.
- [`protosvit/evaluate_robustness_cars_c.py`](./protosvit/evaluate_robustness_cars_c.py): table-ready metrics for ProtoS-ViT, including PAC, weighted alignment, prediction stability, and selection rate.

For the generic metric design and implementation guidance, see [`ProToTTA/INTERPRETABILITY_METRICS.md`](./ProToTTA/INTERPRETABILITY_METRICS.md).

## VLM Evaluation

The VLM analysis pipeline is an important part of this release and is kept in the ProtoViT stack:

- [`ProtoViT/vlm_eval.py`](./ProtoViT/vlm_eval.py): main VLM-based evaluation script for qualitative and quantitative reasoning analysis.
- [`ProtoViT/summarize_vlm_eval.py`](./ProtoViT/summarize_vlm_eval.py): result summarization utilities.
- [`ProtoViT/VLM/vlm.py`](./ProtoViT/VLM/vlm.py): low-level VLM loading example.

This part of the repository is designed to study whether ProtoTTA restores human-aligned semantic focus under distribution shift, and how the proposed prototype metrics correlate with VLM-rated reasoning quality.

For the generic developer guide for explainable TTA, see [`ProToTTA/VLM_EXPLAINABILITY.md`](./ProToTTA/VLM_EXPLAINABILITY.md).

## Notes Before Pushing

- This release intentionally centers the five ProtoTTA-related code paths above and ignores local experiment artifacts, datasets, checkpoints, and unrelated baseline/vendor folders.
- If these subprojects were originally cloned as separate repositories, remove any embedded `.git` directories inside kept folders before making the first top-level commit; otherwise Git may treat them as nested repositories instead of regular source directories.

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@inproceedings{abootorabiprototta,
  title={ProtoTTA: Prototype-Guided Test-Time Adaptation},
  author={Abootorabi, Mohammad Mahdi and Mousavi, Parvin and Abolmaesumi, Purang and Shelhamer, Evan},
  booktitle={Third Workshop on Test-Time Updates (Main Track)}
}
```

## Contact

For questions, please contact `mahdi.abootorabi2@gmail.com`.
