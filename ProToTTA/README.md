# ProToTTA Developer Kit

This folder is a **developer-oriented guide** for adding **ProtoTTA** or **ProtoTTA+** to a new prototype-based model.

The rest of the repository contains paper-specific implementations for:

- ProtoViT
- ProtoPNet-style models
- ProtoLens
- ProtoPFormer
- ProtoS-ViT

This folder distills the **shared implementation pattern** across those backbones so that future users can port ProtoTTA to a new model without reverse-engineering each project separately.

## What Is Here

- [`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md): how to integrate ProtoTTA / ProtoTTA+ into a new architecture.
- [`INTERPRETABILITY_METRICS.md`](./INTERPRETABILITY_METRICS.md): how to implement PAC, PCA, PCA-W, sparsity, calibration, and related metrics.
- [`VLM_EXPLAINABILITY.md`](./VLM_EXPLAINABILITY.md): how to run or adapt the VLM-based explainability pipeline for TTA.
- [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md): links and commands for the current backbone-specific implementations in this repo.
- [`templates/prototta_adapter.py`](./templates/prototta_adapter.py): a reusable implementation skeleton for a new model.

## ProtoTTA in One Sentence

ProtoTTA adapts a prototype model at test time by using **prototype-level signals** instead of only output entropy: it identifies target prototypes from the pseudo-label, filters unreliable samples, and minimizes **binary entropy over prototype activations** so the model restores confident, semantically aligned prototype usage under distribution shift.

## Shared Design Across Backbones

Across ProtoViT, ProtoPNet, ProtoLens, ProtoPFormer, and ProtoS-ViT, the common structure is:

1. expose prototype activations or similarities during inference,
2. choose which parameters are allowed to adapt,
3. build a target-prototype mask from the pseudo-label,
4. filter unreliable samples using prototype support,
5. compute prototype entropy loss,
6. optionally add importance weighting, confidence weighting, or a logit-entropy term,
7. provide evaluation hooks for metrics and explainability.

## Recommended Reading Order

1. Start with [`ADDING_NEW_MODEL.md`](./ADDING_NEW_MODEL.md).
2. If you want to reproduce the analysis section of the paper, then read [`INTERPRETABILITY_METRICS.md`](./INTERPRETABILITY_METRICS.md).
3. If your model is visual and you want explainable TTA narratives, read [`VLM_EXPLAINABILITY.md`](./VLM_EXPLAINABILITY.md).
4. If you simply want to run the existing implementations first, use [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md).

## Source Implementations Used To Derive This Guide

- [`ProtoViT/proto_entropy.py`](../ProtoViT/proto_entropy.py)
- [`ProtoViT/prototype_tta_metrics.py`](../ProtoViT/prototype_tta_metrics.py)
- [`ProtoViT/enhanced_prototype_metrics.py`](../ProtoViT/enhanced_prototype_metrics.py)
- [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py)
- [`ProtoLens/proto_tta.py`](../ProtoLens/proto_tta.py)
- [`ProtoLens/prototype_metrics.py`](../ProtoLens/prototype_metrics.py)
- [`ProtoPFormer/proto_tta.py`](../ProtoPFormer/proto_tta.py)
- [`ProtoPFormer/prototype_tta_metrics.py`](../ProtoPFormer/prototype_tta_metrics.py)
- [`ProtoPFormer/enhanced_prototype_metrics.py`](../ProtoPFormer/enhanced_prototype_metrics.py)
- [`protosvit/run_inference_cars_c.py`](../protosvit/run_inference_cars_c.py)
- [`protosvit/evaluate_robustness_cars_c.py`](../protosvit/evaluate_robustness_cars_c.py)
- [`protopnet/proto_entropy.py`](../protopnet/proto_entropy.py)
- [`protopnet/proto_entropy_enchanced.py`](../protopnet/proto_entropy_enchanced.py)
