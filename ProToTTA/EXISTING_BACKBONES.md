# Existing Backbones and Run Commands

This page is intentionally separate from the generic developer guide. The files below are the **current paper-specific implementations** in this repository.

## ProtoViT

Main files:

- [`ProtoViT/run_inference.py`](../ProtoViT/run_inference.py)
- [`ProtoViT/evaluate_robustness.py`](../ProtoViT/evaluate_robustness.py)
- [`ProtoViT/proto_entropy.py`](../ProtoViT/proto_entropy.py)

Example:

```bash
cd ProtoViT
python run_inference.py \
  -mode proto_importance_confidence \
  -corruption gaussian_noise \
  -severity 5 \
  --use-geometric-filter \
  --geo-filter-threshold 0.92 \
  --consensus-strategy top_k_mean \
  --adaptation-mode layernorm_attn_bias
```

## ProtoLens

Main files:

- [`ProtoLens/run_inference_amazon_c.py`](../ProtoLens/run_inference_amazon_c.py)
- [`ProtoLens/evaluate_robustness_amazonc.py`](../ProtoLens/evaluate_robustness_amazonc.py)
- [`ProtoLens/proto_tta.py`](../ProtoLens/proto_tta.py)

Example:

```bash
cd ProtoLens
python run_inference_amazon_c.py \
  --corruption_type aggressive \
  --severity 80 \
  --methods prototta \
  --geo_filter \
  --geo_threshold 0.1 \
  --sigmoid_temperature 5.0
```

## ProtoPFormer

Main files:

- [`ProtoPFormer/run_inference_dogs.py`](../ProtoPFormer/run_inference_dogs.py)
- [`ProtoPFormer/evaluate_robustness_dogs.py`](../ProtoPFormer/evaluate_robustness_dogs.py)
- [`ProtoPFormer/proto_tta.py`](../ProtoPFormer/proto_tta.py)

Example:

```bash
cd ProtoPFormer
python run_inference_dogs.py \
  --model /path/to/epoch-best.pth \
  --modes proto_tta \
  --clean_dir /path/to/stanford_dogs \
  --data_dir /path/to/stanford_dogs_c \
  --corruption fog \
  --severity 5
```

## ProtoS-ViT

Main files:

- [`protosvit/run_inference_cars_c.py`](../protosvit/run_inference_cars_c.py)
- [`protosvit/evaluate_robustness_cars_c.py`](../protosvit/evaluate_robustness_cars_c.py)

Example:

```bash
cd protosvit
python run_inference_cars_c.py \
  --ckpt /path/to/epoch.ckpt \
  --cars_c_dir /path/to/cars_c \
  --modes proto_tta proto_tta_plus \
  --corruption gaussian_noise \
  --severity 5
```

## ProtoPNet-style Modules

Main files:

- [`protopnet/proto_entropy.py`](../protopnet/proto_entropy.py)
- [`protopnet/proto_entropy_enchanced.py`](../protopnet/proto_entropy_enchanced.py)

These files are module-level implementations rather than a full standalone training/evaluation package in this release.

## VLM Explainability

Main files:

- [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py)
- [`ProtoViT/VLM/vlm.py`](../ProtoViT/VLM/vlm.py)

Use the developer-facing explanation in:

- [`VLM_EXPLAINABILITY.md`](./VLM_EXPLAINABILITY.md)
