# Existing Backbones and Run Commands

These are the four paper-specific implementations. For a new model, prefer
the reusable [`ProtoTTAAdapter`](./adapter.py) instead of copying an entire
backbone directory.

## ProtoViT / CUB-200-C

Main files:

- [`ProtoViT/run_inference.py`](../ProtoViT/run_inference.py)
- [`ProtoViT/evaluate_robustness.py`](../ProtoViT/evaluate_robustness.py)
- [`ProtoViT/proto_entropy.py`](../ProtoViT/proto_entropy.py)

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

## ProtoLens / Amazon-C

Main files:

- [`ProtoLens/run_inference_amazon_c.py`](../ProtoLens/run_inference_amazon_c.py)
- [`ProtoLens/evaluate_robustness_amazonc.py`](../ProtoLens/evaluate_robustness_amazonc.py)
- [`ProtoLens/proto_tta.py`](../ProtoLens/proto_tta.py)

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

## ProtoPFormer / Stanford Dogs-C

Main files:

- [`ProtoPFormer/run_inference_dogs.py`](../ProtoPFormer/run_inference_dogs.py)
- [`ProtoPFormer/evaluate_robustness_dogs.py`](../ProtoPFormer/evaluate_robustness_dogs.py)
- [`ProtoPFormer/proto_tta.py`](../ProtoPFormer/proto_tta.py)

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

## ProtoPNet / SICAPv2-C

Main files:

- [`ProtoPNetVLM/protopnet_tta/evaluate_robustness.py`](../ProtoPNetVLM/protopnet_tta/evaluate_robustness.py)
- [`ProtoPNetVLM/protopnet_tta/proto_entropy_enhanced.py`](../ProtoPNetVLM/protopnet_tta/proto_entropy_enhanced.py)
- [`ProtoPNetVLM/protopnet_tta/README.md`](../ProtoPNetVLM/protopnet_tta/README.md)

The full pipeline expects the upstream `proto_baseline` package on
`PYTHONPATH`. Compact ProtoPNet-oriented adaptation modules are also available
under [`protopnet/`](../protopnet).

## VLM Explainability

The optional VLM evaluation is implemented in:

- [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py)
- [`ProtoViT/VLM/vlm.py`](../ProtoViT/VLM/vlm.py)
- [`VLM_EXPLAINABILITY.md`](./VLM_EXPLAINABILITY.md)
