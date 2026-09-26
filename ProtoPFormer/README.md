# ProtoPFormer with ProtoTTA

This directory contains the ProtoPFormer backbone and its prototype-guided
test-time adaptation implementation for Stanford Dogs-C.

## Dependencies

The original backbone was developed with PyTorch 1.8.1, torchvision 0.9.1,
Pillow 9.1.1, timm 0.5.4, OpenCV 4.6, SciPy 1.8, pandas 1.4, matplotlib 3.5,
and scikit-learn 1.1. Newer compatible versions may also work.

Download Stanford Dogs from its official provider and prepare the clean and
corrupted test folders in ImageFolder-compatible layouts. A trained
ProtoPFormer checkpoint is required and is not bundled with this artifact.

## Evaluation

```bash
python evaluate_robustness_dogs.py \
  --model /path/to/epoch-best.pth \
  --data_dir /path/to/stanford_dogs_c \
  --clean_dir /path/to/stanford_dogs_clean \
  --output /path/to/results.json \
  --modes normal tent eata sar \
          proto_tta_adaptive_source_free_router_coverage_absolute \
  --corruptions all \
  --prototype-metrics \
  --use-enhanced-metrics \
  --track-efficiency \
  --proto_gradient_normalize
```

The key files are:

- `evaluate_robustness_dogs.py`: multi-corruption evaluator;
- `proto_tta.py`: ProtoTTA and adaptive routing;
- `prototype_tta_metrics.py` and `enhanced_prototype_metrics.py`: prototype
  consistency, alignment, calibration, and efficiency metrics;
- `export_vlm_consensus_evidence.py` and `evaluate_vlm_purified.py`:
  label-blind VLM evidence utilities.

## Upstream model

The backbone is based on:

```bibtex
@article{xue2022protopformer,
  title={ProtoPFormer: Concentrating on Prototypical Parts in Vision Transformers for Interpretable Image Recognition},
  author={Xue, Mengqi and Huang, Qihan and Zhang, Haofei and Cheng, Lechao and Song, Jie and Wu, Minghui and Song, Mingli},
  journal={arXiv preprint arXiv:2208.10431},
  year={2022}
}
```

See `LICENSE` for the bundled backbone license.
