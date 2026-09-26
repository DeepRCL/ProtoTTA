# ProtoTTA for ProtoPNet

This package contains the ProtoPNet/SICAPv2-C implementation used to evaluate
prototype-guided test-time adaptation, standard TTA baselines, prototype
metrics, and the label-blind VLM analysis.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- NumPy, SciPy, scikit-image, OpenCV, Pillow, and tqdm
- the upstream `proto_baseline` package on `PYTHONPATH`
- a trained ProtoPNet checkpoint

The VLM-only dependencies are listed in
[`requirements_vlm_prototta.txt`](requirements_vlm_prototta.txt). Keep them in
a separate environment if their PyTorch requirements conflict with the
ProtoPNet environment.

The code never downloads datasets or checkpoints implicitly. Pass all paths on
the command line so that runs are portable and auditable.

## Prepare SICAPv2-C

Generate corruptions from an ImageFolder-compatible clean test set:

```bash
python -m ProtoPNetVLM.protopnet_tta.create_sicap_c \
  --input_dir /path/to/SICAPv2/test \
  --output_dir /path/to/SICAPv2-C \
  --corruption all \
  --severity 5
```

Use `--help` for subset generation and overwrite controls.

## Run the robustness evaluation

The following command evaluates the unadapted model, standard TTA baselines,
and the adaptive ProtoTTA controller with a reproducible stream order:

```bash
python -m ProtoPNetVLM.protopnet_tta.evaluate_robustness \
  --model /path/to/checkpoint.pth \
  --data_dir /path/to/SICAPv2-C \
  --clean_data_dir /path/to/SICAPv2/test \
  --output /path/to/results.json \
  --severity 5 \
  --batch_size 64 \
  --modes Normal Tent EATA SAR MEMO ProtoAbsoluteConsistencyCoverageRouter \
  --prototype-metrics \
  --track-efficiency
```

For the source-free evaluation path, add `--skip-clean-evaluation` and omit
`--prototype-metrics`. The clean set is used only for offline paired metrics;
it is not used by the adaptation objective.

Important reproducibility controls include `--stream-order`, `--corruptions`, `--lambda-proto`,
`--adaptive-delta0`, and `--adaptive-top-k`. The evaluator writes these values,
dataset hashes, and method configurations into the output metadata.

## Main implementation files

- `proto_entropy.py`: prototype-entropy adaptation.
- `proto_entropy_enhanced.py`: hybrid and adaptive controllers.
- `evaluate_robustness.py`: paired clean/corrupted evaluation entry point.
- `prototype_metrics.py` and `enhanced_prototype_metrics.py`: PAC, PCA-W,
  calibration, sparsity, and related diagnostics.
- `vlm_prototta_export.py` and `vlm_prototta_score.py`: label-blind VLM export
  and scoring.

Generated datasets, checkpoints, reasoning boards, sealed labels, logs, and
result files are intentionally excluded from version control.
