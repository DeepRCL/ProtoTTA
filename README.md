# ProtoTTA: Anonymous Reproducibility Package

This repository is the anonymized code artifact for a double-blind paper
submission. It contains the implementation needed to reproduce
prototype-guided test-time adaptation experiments across image, pathology,
and text prototype models. Author names, affiliations, contact details,
publication links, paper drafts, generated results, and machine-specific
cluster launchers are intentionally omitted.

## Method overview

ProtoTTA adapts a pretrained prototype model at inference time using the
model's intermediate prototype evidence. The implementation provides:

- prototype-level binary entropy minimization;
- geometric filtering of unreliable samples;
- prototype-importance and prediction-confidence weighting;
- adaptive routing between prototype and output entropy;
- component-wise gradient normalization and stability guards;
- PAC, PCA-W, calibration, sparsity, selection-rate, and efficiency metrics;
- label-blind VLM export, scoring, and analysis utilities.

The adaptation code does not use target labels. Clean labels or paired clean
samples are used only by optional offline evaluation metrics.

## Repository layout

```text
ProtoTTA/                         Generic integration and metric guides
ProtoViT/                         CUB-200-C implementation and VLM analysis
ProtoPFormer/                     Stanford Dogs-C implementation
ProtoLens/                        Amazon-C text implementation
ProtoPNetVLM/protopnet_tta/       SICAPv2-C and label-blind VLM pipeline
protosvit/                        Stanford Cars-C implementation
protopnet/                        Standalone ProtoPNet adaptation modules
protovit_env.yml                  Main vision environment
vlm_environment.yml               VLM analysis environment
flsh_att_environment.yml          Optional Flash Attention environment
```

Start with [`ProtoTTA/README.md`](ProtoTTA/README.md) for the model-agnostic
interface and [`ProtoTTA/EXISTING_BACKBONES.md`](ProtoTTA/EXISTING_BACKBONES.md)
for backbone-specific entry points.

## Installation

The main vision stack can be created with:

```bash
conda env create -f protovit_env.yml
conda activate protovit
```

For VLM scoring, use the isolated environment:

```bash
conda env create -f vlm_environment.yml
conda activate echofar
```

Backbones have different upstream dependencies. The ProtoPNet pipeline also
expects the upstream `proto_baseline` package on `PYTHONPATH`; see
[`ProtoPNetVLM/protopnet_tta/README.md`](ProtoPNetVLM/protopnet_tta/README.md).
No dataset or checkpoint is downloaded implicitly.

## Data and checkpoints

Download the public source datasets from their official providers and arrange
them in ImageFolder-compatible layouts where applicable:

- CUB-200-2011 for ProtoViT;
- Stanford Dogs for ProtoPFormer;
- Stanford Cars for ProtoS-ViT;
- SICAPv2 for ProtoPNet;
- Yelp/Amazon review data for ProtoLens.

Corruption generators are included for CUB-200-C, Dogs-C, SICAPv2-C, and the
text benchmarks. Dataset, checkpoint, and output paths are command-line
arguments. Checkpoints are not included because of artifact size and upstream
distribution constraints.

## Main evaluation commands

The commands below show the adaptive configuration and expose every path that
must be supplied locally. Output filenames are illustrative and may be changed.

### ProtoViT / CUB-200-C

```bash
cd ProtoViT
python evaluate_robustness.py \
  --model /path/to/protovit_checkpoint.pth \
  --data_dir /path/to/cub200_c \
  --clean_data_dir /path/to/cub200_clean_test \
  --output /path/to/results/protovit.json \
  --modes normal tent eata sar \
          proto_imp_conf_adaptive_source_free_router_coverage_absolute \
  --corruptions all \
  --prototype-metrics \
  --use-enhanced-metrics \
  --track-efficiency \
  --proto-gradient-normalize \
  --proto-adaptive-delta0 0.25 \
  --proto-adaptive-topk 3
```

### ProtoPFormer / Stanford Dogs-C

```bash
cd ProtoPFormer
python evaluate_robustness_dogs.py \
  --model /path/to/protopformer_checkpoint.pth \
  --data_dir /path/to/stanford_dogs_c \
  --clean_dir /path/to/stanford_dogs_clean \
  --output /path/to/results/protopformer.json \
  --modes normal tent eata sar \
          proto_tta_adaptive_source_free_router_coverage_absolute \
  --corruptions all \
  --prototype-metrics \
  --use-enhanced-metrics \
  --track-efficiency \
  --proto_gradient_normalize \
  --proto_adaptive_delta0 0.25 \
  --proto_adaptive_topk 3
```

### ProtoLens / Amazon-C

```bash
cd ProtoLens
python evaluate_robustness_amazonc.py \
  --model_path /path/to/protolens_checkpoint.pth \
  --data_dir /path/to/Amazon-C \
  --output /path/to/results/protolens.json \
  --methods baseline tent eata sar prototta \
  --proto_adaptive_lambda \
  --proto_gradient_normalize \
  --proto_adaptive_strategy source_free_router_coverage_absolute \
  --proto_adaptive_delta0 0.25 \
  --proto_adaptive_topk 3 \
  --force
```

### ProtoPNet / SICAPv2-C

```bash
python -m ProtoPNetVLM.protopnet_tta.evaluate_robustness \
  --model /path/to/protopnet_checkpoint.pth \
  --data_dir /path/to/SICAPv2-C \
  --clean_data_dir /path/to/SICAPv2/test \
  --output /path/to/results/protopnet.json \
  --severity 5 \
  --batch_size 64 \
  --modes Normal Tent EATA SAR MEMO \
          ProtoAbsoluteConsistencyCoverageRouter \
  --prototype-metrics \
  --track-efficiency
```

ProtoS-ViT commands are documented in
[`ProtoTTA/EXISTING_BACKBONES.md`](ProtoTTA/EXISTING_BACKBONES.md).

## Label-free ProtoLens supervisor

The text pipeline exports BEFORE/AFTER predictions and prototype evidence,
scores disagreements without exposing target labels, and opens sealed targets
only during final analysis:

```bash
cd ProtoLens
python protolens_llm_export.py --task-id 0 \
  --data-dir /path/to/Amazon-C \
  --model-path /path/to/protolens_checkpoint.pth \
  --output-dir /path/to/llm_run
python protolens_llm_score.py --task-id 0 \
  --output-dir /path/to/llm_run
python protolens_llm_analyze.py --output-dir /path/to/llm_run
```

Repeat the export and scoring commands for task indices `0` through `19`.
Generated evidence, scores, and sealed targets remain excluded from version
control.

## Reproducibility controls

- Each corruption/method run starts from a fresh checkpoint unless the
  corresponding entry point explicitly documents continuous adaptation.
- Deterministic backend settings and stable data ordering are used where the
  underlying libraries support them.
- Paired clean/corrupted evaluators validate sample count, label order, and
  sample identity before computing paired metrics.
- Result files record method configuration, stream order, and available
  dataset/checkpoint provenance.
- Existing result files are resumed by default in several evaluators; use the
  documented overwrite or force flag when a clean rerun is required.

## Verification

Run syntax and focused unit checks from the repository root:

```bash
python -m compileall -q \
  ProtoTTA ProtoLens ProtoPFormer ProtoViT \
  ProtoPNetVLM/protopnet_tta protopnet protosvit

cd ProtoViT
python -m unittest -v \
  test_failure_detection.py \
  test_fullset_vlm_gate.py \
  test_paired_adaptation_supervision.py

cd ../ProtoLens
python -m unittest -v test_protolens_llm.py
```

Generated datasets, checkpoints, result files, VLM boards, logs, caches, and
private target manifests are excluded by `.gitignore`.

## Double-blind note

Citation and contact information are withheld for review and will be restored
in the public release after the double-blind process.
