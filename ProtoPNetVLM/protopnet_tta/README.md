# ProtoTTA for ProtoPNet (SICAPv2-C)

This package contains the code that produces the ProtoPNet / SICAPv2-C results
of the paper: Tables 1, 2 and 6 (accuracy, interpretability and operational
metrics), Table 3 (right) and Table 10 (controller components and fixed-λ
sweep), Table 8 (label-free judge intervention, Appendix C) and the per-image
overhead of the full controller (Section 5 and Appendix N).

## Files

| File | Role |
|---|---|
| `evaluate_robustness.py` | Evaluator for all methods on SICAPv2-C (accuracy, PAC, PCA-W, prediction stability, selection rate, timing) |
| `proto_entropy_enhanced.py` | Prototype entropy, fixed-λ objective and adaptive controller |
| `tent.py`, `eata_adapt.py`, `sar_adapt.py`, `loss_adapt.py` | Tent, EATA, SAR and MEMO baselines |
| `prototype_metrics.py`, `enhanced_prototype_metrics.py`, `efficiency_metrics.py` | PAC, PCA-W, prediction stability and timing |
| `noise_utils.py`, `create_sicap_c.py` | The corruptions and the SICAPv2-C generator |
| `settings.py`, `preprocess.py` | Shared settings and helpers |
| `vlm_prototta_common.py`, `vlm_prototta_export.py`, `vlm_prototta_score.py`, `vlm_prototta_analyze.py` | Judge intervention of Appendix C (Table 8) |

## Requirements

- Python 3.10+
- PyTorch and torchvision
- NumPy, SciPy, scikit-image, OpenCV, Pillow, and tqdm
- the upstream `proto_baseline` package on `PYTHONPATH`
- a trained ProtoPNet checkpoint (VGG19-BN, 50 prototypes, 63.4% clean test
  accuracy)

The judge dependencies are listed in
[`requirements_vlm_prototta.txt`](requirements_vlm_prototta.txt). Keep them in
a separate environment if their PyTorch requirements conflict with the
ProtoPNet environment.

The code never downloads datasets or checkpoints. All paths are command-line
arguments. Run every command from the parent directory of this package
(`ProtoPNetVLM/`):

```bash
export PYTHONPATH=/path/to/dir_containing_proto_baseline:$PYTHONPATH
cd ProtoPNetVLM
```

## Prepare SICAPv2-C

Generate the 13 corruptions at severity 5 (Appendix A) from the clean SICAPv2
test set in ImageFolder layout (class folders `G3`, `G4`, `G4C`, `G5`, `NC`):

```bash
python -m protopnet_tta.create_sicap_c \
  --input_dir /path/to/SICAPv2/test \
  --output_dir /path/to/SICAPv2-C \
  --corruption gaussian_noise shot_noise impulse_noise speckle_noise \
               gaussian_blur defocus_blur fog frost jpeg_compression \
               pixelate contrast brightness elastic_transform \
  --severity 5
```

## Configuration (Appendix L)

ProtoTTA on ProtoPNet uses:

- the activation u = (1/9) log((d + 1)/(d + 10⁻⁴)) of the minimum squared
  distance d, without sample- or batch-dependent normalization;
- the reliability filter: maximum activation above 0.8 and output entropy below
  0.4 ln 5;
- the prototype loss: uniform mean over the ten predicted-class prototypes,
  without confidence weighting;
- the controller with δ0 = 0.25, Top-3, consistency floor 0.25, the coverage
  guard and gradient normalization (`--modes ProtoAbsoluteConsistencyCoverageRouter
  --adaptive-delta0 0.25 --adaptive-top-k 3`);
- the affine parameters of all 16 VGG19-BN normalization layers and the weights
  of the two 1×1 add-on convolutions; BatchNorm uses current-batch statistics;
- Adam with learning rate 10⁻³, one update per batch, batch size 64, and the
  source checkpoint restored for every corruption;
- seeds 0, 2 and 3; each seed also sets the order of the test stream
  (`--seed s --stream-order-seed s`).

## Commands

Paths are placeholders.

```bash
CKPT=/path/to/protopnet_checkpoint.pth
SICAP_C=/path/to/SICAPv2-C
SICAP_TEST=/path/to/SICAPv2/test
OUT=/path/to/results
mkdir -p $OUT/timing
CORRUPTIONS="gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur fog frost jpeg_compression pixelate contrast brightness elastic_transform"

COMMON=(--model "$CKPT" --data_dir "$SICAP_C" --clean_data_dir "$SICAP_TEST"
        --severity 5 --batch_size 64 --gpuid 0 --corruptions $CORRUPTIONS
        --adaptive-delta0 0.25 --adaptive-top-k 3)
```

For every setting, the output JSON (`results/<mode>/<corruption>/5`) stores
`accuracy`, `PAC_mean`, `PCA_weighted_mean`, `calibration_agreement` (prediction
stability), `adaptation_stats` (selected samples) and `efficiency`
(`time_per_sample_ms`). Table entries average each corruption over the three
seeds and then over the 13 corruptions; the ± values of Tables 1 and 2 are
standard deviations across corruptions. Relative speed is the unadapted time
per sample divided by the method's time per sample.

### Tables 1, 2 and 6: baselines and ProtoTTA with adaptive λ

```bash
for SEED in 0 2 3; do
  for MODE in Normal Tent EATA SAR MEMO; do
    python -m protopnet_tta.evaluate_robustness "${COMMON[@]}" --lambda-proto 0.2 \
      --prototype-metrics --track-efficiency \
      --modes $MODE --seed $SEED --stream-order-seed $SEED \
      --output $OUT/${MODE}_seed_$SEED.json
  done
  # ProtoTTA with adaptive λ (full controller)
  python -m protopnet_tta.evaluate_robustness "${COMMON[@]}" --lambda-proto 0.2 \
    --prototype-metrics --track-efficiency \
    --modes ProtoAbsoluteConsistencyCoverageRouter --seed $SEED --stream-order-seed $SEED \
    --output $OUT/ProtoTTA_final_adaptive_router_seed_$SEED.json
done
```

### Tables 1, 3 (right), 6 and 10: fixed λ

This is the fixed-λ sweep of Table 10 (`--modes ProtoHybrid`, prototype weight
λ and output weight 1 − λ, same reliability filter and parameters). Its
λ = 0.4 runs give the ProtoTTA (Fixed λ) rows of Tables 1 and 6, and its
λ = 0, 0.5 and 1 runs give the fixed-λ rows of Table 3 (right).

```bash
for LAMBDA in 0.0 0.2 0.4 0.5 0.7 0.8 1.0; do
  for SEED in 0 2 3; do
    python -m protopnet_tta.evaluate_robustness "${COMMON[@]}" --lambda-proto $LAMBDA \
      --prototype-metrics --track-efficiency \
      --modes ProtoHybrid --seed $SEED --stream-order-seed $SEED \
      --output $OUT/fixed_lambda_${LAMBDA}_seed_$SEED.json
  done
done
```

### Table 3 (right): controller components

Each variant changes only how λ is set or removes one controller component;
the router row is the adaptive run above.

| Row of Table 3 (right) | `--modes` |
|---|---|
| λ^act only | `ProtoAuditForcedNativeGradNorm` |
| λ^rel only | `ProtoAuditForcedRelativeGradNorm` |
| Mean of the two | `ProtoMeanActRelGradNorm` |
| No coverage guard | `ProtoAbsoluteOnlyRouter` |
| No gradient normalization | `ProtoRouterNoGradNorm` |

```bash
for MODE in ProtoAuditForcedNativeGradNorm ProtoAuditForcedRelativeGradNorm \
            ProtoMeanActRelGradNorm ProtoAbsoluteOnlyRouter ProtoRouterNoGradNorm; do
  for SEED in 0 2 3; do
    python -m protopnet_tta.evaluate_robustness "${COMMON[@]}" --lambda-proto 0.2 \
      --prototype-metrics --track-efficiency \
      --controlled-cohort-name rebuilt_seed0_sicapv2c_controlled_four_seed \
      --refuse-existing-output \
      --modes $MODE --seed $SEED --stream-order-seed $SEED \
      --output $OUT/controller_${MODE}_seed_$SEED.json \
      --clean-reference-cache $OUT/clean_cache/${MODE}_seed$SEED.pt
  done
done
```

### Table 8 (Appendix C): label-free judge intervention

The judge model is Qwen/Qwen3.6-35B-A3B; step 2 runs in the VLM environment.
The exporter checks its predictions against the `Normal` and
`ProtoAbsoluteConsistencyCoverageRouter` runs of Tables 1 and 2, which
`--baseline-dir` must contain as `Unadapted_seed_<s>.json` and
`ProtoTTA_final_adaptive_router_seed_<s>.json`.

```bash
JUDGE=/path/to/judge_intervention
PROTOTYPE_IMAGES=/path/to/prototype_images   # prototype-img<j>.png from the ProtoPNet push step

mkdir -p $OUT/judge_baseline
for SEED in 0 2 3; do
  cp $OUT/Normal_seed_$SEED.json $OUT/judge_baseline/Unadapted_seed_$SEED.json
  cp $OUT/ProtoTTA_final_adaptive_router_seed_$SEED.json $OUT/judge_baseline/
done

for SEED in 0 2 3; do
  for C in $CORRUPTIONS; do
    # 1. Export pre-/post-adaptation predictions, evidence and reasoning boards
    python -m protopnet_tta.vlm_prototta_export \
      --output-dir $JUDGE --seed $SEED --corruption $C \
      --model "$CKPT" --data-dir "$SICAP_C" --prototype-dir $PROTOTYPE_IMAGES \
      --baseline-dir $OUT/judge_baseline --gpuid 0

    # 2. Judge scores (VLM environment)
    python -m protopnet_tta.vlm_prototta_score \
      --output-dir $JUDGE --seed $SEED --corruption $C \
      --model-name Qwen/Qwen3.6-35B-A3B --max-new-tokens 4096 \
      --vlm-batch-size 12 --minimum-vram-gib 85
  done
done

# 3. Accuracy with and without the judge over the three seeds
python -m protopnet_tta.vlm_prototta_analyze --output-dir $JUDGE
```

`$JUDGE/report.md` lists the Table 8 quantities; `$JUDGE/analysis.json` holds
them as fractions: `three_seed_mean_and_sample_std` for ProtoTTA (`prototta`)
and ProtoTTA + Judge (`board_vlm_selector`), and the difference and W/T/L in
`comparisons.board_vs_prototta`.

### Per-image overhead of the full controller (Section 5, Appendix N)

Timed on one NVIDIA B200 GPU over three corruptions with seed 0 and without
interpretability metrics; each run was repeated twice, the second time in
reverse order. The per-image time is `efficiency.time_per_sample_ms` (median of
the two repetitions for each corruption, then the mean over the three
corruptions), and the overhead is the difference between the adaptive run and
the unadapted run. `experiments/timing/make_commands.py overhead` prints the
same commands and `experiments/timing/summarize_overhead.py` computes the
overhead.

```bash
for REP in 0 1; do
  if [ $REP = 0 ]; then ORDER="Normal ProtoAbsoluteConsistencyCoverageRouter"
  else ORDER="ProtoAbsoluteConsistencyCoverageRouter Normal"; fi
  for MODE in $ORDER; do
    python -m protopnet_tta.evaluate_robustness \
      --model "$CKPT" --data_dir "$SICAP_C" --clean_data_dir "$SICAP_TEST" \
      --severity 5 --batch_size 64 --gpuid 0 \
      --lambda-proto 0.2 --adaptive-delta0 0.25 --adaptive-top-k 3 \
      --track-efficiency --modes $MODE \
      --corruptions gaussian_noise defocus_blur fog --seed 0 \
      --output $OUT/timing/${MODE}_rep$REP.json
  done
done
```
