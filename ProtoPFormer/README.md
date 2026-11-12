# ProtoPFormer with ProtoTTA (Stanford Dogs-C)

This directory contains the ProtoPFormer backbone and the code that produces
the ProtoPFormer / Stanford Dogs-C results of the paper: Tables 1, 2 and 6
(accuracy, interpretability and operational metrics), Table 3 (right) and
Table 10 (controller components and fixed-λ sweep), Table 8 (label-free judge
intervention, Appendix C) and the per-image overhead of the full controller
(Section 5 and Appendix N).

## Files

| File | Role |
|---|---|
| `evaluate_robustness_dogs.py` | Evaluator for all methods on Stanford Dogs-C (accuracy, PAC, PCA-W, prediction stability, selection rate, timing) |
| `proto_tta.py` | ProtoTTA (prototype objective, reliability filter, adaptive controller) and the Tent and EATA baselines of this backbone |
| `sar_adapt.py`, `memo_adapt.py` | SAR and MEMO baselines |
| `prototype_tta_metrics.py`, `enhanced_prototype_metrics.py` | PAC, PCA-W and prediction stability |
| `noise_utils.py`, `create_dogs_c.py` | The 13 corruptions and the Stanford Dogs-C generator |
| `protopformer.py`, `tools/` | ProtoPFormer model (upstream code) |
| `main.py` | Upstream ProtoPFormer training entry point (produces the checkpoint) |
| `vlm_prototype_guidance.py`, `export_vlm_consensus_evidence.py`, `analyze_vlm_consensus_selector_multiseed.py` | Judge intervention of Appendix C (Table 8) |

The evaluator also loads `../ProtoViT/efficiency_metrics.py`, and the judge step
of Appendix C runs `../ProtoViT/fullset_vlm_gate.py`, so keep the repository
layout unchanged.

## Dependencies

The original backbone was developed with PyTorch 1.8.1, torchvision 0.9.1,
Pillow 9.1.1, timm 0.5.4, OpenCV 4.6, SciPy 1.8, pandas 1.4, matplotlib 3.5,
and scikit-learn 1.1. Newer compatible versions may also work. The repository
root provides `protovit_env.yml` (evaluation) and `vlm_environment.yml` (VLM
judge).

## Data and checkpoint

Download Stanford Dogs from its official provider. The evaluator reads the test
split through `Images/` and `test_list.mat`, so `/path/to/stanford_dogs` below
is the directory that contains both. Create Stanford Dogs-C (13 corruptions at
severity 5, Appendix A):

```bash
python create_dogs_c.py \
  --input_dir /path/to/stanford_dogs \
  --output_dir /path/to/stanford_dogs_c \
  --corruption all \
  --severity 5
```

A trained ProtoPFormer checkpoint (DeiT-S/16, 1,200 local and 600 global
prototypes, 82.4% clean test accuracy) is required and is not bundled.

## Configuration (Appendix L)

All ProtoTTA runs use:

- the similarity mapping σ(ReLU(u) − 1) on both branches
  (`--proto_mapping sigmoid --proto_sigmoid_center 1.0 --proto_sigmoid_temp 1.0`);
- the reliability filter: mean of the maximum local and the maximum global
  activation above 0.55 (`--proto_threshold 0.55`);
- the prototype loss: uniform mean over the predicted-class prototypes of each
  branch, with both branches weighted equally
  (`--proto_no_importance --proto_branch both`);
- one detached confidence weight shared by both loss terms
  (`--proto_shared_confidence_weighting`);
- gradient normalization of both components (`--proto_gradient_normalize`);
- the controller with δ0 = 0.25, Top-3, consistency floor 0.25, λ in [0, 1] and
  no temporal smoothing (`--proto_adaptive_delta0 0.25 --proto_adaptive_topk 3
  --proto_router_min_consistency 0.25 --proto_lambda_min 0 --proto_lambda_max 1
  --proto_lambda_ema_momentum 0`);
- LayerNorm affine parameters and attention biases adapted in training mode
  (`--adapt_model_mode train`), Adam with learning rate 10⁻³, one update per
  batch, batch size 128, and the source checkpoint restored for every
  corruption;
- seeds 0, 2 and 3.

## Commands

Run all commands from this directory. Paths are placeholders.

```bash
CKPT=/path/to/protopformer_checkpoint.pth
DOGS=/path/to/stanford_dogs          # contains Images/ and test_list.mat
DOGS_C=/path/to/stanford_dogs_c
OUT=/path/to/results
CORRUPTIONS="gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur fog frost brightness jpeg_compression contrast pixelate elastic_transform"

DATA=(--model "$CKPT" --data_dir "$DOGS_C" --clean_dir "$DOGS"
      --severity 5 --batch_size 128 --num_workers 8 --gpuid 0
      --adapt_model_mode train)
METRICS=(--prototype-metrics --use-enhanced-metrics
         --proto-baseline-samples 8580 --track-efficiency)
PROTO=(--lr 1e-3 --steps 1 --proto_threshold 0.55 --proto_mapping sigmoid
       --proto_sigmoid_center 1.0 --proto_sigmoid_temp 1.0 --proto_no_importance
       --proto_branch both --proto_shared_confidence_weighting
       --proto_gradient_normalize)
CONTROLLER=(--proto_adaptive_delta0 0.25 --proto_adaptive_topk 3
            --proto_router_min_consistency 0.25 --proto_lambda_ema_momentum 0
            --proto_lambda_min 0 --proto_lambda_max 1 --proto_record_diagnostics)
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
  # Unadapted, Tent, EATA, SAR
  python evaluate_robustness_dogs.py "${DATA[@]}" "${METRICS[@]}" --steps 1 \
    --seed $SEED --output $OUT/baselines_seed$SEED.json \
    --modes normal tent eata sar \
    --lr 1e-3 --sar-lr 1e-4 --sar-margin 2.8724950456692273 \
    --sar-reset 0.2 --sar-rho 0.05

  # MEMO, one run per corruption
  for C in $CORRUPTIONS; do
    python evaluate_robustness_dogs.py "${DATA[@]}" "${METRICS[@]}" \
      --seed $SEED --output $OUT/memo_${C}_seed$SEED.json \
      --modes memo --corruptions $C \
      --memo-lr 0.00025 --memo-views 16 --memo-steps 1
  done

  # ProtoTTA with adaptive λ (full controller)
  python evaluate_robustness_dogs.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${CONTROLLER[@]}" \
    --seed $SEED --output $OUT/adaptive_seed$SEED.json \
    --modes proto_tta_adaptive_source_free_router_coverage_absolute \
    --proto_adaptive_strategy source_free_router_coverage_absolute
done
```

### Tables 1, 3 (right), 6 and 10: fixed λ

This is the fixed-λ sweep of Table 10. Its λ = 0.2 runs give the ProtoTTA
(Fixed λ) rows of Tables 1 and 6, and its λ = 0, 0.5 and 1 runs give the
fixed-λ rows of Table 3 (right).

```bash
for LAMBDA in 0.0 0.2 0.4 0.5 0.7 0.8 1.0; do
  for SEED in 0 2 3; do
    python evaluate_robustness_dogs.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" \
      --proto_record_diagnostics \
      --seed $SEED --output $OUT/fixed_lambda${LAMBDA}_seed$SEED.json \
      --modes proto_tta --proto_lambda $LAMBDA
  done
done
```

### Table 3 (right): controller components

Each variant changes only how λ is set or removes one controller component;
the router row is the adaptive run above.

```bash
run_variant () {
  local NAME=$1; shift
  for SEED in 0 2 3; do
    python evaluate_robustness_dogs.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${CONTROLLER[@]}" \
      "$@" --seed $SEED --output $OUT/controller_${NAME}_seed$SEED.json
  done
}
# λ^act only
run_variant act_only --modes proto_tta_adaptive_samplewise --proto_adaptive_strategy activation_margin
# λ^rel only
run_variant rel_only --modes proto_tta_adaptive_samplewise --proto_adaptive_strategy relative_evidence
# Mean of the two
run_variant mean     --modes proto_tta_adaptive_samplewise --proto_adaptive_strategy mean_act_rel
# No coverage guard
run_variant no_coverage_guard \
  --modes proto_tta_adaptive_source_free_router_absolute \
  --proto_adaptive_strategy source_free_router_absolute
# No gradient normalization
run_variant no_gradient_norm \
  --modes proto_tta_adaptive_source_free_router_coverage_absolute \
  --proto_adaptive_strategy source_free_router_coverage_absolute \
  --proto_gradient_normalize_override off
```

### Table 8 (Appendix C): label-free judge intervention

The exporter adapts the model with ProtoTTA at fixed λ = 0.2. The judge model is
Qwen/Qwen3.6-35B-A3B; step 3 runs in the VLM environment.

```bash
EVIDENCE=/path/to/prototype_evidence
JUDGE=/path/to/judge_intervention

# 1. Prototype images from the source training split (once)
python vlm_prototype_guidance.py extract \
  --model "$CKPT" --source "$DOGS" --results $EVIDENCE

for SEED in 0 2 3; do
  for C in $CORRUPTIONS; do
    # 2. Adapt, then export pre-/post-adaptation predictions and prototype evidence
    python export_vlm_consensus_evidence.py \
      --seed $SEED --corruption $C \
      --model "$CKPT" --data-dir "$DOGS_C" \
      --prototype-evidence $EVIDENCE/evidence \
      --output-dir $JUDGE/seed$SEED

    # 3. Judge scores (VLM environment): image only, then image + reasoning board
    for CONDITION in image_predictions full_reasoning; do
      (cd ../ProtoViT && python fullset_vlm_gate.py \
        --output-dir $JUDGE/seed$SEED --data-dir "$DOGS_C" --domain-name dog \
        score --seed $SEED --corruption $C --condition $CONDITION \
        --batch-size 12 --max-new-tokens 4096)
    done
  done
done

# 4. Accuracy with and without the judge over the three seeds
python analyze_vlm_consensus_selector_multiseed.py \
  --seed0-dir $JUDGE/seed0 --seed2-dir $JUDGE/seed2 --seed3-dir $JUDGE/seed3 \
  --output $JUDGE/summary_3seed.json
```

`summary_3seed.json` holds the Table 8 quantities: `macro_accuracy_mean` and
`macro_accuracy_seed_std` for ProtoTTA (`prototta`) and ProtoTTA + Judge
(`with_boards`), `macro_accuracy_delta` and `wins_ties_losses`.

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
TIMING=(--model "$CKPT" --data_dir "$DOGS_C" --severity 5 --batch_size 128
        --num_workers 8 --gpuid 0 --adapt_model_mode train --track-efficiency
        "${PROTO[@]}" --proto_adaptive_delta0 0.25 --proto_adaptive_topk 3
        --proto_router_min_consistency 0.25 --proto_lambda_ema_momentum 0
        --proto_lambda_min 0 --proto_lambda_max 1
        --corruptions gaussian_noise defocus_blur fog --seed 0)
UNADAPTED=(--modes normal)
ADAPTIVE=(--modes proto_tta_adaptive_source_free_router_coverage_absolute
          --proto_adaptive_strategy source_free_router_coverage_absolute)

for REP in 0 1; do
  if [ $REP = 0 ]; then ORDER="unadapted adaptive"; else ORDER="adaptive unadapted"; fi
  for METHOD in $ORDER; do
    if [ $METHOD = unadapted ]; then MODE=("${UNADAPTED[@]}"); else MODE=("${ADAPTIVE[@]}"); fi
    python evaluate_robustness_dogs.py "${TIMING[@]}" "${MODE[@]}" \
      --output $OUT/timing/${METHOD}_rep$REP.json
  done
done
```

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
