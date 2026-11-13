# ProtoViT with ProtoTTA (CUB-200-C)

This directory contains the ProtoViT backbone and the code that produces the
ProtoViT / CUB-200-C results of the paper:

- Tables 1, 2 and 6: accuracy, interpretability and operational metrics;
- Table 3 (left): objective × reliability filter, and adapted parameters;
- Table 3 (right) and Table 10: controller components and fixed-λ sweep;
- Table 11: design ablations (Appendix I);
- Table 8: label-free judge intervention (Appendix C);
- Table 9 (left), Figure 2 and Figure 5: VLM scoring of reasoning boards and
  VLM narration (Appendices E.1 and F);
- Figures 3 and 6–8: qualitative analysis and forgetting analysis
  (Appendices J and K);
- the per-run evaluator used by the hyperparameter re-selection runs of
  Appendix M (Table 12).

The part-annotation, human-study, blinded-VLM, narrator-validation and
650-image PCA-W/VLM-correlation code (Tables 4, 5 and 9 right; Appendices E.3,
O and P) is in [`explanation_eval/`](explanation_eval).

## Files

| File | Role |
|---|---|
| `evaluate_robustness.py` | Evaluator for all methods on CUB-200-C: accuracy, PAC, PCA-W, prediction stability, selection rate, timing |
| `proto_entropy.py` | ProtoTTA: prototype objective, reliability filters, adaptive controller, gradient normalization |
| `tent.py`, `eata_adapt.py`, `sar_adapt.py` | Tent, EATA and SAR baselines |
| `evaluate_robustness_memo_only.py`, `memo_adapt.py` | MEMO baseline |
| `prototype_tta_metrics.py`, `enhanced_prototype_metrics.py` | PAC, PCA-W (k = 10, Appendix E.2) and prediction stability (Appendix E.4) |
| `efficiency_metrics.py` | Per-sample time (relative speed) |
| `noise_utils.py`, `create_cub_c.py` | Corruptions and the CUB-200-C generator |
| `preprocessing/` | Cropping, train/test split and training-set augmentation of CUB-200-2011 |
| `model.py`, `tools/`, `settings.py`, `main.py`, `train_and_test.py`, `push_greedy.py`, `helpers.py`, `preprocess.py`, `save.py`, `log.py` | ProtoViT model and training (upstream code; `main.py` produces the checkpoint) |
| `vlm_eval.py` | Reasoning boards, VLM scoring (Table 9 left, Figure 5) and VLM narration (Figure 2) |
| `fullset_vlm_gate.py`, `analyze_judge_intervention.py` | Label-free judge intervention (Table 8) |
| `run_inference.py`, `interpretability_viz.py` | Qualitative figures and per-batch accuracy (Figures 3 and 6–8) |

`../ProtoPFormer` also loads `efficiency_metrics.py` and `fullset_vlm_gate.py`
from this directory, so keep the repository layout unchanged.

## Environment

Create the evaluation environment from the repository root with
`conda env create -f protovit_env.yml` (or install `requirements.txt` and
`robustness_requirements.txt`). The VLM steps (`vlm_eval.py` scoring and
narration, `fullset_vlm_gate.py score`) run in the environment of
`vlm_environment.yml`.

## Data and checkpoint

Download CUB-200-2011 from its official provider, then crop the images with
their bounding boxes, split them into the official train/test sets and augment
the training set:

```bash
python preprocessing/crop_images.py --dataset_dir /path/to/CUB_200_2011 \
  --output_dir ./datasets/cub200_cropped
python preprocessing/split_dataset.py --dataset_dir /path/to/CUB_200_2011 \
  --cropped_dir ./datasets/cub200_cropped/cropped --output_dir ./datasets/cub200_cropped
python preprocessing/image_aug_v2.py --input_dir ./datasets/cub200_cropped/train_cropped \
  --output_dir ./datasets/cub200_cropped/train_cropped_augmented
```

Create CUB-200-C: the 13 corruptions at severity 5 applied to the 5,794 test
photos (75,322 images; Appendix A):

```bash
python create_cub_c.py \
  --input_dir ./datasets/cub200_cropped/test_cropped \
  --output_dir ./datasets/cub200_c \
  --corruption gaussian_noise shot_noise impulse_noise speckle_noise \
               gaussian_blur defocus_blur fog frost brightness \
               jpeg_compression contrast pixelate elastic_transform \
  --severity 5
```

`settings.py`, `main.py` and `run_inference.py` read the data from
`./datasets/cub200_cropped/` and `./datasets/cub200_c/` (relative to this
directory); the other scripts take the paths as arguments.

Checkpoint: ProtoViT with a DeiT-S/16 backbone and 2,000 prototypes (10 per
class, 4 sub-prototypes each), 86.1% clean test accuracy. `python main.py -gpuid 0`
trains it with the settings in `settings.py` and writes
`saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth` (the
checkpoint after the last fine-tuning epoch, named with its test accuracy)
together with the prototype images in `saved_models/deit_small_patch16_224/exp1/img/`.
All results use this checkpoint. The checkpoint is not bundled.

## Configuration (Appendix L)

ProtoTTA on ProtoViT (`--modes proto_imp_conf_adaptive_source_free_router_coverage_absolute`):

- Top-2 mean of the four sub-prototype cosine similarities
  (`--proto-consensus-strategy top_k_mean`, the default);
- reliability filter: maximum prototype activation above 0.92 (the default of
  `--proto-geo-filter-threshold`);
- prototype loss over the predicted-class prototypes with normalized
  classifier-weight importance (on by default);
- one detached confidence weight shared by both loss terms
  (`--proto-shared-confidence-weighting`);
- gradient normalization of both components (`--proto-gradient-normalize`);
- the controller with δ0 = 0.25, Top-3, consistency floor 0.25, coverage
  guard (mean output margin ≥ 0.90 and reliable fraction ≤ 0.40), λ in [0, 1]
  and no temporal smoothing (`--proto-adaptive-strategy
  source_free_router_coverage_absolute --proto-adaptive-delta0 0.25
  --proto-adaptive-topk 3 --proto-router-min-consistency 0.25
  --proto-lambda-min 0 --proto-lambda-max 1 --proto-lambda-ema-momentum 0`);
- LayerNorm affine parameters and attention biases adapted in evaluation mode
  (`--proto-adaptation-mode layernorm_attn_bias`, the default), Adam with
  learning rate 10⁻³ (β = (0.9, 0.999), no weight decay), one update per batch,
  batch size 128, and the source checkpoint restored for every corruption;
- seeds 0, 2 and 3.

The baselines use the settings in `tent.py`, `eata_adapt.py` and `sar_adapt.py`
(learning rate 10⁻³, batch size 128) and, for MEMO, learning rate 2.5 × 10⁻⁴,
16 augmented views and one step per test image.

## Commands

Run all commands from this directory. Paths are placeholders; the output files
are JSON files with one entry per method and corruption.

```bash
CKPT=./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth
CUB_C=./datasets/cub200_c
CLEAN=./datasets/cub200_cropped/test_cropped
PROTODIR=./saved_models/deit_small_patch16_224/exp1/img/epoch-4   # prototype images of the final push
OUT=/path/to/results
VLM=/path/to/vlm_eval              # results directory of vlm_eval.py

DATA=(--model "$CKPT" --data_dir "$CUB_C" --clean_data_dir "$CLEAN" --gpuid 0 --batch_size 128)
METRICS=(--prototype-metrics --use-enhanced-metrics --proto-baseline-samples 5794 --track-efficiency)
PROTO=(--proto-shared-confidence-weighting --proto-gradient-normalize
       --proto-adaptive-delta0 0.25 --proto-adaptive-topk 3
       --proto-router-min-consistency 0.25 --proto-lambda-ema-momentum 0
       --proto-lambda-min 0 --proto-lambda-max 1 --proto-record-diagnostics)
ROUTER=(--modes proto_imp_conf_adaptive_source_free_router_coverage_absolute
        --proto-adaptive-strategy source_free_router_coverage_absolute)
```

`--proto-baseline-samples 5794` pairs every corrupted test image with its clean
photo for PAC and prediction stability.

### Tables 1, 2 and 6: main results

```bash
for SEED in 0 2 3; do
  # Unadapted, Tent, EATA, SAR
  python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" --seed $SEED \
    --modes normal tent eata sar --output $OUT/baselines_seed$SEED.json
  # MEMO (written into a copy of the baseline file)
  python evaluate_robustness_memo_only.py --model "$CKPT" --data_dir "$CUB_C" \
    --clean_data_dir "$CLEAN" --gpuid 0 --seed $SEED \
    --memo-lr 0.00025 --memo-batch-size 16 --memo-steps 1 "${METRICS[@]}" \
    --input $OUT/baselines_seed$SEED.json --output $OUT/memo_seed$SEED.json
  # ProtoTTA, adaptive λ
  python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${ROUTER[@]}" \
    --seed $SEED --output $OUT/adaptive_seed$SEED.json
done
```

ProtoTTA (Fixed λ) is the λ = 1 run of the Table 10 sweep below (the best fixed
λ on ProtoViT). Each table entry averages the three seeds per corruption; the
reported mean and standard deviation are across the 13 corruptions.

Result fields used in Table 2 (per method and corruption):

| Table 2 column | Field |
|---|---|
| PAC | `PAC_mean` |
| PCA-W | `PCA_weighted_mean` |
| Prediction stability | `calibration_agreement` |
| Selection rate | `adaptation_rate` (or `adaptation_stats.adapted_samples / total_samples`), averaged over seeds and corruptions; 100% for MEMO |
| Relative speed | mean `efficiency.time_per_sample_ms` of the unadapted run divided by that of the method (over seeds and corruptions) |

### Table 10 and fixed-λ rows: fixed-λ sweep

```bash
for LAMBDA in 0.0 0.2 0.4 0.5 0.7 0.8 1.0; do
  for SEED in 0 2 3; do
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" \
      --modes proto_imp_conf_v3 --proto-lambda $LAMBDA \
      --seed $SEED --output $OUT/fixed_lambda${LAMBDA}_seed$SEED.json
  done
done
```

λ = 1 gives ProtoTTA (Fixed λ) in Tables 1 and 6; λ = 0, 0.5 and 1 give the
fixed-λ rows of Table 3 (right); λ = 0 gives the "Output entropy / Ours" cell of
Table 3 (left).

### Table 3 (left): objective × reliability filter

The "Ours" column is the λ = 0 run above (output entropy) and the adaptive run
of Table 1 (ProtoTTA). The other four cells:

```bash
for SEED in 0 2 3; do
  for FILTER in none entropy; do
    # Output entropy only (fixed λ = 0)
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" \
      --modes proto_imp_conf_v3 --proto-lambda 0.0 \
      --proto-reliability-filter $FILTER --proto-entropy-filter-factor 0.4 \
      --seed $SEED --output $OUT/output_entropy_filter_${FILTER}_seed$SEED.json
    # ProtoTTA objective (adaptive λ)
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${ROUTER[@]}" \
      --proto-reliability-filter $FILTER --proto-entropy-filter-factor 0.4 \
      --seed $SEED --output $OUT/prototta_filter_${FILTER}_seed$SEED.json
  done
done
```

`--proto-reliability-filter none` adapts on every sample; `entropy` keeps the
samples with output entropy below 0.4 ln C (C = 200), as in EATA.

### Table 3 (left): adapted parameters

"LN + Attn Bias (Ours)" is the adaptive run of Table 1. The other rows change
only the adapted parameters (and, where stated, the learning rate):

```bash
run_params () {
  local NAME=$1; shift
  for SEED in 0 2 3; do
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${ROUTER[@]}" \
      "$@" --seed $SEED --output $OUT/params_${NAME}_seed$SEED.json
  done
}
run_params all_lr1e-3       --proto-adaptation-mode all_parameters
run_params proto_head       --proto-adaptation-mode all_adaptive --proto-lr 0.001
run_params layernorm_only   --proto-adaptation-mode layernorm_only
run_params all_lr1e-5       --proto-adaptation-mode all_parameters --proto-lr 0.00001
```

### Table 3 (right): controller components

The router row is the adaptive run of Table 1 and the fixed-λ rows come from
the Table 10 sweep. The other variants change only how λ is set or remove one
component:

```bash
run_variant () {
  local NAME=$1; shift
  for SEED in 0 2 3; do
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" \
      "$@" --seed $SEED --output $OUT/controller_${NAME}_seed$SEED.json
  done
}
# λ^act only
run_variant act_only --modes proto_imp_conf_adaptive_samplewise --proto-adaptive-strategy activation_margin
# λ^rel only
run_variant rel_only --modes proto_imp_conf_adaptive_samplewise --proto-adaptive-strategy relative_evidence
# Mean of the two
run_variant mean     --modes proto_imp_conf_adaptive_samplewise --proto-adaptive-strategy mean_act_rel
# No coverage guard
run_variant no_coverage_guard \
  --modes proto_imp_conf_adaptive_source_free_router_absolute \
  --proto-adaptive-strategy source_free_router_absolute
# No gradient normalization
run_variant no_gradient_norm "${ROUTER[@]}" --proto-gradient-normalize-override off
```

### Per-image overhead of the full controller (Section 5, Appendix N)

Timed on one NVIDIA B200 GPU over three corruptions with seed 0 and without
interpretability metrics; each run was repeated twice. The per-image time is
`efficiency.time_per_sample_ms` (median of the two repetitions for each
corruption, then the mean over the three corruptions), and the overhead is the
difference between the adaptive run and the unadapted run.

```bash
TIMING=(--model "$CKPT" --data_dir "$CUB_C" --gpuid 0 --batch_size 128 --track-efficiency
        --proto-shared-confidence-weighting --proto-gradient-normalize
        --proto-adaptive-delta0 0.25 --proto-adaptive-topk 3
        --proto-router-min-consistency 0.25 --proto-lambda-ema-momentum 0
        --proto-lambda-min 0 --proto-lambda-max 1
        --corruptions gaussian_noise defocus_blur fog --seed 0)
for REP in 0 1; do
  python evaluate_robustness.py "${TIMING[@]}" --modes normal \
    --output $OUT/timing_unadapted_rep$REP.json
  python evaluate_robustness.py "${TIMING[@]}" "${ROUTER[@]}" \
    --output $OUT/timing_adaptive_rep$REP.json
done
```

### Table 11: design ablations

Top-K Mean, Target Only and Both are the adaptive run of Table 1. The other
rows:

```bash
run_design () {
  local NAME=$1; shift
  for SEED in 0 2 3; do
    python evaluate_robustness.py "${DATA[@]}" "${METRICS[@]}" "${PROTO[@]}" "${ROUTER[@]}" \
      "$@" --seed $SEED --output $OUT/design_${NAME}_seed$SEED.json
  done
}
run_design consensus_mean    --proto-consensus-strategy mean
run_design consensus_max     --proto-consensus-strategy max
run_design all_prototypes    --proto-adapt-all-prototypes
run_design no_weighting      --proto-disable-importance-weighting --proto-disable-confidence-weighting
run_design importance_only   --proto-disable-confidence-weighting
run_design confidence_only   --proto-disable-importance-weighting
```

### Table 8 (Appendix C): label-free judge intervention

`fullset_vlm_gate.py export` adapts the model with ProtoTTA at fixed λ = 1 and exports the pre- and
post-adaptation predictions with their prototype evidence; `score` queries the
judge (Qwen/Qwen3.6-35B-A3B) on the reasoning boards; `analyze_judge_intervention.py`
computes the accuracy with and without the judge. Run `score` in the VLM
environment. `--development-subset` takes the `subset.json` written by
`vlm_eval.py` (next section).

```bash
CORRUPTIONS="gaussian_noise shot_noise impulse_noise speckle_noise defocus_blur gaussian_blur frost fog brightness contrast elastic_transform jpeg_compression pixelate"
JUDGE=/path/to/judge_intervention
for SEED in 0 2 3; do
  for C in $CORRUPTIONS; do
    # 1. Adapt and export pre-/post-adaptation predictions and prototype evidence
    python fullset_vlm_gate.py --output-dir $JUDGE/seed$SEED --data-dir "$CUB_C" \
      --model "$CKPT" --prototype-dir "$PROTODIR" \
      export --corruption $C --seed $SEED --batch-size 128
    # 2. Judge scores with the reasoning boards
    python fullset_vlm_gate.py --output-dir $JUDGE/seed$SEED --data-dir "$CUB_C" \
      --model "$CKPT" --prototype-dir "$PROTODIR" \
      score --corruption $C --seed $SEED --condition full_reasoning \
      --batch-size 12 --max-new-tokens 4096 --vlm-model-id Qwen/Qwen3.6-35B-A3B
  done
done
# 3. Accuracy with and without the judge over the three seeds
python analyze_judge_intervention.py --seeds 0 2 3 \
  --seed-dirs $JUDGE/seed0 $JUDGE/seed2 $JUDGE/seed3 \
  --development-subset $VLM/subset.json --output-dir $JUDGE/summary
```

`$JUDGE/summary/report.md` gives the Table 8 row: ProtoTTA and ProtoTTA + Judge
accuracy (mean and SD over the seed-level macro-averages), their difference and
W/T/L over the 13 corruptions.

### Table 9 (left), Figures 2 and 5: VLM scoring and narration (Appendices E.1, F)

`vlm_eval.py` builds a fixed 100-image subset of CUB-200-C (severity 5,
spread over the 13 corruptions; written to `$VLM/subset.json`), renders for
each image and method the corrupted test image, the predicted-class reasoning
board and the any-class contribution board, and scores them with
Qwen/Qwen3-VL-32B-Thinking (Part A: Focus Relevance, Prototype Match and
Overall Quality on a 1–5 scale). Part B generates the before/after narration of
Figure 2 from the unadapted and ProtoTTA boards.

```bash
COMMON_VLM=(--model "$CKPT" --clean-dir "$CLEAN" --corrupted-dir "$CUB_C" \
            --results-dir $VLM --severity 5 --gpuid 0)
# Part A: Table 9 (left); Figure 5 shows one scored sample
for METHOD in unadapted tent eata prototta; do
  python vlm_eval.py "${COMMON_VLM[@]}" --part A --method $METHOD
done
python vlm_eval.py "${COMMON_VLM[@]}" --part B   # Figure 2
```

`$VLM/README_summary.md` lists the mean and standard deviation of each score
per method.

### Figures 3 and 6–8 (Appendices J and K): qualitative and forgetting analysis

`run_inference.py` adapts the unadapted model, ProtoTTA and EATA on one
corruption stream without intermediate resets, plots the per-batch accuracy of
each method (`per_batch_performance.png`, Figure 8) and writes the
interpretability panels (clean/corrupted activation maps, top prototypes and
contribution boards; Figures 3, 6 and 7) for images where ProtoTTA is correct
and EATA is wrong.

```bash
python run_inference.py -model "$CKPT" -gpuid 0 \
  -corruption gaussian_noise -severity 5 \
  -mode normal,proto_importance_confidence,eata \
  --use-geometric-filter --geo-filter-threshold 0.92 \
  --consensus-strategy top_k_mean --adaptation-mode layernorm_attn_bias \
  --interpretability-num-images 5 --interpretability-mode proto_wins \
  --output-dir /path/to/figures
```

### Appendix M (Table 12): runs of the hyperparameter re-selection

Each configuration of the re-selection grids is one run of this evaluator over
the 13 corruptions with seeds 0, 2 and 3 (no interpretability metrics):

```bash
# Test streams
declare -A STREAMS=(
  [shuffled]="--batch_size 128 --stream-order shuffled"
  [mixed]="--batch_size 128 --stream mixed --stream-order shuffled --corruptions gaussian_noise shot_noise impulse_noise speckle_noise defocus_blur gaussian_blur frost fog brightness contrast elastic_transform pixelate jpeg_compression"
  [bs16]="--batch_size 16 --stream-order shuffled"
)
run_grid () {   # run_grid <data dir> <result dir> <seeds...>
  local DATA_DIR=$1 RES=$2; shift 2
  for NAME in shuffled mixed bs16; do
    read -ra STREAM <<< "${STREAMS[$NAME]}"
    mkdir -p $RES/$NAME
    for SEED in "$@"; do
      # ProtoTTA: filter threshold × learning rate
      for T in 0.86 0.89 0.92 0.94 0.96; do for LR in 3e-4 1e-3 3e-3; do
        python evaluate_robustness.py --model "$CKPT" --data_dir "$DATA_DIR" --gpuid 0 \
          "${PROTO[@]}" "${ROUTER[@]}" --proto-geo-filter-threshold $T --proto-lr $LR \
          "${STREAM[@]}" --seed $SEED --output $RES/$NAME/prototta_t${T}_lr${LR}_seed$SEED.json
      done; done
      # EATA: entropy margin E0 = factor × ln C and learning rate, plus its published configuration
      for E in 0.2 0.3 0.4 0.5 0.6; do for LR in 3e-4 1e-3 3e-3; do
        python evaluate_robustness.py --model "$CKPT" --data_dir "$DATA_DIR" --gpuid 0 \
          --modes eata --baseline-lr $LR --eata-e-margin-factor $E \
          "${STREAM[@]}" --seed $SEED --output $RES/$NAME/eata_e${E}_lr${LR}_seed$SEED.json
      done; done
      python evaluate_robustness.py --model "$CKPT" --data_dir "$DATA_DIR" --gpuid 0 \
        --modes eata "${STREAM[@]}" --seed $SEED --output $RES/$NAME/eata_published_seed$SEED.json
    done
  done
}
# Family protocols (select on three families / on one family): all photos
run_grid "$CUB_C" $OUT/reselection/families 0 2 3
# 20% validation split: validation part (seed 0), then the 80% test part (seeds 0, 2, 3)
run_grid /path/to/cub200_c_val $OUT/reselection/split_val 0
run_grid /path/to/cub200_c_test $OUT/reselection/split_test 0 2 3
```

`--stream-order shuffled` presents the images of each corruption in a fixed
permutation seeded by `--seed` (identical for every method) and restores the
source model for every corruption; `--stream mixed` pools the 13 corruptions
into one shuffled stream without resets. `cub200_c_val` and `cub200_c_test`
hold a class-stratified 20% / 80% partition of the test photos (the same photos
for every corruption), in the layout of CUB-200-C. On the test part only the
selected configurations need to be run. The selection protocols
(select on three families and test on one, select on one and test on three,
20% split; ties broken towards the lower learning rate) are applied to these
result files.
