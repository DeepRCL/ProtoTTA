# ProtoLens / Amazon-C

This folder contains the NLP part of the ProtoTTA experiments: a ProtoLens
model (Wei & Zhu, 2025) with an all-mpnet-base-v2 backbone and 50 prototypes
shared across classes, trained on Yelp (94.0% accuracy) and evaluated on the
Amazon-C benchmark (91.9% accuracy on the clean Amazon reviews). ProtoTTA uses
the shared-prototype objective of Appendix H: classifier-derived importance
weights and soft targets replace hard prototype-class membership.

Configuration (Appendix L): LayerNorm affine parameters and attention-projection
biases are adapted in evaluation mode with Adam (betas 0.9/0.999, no weight
decay), learning rate 5e-6, batch size 128, one update per accepted batch.
Reliability filter: mean of the top 25 of the 50 cosine similarities >= 0.1.
Prototype activations: sigmoid(5 * clip(similarity, -1, 1)). No sample-confidence
weighting. lambda in [0, 1], no smoothing. Adaptation is continual within each
corruption-severity setting, and the source checkpoint is restored between
settings. Seeds 0, 2 and 3.

## Files

| File | Used for |
| --- | --- |
| `evaluate_robustness_amazonc.py` | Tables 1, 2, 3 (right, ProtoLens column) and 7: Unadapted, Tent, EATA, SAR, ProtoTTA (fixed and adaptive lambda, controller variants); PAC, PCA-W, prediction stability, selection rate, timing |
| `proto_tta.py` | ProtoTTA for shared prototypes (Appendix H, controller of Appendix L) |
| `tent.py`, `eata.py` | Tent and EATA baselines (SAR is defined in the evaluator) |
| `adapt_utils.py` | Selection of the adapted parameters (`layernorm_attn_bias`) |
| `prototype_metrics.py` | PAC, PCA-W with the shared-prototype rule of Appendix E.2 (Eqs. 14-15), prediction stability, timing |
| `generate_amazon_c.py` | Amazon-C generation (Appendix A) |
| `protolens_llm_common.py`, `protolens_llm_export.py`, `protolens_llm_score.py`, `protolens_llm_analyze.py` | Label-free judge intervention, ProtoLens row of Table 8 (Appendix C) |
| `PLens.py`, `AdaptiveMask.py`, `DPGMM.py`, `Gumbel_Sigmoid.py`, `utils.py` | ProtoLens model and data utilities |
| `prepare_yelp_data.py`, `generate_prototypes.py`, `split_yelp_data.py`, `args.py`, `experiment.py`, `evaluate_test.py` | Training and testing the Yelp source model |

## Setup

Use the main environment (`../protovit_env.yml`). Amazon-C generation also needs
WildNLP (Rychalska et al., 2019): install it on the `PYTHONPATH`, or clone it
next to this folder as `../WildNLP`. Judge scoring uses the VLM environment
(`../vlm_environment.yml`).

Run every command from this folder. `PLens.py` reads the prototype
initialization files `Datasets/Yelp/all-mpnet-base-v2/Yelp_cluster_50_centers.npy`
and `Datasets/Yelp/all-mpnet-base-v2/Yelp_cluster_50_to_sub_sentence.csv`
relative to the working directory, also when a trained checkpoint is loaded;
`generate_prototypes.py` writes them (step 1).

The reported runs used one NVIDIA B200 GPU at batch size 128 with

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # set for the Unadapted/Tent/EATA/SAR/adaptive runs of Section 3
```

## 1. Source model (Yelp)

Place the Yelp Open Dataset review file at
`Datasets/yelp_academic_dataset_review.json` and run:

```bash
python prepare_yelp_data.py
python generate_prototypes.py --dataset Yelp --num_prototypes 50 --window_size 5
python split_yelp_data.py
python experiment.py -d Yelp -pn 50 -e 15 -bs 128 -lr 0.0005 -ws 5 -i 0
python evaluate_test.py --dataset Yelp \
  --model_path log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth
```

The checkpoint is written to
`log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth`
(`MODEL` below).

## 2. Amazon-C (Appendix A)

Five corruption types (Keyboard: QWERTY; Character: Swap, Remove Char;
Combined: Mixed, Aggressive) at 20/40/60/80% word-level corruption, 20 settings:

```bash
python generate_amazon_c.py \
  --source_file /path/to/Amazon/test.csv \
  --output_dir /path/to/Amazon-C \
  --num_samples 4000 --severities 20 40 60 80 --seed 42
```

The source file is a CSV with columns `review` and `sentiment` (0/1). Besides
`amazon_c_<corruption>_s<severity>.csv`, the generator writes
`amazon_c_clean.csv`, which the evaluator uses as the clean reference.

## 3. Main results: Tables 1, 2 and 7

```bash
MODEL=/path/to/model.pth
DATA=/path/to/Amazon-C
OUT=/path/to/results
COMMON="--model_path $MODEL --data_dir $DATA \
  --learning_rate 0.000005 --adaptation_mode layernorm_attn_bias \
  --geo_filter --geo_threshold 0.1 --sigmoid_temperature 5.0 \
  --batch_size 128 --prototype-metrics --track-efficiency"

for SEED in 0 2 3; do
  # Unadapted, Tent, EATA, SAR
  python evaluate_robustness_amazonc.py --seed $SEED $COMMON \
    --methods baseline tent eata sar --output $OUT/baselines_seed$SEED.json

  # ProtoTTA (adaptive lambda)
  python evaluate_robustness_amazonc.py --seed $SEED $COMMON --methods prototta \
    --proto_lambda 0.5 --proto_adaptive_lambda --proto_samplewise_lambda \
    --proto_gradient_normalize \
    --proto_adaptive_strategy source_free_router_coverage_absolute \
    --proto_adaptive_delta0 0.25 --proto_adaptive_topk 3 \
    --proto_router_min_consistency 0.25 \
    --proto_lambda_ema_momentum 0 --proto_lambda_min 0 --proto_lambda_max 1 \
    --proto_record_diagnostics --output $OUT/adaptive_seed$SEED.json

  # ProtoTTA (fixed lambda = 1)
  python evaluate_robustness_amazonc.py --seed $SEED $COMMON --methods prototta \
    --proto_lambda 1.0 --proto_gradient_normalize \
    --proto_record_diagnostics --output $OUT/fixed_lambda1.0_seed$SEED.json
done
```

`baseline` is the unadapted model; MEMO is not evaluated on ProtoLens. Each
output JSON holds one entry per method, corruption and severity. Table 7 reports
per-setting accuracy averaged over the three seeds; Table 1 reports the mean and
sample standard deviation over the 20 settings of these seed averages. Table 2
averages over settings and seeds:

| Table 2 column | Result key |
| --- | --- |
| Semantic consistency (PAC) | `PAC_mean` |
| Prototype alignment (PCA-W) | `PCA_weighted_mean` |
| Prediction stability | `calibration_agreement` |
| Selection rate | `adaptation_rate` |
| Relative speed | mean `efficiency.time_per_sample_ms` of `baseline` divided by that of the method |

Table 1 means: Unadapted 80.81, SAR 77.98, Tent 79.74, EATA 80.79, ProtoTTA
(fixed lambda) 81.32, ProtoTTA (adaptive lambda) 81.31.

## 4. Controller components: Table 3 (right), ProtoLens column

Every row keeps the filter, adapted parameters and optimizer of the full method
and adds the arguments of one row to `$BASE`:

```bash
BASE="--model_path $MODEL --data_dir $DATA --methods prototta \
  --learning_rate 0.000005 --adaptation_mode layernorm_attn_bias \
  --geo_filter --geo_threshold 0.1 --sigmoid_temperature 5.0 --batch_size 128 \
  --prototype-metrics --track-efficiency \
  --proto_adaptive_delta0 0.25 --proto_adaptive_topk 3 \
  --proto_router_min_consistency 0.25 \
  --proto_lambda_ema_momentum 0 --proto_lambda_min 0 --proto_lambda_max 1 \
  --proto_record_diagnostics"
ADAPT="--proto_lambda 0.5 --proto_adaptive_lambda --proto_samplewise_lambda"

# Example: "Mean of the two"
for SEED in 0 2 3; do
  python evaluate_robustness_amazonc.py --seed $SEED $BASE $ADAPT \
    --proto_adaptive_strategy mean_act_rel --proto_gradient_normalize \
    --output $OUT/mean_act_rel_seed$SEED.json
done
```

| Row | Arguments added to `$BASE` | ProtoLens |
| --- | --- | --- |
| Router (ours) | `$ADAPT --proto_adaptive_strategy source_free_router_coverage_absolute --proto_gradient_normalize` (the adaptive command of Section 3) | 81.31 |
| lambda^act only | `$ADAPT --proto_adaptive_strategy activation_margin --proto_gradient_normalize` | 81.31 |
| lambda^rel only | `$ADAPT --proto_adaptive_strategy relative_evidence --proto_gradient_normalize` | 78.55 |
| Mean of the two | `$ADAPT --proto_adaptive_strategy mean_act_rel --proto_gradient_normalize` | 80.72 |
| No coverage guard | `$ADAPT --proto_adaptive_strategy source_free_router_absolute --proto_gradient_normalize` | 78.56 |
| No gradient normalization | `$ADAPT --proto_adaptive_strategy source_free_router_coverage_absolute` | 78.44 |
| Fixed lambda = 0 (output) | `--proto_gradient_normalize --proto_fixed_lambda_gradnorm --proto_lambda 0.0` | 78.44 |
| Fixed lambda = 0.5 | `--proto_gradient_normalize --proto_fixed_lambda_gradnorm --proto_lambda 0.5` | 80.26 |
| Fixed lambda = 1 (prototype) | the fixed-lambda command of Section 3 | 81.32 |
| Best fixed lambda (sweep using labels) | lambda = 1, the best value of the sweep below | 81.32 |

Fixed-lambda sweep used to select the best fixed lambda:

```bash
for L in 0.0 0.2 0.4 0.5 0.7 0.8 1.0; do
  python evaluate_robustness_amazonc.py --methods prototta --seed 42 \
    --model_path $MODEL --data_dir $DATA \
    --learning_rate 0.000005 --adaptation_mode layernorm_attn_bias \
    --geo_filter --geo_threshold 0.1 --sigmoid_temperature 5.0 \
    --batch_size 128 --proto_lambda $L \
    --output $OUT/lambda_sweep/amazon_c_lambda${L}.json --force
done
```

## 5. Label-free judge intervention: Table 8, ProtoLens row (Appendix C)

The export step runs adaptive ProtoTTA (the configuration of Section 3), the
score step queries Qwen/Qwen3.6-35B-A3B, and the analysis combines the seeds.
Task ids 0-19 index the corruption-severity settings:

```bash
JUDGE=/path/to/judge_run
for SEED in 0 2 3; do
  for TASK in $(seq 0 19); do
    # main environment
    python protolens_llm_export.py --output-dir $JUDGE/seed_$SEED --task-id $TASK \
      --random-state $SEED --batch-size 128 --data-dir $DATA --model-path $MODEL
    # VLM environment
    python protolens_llm_score.py --output-dir $JUDGE/seed_$SEED --task-id $TASK \
      --model-name Qwen/Qwen3.6-35B-A3B --max-new-tokens 2048 \
      --llm-batch-size 12 --minimum-vram-gib 85
  done
done
python protolens_llm_analyze.py \
  --output-dir $JUDGE/seed_0 $JUDGE/seed_2 $JUDGE/seed_3 --summary-dir $JUDGE
```

The analysis writes `analysis.json` and `report.md` with the Table 8 quantities:
ProtoTTA 81.311 +- 0.000 and ProtoTTA + Judge 82.345 +- 0.008 (macro-average over
the 20 settings; SD over the seed-level macro-averages), a difference of +1.034
points, and W/T/L 20/0/0 over the settings.
