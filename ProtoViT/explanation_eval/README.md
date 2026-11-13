# Explanation evaluation on ProtoViT / CUB-200-C

Code for the explanation analyses of ProtoViT on CUB-200-C (severity 5, 13 corruptions): the
part-annotation and clean-reference measures (Table 4, Appendix O), the blinded human and VLM
comparisons of ProtoTTA and EATA and the narrator validation (Table 5, Appendix P), and the PCA-W /
VLM correlation analyses (Table 9 right, Appendix E.3).

| Script | Produces |
|---|---|
| `export_states.py` | per-sample predictions and prototype evidence of the Table 1 runs (Unadapted, EATA, ProtoTTA with adaptive λ; seed 0), checked against the reported runs |
| `export_clean_reference.py` | the unadapted model's explanation of every clean test photo |
| `consistency_stability.py` | Table 4 rows 1–2: part consistency and part stability (Appendix O) |
| `restoration_metrics.py` | Table 4 rows 3–5 and the eight closeness measures of Appendix O |
| `select_items.py`, `select_pairwise.py`, `finalize_pairwise.py`, `render_cards.py` | the 16 images and the explanation cards of the human study |
| `human_study_analysis.py` | Table 5 (left), human row; human–Gemini agreement (Appendix P) |
| `make_items_vlm104.py`, `vlm_judge.py`, `gemini_judge.py`, `vlm104_analysis.py` | Table 5 (left), VLM rows (104 images) |
| `narrator_items.py`, `narrator_board.py`, `narrator_subset.py`, `narrator_run.py`, `narrator_gemini_run.py`, `narrator_analysis.py` | Table 5 (right): narrator validation |
| `pcaw_vlm_correlation.py` | Table 9 (right): PCA-W vs VLM scores on 650 images (Appendix E.3) |
| `pcaw_vlm_partial.py` | Appendix E.3: partial Spearman correlations and the correct-only correlation |
| `pcaw_vs_preference.py` | Appendix E.3: Gemini's 104-image preference vs the PCA-W difference |
| `cub_parts.py`, `paths.py`, `pairwise_stats.py` | shared helpers |

## Setup

Run every command from the `ProtoViT/` directory, in the ProtoViT environment (the Qwen judges and
narrator additionally need `transformers` with Qwen3.6 support and a GPU). Data and checkpoint
locations default to the standard ProtoViT layout and can be overridden with environment variables:

```bash
export PROTOVIT_CHECKPOINT=/path/to/14finetuned0.8609.pth            # ProtoViT CUB checkpoint
export PROTOVIT_PROTOTYPE_DIR=/path/to/img/epoch-4                     # prototype-img-original<p>.png, bb4.npy
export CUB_ROOT=/path/to/CUB_200_2011                                  # images.txt, bounding_boxes.txt, parts/
export CUB_TEST_CROPPED=/path/to/cub200_cropped/test_cropped           # clean cropped test photos
E=explanation_eval
R=results/explanation_eval
```

`render_cards.py` uses the bundled font `assets/DejaVuSans-Bold.ttf`. Stimulus bytes also depend on the
Pillow version; the reported runs rendered them with Pillow 10.0.1 (`narrator_analysis.py` re-renders
the boards and checks them against the records of a run).

## 1. Per-sample states of the Table 1 runs

`--reference` is the `evaluate_robustness.py --output` JSON of the reported seed-0 run of the method
(the unadapted and EATA runs, and the ProtoTTA run with adaptive λ, Appendix L); the export must
reproduce its number of correct predictions in every corruption.

```bash
for M in unadapted eata prototta; do
  python $E/export_states.py --method $M --reference /path/to/table1_${M}_seed0.json --out_root $R/states
  python $E/export_states.py --method $M --reference /path/to/table1_${M}_seed0.json --out_root $R/states_gt --with_gt_protos
done
python $E/export_clean_reference.py --out $R/clean_reference.json
python $E/export_clean_reference.py --full --base $R/clean_reference.json --out $R/clean_reference_full.json
```

## 2. Part annotations and closeness to the clean explanation (Table 4, Appendix O)

The part set of a prototype on an image is the set of visible CUB keypoints that lie in one of its
active sub-prototype cells or in a neighbouring cell; the ten prototypes of the ground-truth class are
used. A prototype is consistent if its most frequent part appears in at least 80% of its class's test
images. Part stability asks whether a true-class prototype's part set on the corrupted photo equals that
of the unadapted model on the clean photo (pairs with an empty clean part set are excluded). Confidence
intervals for consistency resample classes with their ten prototypes; all other intervals are cluster
bootstrap intervals over source photos (2,000 resamples).

```bash
python $E/consistency_stability.py --states_root $R/states_gt --clean $R/clean_reference.json \
    --out $R/consistency_stability.json
python $E/restoration_metrics.py --states_root $R/states_gt --card_states_root $R/states \
    --clean $R/clean_reference_full.json --stability $R/consistency_stability.json \
    --out $R/restoration_metrics.json
```

`consistency_stability.json` (`modes.near`) gives part consistency on clean photos
(`clean_consistency`) and on corrupted photos per method (`consistency_corrupted`), part stability
(`stability`) and the ProtoTTA − EATA intervals (`paired`). `restoration_metrics.py` prints the eight
measures (`card_agreement`, `card_part_jaccard`, `stability`, `any10_agreement`, `gt_spearman`,
`gt_act_mae`, `gt_loc_exact`, `gt_loc_within1`) on all images and on the images where all three
methods keep the clean prediction, with ProtoTTA − EATA intervals and the number of corruptions on
which ProtoTTA is closer than EATA. Table 4 rows 3–5 are `any10_agreement` and `gt_act_mae` (all
images and the same-prediction subset).

## 3. Human study (Table 5 left, Appendix P)

Item selection uses only the exported states; correctness is used only to form strata.

```bash
# photos excluded from all later draws
python $E/select_items.py --mode human --seed 20261106 --states_root $R/states --out $R/items_initial.json
# 16 slots (stratified by correctness) with three backup candidates each
python $E/select_pairwise.py --states_root $R/states --exclude_items $R/items_initial.json \
    --out $R/pairwise/candidates.json
# screening: a candidate is replaced by the next one of its slot only if Gemini returns no valid blinded answer
python $E/finalize_pairwise.py round --candidates $R/pairwise/candidates.json --out $R/pairwise/round1.json
python $E/gemini_judge.py --items $R/pairwise/round1.json --out $R/pairwise/gemini_round1.jsonl \
    --card_background clean --design pairwise --schedule both_orders --terminal_failures
#   (repeat round / gemini_judge with --rounds ... --runs ... until every slot has a passing candidate)
python $E/finalize_pairwise.py final --candidates $R/pairwise/candidates.json \
    --rounds $R/pairwise/round1.json --runs $R/pairwise/gemini_round1.jsonl --out $R/pairwise/items_pairwise.json
# cards (public part: site/; private card key: card_key_PRIVATE.json)
python $E/render_cards.py --items $R/pairwise/items_pairwise.json --releases_dir $R/releases
```

Each screen shows the corrupted photo and the two cards in random order (A/B) with the question
"Which explanation is better?" (A, B, "No clear difference"). Responses are exported as a CSV with
columns `received_at`, `participant_id`, `release`, `payload_json`.

```bash
python $E/human_study_analysis.py --responses /path/to/responses.csv \
    --card_key $R/releases/release_<id>/card_key_PRIVATE.json --items $R/pairwise/items_pairwise.json \
    --exclude <participant ids> --received_before <ISO timestamp> \
    --vlm_records $R/pairwise/gemini_round1.jsonl --vlm_name "Gemini 3.8 Flash"
```

It prints ProtoTTA's share of the judgments that preferred one method, on all 16 images and on the 9
images where both methods are correct, with 95% CIs that resample participants and images (5,000
resamples), and the Spearman correlation between human and Gemini image-level preferences (item
bootstrap CI) with the number of images with non-tied preferences on which they agree.

## 4. Blinded VLM comparison on 104 images (Table 5 left)

```bash
python $E/select_items.py --mode vlm_large --per_corruption 20 --seed 20261108 --states_root $R/states \
    --exclude_items $R/items_initial.json --out $R/items_pool.json
python $E/make_items_vlm104.py --source $R/items_pool.json --human_items $R/pairwise/items_pairwise.json \
    --out $R/vlm104/items_vlm104.json
# Gemini 3.8 Flash
python $E/gemini_judge.py --items $R/vlm104/items_vlm104.json --out $R/vlm104/gemini.jsonl \
    --card_background clean --design pairwise --schedule both_orders --max_records_per_presentation 2
# Qwen3.6-35B-A3B (three shards of the same plan)
for K in 0 1 2; do
  python $E/vlm_judge.py --items $R/vlm104/items_vlm104.json --out $R/vlm104/qwen_s$K.jsonl \
      --model_id Qwen/Qwen3.6-35B-A3B --think_budget 12288 --sampling recommended \
      --card_background clean --design pairwise --schedule both_orders --num_shards 3 --shard_index $K
done
python $E/vlm104_analysis.py --items $R/vlm104/items_vlm104.json \
    --judge "Gemini 3.8 Flash" $R/vlm104/gemini.jsonl \
    --judge "Qwen3.6-35B-A3B" $R/vlm104/qwen_s0.jsonl $R/vlm104/qwen_s1.jsonl $R/vlm104/qwen_s2.jsonl \
    --out $R/vlm104/analysis.json
```

Both judges see the same cards as the participants, in both orders, with the same instructions and
question, without method names or class labels. `vlm104_analysis.py` prints ProtoTTA's share of the
decisive judgments on all 104 images and on the 52 images where both methods are correct, with 95% CIs
that resample images.

Gemini is called through an agentic command-line client (the Antigravity CLI, `agy`) that reads the
images from a temporary directory; the interface is documented in `gemini_judge.py`. Point
`--agent_cli` (or `$GEMINI_AGENT_CLI`) to the client. Every answer is accepted only if the logged tool
calls show that the model read exactly the given images.

## 5. Narrator validation (Table 5 right, Appendix P)

```bash
python $E/narrator_items.py --states_root $R/states --out $R/narrator/items.json \
    --smoke_out $R/narrator/smoke_items.json
python $E/narrator_subset.py --items $R/narrator/items.json --out $R/narrator/items_104.json
# Qwen3.6-35B-A3B (12 shards of the same plan)
for K in $(seq 0 11); do
  python $E/narrator_run.py --items $R/narrator/items_104.json --out $R/narrator/qwen/nv_s$K.jsonl \
      --num_shards 12 --shard_index $K
done
# Gemini 3.8 Flash (install gemini_agent/blind-reader as an agent of the CLI: read_file is its only tool)
python $E/narrator_gemini_run.py --items $R/narrator/items_104.json --out $R/narrator/gemini/gm_s0.jsonl \
    --agent_definition_dir $E/gemini_agent/blind-reader
python $E/narrator_analysis.py --items $R/narrator/items_104.json --records "$R/narrator/qwen/nv_s*.jsonl" \
    --out $R/narrator/analysis_qwen.json
python $E/narrator_analysis.py --items $R/narrator/items_104.json --records "$R/narrator/gemini/gm_s0.jsonl" \
    --out $R/narrator/analysis_gemini.json
```

For each of the 104 corrupted images, the narrator sees the photo and the reasoning boards of the
unadapted model and of ProtoTTA (in both orders) under a neutral
prompt that hides the true class, correctness, method name and which board was adapted, and answers
fixed questions. `narrator_analysis.py` scores the answers against the model's own records and prints
the three entries of Table 5 (right): whether the retrieved training examples changed (accuracy),
whether the evidence for competing classes rose or fell (accuracy), and whether the model's focus moved
(AUC against the measured shift of the boxed cells). It re-renders every board and checks it against
the run's records.

## 6. PCA-W and VLM scores (Table 9 right, Appendix E.3)

The 650 images (50 per corruption) are sampled from the board data that `vlm_eval.py` precomputes for
Table 9 (`results/vlm_eval`), excluding its 100-image subset; the judge is Qwen3.6-35B-A3B and never
sees labels or correctness.

```bash
P=results/pcaw_vlm_correlation
python $E/pcaw_vlm_correlation.py --results-dir $P --source-results-dir results/vlm_eval prepare
for M in unadapted prototta; do
  python $E/pcaw_vlm_correlation.py --results-dir $P --source-results-dir results/vlm_eval export-pcaw --method $M
  python $E/pcaw_vlm_correlation.py --results-dir $P --source-results-dir results/vlm_eval score --method $M
done
python $E/pcaw_vlm_correlation.py --results-dir $P analyze           # Table 9 (right)
python $E/pcaw_vlm_partial.py --metrics $P/correlation_metrics.json   # partial and correct-only Spearman
python $E/pcaw_vs_preference.py --items $R/vlm104/items_vlm104.json --states_root $R/states \
    --analysis $R/vlm104/analysis.json --judge "Gemini 3.8 Flash" --out $R/vlm104/pcaw_vs_preference.json
```

`analyze` prints the Pearson and Spearman correlations of exact sample-level PCA-W with the three VLM
scores, pooled over the 1,300 method–image observations and after centring each method.
`pcaw_vlm_partial.py` prints, for each score, the partial Spearman correlation controlling for
prediction correctness, corruption type and method (pooled over both methods) and within each method
(controlling for correctness and corruption type), and the Spearman correlation among correctly
classified observations (pooled over both methods). `pcaw_vs_preference.py` correlates Gemini's
104-image preference with the PCA-W difference between ProtoTTA and EATA, on all images and on those
where both methods are equally correct (item bootstrap CIs, 2,000 resamples).
