# Experiments of Appendices M and N

Command generators and analyses for the controller ablation (Table 3), the hyperparameter re-selection
on ProtoViT (Table 12), the timing results (Table 13 and the per-image overhead of the controller), and
the calibrate-once mode. Each generator prints portable shell commands (one per line) that call the
backbone evaluators with the configuration of Appendix L; run them from the repository root, collect the
`--output` JSON files, and run the analysis on the output directory. All accuracies are the mean over the
corruption settings of the seed-averaged per-setting accuracy (seeds 0, 2 and 3).

| Directory | Paper |
|---|---|
| `controller_ablation/` | Table 3 (left: objective × reliability filter on ProtoViT; right: controller components on all four backbones, with the largest-shortfall column), Appendix N |
| `hyperparameter_selection/` | Table 12, Appendix M |
| `timing/` | Table 13 (speed) and the per-image overhead of the full controller on one B200 GPU, Appendix N |
| `calibrate_once/` | calibrate-once mode on ProtoViT (Table 13), Appendix N |

Data and checkpoint paths are options of the generators (`--help`); ProtoViT paths default to the
standard layout under `ProtoViT/`. The ProtoPNet commands need the upstream `proto_baseline` package
(`--protopnet-pythonpath`).

## Controller components and objective × filter (Table 3)

```bash
python experiments/controller_ablation/make_commands.py --results $PWD/results/table3 \
    --protopformer-model /path/to/protopformer_dogs_checkpoint.pth --protopformer-data /path/to/stanford_dogs_c \
    --protopnet-model /path/to/protopnet_checkpoint.pth --protopnet-data /path/to/SICAPv2_c_rebuilt_seed0 \
    --protopnet-clean-data /path/to/SICAPv2/test --protopnet-pythonpath /path/to/dir_containing_proto_baseline \
    > table3.sh
bash table3.sh
```

Every variant is the backbone's ProtoTTA command (Appendix L) with the same reliability filter, updated
parameters, optimizer and learning rate, and one change:

| Row of Table 3 | Variant | Change |
|---|---|---|
| Router (ours) | `router` | none (full controller) |
| λ^act only | `act_only` | λ_i = λ^act_i, no routing |
| λ^rel only | `rel_only` | λ_i = λ^rel_i, no routing |
| Mean of the two | `mean` | λ_i = (λ^act_i + λ^rel_i)/2 |
| No coverage guard | `no_coverage_guard` | gradient-consistency routing without the coverage guard |
| No gradient norm. | `no_grad_norm` | no scaling by g_proto and g_out |
| Fixed λ = 0, 0.5, 1 | `fixed_0`, `fixed_0.5`, `fixed_1` | the same λ for every sample (ProtoPNet: the fixed-λ sweep of Table 10, λ ∈ {0, 0.2, 0.4, 0.5, 0.7, 0.8, 1}) |
| Table 3 (left), ProtoViT | `output_nofilter`, `output_entropyfilter` | output entropy (λ = 0) without filter / with the EATA rule (output entropy below 0.4 ln C) |
| | `router_nofilter`, `router_entropyfilter` | ProtoTTA adaptive objective without filter / with the EATA rule |

The "Ours" filter cells of Table 3 (left) are `fixed_0` and `router`.

```bash
python experiments/controller_ablation/analyze.py --results results/table3 \
    --row protovit:router <Table 1 ProtoViT runs, seeds 0 2 3> \
    --row protopformer:router <Table 1 ProtoPFormer runs> \
    --row protolens:router <Table 1 ProtoLens runs> \
    --row protolens:fixed_1 <Table 1 ProtoLens fixed-λ runs> \
    --sweep protovit:<λ> <Table 10 ProtoViT λ runs> ... \
    --sweep protopformer:<λ> <Table 10 ProtoPFormer λ runs> ...
```

The router row and the fixed-λ rows of Table 3 are the runs of Table 1 and of the fixed-λ sweep of
Table 10 (same commands as `router` and `fixed_*`), passed with `--row` (one file per seed) and
`--sweep BACKBONE:λ` (one file per seed, one option per λ); for ProtoPNet these are the `router` and
`fixed_<λ>` runs printed above. "Best fixed λ" is the best λ of the sweep of each backbone. The last
column is the largest shortfall of the row from the best row on any backbone.

## Hyperparameter re-selection (Table 12, Appendix M)

Grids (`hyperparameter_selection/grid.py`): ProtoTTA filter threshold {0.86, 0.89, 0.92, 0.94, 0.96} ×
learning rate {3e-4, 1e-3, 3e-3}; EATA E0 ∈ {0.2, 0.3, 0.4, 0.5, 0.6} · ln C × the same learning rates,
plus its published configuration. Among tied configurations, the lower learning rate is chosen.
Streams: `shuffled` (batch size 128, model reset for each corruption), `mixed` (all 13 corruptions
pooled into one shuffled stream without resets), `bs16` (the shuffled stream with batch size 16).

```bash
H=experiments/hyperparameter_selection
# select on three families / test on one, and select on one family / test on three
python $H/make_commands.py families --results $PWD/results/hp/families > families.sh && bash families.sh
python $H/analyze_families.py --results results/hp/families

# 20% class-stratified validation split of the photos, fixed across corruptions
python $H/make_valsplit.py --src ProtoViT/datasets/cub200_c --val ProtoViT/datasets/cub200_c_val \
    --test ProtoViT/datasets/cub200_c_test --manifest results/hp/valsplit.json
python $H/make_commands.py valsplit-select --results $PWD/results/hp/val > val.sh && bash val.sh
python $H/analyze_valsplit.py select --results results/hp/val --out results/hp/selection.json
python $H/make_commands.py valsplit-test --results $PWD/results/hp/test --selection results/hp/selection.json \
    > test.sh && bash test.sh
python $H/analyze_valsplit.py test --results results/hp/test --selection results/hp/selection.json
```

`analyze_families.py` selects, for each seed, family and method, the configuration with the highest
mean accuracy on the selection corruptions, and prints ProtoTTA − EATA (mean over the 13 corruptions
and the three seeds) for "select on 3 families, test on 1" and "select on 1 family, test on 3" in each
stream, and their averages. `analyze_valsplit.py` selects on the validation photos (seed 0) and
reports ProtoTTA − EATA on the remaining 80% of the photos (seeds 0, 2, 3).

## Timing (Table 13 and per-image overhead, Appendix N)

```bash
T=experiments/timing
# Table 13: one job on one GPU, five repetitions, method order alternating across repetitions
python $T/make_commands.py table13 --results $PWD/results/timing/table13 > table13.sh && bash table13.sh
python $T/summarize_table13.py --results results/timing/table13
# per-image overhead of the full controller (one B200 GPU; two repetitions)
python $T/make_commands.py overhead --results $PWD/results/timing/overhead \
    --protopformer-model ... --protopformer-data ... --protopnet-model ... --protopnet-data ... \
    --protopnet-clean-data ... --protopnet-pythonpath ... > overhead.sh && bash overhead.sh
python $T/summarize_overhead.py --results results/timing/overhead
```

Timing runs use three corruptions (Gaussian noise, defocus blur, fog), seed 0, no interpretability
metrics, and `efficiency.time_per_sample_ms` (forward and adaptation, CUDA-synchronized, data loading
excluded); EATA's Fisher estimation pass is not included. Table 13 reports the speed relative to EATA
(EATA time / method time within each repetition; mean ± SD over the five repetitions); the overhead is
the time per image of ProtoTTA minus that of the unadapted model (median over repetitions per
corruption, mean over corruptions). The accuracies of Table 13 are those of the 13-corruption runs:
ProtoTTA with adaptive λ (Table 1), fixed λ = 1 (Table 10), calibrate-once (below) and EATA (Table 1).

## Calibrate-once (Appendix N)

`calibrate_once/fastmodes.py` runs the unchanged ProtoViT evaluator with the ProtoTTA command and
freezes the routing decision and the two gradient scales after the first reliable batch of each
corruption stream, so every later batch needs a single backward pass. It adds no hyperparameter.

```bash
C=experiments/calibrate_once
python $C/make_commands.py --results $PWD/results/calibrate_once > calibrate_once.sh && bash calibrate_once.sh
python $C/summarize_accuracy.py results/calibrate_once/calibrate_once_seed{0,2,3}.json
```

A single run has the form

```bash
cd ProtoViT && python -u /path/to/experiments/calibrate_once/fastmodes.py --fast-mode calibrate_once \
    --backbone protovit -- evaluate_robustness.py <ProtoTTA arguments of Appendix L> --seed 0 --output out.json
```
