# ProtoTTA: Anonymous Code Artifact

This repository contains the code for the double-blind submission *ProtoTTA: Explainable Test-Time
Adaptation via Prototype Guidance*. It reproduces the results reported in the paper on four prototype
backbones (ProtoViT on CUB-200-C, ProtoPNet on SICAPv2-C, ProtoPFormer on Stanford Dogs-C, and ProtoLens
on Amazon-C) and the explanation evaluations (part annotations, blinded human and VLM comparisons,
narrator validation). Datasets, checkpoints, and generated results are not included.

Adaptation never uses target labels. Ground-truth labels were used to select hyperparameters
(Appendices L and M); otherwise they are used, together with clean counterparts and part annotations,
only by offline evaluation scripts.

## Layout

| Path | Contents |
|---|---|
| `ProtoViT/` | ProtoViT / CUB-200-C: ProtoTTA, baselines, metrics, VLM scoring and narration, judge intervention |
| `ProtoViT/explanation_eval/` | Part annotations and clean-reference measures, human study and blinded VLM comparison, narrator validation, PCA-W–VLM correlation |
| `ProtoPFormer/` | ProtoPFormer / Stanford Dogs-C |
| `ProtoPNetVLM/protopnet_tta/` | ProtoPNet / SICAPv2-C |
| `ProtoLens/` | ProtoLens / Amazon-C (shared-prototype extension) |
| `experiments/` | Command generators and analyses for the controller ablation, hyperparameter re-selection, timing, and calibrate-once mode |
| `ProtoTTA/` | Model-agnostic ProtoTTA implementation and integration guide, with unit tests |
| `protovit_env.yml`, `vlm_environment.yml`, `flsh_att_environment.yml` | Conda environments (main vision stack; VLM judges and narrators; optional Flash Attention) |

Each directory has a README with the exact command for every result it covers. The main-result commands
use the configuration of Appendix L (for example, λ in [0, 1] and a shared confidence weight for both
loss terms); ablation commands change only the component under study. Seeds are 0, 2, and 3 unless
stated otherwise. Paths to data, checkpoints, and outputs are command-line arguments or, where a README
says so, relative to the backbone directory.

## Installation

```bash
conda env create -f protovit_env.yml        # adaptation, metrics, analyses
conda env create -f vlm_environment.yml     # VLM judges, narrators, judge intervention
```

The ProtoPNet code additionally needs the upstream ProtoPNet model package (`proto_baseline`) on
`PYTHONPATH`; see `ProtoPNetVLM/protopnet_tta/README.md`.

## Data and checkpoints

Download CUB-200-2011, Stanford Dogs, SICAPv2, and the review data from their official providers. The
corruption benchmarks are generated with the included scripts (Appendix A):
`ProtoViT/create_cub_c.py`, `ProtoPFormer/create_dogs_c.py`, `ProtoPNetVLM/protopnet_tta/create_sicap_c.py`,
and `ProtoLens/generate_amazon_c.py`. Each backbone README describes how its source model is trained or
obtained.

## Where each result is reproduced

| Paper element | Where |
|---|---|
| Table 1, Table 2, Tables 6–7 (per-corruption results) | `ProtoViT/README.md`, `ProtoPFormer/README.md`, `ProtoPNetVLM/protopnet_tta/README.md`, `ProtoLens/README.md` |
| Table 3 (left): objective × reliability filter; adapted parameters | `ProtoViT/README.md`; `experiments/controller_ablation/` |
| Table 3 (right): controller components on four backbones | backbone READMEs; `experiments/controller_ablation/` (commands and table) |
| Table 4 and Appendix O: part annotations and closeness to the clean explanation | `ProtoViT/explanation_eval/` |
| Table 5 and Appendix P: human study, blinded VLM comparison, narrator validation | `ProtoViT/explanation_eval/` |
| Table 8 (Appendix C): label-free judge intervention | backbone READMEs |
| Table 9, Figures 2 and 5 (Appendix E.1): VLM scoring and narration | `ProtoViT/README.md` (`vlm_eval.py`) |
| Appendix E.3: PCA-W and VLM correlation | `ProtoViT/explanation_eval/` |
| Table 10 (fixed-λ sweep) and Table 11 (design ablations) | backbone READMEs (Table 11: `ProtoViT/README.md`) |
| Figures 3 and 6–8 (Appendices J and K) | `ProtoViT/README.md` |
| Table 12 (Appendix M): hyperparameter re-selection | `experiments/hyperparameter_selection/` |
| Table 13 and per-image overhead (Appendix N) | `experiments/timing/`, `experiments/calibrate_once/` |

## Tests

```bash
python -m unittest -v ProtoTTA.tests.test_adapter
python -m compileall -q ProtoTTA ProtoViT ProtoPFormer ProtoPNetVLM ProtoLens experiments
```

## Double-blind note

Citation and contact information are withheld for review.
