# Backbone Implementations

The results reported in the paper were produced with the four
backbone-specific implementations below. Each backbone README gives the
complete commands for every table and figure that uses the backbone; the
main-result commands use the Appendix L configuration, and seeds are 0, 2,
and 3 unless stated otherwise.

| Backbone / benchmark | Evaluation entry point | ProtoTTA implementation | Commands |
|---|---|---|---|
| ProtoViT / CUB-200-C | [`ProtoViT/evaluate_robustness.py`](../ProtoViT/evaluate_robustness.py) | [`ProtoViT/proto_entropy.py`](../ProtoViT/proto_entropy.py) (`ProtoEntropy`) | [`ProtoViT/README.md`](../ProtoViT/README.md) |
| ProtoPNet / SICAPv2-C | [`ProtoPNetVLM/protopnet_tta/evaluate_robustness.py`](../ProtoPNetVLM/protopnet_tta/evaluate_robustness.py) (run as `python -m protopnet_tta.evaluate_robustness`) | [`ProtoPNetVLM/protopnet_tta/proto_entropy_enhanced.py`](../ProtoPNetVLM/protopnet_tta/proto_entropy_enhanced.py) | [`ProtoPNetVLM/protopnet_tta/README.md`](../ProtoPNetVLM/protopnet_tta/README.md) |
| ProtoPFormer / Stanford Dogs-C | [`ProtoPFormer/evaluate_robustness_dogs.py`](../ProtoPFormer/evaluate_robustness_dogs.py) | [`ProtoPFormer/proto_tta.py`](../ProtoPFormer/proto_tta.py) (`ProtoTTA`) | [`ProtoPFormer/README.md`](../ProtoPFormer/README.md) |
| ProtoLens / Amazon-C | [`ProtoLens/evaluate_robustness_amazonc.py`](../ProtoLens/evaluate_robustness_amazonc.py) | [`ProtoLens/proto_tta.py`](../ProtoLens/proto_tta.py) | [`ProtoLens/README.md`](../ProtoLens/README.md) |

## Method selection

ProtoTTA with adaptive `lambda` (full controller):

| Backbone | Arguments |
|---|---|
| ProtoViT | `--modes proto_imp_conf_adaptive_source_free_router_coverage_absolute --proto-adaptive-strategy source_free_router_coverage_absolute` |
| ProtoPNet | `--modes ProtoAbsoluteConsistencyCoverageRouter` |
| ProtoPFormer | `--modes proto_tta_adaptive_source_free_router_coverage_absolute --proto_adaptive_strategy source_free_router_coverage_absolute` |
| ProtoLens | `--methods prototta --geo_filter --proto_lambda 0.5 --proto_adaptive_lambda --proto_samplewise_lambda --proto_adaptive_strategy source_free_router_coverage_absolute --proto_gradient_normalize` |

These arguments select the method only. The configuration of Appendix L
(for example `lambda` in `[0, 1]` via `--proto-lambda-min 0
--proto-lambda-max 1` and the shared confidence weight via
`--proto-shared-confidence-weighting` on ProtoViT) is set by the additional
flags in the backbone README commands, which must be used as given. The
fixed-`lambda` variant and the controller ablations are listed in the same
READMEs.

## Explanation evaluation

The reasoning boards, VLM scores, and narration of Section 3.5 and
Appendix E.1 (Table 9, left; Figure 2) are produced by
[`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py). The part-annotation and
clean-reference measures (Table 4, Appendix O), the blinded human and VLM
comparisons and narrator validation (Table 5, Appendix P), and the PCA-W/VLM
correlation (Table 9, right; Appendix E.3) are implemented in
[`ProtoViT/explanation_eval/`](../ProtoViT/explanation_eval/); the commands
are in [`ProtoViT/README.md`](../ProtoViT/README.md).
