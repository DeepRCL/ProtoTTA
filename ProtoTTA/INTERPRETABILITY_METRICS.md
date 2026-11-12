# Interpretability and Operational Metrics

This page restates the metrics reported in Table 2 of the paper (Section 3.5,
Appendices E.2 and E.4) for use with a new model. Clean counterparts and
ground-truth labels are used only for offline evaluation and never during
adaptation.

For sample `i`, let `a_i^clean` and `yhat_i^clean` be the prototype activation
vector and prediction of the unadapted model on the clean input, and
`a_i^adapted` and `yhat_i^adapted` the corresponding quantities on the
corrupted input after adaptation. `N` is the number of evaluated samples.

## PCA-W (Eq. 8)

Let `T_i` contain the `k = 10` most activated prototypes on the corrupted
input, `a_ip` the activation of prototype `p`, `W_{y_i,p}` its classifier
weight for the ground-truth class `y_i`, `c_ip = a_ip |W_{y_i,p}|`, and
`class(p)` the class of prototype `p`:

```
PCA-W (%) = 100/N * sum_i [ sum_{p in T_i} c_ip 1[class(p) = y_i] / sum_{p in T_i} c_ip ]
```

For shared prototypes (ProtoLens, Eqs. 14-15), the ten largest similarities
are normalized with a softmax within `T_i`, `alpha_ip`, and

```
PCA-W_PL(i) = sum_{p in T_i} alpha_ip [W_{y_i,p}]_+ / ( sum_{p in T_i} alpha_ip |W_{y_i,p}| + 1e-8 )
```

PCA-W needs only the corrupted input and its label.

## PAC (Eq. 19)

```
PAC (%) = 100/N * sum_i cos(a_i^clean, a_i^adapted)
```

## Prediction stability (Eq. 20)

```
Prediction stability (%) = 100/N * sum_i 1[yhat_i^adapted = yhat_i^clean]
```

## Operational metrics

- **Selection rate**: percentage of test samples selected for adaptation
  updates (for the adapter,
  `100 * stats["adapted_samples"] / stats["seen_samples"]`).
- **Relative speed**: inference throughput of the adapted model as a
  percentage of the unadapted model's throughput.

## Reported values

The four backbone evaluators compute these quantities when run with
`--prototype-metrics` (with `--use-enhanced-metrics` for ProtoViT and
ProtoPFormer) and `--track-efficiency`; the commands are in the backbone
READMEs listed in [`EXISTING_BACKBONES.md`](./EXISTING_BACKBONES.md).

## Reasoning boards and VLM scoring

For a prediction, a reasoning board contains the corrupted input, the
predicted-class prototype matches with their spatial focus, and the strongest
any-class contributions. In the scoring protocol of
Appendix E.1, a VLM scores each board from 1 to 5 for Focus Relevance,
Prototype Match, and Overall Adaptation Quality; the same agent can narrate
the change between unadapted and adapted boards (Figure 2). For ProtoViT this
is implemented in [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py). The
blinded comparisons, part-annotation measures, and PCA-W/VLM correlation of
Section 5 and Appendices E.3, O, and P are implemented in
[`ProtoViT/explanation_eval/`](../ProtoViT/explanation_eval/).
