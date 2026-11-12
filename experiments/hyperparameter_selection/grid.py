"""Search grids and test streams of the hyperparameter re-selection on ProtoViT/CUB-200-C (Appendix M).

ProtoTTA: filter threshold {0.86, 0.89, 0.92, 0.94, 0.96} x learning rate {3e-4, 1e-3, 3e-3}, all other
settings as in Appendix L. EATA: entropy margin E0 in {0.2, ..., 0.6} x ln C x the same learning rates,
plus its published configuration (E0 = ln(1000)/2 - 1, lr 1e-3). Each method keeps its own objective,
filter and updated parameters. Candidates are listed in tie-break order: among tied configurations the
lower learning rate is chosen, then the earlier entry of this list.
"""
MODEL = "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
PROTOTTA = ["--proto-shared-confidence-weighting", "--proto-gradient-normalize",
            "--proto-adaptive-delta0", "0.25", "--proto-adaptive-topk", "3",
            "--proto-router-min-consistency", "0.25", "--proto-lambda-ema-momentum", "0",
            "--proto-lambda-min", "0", "--proto-lambda-max", "1", "--proto-record-diagnostics",
            "--modes", "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
            "--proto-adaptive-strategy", "source_free_router_coverage_absolute"]
LRS = ("3e-4", "1e-3", "3e-3")
THRESHOLDS = ("0.86", "0.89", "0.92", "0.94", "0.96")
E0 = ("0.2", "0.3", "0.4", "0.5", "0.6")
# Test streams: (batch size, stream arguments). The mixed stream pools all 13 corruptions into one
# shuffled stream without resets (corruptions listed in the ImageNet-C order of the families).
ORDER13 = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "defocus_blur",
           "gaussian_blur", "frost", "fog", "brightness", "contrast", "elastic_transform",
           "pixelate", "jpeg_compression"]
SHUF = ["--stream-order", "shuffled"]
STREAMS = {
    "shuffled": (128, SHUF),
    "mixed": (128, ["--stream", "mixed"] + SHUF + ["--corruptions"] + ORDER13),
    "bs16": (16, SHUF),
}


def grid():
    """[(method, config_id, knobs, args)] in tie-break order within each method."""
    out = []
    for lr in LRS:
        for t in THRESHOLDS:
            out.append(("prototta", f"prototta_t{t}_lr{lr}", {"threshold": t, "lr": lr},
                        PROTOTTA + ["--proto-geo-filter-threshold", t, "--proto-lr", lr]))
    for lr in LRS:
        for e in E0:
            out.append(("eata", f"eata_e{e}_lr{lr}", {"e0_factor": e, "lr": lr},
                        ["--modes", "eata", "--baseline-lr", lr, "--eata-e-margin-factor", e]))
    out.append(("eata", "eata_published", {"e0": "ln(1000)/2-1", "lr": "1e-3"}, ["--modes", "eata"]))
    return out


def ordered(method):
    """Config ids of a method sorted by (learning rate, list position): the tie-break order."""
    g = [(i, c, k) for i, (m, c, k, _) in enumerate(grid()) if m == method]
    return [c for i, c, k in sorted(g, key=lambda x: (float(x[2]["lr"]), x[0]))]


def evaluator_args(data_dir, stream, config_args, seed, output, model=MODEL):
    bs, stream_args = STREAMS[stream]
    return (["--model", model, "--gpuid", "0", "--data_dir", data_dir, "--batch_size", str(bs)]
            + list(config_args) + list(stream_args) + ["--seed", str(seed), "--output", str(output)])
