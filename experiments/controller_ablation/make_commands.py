#!/usr/bin/env python
"""Print the commands of the controller-component and objective x filter ablations (Table 3; Appendix N).

Every command is the backbone's main ProtoTTA command (Appendix L) plus exactly the variant's
arguments; each variant runs with seeds 0, 2 and 3 on all corruption settings. Output files:
<results>/<backbone>/<variant>_seed<s>.json (read by analyze.py).

Variants (Table 3, right; all backbones):
  router              full controller (ProtoTTA, adaptive lambda)
  act_only            lambda^act only
  rel_only            lambda^rel only
  mean                mean of the two estimates
  no_coverage_guard   routing without the coverage guard
  no_grad_norm        no gradient normalization
  fixed_<lambda>      fixed lambda, same filter, parameters and optimizer: lambda = 0, 0.5, 1 on ProtoViT,
                      ProtoPFormer and ProtoLens; on ProtoPNet the fixed-lambda sweep of Table 10
                      (--modes ProtoHybrid, lambda = 0, 0.2, 0.4, 0.5, 0.7, 0.8, 1)
Objective x reliability filter (Table 3, left; ProtoViT):
  output_nofilter, output_entropyfilter   output entropy (lambda = 0) without filter / with the EATA rule
  router_nofilter, router_entropyfilter   ProtoTTA adaptive objective without filter / with the EATA rule
  (the "Ours" filter cells are fixed_0 and router)

  python make_commands.py --results /path/to/results > commands.sh
Commands are written relative to the repository root (--repo); paths of data and checkpoints are
options. ProtoPNet commands need the upstream proto_baseline package on PYTHONPATH (--protopnet-pythonpath).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SEEDS, VISION_CORRUPTIONS, command  # noqa: E402

# ---------------------------------------------------------------- ProtoViT (CUB-200-C)
PV_TAIL = ["--gpuid", "0", "--batch_size", "128",
           "--prototype-metrics", "--use-enhanced-metrics", "--proto-baseline-samples", "5794",
           "--track-efficiency", "--proto-shared-confidence-weighting", "--proto-gradient-normalize",
           "--proto-adaptive-delta0", "0.25", "--proto-adaptive-topk", "3",
           "--proto-router-min-consistency", "0.25", "--proto-lambda-ema-momentum", "0",
           "--proto-lambda-min", "0", "--proto-lambda-max", "1", "--proto-record-diagnostics"]
PV_R = ["--modes", "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
        "--proto-adaptive-strategy", "source_free_router_coverage_absolute"]
PV_SW = ["--modes", "proto_imp_conf_adaptive_samplewise", "--proto-adaptive-strategy"]
PV_FIX = ["--modes", "proto_imp_conf_v3", "--proto-lambda"]
PV = {
    "router": PV_R,
    "act_only": PV_SW + ["activation_margin"],
    "rel_only": PV_SW + ["relative_evidence"],
    "mean": PV_SW + ["mean_act_rel"],
    "no_coverage_guard": ["--modes", "proto_imp_conf_adaptive_source_free_router_absolute",
                          "--proto-adaptive-strategy", "source_free_router_absolute"],
    "no_grad_norm": PV_R + ["--proto-gradient-normalize-override", "off"],
    "fixed_0": PV_FIX + ["0.0"], "fixed_0.5": PV_FIX + ["0.5"], "fixed_1": PV_FIX + ["1.0"],
    "output_nofilter": PV_FIX + ["0.0", "--proto-reliability-filter", "none"],
    "output_entropyfilter": PV_FIX + ["0.0", "--proto-reliability-filter", "entropy"],
    "router_nofilter": PV_R + ["--proto-reliability-filter", "none"],
    "router_entropyfilter": PV_R + ["--proto-reliability-filter", "entropy"],
}

# ---------------------------------------------------------------- ProtoPFormer (Stanford Dogs-C)
PF_TAIL = ["--severity", "5", "--batch_size", "128", "--num_workers", "8", "--gpuid", "0",
           "--adapt_model_mode", "train", "--lr", "1e-3", "--steps", "1",
           "--prototype-metrics", "--use-enhanced-metrics", "--proto-baseline-samples", "8580",
           "--track-efficiency", "--proto_threshold", "0.55", "--proto_mapping", "sigmoid",
           "--proto_sigmoid_center", "1.0", "--proto_sigmoid_temp", "1.0", "--proto_no_importance",
           "--proto_branch", "both", "--proto_shared_confidence_weighting", "--proto_gradient_normalize",
           "--proto_adaptive_delta0", "0.25", "--proto_adaptive_topk", "3",
           "--proto_router_min_consistency", "0.25", "--proto_lambda_ema_momentum", "0",
           "--proto_lambda_min", "0", "--proto_lambda_max", "1", "--proto_record_diagnostics"]
PF_R = ["--modes", "proto_tta_adaptive_source_free_router_coverage_absolute",
        "--proto_adaptive_strategy", "source_free_router_coverage_absolute"]
PF_SW = ["--modes", "proto_tta_adaptive_samplewise", "--proto_adaptive_strategy"]
PF_FIX = ["--modes", "proto_tta", "--proto_lambda"]
PF = {
    "router": PF_R,
    "act_only": PF_SW + ["activation_margin"],
    "rel_only": PF_SW + ["relative_evidence"],
    "mean": PF_SW + ["mean_act_rel"],
    "no_coverage_guard": ["--modes", "proto_tta_adaptive_source_free_router_absolute",
                          "--proto_adaptive_strategy", "source_free_router_absolute"],
    "no_grad_norm": PF_R + ["--proto_gradient_normalize_override", "off"],
    "fixed_0": PF_FIX + ["0.0"], "fixed_0.5": PF_FIX + ["0.5"], "fixed_1": PF_FIX + ["1.0"],
}

# ---------------------------------------------------------------- ProtoLens (Amazon-C)
PL_BASE = ["--methods", "prototta", "--learning_rate", "0.000005",
           "--adaptation_mode", "layernorm_attn_bias", "--geo_threshold", "0.1",
           "--sigmoid_temperature", "5.0", "--batch_size", "128", "--prototype-metrics",
           "--track-efficiency", "--proto_adaptive_delta0", "0.25", "--proto_adaptive_topk", "3",
           "--proto_router_min_consistency", "0.25", "--proto_lambda_ema_momentum", "0",
           "--proto_lambda_min", "0", "--proto_lambda_max", "1", "--proto_record_diagnostics"]
PL_AD = ["--proto_lambda", "0.5", "--proto_adaptive_lambda", "--proto_samplewise_lambda",
         "--proto_adaptive_strategy"]
PL_GN = ["--proto_gradient_normalize"]
PL_R = PL_AD + ["source_free_router_coverage_absolute"]
PL_FIX = ["--proto_gradient_normalize", "--proto_fixed_lambda_gradnorm", "--proto_lambda"]
PL = {
    "router": ["--geo_filter"] + PL_R + PL_GN,
    "act_only": ["--geo_filter"] + PL_AD + ["activation_margin"] + PL_GN,
    "rel_only": ["--geo_filter"] + PL_AD + ["relative_evidence"] + PL_GN,
    "mean": ["--geo_filter"] + PL_AD + ["mean_act_rel"] + PL_GN,
    "no_coverage_guard": ["--geo_filter"] + PL_AD + ["source_free_router_absolute"] + PL_GN,
    "no_grad_norm": ["--geo_filter"] + PL_R,
    "fixed_0": ["--geo_filter"] + PL_FIX + ["0.0"],
    "fixed_0.5": ["--geo_filter"] + PL_FIX + ["0.5"],
    "fixed_1": ["--geo_filter"] + PL_FIX + ["1.0"],
}

# ---------------------------------------------------------------- ProtoPNet (SICAPv2-C; TITAN RTX)
PPN = {
    "router": ["--modes", "ProtoAbsoluteConsistencyCoverageRouter"],
    "act_only": ["--modes", "ProtoAuditForcedNativeGradNorm"],
    "rel_only": ["--modes", "ProtoAuditForcedRelativeGradNorm"],
    "mean": ["--modes", "ProtoMeanActRelGradNorm"],
    "no_coverage_guard": ["--modes", "ProtoAbsoluteOnlyRouter"],
    "no_grad_norm": ["--modes", "ProtoRouterNoGradNorm"],
}

PPN_SWEEP = ("0.0", "0.2", "0.4", "0.5", "0.7", "0.8", "1.0")   # Table 10 (ProtoHybrid)
TABLE_RIGHT = ("router", "act_only", "rel_only", "mean", "no_coverage_guard", "no_grad_norm",
               "fixed_0", "fixed_0.5", "fixed_1")
TABLE_LEFT = ("output_nofilter", "output_entropyfilter", "router_nofilter", "router_entropyfilter")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="output root (made absolute in the printed commands)")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--python", default="python")
    ap.add_argument("--backbones", nargs="+", default=["protovit", "protopnet", "protopformer", "protolens"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--protovit-model", default="saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
    ap.add_argument("--protovit-data", default="./datasets/cub200_c/")
    ap.add_argument("--protopformer-model", default="/path/to/protopformer_dogs_checkpoint.pth")
    ap.add_argument("--protopformer-data", default="/path/to/stanford_dogs_c")
    ap.add_argument("--protopnet-model", default="/path/to/protopnet_checkpoint.pth")
    ap.add_argument("--protopnet-data", default="/path/to/SICAPv2_c_rebuilt_seed0")
    ap.add_argument("--protopnet-clean-data", default="/path/to/SICAPv2/test")
    ap.add_argument("--protopnet-pythonpath", default="/path/to/dir_containing_proto_baseline")
    a = ap.parse_args()
    repo, res = Path(a.repo).resolve(), Path(a.results).resolve()
    for b in a.backbones:
        for v in (TABLE_RIGHT + (TABLE_LEFT if b == "protovit" else ())):
            for s in a.seeds:
                out = res / b / f"{v}_seed{s}.json"
                if b == "protovit" and v in PV:
                    args = (["--model", a.protovit_model, "--data_dir", a.protovit_data] + PV_TAIL + PV[v]
                            + ["--seed", str(s), "--output", out])
                    print(command(repo / "ProtoViT", ["evaluate_robustness.py"], args, a.python))
                elif b == "protopformer" and v in PF:
                    args = (["--model", a.protopformer_model, "--data_dir", a.protopformer_data] + PF_TAIL + PF[v]
                            + ["--seed", str(s), "--output", out])
                    print(command(repo / "ProtoPFormer", ["evaluate_robustness_dogs.py"], args, a.python))
                elif b == "protolens" and v in PL:
                    args = PL_BASE + PL[v] + ["--seed", str(s), "--output", out]
                    print(command(repo / "ProtoLens", ["evaluate_robustness_amazonc.py"], args, a.python,
                                  env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
                elif b == "protopnet" and v in PPN:
                    args = (["--model", a.protopnet_model, "--data_dir", a.protopnet_data,
                             "--clean_data_dir", a.protopnet_clean_data, "--severity", "5", "--batch_size", "64",
                             "--gpuid", "0", "--corruptions", *VISION_CORRUPTIONS,
                             "--lambda-proto", "0.2", "--adaptive-delta0", "0.25", "--adaptive-top-k", "3",
                             "--prototype-metrics", "--track-efficiency",
                             "--controlled-cohort-name", "rebuilt_seed0_sicapv2c_controlled_four_seed",
                             "--refuse-existing-output"] + PPN[v]
                            + ["--seed", str(s), "--stream-order-seed", str(s), "--output", out,
                               "--clean-reference-cache", res / b / "clean_cache" / f"{v}_seed{s}.pt"])
                    print(command(repo / "ProtoPNetVLM", ["-m", "protopnet_tta.evaluate_robustness"], args, a.python,
                                  env={"PYTHONPATH": f".:{a.protopnet_pythonpath}"}))
        if b == "protopnet":
            for lam in PPN_SWEEP:
                for s in a.seeds:
                    out = res / b / f"fixed_{float(lam):g}_seed{s}.json"
                    args = (["--model", a.protopnet_model, "--data_dir", a.protopnet_data,
                             "--clean_data_dir", a.protopnet_clean_data, "--severity", "5", "--batch_size", "64",
                             "--gpuid", "0", "--corruptions", *VISION_CORRUPTIONS,
                             "--adaptive-delta0", "0.25", "--adaptive-top-k", "3", "--lambda-proto", lam,
                             "--prototype-metrics", "--track-efficiency", "--modes", "ProtoHybrid",
                             "--seed", str(s), "--stream-order-seed", str(s), "--output", out])
                    print(command(repo / "ProtoPNetVLM", ["-m", "protopnet_tta.evaluate_robustness"], args, a.python,
                                  env={"PYTHONPATH": f".:{a.protopnet_pythonpath}"}))


if __name__ == "__main__":
    main()
