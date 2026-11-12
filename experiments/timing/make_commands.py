#!/usr/bin/env python
"""Print the timing commands of Appendix N (one GPU; run the printed commands sequentially in one job).

  table13   Table 13 (ProtoViT/CUB-200-C): unadapted, EATA, ProtoTTA with adaptive lambda, ProtoTTA with
            fixed lambda = 1, and ProtoTTA calibrate-once (../calibrate_once/fastmodes.py), five
            repetitions; the order of the methods alternates between forward and reverse across
            repetitions. -> <results>/<method>_rep<r>.json (summarize_table13.py)
  overhead  per-image overhead of the full controller: unadapted and ProtoTTA (adaptive lambda) on
            ProtoViT, ProtoPFormer and ProtoPNet, two repetitions (second in reverse order).
            -> <results>/<backbone>/<method>_rep<r>.json (summarize_overhead.py)
Timing protocol: three corruptions (gaussian_noise, defocus_blur, fog; severity 5), seed 0, no
interpretability metrics, efficiency tracking on (efficiency.time_per_sample_ms: forward + adaptation,
CUDA-synchronized, data loading excluded).

  python make_commands.py table13 --results /path/to/timing_table13 > table13.sh
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import command  # noqa: E402

SUBSET = ["--corruptions", "gaussian_noise", "defocus_blur", "fog"]
PV_BASE = ["--gpuid", "0", "--batch_size", "128", "--track-efficiency",
           "--proto-shared-confidence-weighting", "--proto-gradient-normalize",
           "--proto-adaptive-delta0", "0.25", "--proto-adaptive-topk", "3",
           "--proto-router-min-consistency", "0.25", "--proto-lambda-ema-momentum", "0",
           "--proto-lambda-min", "0", "--proto-lambda-max", "1"]
PV_ROUTER = ["--modes", "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
             "--proto-adaptive-strategy", "source_free_router_coverage_absolute"]
PV_METHODS = {"unadapted": ["--modes", "normal"], "eata": ["--modes", "eata"], "adaptive": PV_ROUTER,
              "fixed_1": ["--modes", "proto_imp_conf_v3", "--proto-lambda", "1.0"], "calibrate_once": PV_ROUTER}
TABLE13 = ("unadapted", "eata", "adaptive", "fixed_1", "calibrate_once")
PF_BASE = ["--severity", "5", "--batch_size", "128", "--num_workers", "8", "--gpuid", "0",
           "--adapt_model_mode", "train", "--lr", "1e-3", "--steps", "1", "--track-efficiency",
           "--proto_threshold", "0.55", "--proto_mapping", "sigmoid", "--proto_sigmoid_center", "1.0",
           "--proto_sigmoid_temp", "1.0", "--proto_no_importance", "--proto_branch", "both",
           "--proto_shared_confidence_weighting", "--proto_gradient_normalize",
           "--proto_adaptive_delta0", "0.25", "--proto_adaptive_topk", "3",
           "--proto_router_min_consistency", "0.25", "--proto_lambda_ema_momentum", "0",
           "--proto_lambda_min", "0", "--proto_lambda_max", "1"]
PF_METHODS = {"unadapted": ["--modes", "normal"],
              "adaptive": ["--modes", "proto_tta_adaptive_source_free_router_coverage_absolute",
                           "--proto_adaptive_strategy", "source_free_router_coverage_absolute"]}
PPN_METHODS = {"unadapted": ["--modes", "Normal"], "adaptive": ["--modes", "ProtoAbsoluteConsistencyCoverageRouter"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["table13", "overhead"])
    ap.add_argument("--results", required=True, help="output root (made absolute in the printed commands)")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--python", default="python")
    ap.add_argument("--reps", type=int, default=None, help="default: 5 (table13), 2 (overhead)")
    ap.add_argument("--protovit-model", default="saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
    ap.add_argument("--protovit-data", default="./datasets/cub200_c/")
    ap.add_argument("--protopformer-model", default="/path/to/protopformer_dogs_checkpoint.pth")
    ap.add_argument("--protopformer-data", default="/path/to/stanford_dogs_c")
    ap.add_argument("--protopnet-model", default="/path/to/protopnet_checkpoint.pth")
    ap.add_argument("--protopnet-data", default="/path/to/SICAPv2-C")
    ap.add_argument("--protopnet-clean-data", default="/path/to/SICAPv2/test")
    ap.add_argument("--protopnet-pythonpath", default="/path/to/dir_containing_proto_baseline")
    a = ap.parse_args()
    repo, res = Path(a.repo).resolve(), Path(a.results).resolve()
    fastmodes = (repo / "experiments" / "calibrate_once" / "fastmodes.py").resolve()

    def protovit(method, out):
        args = (["--model", a.protovit_model, "--data_dir", a.protovit_data] + PV_BASE + PV_METHODS[method]
                + SUBSET + ["--seed", "0", "--output", out])
        entry = (["evaluate_robustness.py"] if method != "calibrate_once" else
                 [fastmodes, "--fast-mode", "calibrate_once", "--backbone", "protovit", "--", "evaluate_robustness.py"])
        return command(repo / "ProtoViT", entry, args, a.python)

    def protopformer(method, out):
        args = (["--model", a.protopformer_model, "--data_dir", a.protopformer_data] + PF_BASE + PF_METHODS[method]
                + SUBSET + ["--seed", "0", "--output", out])
        return command(repo / "ProtoPFormer", ["evaluate_robustness_dogs.py"], args, a.python)

    def protopnet(method, out):
        args = (["--model", a.protopnet_model, "--data_dir", a.protopnet_data,
                 "--clean_data_dir", a.protopnet_clean_data, "--severity", "5", "--batch_size", "64", "--gpuid", "0",
                 "--lambda-proto", "0.2", "--adaptive-delta0", "0.25", "--adaptive-top-k", "3",
                 "--track-efficiency"] + PPN_METHODS[method] + SUBSET + ["--seed", "0", "--output", out])
        return command(repo / "ProtoPNetVLM", ["-m", "protopnet_tta.evaluate_robustness"], args, a.python,
                       env={"PYTHONPATH": f".:{a.protopnet_pythonpath}"})

    if a.stage == "table13":
        for r in range(a.reps or 5):
            for m in (TABLE13 if r % 2 == 0 else TABLE13[::-1]):
                print(protovit(m, res / f"{m}_rep{r}.json"))
    else:
        for name, fn in (("protovit", protovit), ("protopformer", protopformer), ("protopnet", protopnet)):
            order = ("unadapted", "adaptive")
            for r in range(a.reps or 2):
                for m in (order if r % 2 == 0 else order[::-1]):
                    print(fn(m, res / name / f"{m}_rep{r}.json"))


if __name__ == "__main__":
    main()
