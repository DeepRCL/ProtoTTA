#!/usr/bin/env python
"""Print the accuracy runs of ProtoTTA calibrate-once on ProtoViT/CUB-200-C (Table 13, accuracy column):
the ProtoTTA (adaptive lambda) command of Appendix L on all 13 corruptions, seeds 0, 2, 3, wrapped by
fastmodes.py. -> <results>/calibrate_once_seed<s>.json (summarize with summarize_accuracy.py).
The timing runs of Table 13 are printed by ../timing/make_commands.py table13.

  python make_commands.py --results /path/to/calibrate_once > calibrate_once.sh
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SEEDS, command  # noqa: E402

PV_ARGS = ["--gpuid", "0", "--batch_size", "128",
           "--prototype-metrics", "--use-enhanced-metrics", "--proto-baseline-samples", "5794",
           "--track-efficiency", "--proto-shared-confidence-weighting", "--proto-gradient-normalize",
           "--proto-adaptive-delta0", "0.25", "--proto-adaptive-topk", "3",
           "--proto-router-min-consistency", "0.25", "--proto-lambda-ema-momentum", "0",
           "--proto-lambda-min", "0", "--proto-lambda-max", "1", "--proto-record-diagnostics",
           "--modes", "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
           "--proto-adaptive-strategy", "source_free_router_coverage_absolute"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="output root (made absolute in the printed commands)")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--python", default="python")
    ap.add_argument("--protovit-model", default="saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
    ap.add_argument("--protovit-data", default="./datasets/cub200_c/")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    wrapper = (repo / "experiments" / "calibrate_once" / "fastmodes.py").resolve()
    for s in SEEDS:
        args = (["--model", a.protovit_model, "--data_dir", a.protovit_data] + PV_ARGS
                + ["--seed", str(s), "--output", Path(a.results).resolve() / f"calibrate_once_seed{s}.json"])
        print(command(repo / "ProtoViT", [wrapper, "--fast-mode", "calibrate_once", "--backbone", "protovit", "--",
                                         "evaluate_robustness.py"], args, a.python))


if __name__ == "__main__":
    main()
