#!/usr/bin/env python
"""Export per-sample predictions and predicted-class prototype evidence of the Table-1 methods
(Unadapted, EATA, adaptive ProtoTTA) on CUB-200-C, severity 5, ProtoViT, seed 0.

Fidelity design
---------------
* Each method is run through ``evaluate_robustness.main()`` with exactly the
  adaptation flags of the reported Table-1 runs, so model construction, EATA
  Fisher estimation, seeding, data order and the online (non-episodic,
  reset-per-corruption) protocol are the evaluator's own code.
* Only ``evaluate_robustness.evaluate_model`` is replaced. Before every TTA
  wrapper call we run a no-grad forward and ``push_forward`` on the *current*
  (pre-update) network. Both the EATA and ProtoTTA wrappers return the logits of
  their pre-update forward, so this captured state is exactly the state that
  produced the reported prediction. We verify this per batch (argmax agreement
  and max |logit difference|) and store the check per sample.
* Accuracy is computed from the wrapper's returned logits and compared against
  the evaluator output (--output JSON) of the reported seed-0 run (--reference):
  the number of correct predictions must be identical in every corruption.

Relative paths are interpreted relative to the ProtoViT/ directory (the evaluator's working directory).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(PV))
os.chdir(PV)  # evaluate_robustness relies on paths relative to ProtoViT/

import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_robustness as er  # noqa: E402

MODEL = "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
DATA_DIR = "./datasets/cub200_c/"


def common_flags(model=MODEL, data_dir=DATA_DIR):
    return [
        "--model", model,
        "--data_dir", data_dir,
        "--gpuid", "0", "--batch_size", "128", "--seed", "0",
        "--prototype-metrics", "--use-enhanced-metrics",
        "--proto-baseline-samples", "5794", "--track-efficiency",
    ]


METHOD_FLAGS = {
    "unadapted": ["--modes", "normal"],
    "eata": ["--modes", "eata"],
    "prototta": [
        "--modes", "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
        "--proto-shared-confidence-weighting", "--proto-gradient-normalize",
        "--proto-adaptive-strategy", "source_free_router_coverage_absolute",
        "--proto-adaptive-delta0", "0.25", "--proto-adaptive-topk", "3",
        "--proto-router-min-consistency", "0.25",
        "--proto-lambda-ema-momentum", "0", "--proto-lambda-min", "0", "--proto-lambda-max", "1",
        "--proto-record-diagnostics",
    ],
}
MODE_NAME = {
    "unadapted": "normal",
    "eata": "eata",
    "prototta": "proto_imp_conf_adaptive_source_free_router_coverage_absolute",
}
TOP_K = 5
TOP_ANY = 10            # top-10 most activated prototypes of ANY class (PCA-W uses k=10)
LOGIT_ATOL = 1e-3
SEVERITY = 5            # evaluate_robustness.main() hard-codes severity 5; asserted below
GATE_VERSION = "exact_count_v2"
ACC_TOL = 0             # exact: correct-prediction counts must equal the reference run (deterministic on same hardware)
EXPECTED_CORRUPTIONS = 13

CTX = {"out_dir": None, "max_batches": None, "with_gt_protos": False}


def _parse_description(description):
    # evaluate_single_combination passes f"{mode_name} on {corruption_type}-{severity}"
    mode_name, rest = description.split(" on ", 1)
    corruption, severity = rest.rsplit("-", 1)
    return mode_name, corruption, int(severity)


def _predicted_class_top_k(cosine_act_b, pred, weights, proto_class, k=TOP_K):
    """Same rule as vlm_eval.select_predicted_class_prototypes: prototypes of the
    predicted class with positive last-layer weight, ranked by activation*weight."""
    w_c = weights[pred]
    mask = (proto_class == pred) & (w_c > 0)
    rule = "predicted_class"
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        idx = torch.where(w_c > 0)[0]
        rule = "positive_weight_fallback"
    if idx.numel() == 0:
        idx = torch.arange(cosine_act_b.numel(), device=cosine_act_b.device)
        rule = "activation_fallback"
        score = cosine_act_b[idx]
    else:
        score = cosine_act_b[idx] * w_c[idx]
    order = torch.argsort(score, descending=True)[:k]
    return idx[order], score[order], rule


def capturing_evaluate_model(model, loader, description="Inference", verbose=True,
                             proto_evaluator=None, compute_proto_metrics=False,
                             efficiency_tracker=None):
    mode_name, corruption, severity = _parse_description(description)
    if severity != SEVERITY:
        raise RuntimeError(f"Unexpected severity {severity}; the study uses severity {SEVERITY} only")
    out = Path(CTX["out_dir"]) / f"{corruption}_s{severity}.json"
    if out.exists():
        raise RuntimeError(f"Duplicate export identity {out}")
    model.eval()
    actual = model.model if hasattr(model, "model") else model
    device = next(actual.parameters()).device
    dataset = loader.dataset
    root = Path(dataset.root)

    records = []
    n_correct = 0
    n_examples = 0
    n_argmax_mismatch = 0
    worst_logit_diff = 0.0
    cursor = 0
    for batch_id, (images, labels) in enumerate(loader):
        if CTX["max_batches"] is not None and batch_id >= CTX["max_batches"]:
            break
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # Pre-update state: identical to the forward the wrapper uses for its returned logits.
            pre_logits, _, pre_values = actual(images)
            _, _, pre_indices = actual.push_forward(images)
            slots = torch.sigmoid(actual.patch_select * actual.temp).squeeze(0)  # [P, S]
            n_sub = actual.prototype_shape[2]
            factor = slots.sum(-1).unsqueeze(-1) + 1e-10  # [P, 1]
            weights = actual.last_layer.weight.detach().clone()  # [C, P]
            proto_class = actual.prototype_class_identity.argmax(dim=1).to(weights.device)  # [P]
            cosine_act = (pre_values * (slots.unsqueeze(0) * n_sub / factor.unsqueeze(0))).sum(-1)  # [B, P]

            outputs = model(images)  # the paper's TTA step (adapts internally)
        logits = er._output_logits(outputs)
        pred = logits.argmax(dim=1)
        pre_pred = pre_logits.argmax(dim=1)
        diff = (logits.float() - pre_logits.float()).abs().amax(dim=1)
        probs = torch.softmax(logits.float(), dim=1)

        n_correct += pred.eq(labels).sum().item()
        n_examples += labels.size(0)
        for b in range(labels.size(0)):
            path, label_from_ds = dataset.samples[cursor + b]
            assert int(label_from_ds) == int(labels[b].item())
            p = int(pred[b].item())
            match = bool(pre_pred[b].item() == p) and float(diff[b].item()) <= LOGIT_ATOL
            n_argmax_mismatch += int(pre_pred[b].item() != p)
            worst_logit_diff = max(worst_logit_diff, float(diff[b].item()))
            top_idx, top_score, rule = _predicted_class_top_k(cosine_act[b], p, weights, proto_class)
            top = []
            for proto_idx, score in zip(top_idx.tolist(), top_score.tolist()):
                top.append({
                    "p": int(proto_idx),
                    "cls": int(proto_class[proto_idx].item()),
                    "act": float(cosine_act[b, proto_idx].item()),
                    "w": float(weights[p, proto_idx].item()),
                    "score": float(score),
                    "loc": [int(v) for v in pre_indices[b, proto_idx].reshape(-1).tolist()],  # flat 14x14 ids
                    "slots": [float(v) for v in slots[proto_idx].tolist()],
                })
            # top-10 prototypes of any class by activation, from the same captured state
            # (enables exact-state PCA-W; classifier weights are not adapted, so W comes from the checkpoint)
            any_act, any_idx = torch.topk(cosine_act[b], TOP_ANY)
            top_any = [{
                "p": int(pi),
                "cls": int(proto_class[pi].item()),
                "act": float(a),
                "loc": [int(v) for v in pre_indices[b, pi].reshape(-1).tolist()],
            } for a, pi in zip(any_act.tolist(), any_idx.tolist())]
            extra = {}
            if CTX["with_gt_protos"]:
                # all prototypes of the ground-truth class (for consistency/stability metrics, Huang et al. ICCV'23);
                # offline evaluation only, from the same captured pre-update state
                y = int(labels[b].item())
                gt_idx = torch.where(proto_class == y)[0].tolist()
                extra["gt_protos"] = [{
                    "p": int(pi), "act": float(cosine_act[b, pi].item()),
                    "loc": [int(v) for v in pre_indices[b, pi].reshape(-1).tolist()],
                    "slots": [float(v) for v in slots[pi].tolist()],
                } for pi in gt_idx]
            records.append({
                **extra,
                "i": cursor + b,
                "path": os.path.relpath(path, root),
                "label": int(labels[b].item()),
                "pred": p,
                "conf": float(probs[b, p].item()),
                "state_check": match,
                "max_abs_logit_diff": float(diff[b].item()),
                "top_rule": rule,
                "top": top,
                "top_any": top_any,
            })
        cursor += labels.size(0)

    if CTX["max_batches"] is None and cursor != len(dataset):
        raise RuntimeError(f"Incomplete export for {corruption}: {cursor} of {len(dataset)} samples")
    acc = n_correct / max(n_examples, 1)
    payload = {
        "mode_name": mode_name, "corruption": corruption, "severity": severity,
        "n": n_examples, "accuracy": acc,
        "n_argmax_mismatch_pre_vs_returned": n_argmax_mismatch,
        "worst_abs_logit_diff": worst_logit_diff,
        "class_names": list(dataset.classes),
        "dataset_root": str(root),
        "samples": records,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out)
    print(f"[export] {mode_name} {corruption}: acc={acc*100:.2f}% n={n_examples} "
          f"argmax_mismatch={n_argmax_mismatch} worst_logit_diff={worst_logit_diff:.2e}", flush=True)
    return acc, {}


def file_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_against_reference(method, out_dir, reference):
    """Hard pass/fail check that the export reproduces the reported run it claims to be."""
    ref = json.loads(Path(reference).read_text())["results"][MODE_NAME[method]]
    ref_corruptions = sorted(c for c, v in ref.items() if v.get(str(SEVERITY)))
    rows, failures, max_abs, hashes = [], [], 0.0, {}
    exports = {}
    for f in sorted(Path(out_dir).glob(f"*_s{SEVERITY}.json")):
        d = json.loads(f.read_text())
        exports[d["corruption"]] = (f, d)
    if len(ref_corruptions) != EXPECTED_CORRUPTIONS:
        failures.append(f"reference has {len(ref_corruptions)} corruptions at severity {SEVERITY}")
    for c in ref_corruptions:
        if c not in exports:
            failures.append(f"missing export for {c}")
            continue
        f, d = exports[c]
        hashes[f.name] = file_sha256(f)
        if d["mode_name"] != MODE_NAME[method]:
            failures.append(f"{c}: mode {d['mode_name']} != {MODE_NAME[method]}")
        # completeness: the export must cover the full evaluation set, in loader order
        manifest = er.datasets.ImageFolder(str(Path(d["dataset_root"])))
        expected = [(os.path.relpath(p, manifest.root), int(y)) for p, y in manifest.samples]
        got = [(s["path"], int(s["label"])) for s in d["samples"]]
        if got != expected:
            failures.append(f"{c}: exported samples ({len(got)}) differ from dataset manifest ({len(expected)})")
        ref_acc = ref[c][str(SEVERITY)]["accuracy"]
        delta = d["accuracy"] - ref_acc
        max_abs = max(max_abs, abs(delta))
        # exact gate: number of correct predictions must equal the reference run's count
        ref_correct, exp_correct = round(ref_acc * d["n"]), round(d["accuracy"] * d["n"])
        if ref_correct != exp_correct:
            failures.append(f"{c}: {exp_correct} correct vs reference {ref_correct} (of {d['n']})")
        n_state_fail = sum(not s["state_check"] for s in d["samples"])
        if d["severity"] != SEVERITY:
            failures.append(f"{c}: severity {d['severity']}")
        if d["n"] != len(d["samples"]):
            failures.append(f"{c}: n={d['n']} but {len(d['samples'])} records")
        if n_state_fail:
            failures.append(f"{c}: {n_state_fail} samples failed the pre-update state check")
        rows.append({"corruption": c, "n": d["n"], "export_acc": d["accuracy"], "reference_acc": ref_acc,
                     "delta": delta, "state_check_failures": n_state_fail,
                     "argmax_mismatch": d["n_argmax_mismatch_pre_vs_returned"],
                     "worst_logit_diff": d["worst_abs_logit_diff"]})
    extra = sorted(set(exports) - set(ref_corruptions))
    if extra:
        failures.append(f"unexpected exports: {extra}")
    report = {"method": method, "reference": str(reference), "severity": SEVERITY,
              "gate": GATE_VERSION, "acc_tolerance": ACC_TOL, "max_abs_acc_delta": max_abs,
              "passed": not failures, "failures": failures, "export_sha256": hashes, "rows": rows}
    (Path(out_dir) / "_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"VERIFICATION FAILED for {method}: {failures}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHOD_FLAGS), required=True)
    ap.add_argument("--reference", required=True,
                    help="evaluate_robustness.py --output JSON of the reported seed-0 run of this method")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out_root", default="results/explanation_eval/states")
    ap.add_argument("--corruptions", nargs="+", default=None, help="subset for smoke tests")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke test only")
    ap.add_argument("--with_gt_protos", action="store_true",
                    help="also store all ground-truth-class prototypes (use with a separate --out_root)")
    ap.add_argument("--verify_only", action="store_true",
                    help="re-run the verification gate on an existing export (no GPU work)")
    args = ap.parse_args()
    CTX["with_gt_protos"] = args.with_gt_protos
    if args.verify_only:
        verify_against_reference(args.method, (PV / args.out_root / args.method).resolve(), PV / args.reference)
        return

    out_dir = (PV / args.out_root / args.method).resolve()
    if args.max_batches is not None:
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.json")):
        raise SystemExit(f"Refusing to overwrite existing export in {out_dir} (delete it explicitly first)")
    CTX["out_dir"] = str(out_dir)
    CTX["max_batches"] = args.max_batches

    er.evaluate_model = capturing_evaluate_model  # the only behavioural substitution
    argv = ["evaluate_robustness.py", *common_flags(args.model, args.data_dir), *METHOD_FLAGS[args.method],
            "--output", str(out_dir / "_evaluate_robustness_output.json")]
    if args.corruptions:
        argv += ["--corruptions", *args.corruptions]
    sys.argv = argv
    er.main()
    if args.max_batches is None:
        verify_against_reference(args.method, out_dir, PV / args.reference)


if __name__ == "__main__":
    main()
