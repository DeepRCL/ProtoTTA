#!/usr/bin/env python3
"""Aggregate paper replay and causal PRE/POST ProtoTTA diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import fullset_vlm_gate as gate


ROOT = Path(__file__).resolve().parent


def read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    output = ROOT / "results" / "prototta_prepost_audit"
    audits = {c: read(output / f"{c}.json") for c in gate.CORRUPTIONS}
    if not all(item.get("complete") for item in audits.values()):
        raise RuntimeError("At least one PRE/POST corruption audit is incomplete")

    total_n = sum(item["num_samples"] for item in audits.values())
    pre_correct = sum(round(item["pre_accuracy"] * item["num_samples"]) for item in audits.values())
    post_correct = sum(round(item["post_same_batch_accuracy"] * item["num_samples"]) for item in audits.values())
    next_n = sum(sum(r["next_n"] for r in item["records"] if "next_n" in r) for item in audits.values())
    commit = sum(item["next_batch_commit_correct"] for item in audits.values())
    rollback = sum(item["next_batch_rollback_correct"] for item in audits.values())
    oracle = sum(item["next_batch_oracle_correct"] for item in audits.values())
    outcomes = {
        key: sum(item["next_batch_update_outcomes"][key] for item in audits.values())
        for key in ("beneficial", "harmful", "neutral")
    }

    replay_corruptions = ["gaussian_noise", "gaussian_blur", "jpeg_compression"]
    saved = read(ROOT / "results" / "fixed_lambda_metrics" / "10819" / "cub200c_fixed_lambda1.0_seed0.json")
    saved = saved["results"]["proto_imp_conf_v3"]
    replay_rows = []
    for corruption in replay_corruptions:
        fresh = read(ROOT / "results" / "vlm_gate_diagnostics" / "paper_replay" / f"{corruption}.json")
        fresh = fresh["results"]["proto_imp_conf_v3"][corruption]["5"]["accuracy"]
        exported = read(ROOT / "results" / "vlm_fullset_gate" / "export" / "seed0" / f"{corruption}.json")["after_accuracy"]
        replay_rows.append(
            {
                "corruption": corruption,
                "saved_paper": float(saved[corruption]["5"]["accuracy"]),
                "fresh_original_evaluator": float(fresh),
                "old_board_exporter": float(exported),
            }
        )

    corrected_export_path = (
        ROOT / "results" / "vlm_gate_diagnostics" / "export_replay" / "export" / "seed0" / "gaussian_blur.json"
    )
    corrected_export = read(corrected_export_path)["after_accuracy"] if corrected_export_path.exists() else None
    payload = {
        "schema_version": 1,
        "paper_prediction_semantics": "PRE logits from current adapted state; update affects future batches",
        "pre_accuracy": pre_correct / total_n,
        "post_same_batch_accuracy": post_correct / total_n,
        "pre_post_prediction_changes": sum(item["total_pre_post_prediction_changes"] for item in audits.values()),
        "next_batch_n": next_n,
        "next_batch_commit_accuracy": commit / next_n,
        "next_batch_rollback_accuracy": rollback / next_n,
        "next_batch_batchwise_oracle_accuracy": oracle / next_n,
        "next_batch_update_outcomes": outcomes,
        "paper_replay": replay_rows,
        "corrected_export_gaussian_blur": corrected_export,
    }
    destination = output / "summary.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# ProtoTTA PRE/POST and Paper-Replay Diagnostics",
        "",
        f"- Standard paper/FULLSET prediction semantics: **{payload['paper_prediction_semantics']}**.",
        f"- PRE accuracy: {100*payload['pre_accuracy']:.3f}%.",
        f"- POST accuracy on the same batches: {100*payload['post_same_batch_accuracy']:.3f}%.",
        f"- Predictions changed immediately after one update: {payload['pre_post_prediction_changes']}.",
        f"- Next-batch commit accuracy: {100*payload['next_batch_commit_accuracy']:.3f}%.",
        f"- Next-batch rollback accuracy: {100*payload['next_batch_rollback_accuracy']:.3f}%.",
        f"- Batchwise oracle next-batch accuracy: {100*payload['next_batch_batchwise_oracle_accuracy']:.3f}%.",
        f"- Update outcomes by next-batch correctness: {outcomes}.",
        "",
        "## Exact paper replay",
        "",
        "Corruption | Saved paper | Fresh original evaluator | Old board exporter",
        "---------- | ----------- | ------------------------ | ------------------",
    ]
    for row in replay_rows:
        lines.append(
            f"{row['corruption']} | {100*row['saved_paper']:.3f}% | "
            f"{100*row['fresh_original_evaluator']:.3f}% | {100*row['old_board_exporter']:.3f}%"
        )
    if corrected_export is not None:
        lines.append(
            f"\nCorrected record-diagnostics-matched Gaussian-blur exporter: {100*corrected_export:.3f}%."
        )
    report = "\n".join(lines) + "\n"
    (output / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
