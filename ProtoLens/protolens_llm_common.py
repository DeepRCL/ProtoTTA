"""Shared label-free utilities for the ProtoLens LLM supervisor."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORRUPTIONS = ("qwerty", "swap", "remove_char", "mixed", "aggressive")
SEVERITIES = (20, 40, 60, 80)
CLASS_NAMES = ("negative", "positive")
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
OUTPUT_THRESHOLD = -4
PROTOTYPE_THRESHOLD = -3


def task_for_index(task_id: int) -> tuple[str, int]:
    task_count = len(CORRUPTIONS) * len(SEVERITIES)
    if not 0 <= task_id < task_count:
        raise ValueError(f"task id {task_id} outside 0..{task_count - 1}")
    corruption = CORRUPTIONS[task_id // len(SEVERITIES)]
    severity = SEVERITIES[task_id % len(SEVERITIES)]
    return corruption, severity


def stream_dir(root: Path, corruption: str, severity: int) -> Path:
    return root / corruption / f"severity_{severity}"


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def reject_label_fields(value: Any, location: str = "root") -> None:
    forbidden = ("label", "ground_truth", "ground-truth", "correct")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(token in str(key).lower() for token in forbidden):
                raise ValueError(f"label-bearing key at {location}.{key}")
            reject_label_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_label_fields(child, f"{location}[{index}]")


def select_prediction(
    before: int, after: int, output_score: int,
    prototype_score: int | None = None, mode: str = "output_only",
) -> int:
    if before == after:
        return after
    output_rejects = output_score <= OUTPUT_THRESHOLD
    if mode == "output_only":
        rollback = output_rejects
    elif mode == "dual_view":
        if prototype_score is None:
            raise ValueError("dual_view requires prototype_score")
        rollback = output_rejects and prototype_score <= PROTOTYPE_THRESHOLD
    else:
        raise ValueError(mode)
    return before if rollback else after


def parse_decisions(text: str, expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    payloads = []
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
            payloads.append(payload)
    if not payloads:
        raise ValueError("response has no JSON decisions object")
    payload = max(payloads, key=lambda item: len(item["decisions"]))
    decisions = payload["decisions"]
    if len(decisions) != len(expected_ids):
        raise ValueError("decision count mismatch")
    output = []
    for expected, decision in zip(expected_ids, decisions):
        if not isinstance(decision, dict):
            raise ValueError("decision is not an object")
        identifier = str(decision.get("id", ""))
        action = str(decision.get("action", "")).upper().strip()
        score = decision.get("adaptation_score")
        rationale = str(decision.get("rationale", "")).strip()
        if identifier != expected:
            raise ValueError(f"expected {expected}, received {identifier}")
        if action not in {"ACCEPT", "ROLLBACK"}:
            raise ValueError(f"invalid action for {expected}")
        if isinstance(score, bool) or not isinstance(score, int) or not -5 <= score <= 5:
            raise ValueError(f"invalid score for {expected}")
        if not rationale:
            raise ValueError(f"empty rationale for {expected}")
        output.append({
            "id": identifier, "action": action,
            "adaptation_score": score, "rationale": rationale,
        })
    return output


def _evidence_text(side: str, entries: Sequence[Mapping[str, Any]]) -> str:
    pieces = []
    for entry in entries[:5]:
        phrases = "; ".join(str(x) for x in entry.get("phrases", [])[:2])
        pieces.append(
            f"P{entry['prototype_index']} [{phrases}] "
            f"similarity={float(entry['similarity']):.3f}, "
            f"class-weight={float(entry['class_weight']):+.3f}, "
            f"contribution={float(entry['contribution']):+.3f} "
            f"({entry['effect']})"
        )
    return f"{side} prototype evidence: " + " | ".join(pieces)


def build_prompt(records: Sequence[Mapping[str, Any]], mode: str) -> str:
    if mode not in {"output_only", "prototype_evidence"}:
        raise ValueError(mode)
    lines = [
        "You are a label-free supervisor for a prototype-based sentiment classifier.",
        "For each corrupted Amazon review, decide whether to ACCEPT the AFTER continuous ProtoTTA prediction or ROLLBACK to BEFORE.",
        "BEFORE is the frozen source model. AFTER accumulated online updates from preceding batches in the same corruption stream and was not reset per review.",
        "Ground truth and correctness are withheld. Adaptation may help or harm; do not assume AFTER is better.",
        "Judge which prediction is better supported by the review text and supplied model evidence.",
        "",
    ]
    if mode == "output_only":
        lines.append("No prototype evidence is supplied in this condition.")
    else:
        lines.append(
            "Prototype phrases are training-set concepts retrieved by ProtoLens. Positive contribution supports that side's predicted sentiment; negative contribution opposes it. Compare BEFORE and AFTER evidence for semantic relevance to the review."
        )
    for index, record in enumerate(records, 1):
        identifier = f"S{index:02d}"
        review = " ".join(str(record["corrupted_review"]).split())[:2800]
        lines.extend([
            "",
            f"ID {identifier}",
            f"CORRUPTED REVIEW: {review}",
            f"BEFORE predicts {record['before_prediction']} (MSP {float(record['before_msp']):.3f}).",
            f"AFTER predicts {record['after_prediction']} (MSP {float(record['after_msp']):.3f}).",
        ])
        if mode == "prototype_evidence":
            lines.append(_evidence_text("BEFORE", record["before_evidence"]))
            lines.append(_evidence_text("AFTER", record["after_evidence"]))
    lines.extend([
        "",
        "Return exactly one JSON object with key decisions. The value must be an array in the same ID order.",
        "Every decision must contain exactly: id, action (ACCEPT or ROLLBACK), adaptation_score (integer -5 to +5), and rationale (one concise evidence-grounded sentence).",
        "Use a positive score when AFTER is more likely correct and a negative score when BEFORE is more likely correct. Return JSON only.",
    ])
    return "\n".join(lines)
