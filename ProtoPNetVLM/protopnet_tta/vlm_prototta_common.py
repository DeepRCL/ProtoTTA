"""Shared, label-free utilities for the VLM-guided ProtoTTA evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "gaussian_blur", "defocus_blur", "fog", "frost",
    "jpeg_compression", "pixelate", "contrast", "brightness",
    "elastic_transform",
)
CLASS_NAMES = ("G3", "G4", "G4C", "G5", "NC")
IMAGE_THRESHOLD = -4
BOARD_THRESHOLD = -3
VLM_MODEL = "Qwen/Qwen3.6-35B-A3B"
VLM_BATCH_SIZE = 12
VLM_DOMAIN_NAME = "prostate histopathology"


def select_prediction(
    before_prediction: int,
    after_prediction: int,
    image_score: int,
    board_score: int | None = None,
    mode: str = "image_only",
) -> int:
    """Apply the fixed, label-free prediction-level rollback policy."""
    if before_prediction == after_prediction:
        return after_prediction

    image_rejects = image_score <= IMAGE_THRESHOLD

    if mode == "image_only":
        rollback = image_rejects
    elif mode == "dual_view":
        if board_score is None:
            raise ValueError("dual_view requires board_score")
        rollback = image_rejects and board_score <= BOARD_THRESHOLD
    else:
        raise ValueError(mode)

    return before_prediction if rollback else after_prediction


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON durably and replace the destination atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def reject_label_bearing_payload(value: Any, location: str = "root") -> None:
    """Refuse label/correctness fields before any data reach the VLM."""
    forbidden = ("label", "ground_truth", "ground-truth", "correct")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                raise ValueError(f"forbidden label-derived field at {location}.{key}")
            reject_label_bearing_payload(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_label_bearing_payload(child, f"{location}[{index}]")


def validate_decisions(payload: Any, expected_ids: Sequence[str]) -> list[dict]:
    """Strictly validate one generated JSON response."""
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise ValueError("response must contain exactly the key 'decisions'")
    decisions = payload["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(expected_ids):
        raise ValueError("decision count does not match the requested IDs")
    validated = []
    for expected_id, decision in zip(expected_ids, decisions):
        required = {"id", "action", "adaptation_score", "rationale"}
        if not isinstance(decision, dict) or set(decision) != required:
            raise ValueError(f"decision for {expected_id} has invalid fields")
        if decision["id"] != expected_id:
            raise ValueError(f"expected {expected_id}, received {decision['id']!r}")
        if decision["action"] not in {"ACCEPT", "ROLLBACK"}:
            raise ValueError(f"invalid action for {expected_id}")
        score = decision["adaptation_score"]
        if isinstance(score, bool) or not isinstance(score, int) or not -5 <= score <= 5:
            raise ValueError(f"invalid adaptation score for {expected_id}")
        rationale = decision["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"empty rationale for {expected_id}")
        validated.append(dict(decision))
    return validated


def parse_and_validate_response(text: str, expected_ids: Sequence[str]) -> list[dict]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    # Qwen occasionally emits a short preamble even after being instructed to
    # return JSON only. Extract the longest valid object so nested decision
    # objects cannot be mistaken for the complete response.
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for match in re.finditer(r"\{", stripped):
        try:
            payload, end = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(stripped[match.start():match.start() + end])
    candidate = max(candidates, key=len) if candidates else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    return validate_decisions(payload, expected_ids)


def build_prompt(records: Sequence[Mapping[str, Any]], mode: str) -> str:
    if mode == "image_only":
        condition = (
            f"Each image is the corrupted test {VLM_DOMAIN_NAME} image. "
            "No prototype reasoning board is supplied."
        )
    elif mode == "board":
        condition = (
            "Each image is a paired reasoning canvas: blue/left is BEFORE and "
            "orange/right is AFTER. Compare localization, predicted-class "
            "prototype match, competing prototypes, and confidence."
        )
    else:
        raise ValueError(mode)

    sample_lines = []
    for index, record in enumerate(records, 1):
        sample_lines.append(
            f"IMAGE {index} / ID S{index:02d}: BEFORE predicts "
            f"{record['before_class']} (MSP {record['before_msp']:.3f}); "
            f"AFTER predicts {record['after_class']} "
            f"(MSP {record['after_msp']:.3f})."
        )
    # Keep this template identical to ProtoViT/fullset_vlm_gate.py and the
    # untouched ProtoPFormer transfer, changing only the dataset domain term.
    return "\n".join((
        f"You are a label-free supervisor for a prototype-based {VLM_DOMAIN_NAME} classifier.",
        "For every sample, decide whether to ACCEPT its AFTER continuous ProtoTTA prediction or ROLLBACK to BEFORE.",
        "BEFORE is the frozen source model. AFTER accumulated online updates from preceding batches in the corruption stream and was not reset per image.",
        "Adaptation may help or harm. Ground truth and correctness are withheld. Do not assume AFTER is better.",
        "",
        condition,
        "",
        "Samples and images are aligned in this exact order:",
        *sample_lines,
        "",
        "Return exactly one JSON object with key 'decisions'. Its value must be an array containing one object per ID, in the same order.",
        "Each decision object must contain: id, action (exactly ACCEPT or ROLLBACK), adaptation_score (integer -5 to +5), and rationale (one concise evidence-grounded sentence).",
        "ACCEPT only when AFTER is more likely correct than BEFORE. Otherwise ROLLBACK. Return no markdown or text outside JSON.",
    ))
