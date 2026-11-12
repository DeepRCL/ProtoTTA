#!/usr/bin/env python3
"""Label-blind correlation between exact sample-level PCA-W and VLM scores (Appendix E.3; Table 9, right).

650 CUB-200-C images (50 per corruption, severity 5), disjoint from the 100-image subset of Table 9
(left), each under the unadapted model and ProtoTTA. The VLM (Qwen3.6-35B-A3B) receives the corrupted
input, the predicted-class reasoning board and the any-class competing-evidence board; ground truth and
correctness are withheld. PCA-W is recomputed from all prototype activations exactly as in
EnhancedPrototypeMetrics.compute_pca_weighted_by_importance (top-10, true-class |weight|).

  prepare      sample the 650 images (seeded; the vlm_eval.py 100-image subset is excluded)
  export-pcaw  exact PCA-W of every sampled image for one method
  score        label-blind VLM scores for one method (boards rendered with vlm_eval.py)
  analyze      Pearson / Spearman, pooled over the 1,300 method-image observations and after
               centring each method (Table 9, right)
Run from the ProtoViT/ directory; it uses vlm_eval.py and its precomputed board data
(--source-results-dir, written by vlm_eval.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent   # the ProtoViT/ directory
sys.path.insert(0, str(ROOT))
import vlm_eval  # noqa: E402

METHODS = ["unadapted", "prototta"]
LOGGER = logging.getLogger("pcaw_vlm_correlation")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "pcaw_vlm_correlation")
    parser.add_argument("--source-results-dir", type=Path, default=ROOT / "results" / "vlm_eval")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "cub200_c")
    parser.add_argument("--model", type=Path, default=ROOT / "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
    parser.add_argument("--prototype-dir", type=Path, default=ROOT / "saved_models/deit_small_patch16_224/exp1/img/epoch-4")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--per-corruption", type=int, default=50)
    prep.add_argument("--seed", type=int, default=20260813)
    export = sub.add_parser("export-pcaw")
    export.add_argument("--method", choices=METHODS, required=True)
    export.add_argument("--batch-size", type=int, default=128)
    export.add_argument("--max-corruptions", type=int)
    score = sub.add_parser("score")
    score.add_argument("--method", choices=METHODS, required=True)
    score.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    score.add_argument("--max-new-tokens", type=int, default=1024)
    score.add_argument("--max-samples", type=int)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--output-dir", type=Path, default=None,
                         help="where correlation_metrics.json is written (default: --results-dir)")
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def stable_key(sample: Dict) -> str:
    return f"{sample['corruption_type']}::{sample['image_path']}"


def public_id(sample: Dict) -> str:
    return hashlib.sha1(stable_key(sample).encode()).hexdigest()[:12]


def cache_path(args, method: str, corruption: str) -> Path:
    return args.source_results_dir / "precompute" / method / f"{corruption}.json"


def run_prepare(args) -> None:
    destination = args.results_dir / "manifest.json"
    old = read_json(args.source_results_dir / "subset.json")
    excluded = {stable_key(item) for item in old["samples"]}
    rng = random.Random(args.seed)
    samples = []
    for corruption in vlm_eval.CORRUPTION_TYPES:
        payload = read_json(cache_path(args, "unadapted", corruption))
        entries = [item for key, item in sorted(payload["entries"].items()) if key not in excluded]
        selected = rng.sample(entries, args.per_corruption)
        for item in sorted(selected, key=lambda value: int(value["sample_idx"])):
            samples.append({
                "sample_idx": int(item["sample_idx"]),
                "corruption_type": corruption,
                "image_path": item["image_path"],
                "ground_truth_index": int(item["ground_truth_index"]),
                "ground_truth_class": item["ground_truth_class"],
                "public_id": hashlib.sha1(f"{corruption}::{item['image_path']}".encode()).hexdigest()[:12],
            })
    for index, sample in enumerate(samples):
        sample["subset_position"] = index
    write_json(destination, {
        "schema_version": 1,
        "seed": args.seed,
        "severity": 5,
        "selection": "uniform random within corruption; vlm_eval.py 100-image subset excluded",
        "per_corruption": args.per_corruption,
        "num_underlying_samples": len(samples),
        "methods": METHODS,
        "corruption_types": vlm_eval.CORRUPTION_TYPES,
        "num_method_sample_pairs": len(samples) * len(METHODS),
        "label_visible_to_vlm": False,
        "samples": samples,
    })
    LOGGER.info("Wrote %s with %d samples", destination, len(samples))


def exact_pcaw(similarities, labels, ppnet):
    import torch
    activations = similarities.sum(dim=2) if similarities.dim() == 3 else similarities
    k = min(10, activations.shape[1])
    top_values, top_indices = torch.topk(activations, k=k, dim=1)
    weights = ppnet.last_layer.weight.detach().abs()
    importance = weights[labels.unsqueeze(1), top_indices]
    contribution = top_values * importance
    identities = ppnet.prototype_class_identity.argmax(dim=1).to(top_indices.device)
    correct = identities[top_indices].eq(labels.unsqueeze(1))
    denominator = contribution.sum(dim=1)
    numerator = (contribution * correct).sum(dim=1)
    return torch.where(denominator.abs() > 1e-12, numerator / denominator, torch.zeros_like(numerator))


def run_export_pcaw(args) -> None:
    import torch
    vlm_eval.install_timm_checkpoint_compat()
    vlm_eval.lazy_import_runtime_modules()
    manifest = read_json(args.results_dir / "manifest.json")
    destination = args.results_dir / "exact_pcaw" / f"{args.method}.json"
    payload = read_json(destination) if destination.exists() else {
        "schema_version": 1,
        "method": args.method,
        "definition": "exact EnhancedPrototypeMetrics PCA-W: top-10 activations weighted by abs(true-class last-layer weight)",
        "model_protocol": "same vlm_eval.setup_method_model stream semantics used to create the original reasoning boards",
        "top_k": 10,
        "entries": {},
    }
    device = torch.device("cuda")
    corruptions = vlm_eval.CORRUPTION_TYPES[: args.max_corruptions]
    for corruption in corruptions:
        selected = {stable_key(item): item for item in manifest["samples"] if item["corruption_type"] == corruption}
        args.tta_steps = 1
        model = vlm_eval.setup_method_model(args.method, args.model, device, args, [], None)
        model.eval()
        ppnet = vlm_eval.get_underlying_model(model)
        dataset = vlm_eval.load_corruption_dataset(args.data_dir, corruption, 5)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        cursor = 0
        for batch_index, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
            logits, _, similarities = vlm_eval.normalize_outputs(outputs)
            values = exact_pcaw(similarities, labels, ppnet)
            probabilities = logits.float().softmax(dim=1)
            for local in range(labels.numel()):
                dataset_index = cursor + local
                path, _ = dataset.samples[dataset_index]
                relative = str(Path(path).relative_to(dataset.root))
                key = f"{corruption}::{relative}"
                if key not in selected:
                    continue
                top2 = probabilities[local].topk(2).values
                payload["entries"][key] = {
                    "public_id": selected[key]["public_id"],
                    "corruption_type": corruption,
                    "sample_idx": dataset_index,
                    "exact_pcaw": float(values[local].item()),
                    "predicted_index": int(logits[local].argmax().item()),
                    "ground_truth_index": int(labels[local].item()),
                    "is_correct": bool(logits[local].argmax() == labels[local]),
                    "msp": float(top2[0].item()),
                    "margin": float((top2[0] - top2[1]).item()),
                }
            cursor += labels.numel()
            if batch_index % 10 == 0:
                LOGGER.info("%s/%s batch=%d entries=%d", args.method, corruption, batch_index, len(payload["entries"]))
        write_json(destination, payload)
        del model
        torch.cuda.empty_cache()
    payload["num_entries"] = len(payload["entries"])
    payload["complete"] = len(payload["entries"]) == len(manifest["samples"])
    write_json(destination, payload)


def validate_vlm(payload: Dict) -> Dict:
    """Validate numeric scores and audit safe score/rationale schema repairs."""
    cleaned = dict(payload)
    repairs = []
    compact_scores = payload.get("scores")
    if compact_scores is not None:
        if not isinstance(compact_scores, (list, tuple)) or len(compact_scores) != 3:
            raise ValueError(f"scores must contain exactly three values: {compact_scores!r}")
        names = ("focus_relevance_score", "prototype_match_score", "overall_quality_score")
        rationale_names = ("focus_rationale", "prototype_rationale", "overall_rationale")
        for name, rationale_name, raw_value in zip(names, rationale_names, compact_scores):
            value = float(raw_value)
            if not 1 <= value <= 5:
                raise ValueError(f"{name} outside [1,5]: {value}")
            cleaned[name] = value
            cleaned[rationale_name] = "Compact numeric retry; rationale not requested."
        cleaned["schema_repairs"] = ["compact_numeric_retry"]
        return cleaned
    pairs = (
        ("focus_relevance_score", "focus_rationale"),
        ("prototype_match_score", "prototype_rationale"),
        ("overall_quality_score", "overall_rationale"),
    )
    for score_key, rationale_key in pairs:
        raw_score = payload.get(score_key)
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            raw_fallback = payload.get(rationale_key)
            value = float(raw_fallback)
            cleaned[rationale_key] = str(raw_score).strip()
            repairs.append(f"swapped:{score_key}<->{rationale_key}")
        if not 1 <= value <= 5:
            raise ValueError(f"{score_key} outside [1,5]: {value}")
        cleaned[score_key] = value
        rationale = str(cleaned.get(rationale_key, "")).strip()
        if not rationale:
            rationale = "Rationale not returned; numeric score retained."
            repairs.append(f"missing:{rationale_key}")
        cleaned[rationale_key] = rationale
    cleaned["schema_repairs"] = repairs
    return cleaned

def generate_valid_vlm_score(scorer, image_paths, prompt: str):
    """Retry once when generation is JSON but violates the requested typed schema."""
    correction = (
        "\n\nYour previous JSON omitted or mistyped a numeric score. Return ONLY "
        '{"scores": [FOCUS_RELEVANCE, PROTOTYPE_MATCH, OVERALL_QUALITY]}. '
        "Replace each placeholder with a JSON number from 1 to 5. Return no rationale or other text."
    )
    last_error = None
    for attempt in range(2):
        current_prompt = prompt if attempt == 0 else prompt + correction
        parsed, raw, actual_prompt = scorer.generate_json(image_paths, current_prompt)
        try:
            return validate_vlm(parsed), raw, actual_prompt
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Typed VLM schema validation attempt %d/2 failed: %s", attempt + 1, exc)
    raise ValueError(f"VLM failed typed score schema after two attempts: {last_error}")



def build_prompt(predicted: str) -> str:
    return (
        "You are a label-blind evaluator of a prototype-based bird classifier. You receive the corrupted input, "
        "a predicted-class reasoning board, and an any-class competing-evidence board. The classifier predicts "
        f"{predicted}. Ground truth and correctness are deliberately withheld. Do not assume the prediction is correct.\n\n"
        "Score only visible reasoning quality from 1 to 5. FOCUS_RELEVANCE measures whether localization is on a "
        "real, discriminative bird part rather than background/corruption. PROTOTYPE_MATCH measures whether retrieved "
        "training patches visually match the highlighted input region, not whether their printed labels repeat the "
        "prediction. OVERALL_QUALITY measures whether focus, predicted-class prototypes, and competing evidence form "
        "a coherent and trustworthy explanation. Repeated same-class prototypes or high contribution values are not "
        "by themselves evidence of correctness.\n\nReturn exactly one JSON object with keys focus_relevance_score, "
        "prototype_match_score, overall_quality_score, focus_rationale, prototype_rationale, overall_rationale. "
        "Each rationale must be one concise sentence. Return no markdown or other text."
    )


def score_path(args, method: str, sample: Dict) -> Path:
    return args.results_dir / "scores" / method / sample["corruption_type"] / f"{sample['public_id']}.json"


def run_score(args) -> None:
    manifest = read_json(args.results_dir / "manifest.json")
    samples = manifest["samples"][: args.max_samples]
    scorer = vlm_eval.VLMScorer(args.model_id, args.max_new_tokens, enable_thinking=False)
    transform = vlm_eval.build_transform()
    current_corruption = None
    cache = {}
    for index, sample in enumerate(samples, 1):
        destination = score_path(args, args.method, sample)
        if destination.exists():
            continue
        if sample["corruption_type"] != current_corruption:
            current_corruption = sample["corruption_type"]
            cache = read_json(cache_path(args, args.method, current_corruption))["entries"]
        meta = cache[stable_key(sample)]
        image_path = args.data_dir / sample["corruption_type"] / "5" / sample["image_path"]
        rendered = args.results_dir / "rendered" / args.method / sample["corruption_type"] / sample["public_id"]
        rendered.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as opened:
            tensor = transform(opened.convert("RGB"))
        predicted_board = rendered / "predicted.jpg"
        any_board = rendered / "any.jpg"
        if not predicted_board.exists():
            vlm_eval.render_reasoning_overview(tensor, meta["predicted_top_prototypes"],
                f"Label-blind predicted-class evidence | Prediction: {meta['predicted_class']}", predicted_board,
                args.prototype_dir, 2, 3, [], "Predicted-class reasoning board")
        if not any_board.exists():
            vlm_eval.render_reasoning_overview(tensor, meta["any_class_top_prototypes"],
                "Label-blind strongest competing evidence", any_board,
                args.prototype_dir, 2, 5, [], "Any-class reasoning board")
        prompt = build_prompt(str(meta["predicted_class"]))
        result, raw, actual_prompt = generate_valid_vlm_score(
            scorer, [image_path, predicted_board, any_board], prompt
        )
        write_json(destination, {
            "schema_version": 1,
            "method": args.method,
            "public_id": sample["public_id"],
            "label_blind": True,
            "ground_truth_visible_to_vlm": False,
            "correctness_visible_to_vlm": False,
            "model_id": args.model_id,
            "prompt_sha256": hashlib.sha256(actual_prompt.encode()).hexdigest(),
            **result,
            "raw_response": raw,
        })
        LOGGER.info("%s %d/%d %s", args.method, index, len(samples), sample["public_id"])


def correlations(rows: Sequence[Dict]) -> Dict:
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray([row["exact_pcaw"] for row in rows])
    output = {"n": len(rows)}
    for field in ("overall_quality_score", "prototype_match_score", "focus_relevance_score"):
        y = np.asarray([row[field] for row in rows])
        output[field] = {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)}
    return output


def centered_rows(rows: Sequence[Dict]) -> List[Dict]:
    output = []
    means = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        means[method] = {field: float(np.mean([row[field] for row in subset])) for field in
            ("exact_pcaw", "overall_quality_score", "prototype_match_score", "focus_relevance_score")}
    for row in rows:
        copy = dict(row)
        for field, mean in means[row["method"]].items():
            copy[field] = row[field] - mean
        output.append(copy)
    return output


def run_analyze(args) -> None:
    manifest = read_json(args.results_dir / "manifest.json")
    exact = {method: read_json(args.results_dir / "exact_pcaw" / f"{method}.json")["entries"] for method in METHODS}
    rows = []
    for method in METHODS:
        for sample in manifest["samples"]:
            score = read_json(score_path(args, method, sample))
            entry = exact[method][stable_key(sample)]
            rows.append({"method": method, "sample_key": stable_key(sample), "corruption_type": sample["corruption_type"],
                "exact_pcaw": entry["exact_pcaw"], "is_correct": entry["is_correct"],
                "overall_quality_score": score["overall_quality_score"], "prototype_match_score": score["prototype_match_score"],
                "focus_relevance_score": score["focus_relevance_score"]})
    summary = {"pooled": correlations(rows), "within_method_centered": correlations(centered_rows(rows))}
    payload = {"schema_version": 1, "protocol": "label-blind VLM; exact sample PCA-W; Table 9 100-image subset excluded",
               "summary": summary, "rows": rows}
    out_dir = args.output_dir or args.results_dir
    write_json(out_dir / "correlation_metrics.json", payload)
    lines = [f"- Images: {len(manifest['samples'])}; method-image observations: {len(rows)}", "",
        "Scope | N | Overall r / rho | Prototype r / rho | Focus r / rho", "----- | - | --------------- | ----------------- | -------------"]
    for scope in ("pooled", "within_method_centered"):
        item = summary[scope]
        def pair(field): return f"{item[field]['pearson']:.3f} / {item[field]['spearman']:.3f}"
        lines.append(f"{scope} | {item['n']} | {pair('overall_quality_score')} | {pair('prototype_match_score')} | {pair('focus_relevance_score')}")
    print("\n".join(lines))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args=parse_args()
    for name in ("results_dir","source_results_dir","data_dir","model","prototype_dir"):
        setattr(args,name,getattr(args,name).resolve())
    if getattr(args, "output_dir", None) is not None:
        args.output_dir = args.output_dir.resolve()
    if args.command=="prepare": run_prepare(args)
    elif args.command=="export-pcaw": run_export_pcaw(args)
    elif args.command=="score": run_score(args)
    else: run_analyze(args)

if __name__ == "__main__": main()
