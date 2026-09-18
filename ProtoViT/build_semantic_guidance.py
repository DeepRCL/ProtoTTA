#!/usr/bin/env python3
"""Build the offline VLM prototype audit and distilled compatibility scorer."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import pearsonr, spearmanr

import large_labelblind_correlation as corr
import vlm_eval
from semantic_prototype_guidance import FEATURE_NAMES, cached_features, read_json, write_json


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("semantic_guidance")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "semantic_prototta")
    parser.add_argument("--prototype-dir", type=Path, default=ROOT / "saved_models/deit_small_patch16_224/exp1/img/epoch-4")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/cub200_c")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("--max-classes", type=int)
    score = sub.add_parser("score")
    score.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    score.add_argument("--max-new-tokens", type=int, default=1536)
    score.add_argument("--shard-index", type=int, default=0)
    score.add_argument("--num-shards", type=int, default=1)
    score.add_argument("--max-classes", type=int)
    rank = sub.add_parser("rank-score")
    rank.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    rank.add_argument("--max-new-tokens", type=int, default=768)
    rank.add_argument("--shard-index", type=int, default=0)
    rank.add_argument("--num-shards", type=int, default=1)
    rank.add_argument("--max-classes", type=int)
    sub.add_parser("aggregate")
    sub.add_parser("rank-aggregate")
    train = sub.add_parser("train-student")
    train.add_argument("--correlation-dir", type=Path, default=ROOT / "results" / "vlm_correlation_large")
    train.add_argument("--precompute-dir", type=Path, default=ROOT / "results" / "vlm_eval/precompute")
    return parser.parse_args()


def class_names(args):
    root = args.data_dir / "gaussian_noise" / "5"
    return [item.name for item in sorted(root.iterdir()) if item.is_dir()]


def font(size=22):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_class(args, class_index, class_name):
    output = args.results_dir / "audit" / "boards" / f"class_{class_index:03d}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1900, 920), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), f"Class {class_index}: {class_name.replace('_', ' ')}", fill="black", font=font(30))
    for local in range(10):
        proto = class_index * 10 + local
        column, row = local % 5, local // 5
        x, y = 20 + column * 376, 70 + row * 415
        draw.rectangle((x, y, x + 360, y + 395), outline=(80, 80, 80), width=2)
        draw.text((x + 8, y + 6), f"P{proto} (prototype {local})", fill="black", font=font(20))
        paths = [
            args.prototype_dir / f"prototype-img-original{proto}.png",
            args.prototype_dir / f"prototype-img_vis_{proto}.png",
        ]
        labels = ["source image", "localized evidence"]
        for side, (path, label) in enumerate(zip(paths, labels)):
            if not path.exists():
                raise FileNotFoundError(path)
            image = Image.open(path).convert("RGB")
            image.thumbnail((165, 315))
            ix = x + 8 + side * 176
            iy = y + 48
            canvas.paste(image, (ix, iy))
            draw.text((ix, y + 365), label, fill="black", font=font(15))
    canvas.save(output, quality=92)
    return output


def run_render(args):
    names = class_names(args)
    if args.max_classes is not None:
        names = names[: args.max_classes]
    for index, name in enumerate(names):
        render_class(args, index, name)
    LOGGER.info("Rendered %d prototype-bank boards", len(names))


def validate_scores(payload, count=10):
    keys = ["part", "class_match", "discriminative", "background", "quality"]
    clean = {}
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"{key} must have exactly {count} entries")
        clean[key] = [float(np.clip(float(value), 1, 5)) for value in values]
    return clean


def run_score(args):
    names = class_names(args)
    selected = [(i, n) for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
    if args.max_classes is not None:
        selected = selected[: args.max_classes]
    scorer = vlm_eval.VLMScorer(args.model_id, args.max_new_tokens, enable_thinking=False)
    for class_index, name in selected:
        output = args.results_dir / "audit" / "classes" / f"class_{class_index:03d}.json"
        if output.exists():
            continue
        board = args.results_dir / "audit" / "boards" / f"class_{class_index:03d}.jpg"
        if not board.exists():
            board = render_class(args, class_index, name)
        prompt = f"""You are auditing the fixed prototype bank of a bird classifier.
The board contains all ten learned prototypes assigned to class '{name.replace('_', ' ')}'.
For each prototype P{class_index * 10} through P{class_index * 10 + 9}, score five criteria from 1 (bad) to 5 (excellent):
- part: localized evidence lies on a recognizable bird part rather than background/artifact;
- class_match: evidence is visually compatible with the named bird class;
- discriminative: evidence is likely useful for distinguishing bird classes;
- background: reliance on background or corruption (1=none/clean, 5=strong background reliance; lower is better);
- quality: overall prototype quality.
Judge the highlighted/localized evidence, not merely whether a bird exists in the source image.
Return exactly one JSON object with five arrays in prototype order, each containing exactly ten numbers:
{{"part":[...],"class_match":[...],"discriminative":[...],"background":[...],"quality":[...]}}"""
        last_error = None
        for _ in range(3):
            try:
                parsed, raw, _ = scorer.generate_json([board], prompt)
                clean = validate_scores(parsed)
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(f"Invalid VLM audit for class {class_index}: {last_error}")
        write_json(output, {"schema_version": 1, "model_id": args.model_id, "enable_thinking": False,
                            "class_index": class_index, "class_name": name, "scores": clean, "raw_response": raw})
        LOGGER.info("Scored class %d (%s)", class_index, name)


def run_aggregate(args):
    names = class_names(args)
    quality = np.zeros(len(names) * 10, dtype=np.float64)
    dimensions = {key: [] for key in ("part", "class_match", "discriminative", "background", "quality")}
    for class_index, _ in enumerate(names):
        path = args.results_dir / "audit" / "classes" / f"class_{class_index:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing VLM audit: {path}")
        scores = validate_scores(read_json(path)["scores"])
        for key, values in scores.items():
            dimensions[key].extend(values)
        for local in range(10):
            # Background score is reversed. A small floor keeps every prototype usable.
            value = (scores["part"][local] + scores["class_match"][local] +
                     scores["discriminative"][local] + (6.0 - scores["background"][local]) - 4.0) / 16.0
            quality[class_index * 10 + local] = np.clip(value, 0.05, 1.0)
    write_json(args.results_dir / "prototype_quality.json", {
        "schema_version": 1,
        "definition": "mean normalized part relevance, class match, discriminativeness, and reversed background reliance",
        "label_usage": "training prototype class identity only; no test image labels",
        "prototype_quality": quality.tolist(),
        "dimension_means": {key: float(np.mean(values)) for key, values in dimensions.items()},
        "num_prototypes": int(quality.size),
    })
    LOGGER.info("Aggregated %d prototype scores", quality.size)


def validate_ranking(payload, class_index=None):
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 10:
        raise ValueError("ranking must contain exactly ten local prototype indices")
    ranking = [int(value) for value in ranking]
    # Qwen occasionally follows the visible P123 labels and returns global
    # prototype IDs despite the request for local 0--9 indices. This contains
    # the same ranking information, so normalize it rather than rerunning an
    # otherwise valid judgment.
    if class_index is not None:
        start = int(class_index) * 10
        ranking = [
            value - start if start <= value < start + 10 else value
            for value in ranking
        ]
    if sorted(ranking) == list(range(10)):
        return ranking
    # Deterministic decoding can repeat one item and omit another on otherwise
    # valid responses. Preserve the VLM's valid relative order, discard only
    # duplicates/out-of-range tokens, then append omitted IDs. This avoids an
    # infinite retry on the same deterministic generation.
    repaired = []
    for value in ranking:
        if 0 <= value < 10 and value not in repaired:
            repaired.append(value)
    repaired.extend(value for value in range(10) if value not in repaired)
    if len(repaired) != 10:
        raise ValueError("ranking could not be repaired to a permutation of 0,...,9")
    return repaired


def run_rank_score(args):
    """Forced within-class ranking avoids the ceiling bias of independent 1--5 scores."""
    names = class_names(args)
    selected = [(i, n) for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
    if args.max_classes is not None:
        selected = selected[: args.max_classes]
    scorer = vlm_eval.VLMScorer(args.model_id, args.max_new_tokens, enable_thinking=False)
    for class_index, name in selected:
        output = args.results_dir / "audit_ranked" / "classes" / f"class_{class_index:03d}.json"
        if output.exists():
            continue
        board = args.results_dir / "audit" / "boards" / f"class_{class_index:03d}.jpg"
        if not board.exists():
            board = render_class(args, class_index, name)
        prompt = f"""You are purifying the ten learned prototypes assigned to bird class '{name.replace('_', ' ')}'.
Rank prototype P{class_index * 10} through P{class_index * 10 + 9} from BEST to WORST semantic evidence.
The best prototypes localize a recognizable, class-compatible, discriminative bird part. The worst localize water, sky, branches, blank regions, image borders, or nondiscriminative artifacts.
Judge the bright localized evidence square, not merely whether the source image contains a bird.
You MUST make a strict relative ranking even if several prototypes look good. Use each LOCAL index 0 through 9 exactly once.
Return exactly: {{"ranking":[ten local indices from best to worst],"rationale":"one short sentence identifying the main background prototypes"}}"""
        last_error = None
        for _ in range(3):
            try:
                parsed, raw, _ = scorer.generate_json([board], prompt)
                ranking = validate_ranking(parsed, class_index)
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(f"Invalid forced ranking for class {class_index}: {last_error}")
        write_json(output, {
            "schema_version": 1, "model_id": args.model_id, "enable_thinking": False,
            "class_index": class_index, "class_name": name, "ranking": ranking,
            "rationale": parsed.get("rationale", ""), "raw_response": raw,
        })
        LOGGER.info("Ranked class %d (%s)", class_index, name)


def run_rank_aggregate(args):
    names = class_names(args)
    # Strong but nonzero attenuation: the best four remain dominant, while the
    # bottom three are explicitly designated as likely spurious evidence.
    rank_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.25, 0.10, 0.05]
    quality = np.zeros(len(names) * 10, dtype=np.float64)
    rankings = {}
    for class_index, name in enumerate(names):
        path = args.results_dir / "audit_ranked" / "classes" / f"class_{class_index:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing forced ranking: {path}")
        ranking = validate_ranking(read_json(path), class_index)
        rankings[str(class_index)] = ranking
        for position, local_index in enumerate(ranking):
            quality[class_index * 10 + local_index] = rank_weights[position]
    write_json(args.results_dir / "prototype_quality_ranked.json", {
        "schema_version": 1,
        "definition": "forced Qwen within-class semantic ranking mapped to fixed rank weights",
        "rank_weights_best_to_worst": rank_weights,
        "label_usage": "fixed training prototype class identity only; no target labels",
        "num_classes": len(names), "num_prototypes": int(quality.size),
        "prototype_quality": quality.tolist(), "class_rankings": rankings,
    })
    LOGGER.info("Aggregated %d forced-ranked prototype scores", quality.size)


def collect_student_data(args, quality):
    manifest = read_json(args.correlation_dir / "manifest.json")
    rows = []
    for method in ("unadapted", "prototta"):
        exact = read_json(args.correlation_dir / "exact_pcaw" / f"{method}.json")["entries"]
        for corruption in vlm_eval.CORRUPTION_TYPES:
            cache_entries = read_json(args.precompute_dir / method / f"{corruption}.json")["entries"]
            samples = [item for item in manifest["samples"] if item["corruption_type"] == corruption]
            for sample in samples:
                key = corr.stable_key(sample)
                if key not in exact:
                    continue
                cache = cache_entries[key]
                score_path = args.correlation_dir / "scores" / method / corruption / f"{sample['public_id']}.json"
                score = read_json(score_path)
                rows.append((cached_features(cache, exact[key]["msp"], exact[key]["margin"], quality),
                             (float(score["prototype_match_score"]) - 1.0) / 4.0,
                             corruption, method))
    return rows


def ridge_fit(x, y, penalty):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    reg = np.eye(design.shape[1]) * penalty
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return mean, scale, coef[0], coef[1:]


def predict(x, model):
    mean, scale, bias, weights = model
    return np.clip(((x - mean) / scale) @ weights + bias, 0.0, 1.0)


def run_train(args):
    quality_file = args.results_dir / "prototype_quality.json"
    # The student can be trained before the audit finishes; neutral quality makes
    # its board-only features usable, while aggregate+retrain activates VLM quality features.
    quality = read_json(quality_file)["prototype_quality"] if quality_file.exists() else [1.0] * 2000
    rows = collect_student_data(args, quality)
    x = np.stack([row[0] for row in rows]).astype(np.float64)
    y = np.asarray([row[1] for row in rows], dtype=np.float64)
    groups = np.asarray([row[2] for row in rows])
    best = None
    for penalty in (0.01, 0.1, 1.0, 10.0, 100.0):
        predictions = np.zeros_like(y)
        for group in sorted(set(groups)):
            train, test = groups != group, groups == group
            predictions[test] = predict(x[test], ridge_fit(x[train], y[train], penalty))
        score = float(spearmanr(y, predictions).statistic)
        if best is None or score > best[0]:
            best = (score, penalty, predictions)
    model = ridge_fit(x, y, best[1])
    metrics = {"pearson_r": float(pearsonr(y, best[2]).statistic),
               "spearman_rho": float(spearmanr(y, best[2]).statistic),
               "mae": float(np.abs(y - best[2]).mean())}
    write_json(args.results_dir / "compatibility_student.json", {
        "schema_version": 1, "teacher": "label-blind Qwen prototype_match_score",
        "training_observations": len(rows), "validation": "leave-one-corruption-out",
        "ridge_penalty": best[1], "feature_names": FEATURE_NAMES,
        "mean": model[0].tolist(), "scale": model[1].tolist(), "bias": float(model[2]),
        "weights": model[3].tolist(), "validation_metrics": metrics,
        "uses_vlm_prototype_quality": quality_file.exists(),
    })
    # Save the reviewer-safe deployment models: the student used for a target
    # corruption has never seen any VLM board from that corruption.
    for held_out in sorted(set(groups)):
        train = groups != held_out
        held_model = ridge_fit(x[train], y[train], best[1])
        destination = args.results_dir / "compatibility_students_loco" / f"{held_out}.json"
        write_json(destination, {
            "schema_version": 1,
            "teacher": "label-blind Qwen prototype_match_score",
            "held_out_corruption": held_out,
            "training_corruptions": sorted(set(groups) - {held_out}),
            "training_observations": int(train.sum()),
            "ridge_penalty": best[1],
            "feature_names": FEATURE_NAMES,
            "mean": held_model[0].tolist(), "scale": held_model[1].tolist(),
            "bias": float(held_model[2]), "weights": held_model[3].tolist(),
            "uses_vlm_prototype_quality": quality_file.exists(),
        })
    LOGGER.info("Student: N=%d LOCO r=%.3f rho=%.3f MAE=%.3f", len(rows), metrics["pearson_r"], metrics["spearman_rho"], metrics["mae"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    {"render": run_render, "score": run_score, "rank-score": run_rank_score,
     "aggregate": run_aggregate, "rank-aggregate": run_rank_aggregate,
     "train-student": run_train}[args.command](args)


if __name__ == "__main__":
    main()
