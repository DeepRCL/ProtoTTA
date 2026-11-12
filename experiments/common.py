"""Shared helpers for the experiment command generators and analyses.

Every evaluator (ProtoViT, ProtoPFormer, ProtoLens, ProtoPNet) writes one JSON per run (--output) whose
"results" map holds one method -> corruption -> severity -> {"accuracy": fraction, ...,
"efficiency": {"time_per_sample_ms": ...}}.
"""
import json
import math
import shlex
from pathlib import Path

SEEDS = (0, 2, 3)
VISION_CORRUPTIONS = ("gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur "
                      "fog frost jpeg_compression pixelate contrast brightness elastic_transform").split()
# The four corruption families (Appendix A) used by the hyperparameter re-selection.
FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["gaussian_blur", "defocus_blur"],
    "weather": ["fog", "frost", "brightness"],
    "digital": ["contrast", "jpeg_compression", "pixelate", "elastic_transform"],
}


def settings(path):
    """{"<corruption>/<severity>": accuracy in %} of a single-method result file."""
    d = json.loads(Path(path).read_text())
    res = d["results"]
    if len(res) != 1:
        raise ValueError(f"{path}: expected one method, got {list(res)}")
    out = {}
    for corr, sev in next(iter(res.values())).items():
        for s, v in sev.items():
            if not isinstance(v, dict) or "accuracy" not in v:
                continue
            acc = float(v["accuracy"])
            if not 0.0 <= acc <= 1.0:
                raise ValueError(f"{path}: accuracy {acc} is not a fraction")
            out[f"{corr}/{s}"] = 100.0 * acc
    return out


def times(path):
    """{"<corruption>/<severity>": time_per_sample_ms} of a single-method result file."""
    d = json.loads(Path(path).read_text())
    (res,) = d["results"].values()
    return {f"{c}/{s}": float(v["efficiency"]["time_per_sample_ms"])
            for c, sev in res.items() for s, v in sev.items() if isinstance(v, dict) and "efficiency" in v}


def mean_sd(xs):
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def seed_averaged(files, n_settings=None):
    """Mean over settings of the seed-averaged per-setting accuracy (%), as in the paper's tables.
    Returns (mean, sd across settings, per-setting dict) or None if a file is missing."""
    per_seed = []
    for f in files:
        if not Path(f).exists():
            return None
        st = settings(f)
        if n_settings is not None and len(st) != n_settings:
            raise ValueError(f"{f}: {len(st)} settings, expected {n_settings}")
        per_seed.append(st)
    if not per_seed:
        return None
    keys = sorted(per_seed[0])
    if any(sorted(p) != keys for p in per_seed):
        raise ValueError(f"settings differ across seeds: {files}")
    acc = {k: sum(p[k] for p in per_seed) / len(per_seed) for k in keys}
    m, sd = mean_sd(acc.values())
    return m, sd, acc


def command(workdir, entry, args, python="python", env=None):
    """One self-contained shell command: (cd <absolute workdir> && [ENV=...] <python> -u <entry> <args>).
    The subshell keeps the caller's directory, so the printed commands can run one after another."""
    pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    head = f"(cd {shlex.quote(str(Path(workdir).resolve()))} && " + (pre + " " if pre else "")
    return head + " ".join([python, "-u"] + [shlex.quote(str(x)) for x in list(entry) + list(args)]) + ")"
