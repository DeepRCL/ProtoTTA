#!/usr/bin/env python
"""Class-stratified 20% validation / 80% test split of the CUB-200-C photos (Appendix M).

Each CUB test photo goes to exactly one part for all 13 corruptions at once. Stratified by class:
per class, photos sorted by sha256("<SPLIT_SEED>:<class>/<file>"), the first round(FRAC * n_c) are
validation. Builds symlink trees <val>/<corruption>/5/<class>/<file> and <test>/... pointing into the
CUB-200-C tree, and writes a manifest of both lists. --check only verifies existing trees.

  python make_valsplit.py --src ProtoViT/datasets/cub200_c --val ProtoViT/datasets/cub200_c_val \
      --test ProtoViT/datasets/cub200_c_test --manifest valsplit.json
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

CORRUPTIONS = ["gaussian_noise", "fog", "gaussian_blur", "elastic_transform", "brightness",
               "jpeg_compression", "contrast", "defocus_blur", "frost", "impulse_noise",
               "pixelate", "shot_noise", "speckle_noise"]
SEV = "5"
FRAC = 0.2
SPLIT_SEED = 20261108


def photos(src, corruption):
    root = src / corruption / SEV
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def make_split(src):
    ref = photos(src, CORRUPTIONS[0])
    for c in CORRUPTIONS[1:]:
        if photos(src, c) != ref:
            raise SystemExit(f"{c}: photo list differs from {CORRUPTIONS[0]}")
    by_class = defaultdict(list)
    for rel in ref:
        by_class[rel.split("/")[0]].append(rel)
    val, test = [], []
    for cls in sorted(by_class):
        ranked = sorted(by_class[cls],
                        key=lambda r: hashlib.sha256(f"{SPLIT_SEED}:{r}".encode()).hexdigest())
        k = round(FRAC * len(ranked))
        assert 1 <= k < len(ranked), (cls, len(ranked))
        val += ranked[:k]
        test += ranked[k:]
    return ref, sorted(val), sorted(test)


def lists_sha(val, test):
    return hashlib.sha256(json.dumps({"val": val, "test": test}).encode()).hexdigest()


def check(manifest, src, parts):
    """Raise unless the trees on disk match the manifest exactly."""
    val, test = manifest["val"], manifest["test"]
    assert lists_sha(val, test) == manifest["sha256"], "manifest lists do not match their sha256"
    assert (manifest["seed"], manifest["frac"]) == (SPLIT_SEED, FRAC), "split seed/fraction changed"
    assert (manifest["n_val"], manifest["n_test"]) == (len(val), len(test)), "manifest counts wrong"
    full, val_re, test_re = make_split(src)          # the deterministic partition, recomputed
    assert len(full) == manifest["n_full"] == 5794, "photo universe changed"
    assert (val_re, test_re) == (val, test), "manifest is not the specified partition"
    assert not set(val) & set(test), "val and test overlap"
    assert sorted(val + test) == full, "val + test != all photos"
    classes = {r.split("/")[0] for r in full}
    for part in (val, test):
        assert {r.split("/")[0] for r in part} == classes, "a class is missing from a part"
    for c in CORRUPTIONS:
        assert photos(src, c) == full, f"{c}: source photo list changed"
        for name, lst in (("val", val), ("test", test)):
            root = parts[name] / c / SEV
            on_disk = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                             if p.is_file() or p.is_symlink())
            assert on_disk == lst, f"{name}/{c}: files differ from the manifest"
            for rel in lst:
                p = root / rel
                assert p.is_symlink() and Path(os.readlink(p)) == src / c / SEV / rel, (name, c, rel)
            assert sorted(d.name for d in root.iterdir()) == sorted(classes), (name, c)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CUB-200-C root (<corruption>/5/<class>/<file>)")
    ap.add_argument("--val", required=True, help="output tree of the validation photos")
    ap.add_argument("--test", required=True, help="output tree of the test photos")
    ap.add_argument("--manifest", required=True, help="JSON with the two photo lists")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    parts = {"val": Path(a.val).resolve(), "test": Path(a.test).resolve()}
    out = Path(a.manifest)
    if a.check:
        check(json.loads(out.read_text()), src, parts)
        print("split OK")
        return
    if out.exists() or any(p.exists() for p in parts.values()):
        raise SystemExit("split already exists; use --check")
    full, val, test = make_split(src)
    for name, lst in (("val", val), ("test", test)):
        for c in CORRUPTIONS:
            for rel in lst:
                dst = parts[name] / c / SEV / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src / c / SEV / rel, dst)
    manifest = {"seed": SPLIT_SEED, "frac": FRAC, "n_full": len(full), "n_val": len(val),
                "n_test": len(test), "corruptions": CORRUPTIONS, "val": val, "test": test,
                "sha256": lists_sha(val, test)}
    out.write_text(json.dumps(manifest, indent=0) + "\n")
    check(manifest, src, parts)
    print(f"val {len(val)}, test {len(test)} photos x {len(CORRUPTIONS)} corruptions; sha {manifest['sha256']}")


if __name__ == "__main__":
    main()
