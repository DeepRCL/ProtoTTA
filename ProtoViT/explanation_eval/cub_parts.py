#!/usr/bin/env python
"""CUB part keypoints in the ProtoViT input frame, and per-sample helpers shared by the analyses.

Keypoints are mapped exactly like the model input: CUB bounding-box crop (int coords, clipped) ->
the cropped/corrupted image keeps that frame -> Resize(224, 224). Sub-patch cells live on the 14x14
grid of 16x16 patches; a sub-prototype is active (drawn on the cards) if its slot weight is > 0.5.
pcaw() is the exact sample-level PCA-W (k = 10) from the exported top-10 any-class prototypes.
"""
import numpy as np

from paths import CUB_ROOT as CUB, CUB_TEST_CROPPED as CROPPED  # noqa: F401  (also used by narrator_items)

METHODS = ("unadapted", "eata", "prototta")
GRID, PATCH, IMG, SEVERITY = 14, 16, 224, 5
SLOT_ACTIVE = 0.5


def load_keypoints():
    from PIL import Image
    images = dict(l.split() for l in (CUB / "images.txt").read_text().splitlines())        # id -> rel path
    bbox = {k: tuple(map(float, v)) for k, *v in (l.split() for l in (CUB / "bounding_boxes.txt").read_text().splitlines())}
    parts = {}
    for l in (CUB / "parts/part_locs.txt").read_text().splitlines():
        img_id, part_id, x, y, vis = l.split()
        if int(vis):
            parts.setdefault(img_id, []).append((int(part_id), float(x), float(y)))
    kp = {}
    for img_id, rel in images.items():
        crop_path = CROPPED / rel
        if not crop_path.exists():
            continue  # training image
        with Image.open(CUB / "images" / rel) as im:
            W0, H0 = im.size
        x, y, w, h = bbox[img_id]
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(W0, int(x) + int(w)), min(H0, int(y) + int(h))
        with Image.open(crop_path) as im:
            cw, ch = im.size
        assert (cw, ch) == (right - left, bottom - top), (rel, cw, ch, right - left, bottom - top)
        pts = []
        for pid, px, py in parts.get(img_id, []):
            if left <= px < right and top <= py < bottom:
                pts.append((pid, (px - left) * IMG / cw, (py - top) * IMG / ch))
        kp[rel] = pts
    return kp


def cell_hits(flat, pts):
    r, c = divmod(int(flat), GRID)
    x0, y0 = c * PATCH, r * PATCH
    cx, cy = x0 + PATCH / 2, y0 + PATCH / 2
    strict = any(x0 <= px < x0 + PATCH and y0 <= py < y0 + PATCH for _, px, py in pts)
    near = any(max(abs(px - cx), abs(py - cy)) <= PATCH * 1.5 for _, px, py in pts)  # cell or 8-neighbours
    return near, strict


def sample_metrics(rec, pts):
    """Share of the card's active sub-patches near an annotated part (near, strict, top-1 prototype);
    None if the card shows no active sub-patch."""
    near, strict, top1 = [], [], []
    for k, proto in enumerate(rec["top"]):
        for flat, s in zip(proto["loc"], proto["slots"]):
            if s <= SLOT_ACTIVE:
                continue
            n, st = cell_hits(flat, pts)
            near.append(n)
            strict.append(st)
            if k == 0:
                top1.append(n)
    if not near or not top1:
        return None
    return float(np.mean(near)), float(np.mean(strict)), float(np.mean(top1))


def pcaw(rec, W):
    y = rec["label"]
    c = np.array([t["act"] * abs(W[y, t["p"]]) for t in rec["top_any"]])
    same = np.array([t["cls"] == y for t in rec["top_any"]])
    return float(c[same].sum() / c.sum()) if c.sum() > 0 else 0.0
