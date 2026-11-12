#!/usr/bin/env python
"""Render label-free explanation cards for the selected study items.

For every item and method one card is rendered:
  [the CLEAN counterpart of the test image with the sub-patch boxes of the top-5
   predicted-class prototypes]  +  [the 5 prototype training examples with their
   own sub-patch boxes], prototype k drawn in colour k on both sides.
The boxes are computed by the model on the CORRUPTED image (exported states); they
are drawn on the clean counterpart (same crop/224 frame; corruptions are
pixel-wise, except small local displacements for elastic_transform) so people can
see which bird parts they fall on. The corrupted image itself is shown once per
item ("what the AI saw"). All three methods are rendered identically.
No class names, scores, prototype ids or method names appear on the cards.
Card files get opaque names; the name -> method key is written to a private
key file that must NOT be copied to the public study website.
"""
import argparse
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paths import CUB_TEST_CROPPED, PROTOTYPE_DIR

HERE = Path(__file__).resolve().parent
PV = HERE.parent
PROTO_DIR = PROTOTYPE_DIR
CLEAN_ROOT = CUB_TEST_CROPPED   # clean counterparts (same relative paths)
IMG = 224          # model input resolution; all boxes live in this frame
PATCH = 16         # ViT patch size -> 14x14 grid
TEST_SCALE = 1.5   # test image shown at 336 px
TILE = 200         # exemplar tile size
BORDER = 7
GAP = 14
SLOT_ACTIVE = 0.5
# Okabe-Ito colour-blind-safe palette (orange, sky blue, green, vermillion, purple)
COLORS = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (213, 94, 0), (204, 121, 167)]
METHODS = ("unadapted", "eata", "prototta")


FONT_PATH = HERE / "assets" / "DejaVuSans-Bold.ttf"   # bundled: rendering must not depend on system fonts


def font(size):
    # BASIC layout: text bytes must not depend on whether libraqm is installed
    return ImageFont.truetype(str(FONT_PATH), size, layout_engine=ImageFont.Layout.BASIC)


def draw_box(draw, xy, color, width):
    x0, y0, x1, y1 = xy
    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=(255, 255, 255), width=1)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)


def badge(draw, x, y, k, color, size=22):
    draw.rectangle((x, y, x + size, y + size), fill=color, outline=(255, 255, 255), width=1)
    f = font(int(size * 0.75))
    tw = draw.textlength(str(k), font=f)
    draw.text((x + (size - tw) / 2, y + 1), str(k), fill=(0, 0, 0), font=f)


def load_test_image(dataset_root, rel_path):
    # Same geometry as the model input: Resize((224, 224)).
    return Image.open(Path(dataset_root) / rel_path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)


def item_images(dataset_root, rel_path, tops, bb, background="clean"):
    """Single source of truth for the stimuli (website release and both VLM judges):
    returns {'test': corrupted image as shown on top, method: card}. Cards draw the
    boxes on the clean counterpart (background="clean", human study) or on the
    corrupted image the model saw (background="corrupted", VLM judges)."""
    assert background in ("clean", "corrupted")
    corrupted = load_test_image(dataset_root, rel_path)
    out = {"test": corrupted.resize((int(IMG * TEST_SCALE),) * 2, Image.BILINEAR)}
    if background == "clean":
        assert Image.open(Path(dataset_root) / rel_path).size == Image.open(CLEAN_ROOT / rel_path).size, rel_path
        base = load_test_image(CLEAN_ROOT, rel_path)
    else:
        base = corrupted
    for m, top in tops.items():
        out[m] = render_card(base, top, bb)
    return out


def active_mask(proto):
    """Single visualization mask shared by both halves of a card (sub-prototype on/off)."""
    return [float(s) > SLOT_ACTIVE for s in proto["slots"]]


def test_panel(test_img, top):
    s = TEST_SCALE
    panel = test_img.resize((int(IMG * s), int(IMG * s)), Image.BILINEAR)
    d = ImageDraw.Draw(panel)
    for k, proto in enumerate(top):
        color = COLORS[k]
        first = None
        for flat, on in zip(proto["loc"], active_mask(proto)):
            if not on:
                continue
            r, c = divmod(int(flat), IMG // PATCH)
            inset = k  # small per-prototype inset keeps overlapping boxes visible
            x0, y0 = c * PATCH * s + inset, r * PATCH * s + inset
            x1, y1 = (c + 1) * PATCH * s - 1 - inset, (r + 1) * PATCH * s - 1 - inset
            draw_box(d, (x0, y0, x1, y1), color, 3)
            first = first or (x0, y0)
        if first is not None:
            badge(d, max(0, first[0] - 2), max(0, first[1] - 24), k + 1, color, size=20)
    return panel


def exemplar_tile(proto_idx, k, bb, mask):
    img = Image.open(PROTO_DIR / f"prototype-img-original{proto_idx}.png").convert("RGB")
    assert img.size == (IMG, IMG), img.size
    assert len(mask) == bb.shape[2], (proto_idx, len(mask), bb.shape)
    d = ImageDraw.Draw(img)
    color = COLORS[k]
    for slot, on in enumerate(mask):
        if not on:
            continue
        y0, y1, x0, x1 = (int(bb[proto_idx, 1, slot]), int(bb[proto_idx, 2, slot]),
                          int(bb[proto_idx, 3, slot]), int(bb[proto_idx, 4, slot]))
        if min(y0, y1, x0, x1) < 0:
            raise ValueError(f"prototype {proto_idx}: slot {slot} active in model but has no exemplar box")
        draw_box(d, (x0, y0, x1 - 1, y1 - 1), color, 2)
    img = img.resize((TILE, TILE), Image.BILINEAR)
    tile = Image.new("RGB", (TILE + 2 * BORDER, TILE + 2 * BORDER), color)
    tile.paste(img, (BORDER, BORDER))
    badge(ImageDraw.Draw(tile), 0, 0, k + 1, color, size=26)
    return tile


def render_card(test_img, top, bb):
    left = test_panel(test_img, top)
    tiles = [exemplar_tile(p["p"], k, bb, active_mask(p)) for k, p in enumerate(top)]
    tw = tiles[0].width
    width = left.width + GAP * 2 + len(tiles) * tw + (len(tiles) - 1) * GAP // 2
    height = max(left.height, tw) + 2 * GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left, (GAP, (height - left.height) // 2))
    x = left.width + GAP * 2
    for t in tiles:
        canvas.paste(t, (x, (height - tw) // 2))
        x += tw + GAP // 2
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="final items file of the human study (finalize_pairwise.py final)")
    ap.add_argument("--releases_dir", default=str(PV / "results/explanation_eval/releases"))
    ap.add_argument("--allow_additional_release", action="store_true",
                    help="only if no participant has seen an earlier release")
    args = ap.parse_args()
    items_bytes = Path(args.items).read_bytes()
    items = json.loads(items_bytes)
    methods = tuple(items.get("methods_compared", METHODS))   # pairwise study: ("eata", "prototta")
    bb = np.load(PROTO_DIR / "bb4.npy")
    releases = Path(args.releases_dir)
    if releases.exists() and any(releases.iterdir()) and not args.allow_additional_release:
        raise SystemExit(f"A release already exists in {releases}; refusing to create another "
                         "(its private key decodes collected responses).")
    salt = secrets.token_hex(8)
    release = hashlib.sha256(salt.encode() + items_bytes).hexdigest()[:10]
    rel_dir = releases / f"release_{release}"
    out = rel_dir / "site"          # public: copy to the website
    key_path = rel_dir / "card_key_PRIVATE.json"   # private: never deploy
    (out / "cards").mkdir(parents=True, exist_ok=False)
    key, site_items = {}, []
    for it in items["items"]:
        imgs = item_images(it["dataset_root"], it["path"], {m: it["methods"][m]["top"] for m in methods}, bb,
                           background="clean")
        tid = hashlib.sha256(f"{salt}:{it['item_id']}:test".encode()).hexdigest()[:12]
        imgs["test"].save(out / "cards" / f"{tid}.jpg", quality=92)
        cards = []
        for m in methods:
            cid = hashlib.sha256(f"{salt}:{it['item_id']}:{m}".encode()).hexdigest()[:12]
            imgs[m].save(out / "cards" / f"{cid}.jpg", quality=90)
            key[cid] = {"item_id": it["item_id"], "method": m}
            cards.append(cid)
        site_items.append({"item_id": it["item_id"], "test": tid, "cards": sorted(cards)})
    tut = items["tutorial"]
    tut_imgs = item_images(tut["dataset_root"], tut["path"], {"card": tut["top"]}, bb, background="clean")
    tut_imgs["card"].save(out / "cards" / "tutorial_card.jpg", quality=90)
    tut_imgs["test"].save(out / "cards" / "tutorial_test.jpg", quality=92)
    config = {"release": release, "design": items.get("design", "pairwise"), "card_background": "clean",
              "items": site_items, "repeat_item_ids": items.get("repeat_item_ids", []),
              "tutorial": {"test": "tutorial_test", "card": "tutorial_card"}}
    (out / "study_config.json").write_text(json.dumps(config, indent=2))
    key_path.write_text(json.dumps({"release": release, "salt": salt,
                                    "items_sha256": hashlib.sha256(items_bytes).hexdigest(),
                                    "cards": key}, indent=2))
    print(f"Release {release}: rendered {len(site_items)} items -> {out}; private key -> {key_path}")


if __name__ == "__main__":
    main()
