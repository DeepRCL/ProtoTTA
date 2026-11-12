#!/usr/bin/env python
"""Reasoning boards for the narrator validation (label-free; no method names).

One board per model state, shown to the narrator as one image:
  * header      "STATE <n>   predicted class: <name>" (the model's own prediction; the
                ground-truth class and correctness are never shown);
  * evidence    render_cards.render_card on the CORRUPTED image the model saw: the
                sub-patch boxes of the top-5 predicted-class prototypes and their
                training examples, prototype k in colour k (the paper's predicted-class
                reasoning board);
  * any-class   the ten strongest prototype matches from any class (prototype's class
                name + match strength bar on a FIXED 0..4 scale shared by all boards, so
                bars are comparable between the two states).
The two states of a presentation are rendered with the same code; only the header
label ("STATE 1" / "STATE 2") and the evidence differ.
"""
from PIL import Image, ImageDraw

from render_cards import IMG, TEST_SCALE, font, load_test_image, render_card

ACT_MAX = 4.0        # ProtoViT activation = sum of 4 sub-prototype cosine similarities <= 4
N_ANY = 10
BAR_GREY = (95, 95, 95)


def pretty(class_name):
    """'014.Indigo_Bunting' -> 'Indigo Bunting'."""
    return class_name.split(".", 1)[1].replace("_", " ")


def damaged_photo(dataset_root, rel_path):
    """IMAGE 1: the corrupted photo, no boxes, same size as on the boards' evidence panel."""
    return load_test_image(dataset_root, rel_path).resize((int(IMG * TEST_SCALE),) * 2, Image.BILINEAR)


def any_class_panel(top_any, class_names, width):
    rows = top_any[:N_ANY]
    if len(rows) != N_ANY:
        raise ValueError(f"expected {N_ANY} any-class entries, got {len(rows)}")
    pad, title_h, row_h = 14, 44, 30
    im = Image.new("RGB", (width, title_h + row_h * len(rows) + pad), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((pad, 10), "Ten strongest prototype matches from ANY class "
                      "(class the prototype belongs to; match strength):", fill=(0, 0, 0), font=font(20))
    f = font(19)
    name_x, bar_x = pad + 44, pad + 44 + 430
    bar_w = width - bar_x - 110
    for j, t in enumerate(rows):
        act = float(t["act"])
        if not (-1e-6 <= act <= ACT_MAX + 1e-4):
            raise ValueError(f"activation {act} outside the fixed bar scale [0, {ACT_MAX}]")
        y = title_h + j * row_h
        d.text((pad, y + 3), f"{j + 1}.", fill=(0, 0, 0), font=f)
        d.text((name_x, y + 3), pretty(class_names[int(t["cls"])]), fill=(0, 0, 0), font=f)
        length = int(round(bar_w * max(act, 0.0) / ACT_MAX))
        d.rectangle((bar_x, y + 6, bar_x + bar_w, y + row_h - 6), outline=(200, 200, 200), width=1)
        if length > 0:
            d.rectangle((bar_x, y + 6, bar_x + length, y + row_h - 6), fill=BAR_GREY)
        d.text((bar_x + bar_w + 10, y + 3), f"{act:.2f}", fill=(0, 0, 0), font=f)
    return im


def state_board(dataset_root, rel_path, state, class_names, bb, label):
    """state: {'pred', 'top', 'top_any'} from the exported model states; label: 'STATE 1' / 'STATE 2'."""
    corrupted = load_test_image(dataset_root, rel_path)
    card = render_card(corrupted, state["top"], bb)
    panel = any_class_panel(state["top_any"], class_names, card.width)
    head_h = 58
    board = Image.new("RGB", (card.width, head_h + card.height + panel.height), (255, 255, 255))
    d = ImageDraw.Draw(board)
    d.text((14, 12), f"{label}    predicted class: {pretty(class_names[int(state['pred'])])}",
           fill=(0, 0, 0), font=font(30))
    d.line((0, head_h - 2, card.width, head_h - 2), fill=(0, 0, 0), width=2)
    board.paste(card, (0, head_h))
    board.paste(panel, (0, head_h + card.height))
    return board
