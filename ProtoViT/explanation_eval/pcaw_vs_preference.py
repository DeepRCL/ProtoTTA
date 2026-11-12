#!/usr/bin/env python
"""Does a VLM judge's blinded preference between ProtoTTA and EATA (104 images) follow their PCA-W
difference? (Appendix E.3, last paragraph.)

PCA-W per image is the exact sample-level definition (cub_parts.pcaw: top-10 any-class prototypes from
the verified state exports, weights from the frozen classifier). The judge's item score comes from
vlm104_analysis.py (positive = ProtoTTA preferred, averaged over both presentation orders). Spearman
correlation between the item score and PCA-W(ProtoTTA) - PCA-W(EATA), with an item bootstrap CI
(2,000 resamples, seed 0), on all images and on the images where both methods are equally correct.

  python pcaw_vs_preference.py --items items_pairwise104.json --states_root STATES \
      --analysis vlm104_analysis.json --judge "Gemini 3.8 Flash" --out pcaw_vs_preference.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PV = HERE.parent
sys.path.insert(0, str(PV))
sys.path.insert(0, str(HERE))
import cub_parts as LM  # noqa: E402
from paths import CHECKPOINT  # noqa: E402


def spearman(x, y):
    from scipy.stats import spearmanr
    v = spearmanr(x, y).correlation
    return float(v) if v == v else float("nan")


def boot(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    st = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        v = spearman(x[i], y[i])
        if v == v:
            st.append(v)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="make_items_vlm104.py output")
    ap.add_argument("--states_root", required=True, help="export_states.py output root (states/<method>/...)")
    ap.add_argument("--analysis", required=True, help="vlm104_analysis.py output")
    ap.add_argument("--judge", required=True, help="judge name as given to vlm104_analysis.py --judge")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.loads(Path(a.items).read_text())["items"]
    ana = json.loads(Path(a.analysis).read_text())

    import torch
    import model  # noqa: F401  (for torch.load)
    net = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    W = net.last_layer.weight.detach().cpu().numpy()

    cache = {}

    def state(method, corr):
        key = (method, corr)
        if key not in cache:
            cache[key] = json.loads((Path(a.states_root) / method / f"{corr}_s5.json").read_text())["samples"]
        return cache[key]

    rows = []
    for it in items:
        r = {"item": it["item_id"], "corr": it["corruption"]}
        for m in ("prototta", "eata"):
            rec = state(m, it["corruption"])[it["sample_index"]]
            assert rec["path"] == it["path"], (it["item_id"], m, rec["path"], it["path"])
            assert rec["label"] == it["gt_label"]
            assert rec["pred"] == it["methods"][m]["pred"], (it["item_id"], m)
            r[f"pcaw_{m}"] = LM.pcaw(rec, W)
            r[f"correct_{m}"] = bool(it["methods"][m]["correct"])
        r["d_pcaw"] = r["pcaw_prototta"] - r["pcaw_eata"]
        rows.append(r)

    subsets = {
        "all": lambda r: True,
        "same_correctness": lambda r: r["correct_prototta"] == r["correct_eata"],
    }
    scores = ana["judges"][a.judge]["subsets"]["all"]["primary"]["item_scores"]
    res = {"judge": a.judge, "n_items": len(rows), "subsets": {}}
    for sname, f in subsets.items():
        sel = [r for r in rows if f(r) and r["item"] in scores]
        x = [r["d_pcaw"] for r in sel]
        y = [scores[r["item"]] for r in sel]
        out = {"n": len(sel), "spearman_d_pcaw": spearman(x, y), "ci_d_pcaw": boot(x, y)}
        res["subsets"][sname] = out
        print(f"{a.judge} {sname}: n={out['n']} Spearman rho {out['spearman_d_pcaw']:.2f} "
              f"[{out['ci_d_pcaw'][0]:.2f}, {out['ci_d_pcaw'][1]:.2f}]")
    res["rows"] = rows
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
