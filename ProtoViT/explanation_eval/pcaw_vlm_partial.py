#!/usr/bin/env python
"""PCA-W vs VLM scores after controlling for correctness (Appendix E.3, "Controlling for Correctness").

Reads the 1,300 method-image observations written by pcaw_vlm_correlation.py analyze
(correlation_metrics.json, "rows") and prints, for each of the three VLM scores:
  * the partial Spearman correlation of PCA-W and the score: both are rank-transformed, the ranks are
    regressed (least squares) on an intercept, prediction correctness and corruption-type indicators
    (and a method indicator when both methods are pooled), and the residuals are correlated (Pearson);
    computed pooled over both methods and within each method;
  * the Spearman correlation among correctly classified observations, pooled over both methods.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

DIMS = ("overall_quality_score", "prototype_match_score", "focus_relevance_score")
SCOPES = ("pooled", "unadapted", "prototta")


def arr(rs, k):
    return np.array([r[k] for r in rs], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="correlation_metrics.json from pcaw_vlm_correlation.py analyze")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = json.loads(Path(a.metrics).read_text())["rows"]
    rs = [r for r in rows if all(r.get(k) is not None for k in DIMS + ("exact_pcaw",))]
    out = {}
    for scope in SCOPES:
        sub = [r for r in rs if scope == "pooled" or r["method"] == scope]
        c = arr(sub, "is_correct")
        cor = [r["corruption_type"] for r in sub]
        X = [np.ones(len(sub)), c] + [np.array([x == k for x in cor], float) for k in sorted(set(cor))[1:]]
        if scope == "pooled":
            X.append(np.array([r["method"] == "prototta" for r in sub], float))
        X = np.stack(X, 1)

        def res(y):
            return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        out[scope] = {}
        for dim in DIMS:
            v, p = arr(sub, dim), arr(sub, "exact_pcaw")
            partial = pearsonr(res(rankdata(p)), res(rankdata(v)))[0]
            out[scope][dim] = {"n": len(sub), "partial_spearman": float(partial)}
            controls = "correctness, corruption" + (", method" if scope == "pooled" else "")
            line = f"{scope:9s} {dim:22s} partial rho (controls: {controls}) {partial:.3f}"
            if scope == "pooled":
                correct_only = spearmanr(p[c == 1], v[c == 1])[0]
                out[scope][dim].update({"spearman_correct_only": float(correct_only), "n_correct": int(c.sum())})
                line += f" | rho among correct (n={int(c.sum())}) {correct_only:.3f}"
            print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
