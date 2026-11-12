#!/usr/bin/env python
"""Calibrate-once mode of ProtoTTA on ProtoViT (Appendix N, Table 13), applied at run time.

No source file is edited: this wrapper runs the unchanged evaluator as __main__ with the ProtoTTA
(adaptive lambda) command-line arguments; an import hook replaces two methods of the adapter class right
after the evaluator itself imports the adapter module (import order and module-level code unchanged).
The run fails if the module was never imported (nothing patched).

Per batch the full controller runs, in a fixed order, two gradient-agreement tests (prototype term,
then output term; 2 interleaved half-batch backward passes each) and, for gradient normalization, two
gradient norms (prototype term, then output term; 1 backward pass each), then one update backward pass.

  calibrate_once  The two gradient-agreement values are computed on the first batch of a stream in
                  which the test is defined (>= 4 reliable samples) and reused for every later batch
                  of that stream; the two gradient norms are computed on the first batch of a stream in
                  which both are positive and finite and reused for every later batch of that stream.
                  All cheap per-batch quantities (lambda estimates, the coverage guard, filters,
                  weights) are still computed per batch. Later batches: 1 backward pass (the update).

A stream = one adapter instance; the evaluator builds a fresh adapter per corruption setting, so the
frozen values never cross streams. Before freezing, every call runs the original method, so the first
batch is identical to the full controller. Counts of computed/reused calls are printed at exit.

Usage (from the ProtoViT/ directory, same environment and arguments as the ProtoTTA command):
  python /path/to/experiments/calibrate_once/fastmodes.py --fast-mode calibrate_once --backbone protovit \
      -- evaluate_robustness.py <evaluator arguments ...>
"""
import argparse
import atexit
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
import runpy
import sys

# backbone: (module imported by the evaluator, adapter class, agreement-test method, norm method)
SPEC = {
    "protovit": ("proto_entropy", "ProtoEntropy", "_gradient_consistency", "_gradient_norm"),
}
MIN_RELIABLE = 4   # the agreement test returns 0 (no information) below this many reliable samples
COUNTS = {"adapters": 0, "q_computed": 0, "q_passthrough_undefined": 0, "q_reused": 0, "streams_q_frozen": 0,
          "norm_computed": 0, "norm_passthrough_degenerate": 0, "norm_reused": 0, "streams_norm_frozen": 0}


def _state(adapter):
    st = adapter.__dict__.get("_fastmode_state")
    if st is None:
        st = {"q_calls": 0, "q_pending": None, "q_frozen": None,
              "n_calls": 0, "n_pending": None, "n_frozen": None}
        adapter.__dict__["_fastmode_state"] = st      # plain attribute; not a parameter or buffer
        COUNTS["adapters"] += 1
    return st


def make_agreement(orig):
    def agreement(self, per_sample_loss, reliable_mask, *args, **kwargs):
        st = _state(self)
        slot = st["q_calls"] % 2                     # 0 = prototype term, 1 = output term
        st["q_calls"] += 1
        if st["q_frozen"] is not None:
            COUNTS["q_reused"] += 1
            return st["q_frozen"][slot]
        value = orig(self, per_sample_loss, reliable_mask, *args, **kwargs)
        if int(reliable_mask.bool().sum().item()) < MIN_RELIABLE:
            COUNTS["q_passthrough_undefined"] += 1    # test undefined on this batch: do not freeze it
            st["q_pending"] = None
            return value
        COUNTS["q_computed"] += 1
        if slot == 0:
            st["q_pending"] = value.detach().clone()
        elif st["q_pending"] is not None:
            st["q_frozen"] = (st["q_pending"], value.detach().clone())
            st["q_pending"] = None
            COUNTS["streams_q_frozen"] += 1
            print(f"[fastmode] agreement test frozen at call pair {st['q_calls'] // 2}: "
                  f"q_proto={float(st['q_frozen'][0]):.6g} q_out={float(st['q_frozen'][1]):.6g}", flush=True)
        return value
    return agreement


def make_norm(orig):
    def norm(self, loss, *args, **kwargs):
        st = _state(self)
        slot = st["n_calls"] % 2                     # 0 = prototype term, 1 = output term
        st["n_calls"] += 1
        if st["n_frozen"] is not None:
            COUNTS["norm_reused"] += 1
            return st["n_frozen"][slot]
        value = orig(self, loss, *args, **kwargs)
        v = float(value)
        if not (math.isfinite(v) and v > 0):
            COUNTS["norm_passthrough_degenerate"] += 1
            st["n_pending"] = None
            return value
        COUNTS["norm_computed"] += 1
        if slot == 0:
            st["n_pending"] = value.detach().clone()
        elif st["n_pending"] is not None:
            st["n_frozen"] = (st["n_pending"], value.detach().clone())
            st["n_pending"] = None
            COUNTS["streams_norm_frozen"] += 1
            print(f"[fastmode] gradient norms frozen at call pair {st['n_calls'] // 2}: "
                  f"proto={float(st['n_frozen'][0]):.6g} output={float(st['n_frozen'][1]):.6g}", flush=True)
        return value
    return norm


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Patches the adapter class right after the evaluator itself imports the adapter module, so the
    evaluator's import order and module-level code run exactly as without the wrapper."""

    def __init__(self, module, patch):
        self.module, self.patch, self.done = module, patch, False

    def find_spec(self, fullname, path, target=None):
        if fullname != self.module or self.done:
            return None
        sys.meta_path.remove(self)                  # let the normal finders locate the module
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        original_exec = spec.loader.exec_module

        def exec_module(mod):
            original_exec(mod)
            self.patch(mod)
            self.done = True
        spec.loader.exec_module = exec_module
        return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-mode", required=True, choices=["calibrate_once"])
    ap.add_argument("--backbone", required=True, choices=sorted(SPEC))
    ap.add_argument("entry", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    entry = a.entry[1:] if a.entry[:1] == ["--"] else a.entry
    if not entry:
        raise SystemExit("missing evaluator entry")
    sys.path[0] = os.getcwd()          # as when the evaluator is started from its own directory
    module, cls_name, q_name, n_name = SPEC[a.backbone]
    names = [q_name] + ([n_name] if a.fast_mode == "calibrate_once" else [])
    if module in sys.modules:
        raise SystemExit(f"{module} already imported before the evaluator; refusing")

    def patch(mod):
        cls = getattr(mod, cls_name)
        for name in names:
            if name not in cls.__dict__:
                raise SystemExit(f"{cls_name}.{name} is not defined on the class itself; refusing to patch")
        setattr(cls, q_name, make_agreement(cls.__dict__[q_name]))
        if a.fast_mode == "calibrate_once":
            setattr(cls, n_name, make_norm(cls.__dict__[n_name]))
        print(f"[fastmode] mode={a.fast_mode} backbone={a.backbone} patched {module}.{cls_name}: {names}",
              flush=True)

    hook = PatchOnImport(module, patch)
    sys.meta_path.insert(0, hook)

    def report():
        print("[fastmode] counts " + json.dumps(dict(COUNTS, patched=hook.done)), flush=True)
    atexit.register(report)
    if entry[0] == "-m":
        sys.argv = [entry[1]] + entry[2:]
        runpy.run_module(entry[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = list(entry)
        runpy.run_path(entry[0], run_name="__main__")
    if not hook.done:
        raise SystemExit(f"[fastmode] {module} was never imported by the evaluator; nothing was patched")


if __name__ == "__main__":
    main()
