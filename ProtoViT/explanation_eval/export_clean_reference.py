#!/usr/bin/env python
"""Clean-image reference for the consistency/stability metrics: the UNADAPTED source
model on the clean CUB-200 test images (bbox-cropped; same ImageFolder order and
relative paths as the corrupted sets). Stores, per image, the prediction and the
locations/activations of all ground-truth-class prototypes, computed exactly like
export_states.py (slot-weighted activation; push_forward greedy sub-patch indices).

--full additionally stores the clean explanation card (top-5 predicted-class
prototypes, same rule as export_states._predicted_class_top_k) and the top-10
any-class prototypes, into clean_reference_full.json (restoration metrics). Its
GT-class prototype records must reproduce the base reference (--base) exactly.
Relative paths are interpreted relative to the ProtoViT/ directory."""
import argparse
import json
import os
import sys
from pathlib import Path

PV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PV))
os.chdir(PV)

import torch  # noqa: E402
import torchvision.datasets as datasets  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

import model  # noqa: E402,F401
from preprocess import mean, std  # noqa: E402
from settings import img_size, test_dir  # noqa: E402

MODEL = "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
REF_CLEAN_ACC = 0.8609      # checkpoint name / paper (86.1% clean accuracy)
CLEAN_ACC_TOL = 0.0006      # the reference is reported to 4 decimals
N_GT_PROTOS = 10            # ProtoViT CUB: 10 prototypes per class


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--test_dir", default=test_dir, help="clean cropped CUB test images (ImageFolder)")
    ap.add_argument("--out", default=None, help="default: results/explanation_eval/clean_reference[_full].json")
    ap.add_argument("--base", default="results/explanation_eval/clean_reference.json",
                    help="--full only: the verified clean reference written without --full")
    args = ap.parse_args()
    out = PV / (args.out or ("results/explanation_eval/clean_reference_full.json" if args.full
                             else "results/explanation_eval/clean_reference.json"))
    if args.full:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_states import TOP_ANY, _predicted_class_top_k  # single source of the card rule
        base_path = PV / args.base
        base_ver = json.loads(base_path.with_suffix(".verification.json").read_text())
        if not base_ver.get("passed") or sha256_file(base_path) != base_ver["file_sha256"]:
            raise SystemExit("base clean reference failed verification")
        base = {r["path"]: r for r in json.loads(base_path.read_text())["samples"]}
    ver_path = out.with_suffix(".verification.json")
    if out.exists() or ver_path.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    tf = transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(),
                             transforms.Normalize(mean=mean, std=std)])   # same as the corrupted loader
    ds = datasets.ImageFolder(args.test_dir, tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
    net = torch.load(args.model, weights_only=False).cuda().eval()
    slots = torch.sigmoid(net.patch_select * net.temp).squeeze(0)
    n_sub = net.prototype_shape[2]
    factor = slots.sum(-1).unsqueeze(-1) + 1e-10
    proto_class = net.prototype_class_identity.argmax(dim=1).to(slots.device)
    weights = net.last_layer.weight.detach().clone()
    records, cursor, correct = [], 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            logits, _, values = net(images)
            _, _, indices = net.push_forward(images)
            act = (values * (slots.unsqueeze(0) * n_sub / factor.unsqueeze(0))).sum(-1)
            pred = logits.argmax(1)
            correct += pred.eq(labels).sum().item()
            for b in range(labels.size(0)):
                path, y = ds.samples[cursor + b]
                gt_idx = torch.where(proto_class == int(y))[0].tolist()
                rec = {
                    "i": cursor + b, "path": os.path.relpath(path, ds.root), "label": int(y),
                    "pred": int(pred[b].item()),
                    "gt_protos": [{"p": int(p), "act": float(act[b, p].item()),
                                   "loc": [int(v) for v in indices[b, p].reshape(-1).tolist()],
                                   "slots": [float(v) for v in slots[p].tolist()]} for p in gt_idx],
                }
                if args.full:
                    pb = rec["pred"]
                    top_idx, top_score, rule = _predicted_class_top_k(act[b], pb, weights, proto_class)
                    rec["top_rule"] = rule
                    rec["top"] = [{"p": int(pi), "cls": int(proto_class[pi].item()),
                                   "act": float(act[b, pi].item()), "w": float(weights[pb, pi].item()),
                                   "score": float(sc),
                                   "loc": [int(v) for v in indices[b, pi].reshape(-1).tolist()],
                                   "slots": [float(v) for v in slots[pi].tolist()]}
                                  for pi, sc in zip(top_idx.tolist(), top_score.tolist())]
                    any_act, any_idx = torch.topk(act[b], TOP_ANY)
                    rec["top_any"] = [{"p": int(pi), "cls": int(proto_class[pi].item()), "act": float(a),
                                       "loc": [int(v) for v in indices[b, pi].reshape(-1).tolist()]}
                                      for a, pi in zip(any_act.tolist(), any_idx.tolist())]
                records.append(rec)
            cursor += labels.size(0)
    manifest = [(os.path.relpath(p, ds.root), int(y)) for p, y in ds.samples]
    failures = []
    if cursor != len(ds):
        failures.append(f"covered {cursor} of {len(ds)} images")
    if len({r["path"] for r in records}) != len(records):
        failures.append("duplicate paths")
    if [(r["path"], r["label"]) for r in records] != manifest:
        failures.append("records differ from dataset manifest")
    bad = [r["path"] for r in records if len(r["gt_protos"]) != N_GT_PROTOS
           or len({g["p"] for g in r["gt_protos"]}) != N_GT_PROTOS]
    if bad:
        failures.append(f"{len(bad)} images without exactly {N_GT_PROTOS} distinct GT-class prototypes")
    if args.full:
        # the GT-class records must reproduce the verified base reference
        n_loc_diff, max_act_diff = 0, 0.0
        for r in records:
            b0 = base.get(r["path"])
            if b0 is None or b0["pred"] != r["pred"] or [g["p"] for g in b0["gt_protos"]] != [g["p"] for g in r["gt_protos"]]:
                failures.append(f"{r['path']}: differs from base clean reference")
                continue
            for g0, g in zip(b0["gt_protos"], r["gt_protos"]):
                n_loc_diff += int(g0["loc"] != g["loc"])
                max_act_diff = max(max_act_diff, abs(g0["act"] - g["act"]))
        if n_loc_diff or max_act_diff > 1e-4:
            failures.append(f"GT-class prototypes differ from base reference: {n_loc_diff} loc, max act diff {max_act_diff:.2e}")
        bad_top = [r["path"] for r in records if r["top_rule"] != "predicted_class" or len(r["top"]) != 5]
        if bad_top:
            failures.append(f"{len(bad_top)} images without a 5-prototype predicted-class card")
    acc = correct / max(cursor, 1)
    if abs(acc - REF_CLEAN_ACC) > CLEAN_ACC_TOL:
        failures.append(f"clean accuracy {acc:.5f} differs from reference {REF_CLEAN_ACC}")
    payload = {"model": args.model, "model_sha256": sha256_file(PV / args.model), "test_dir": args.test_dir,
               "manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
               "n": cursor, "clean_accuracy": acc, "samples": records}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    ver = {"passed": not failures, "failures": failures, "clean_accuracy": acc, "reference": REF_CLEAN_ACC,
           "model_sha256": payload["model_sha256"], "manifest_sha256": payload["manifest_sha256"],
           "file_sha256": sha256_file(out)}
    ver_path.write_text(json.dumps(ver, indent=2))
    print(json.dumps(ver, indent=2))
    if failures:
        raise SystemExit(f"CLEAN REFERENCE VERIFICATION FAILED: {failures}")


if __name__ == "__main__":
    main()
