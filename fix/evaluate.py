#!/usr/bin/env python3
"""Before/after evaluation of the synthetic-data fix -> results/fix.json.

Methods (all yolo11s, conf 0.4, vehicle classes, same imgsz):
  base      = stock yolo11s (the event's hosted model)
  clahe     = stock yolo11s on CLAHE+gamma preprocessed frames (cheap non-learning baseline)
  spore     = yolo11s fine-tuned on Cosmos-Transfer variants with free seed labels (fix/train_modal.py)

Sets:
  test variants  = held-out cameras (i24 p1c3, nyc) synthetic variants, recall vs seed pseudo-GT (yolo11n on clean seed)
  clean seeds    = held-out clean seeds: recall vs same pseudo-GT + mean count (daytime regression check)
  real           = data/real_check/*.mp4 if present: detections/frame, no GT

Usage: eval/.venv/bin/python fix/evaluate.py [--device mps] [--imgsz 960]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "fix"))
from yolo_eval import frame_recall  # noqa: E402
from build_dataset import clips, seed_labels, split_of  # noqa: E402

VEHICLES = [1, 2, 3, 5, 7]
CONF = 0.4
OUT = ROOT / "results" / "fix"
BOX = OUT / "boxes"


def clahe_gamma(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean = l.mean()
    if mean < 90:  # dark frame: lift shadows first
        g = np.clip(np.log(110 / 255) / np.log(max(mean, 5) / 255), 0.3, 1.0)
        l = (255 * (l / 255.0) ** g).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def read(path, max_frames=None, stride=1):
    cap, fr, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(fr) >= max_frames):
            break
        if i % stride == 0:
            fr.append(f)
        i += 1
    cap.release()
    return fr


_M = {}


def predict(method, path, key, a, **rk):
    out = BOX / method / f"{key}.json"
    wt = Path(a.weights[method]).stat().st_mtime if method == "spore" else 0  # base weights are the stock file
    if out.exists() and out.stat().st_mtime >= max(Path(path).stat().st_mtime, wt):
        return json.loads(out.read_text())
    from ultralytics import YOLO
    w = a.weights[method]
    if w not in _M:
        _M[w] = YOLO(w)
    frames = read(path, **rk)
    if method == "clahe":
        frames = [clahe_gamma(f) for f in frames]
    dets, t0 = [], time.time()
    for i in range(0, len(frames), 16):
        for r in _M[w].predict(frames[i:i + 16], conf=CONF, classes=VEHICLES, imgsz=a.imgsz, device=a.device, verbose=False):
            b = r.boxes
            xy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
            dets.append([[round(float(v), 1) for v in xy[k]] + [round(float(cf[k]), 3), int(cl[k])] for k in range(len(cf))])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dets))
    print(f"  {method:6s} {key}: {len(dets)} frames {time.time()-t0:.1f}s", flush=True)
    return dets


def clip_recall(gt, pred):
    v = [frame_recall(gt[i], pred[i]) for i in range(min(len(gt), len(pred)))]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()
    a.weights = {"base": str(OUT / "yolo11s_base.pt"), "clahe": str(OUT / "yolo11s_base.pt"),
                 "spore": str(OUT / "yolo11s_spore.pt")}
    methods = ["base", "clahe", "spore"] if Path(a.weights["spore"]).exists() else ["base", "clahe"]
    t0 = time.time()
    test = [c for c in clips() if split_of(c["seed_id"]) == "test" and (c["kind"] == "seed" or c.get("fidelity_pass") is True)]
    per_clip = []
    for c in test:
        gt = seed_labels(c["seed_id"])
        row = {"clip_id": c["clip_id"], "cond": c["cond"], "camera": c["seed_id"].rsplit("_", 1)[0],
               "gt_mean": float(np.mean([len(g) for g in gt]))}
        for m in methods:
            p = predict(m, ROOT / c["src"], c["clip_id"], a)
            row[f"recall_{m}"] = clip_recall(gt, p)
            row[f"count_{m}"] = float(np.mean([len(f) for f in p]))
        per_clip.append(row)

    results = []

    def agg(rows, set_name):
        if not rows:
            return
        for metric, k in (("vehicle recall vs seed pseudo-GT", "recall"), ("vehicles detected / frame", "count")):
            vals = {m: [r[f"{k}_{m}"] for r in rows if r[f"{k}_{m}"] is not None] for m in methods}
            if not vals["base"]:
                continue
            results.append({"metric": metric, "set": set_name, "n_clips": len(rows),
                            "before": round(float(np.mean(vals["base"])), 4),
                            "after": round(float(np.mean(vals["spore"])), 4) if vals.get("spore") else None,
                            "clahe_baseline": round(float(np.mean(vals["clahe"])), 4),
                            "gt_vehicles_per_frame": round(float(np.mean([r["gt_mean"] for r in rows])), 3)})

    var = [r for r in per_clip if r["cond"] != "clean"]
    agg(var, "held-out synthetic variants (all conditions)")
    for cond in sorted({r["cond"] for r in var}):
        agg([r for r in var if r["cond"] == cond], f"held-out synthetic: {cond}")
    for cam in sorted({r["camera"] for r in var}):
        agg([r for r in var if r["camera"] == cam], f"held-out synthetic: camera {cam}")
    agg([r for r in per_clip if r["cond"] == "clean"], "held-out CLEAN seeds (daytime regression check)")

    real = sorted((ROOT / "data" / "real_check").glob("*.mp4")) if (ROOT / "data" / "real_check").exists() else []
    real_rows = []
    for p in real:
        row = {"clip_id": p.stem}
        for m in methods:
            d = predict(m, p, "real__" + p.stem, a, max_frames=200)
            row[f"count_{m}"] = float(np.mean([len(f) for f in d])) if d else 0.0
        real_rows.append(row)
    if real_rows:
        results.append({"metric": "detections/frame on real night/rain clips, no GT", "set": "data/real_check",
                        "n_clips": len(real_rows),
                        "before": round(float(np.mean([r["count_base"] for r in real_rows])), 3),
                        "after": round(float(np.mean([r.get("count_spore", np.nan) for r in real_rows])), 3),
                        "clahe_baseline": round(float(np.mean([r["count_clahe"] for r in real_rows])), 3),
                        "per_clip": real_rows})

    train_log = json.loads((OUT / "train_log.json").read_text()) if (OUT / "train_log.json").exists() else {}
    notes = [
        "Pseudo-GT = stock yolo11s (conf>=0.4, imgsz 960) vehicle boxes on the CLEAN seed, copied to the same frame of each "
        "structure-preserving variant (Cosmos-Transfer gen_* with edge-SSIM pass, or physics phys_* with exact geometry). "
        "Recall = 'fraction of what the stock model sees in clear daylight that it still sees under the condition'. "
        "On clean seeds the base model's recall is 1.0 by construction; the clean row measures agreement/regression only.",
        "All methods are yolo11s, conf 0.4, vehicle classes, imgsz %d; before = stock yolo11s; after = yolo11s fine-tuned "
        "only on train cameras (i24 p1c1/p1c2 variants + clean seeds); test = held-out camera i24 p1c3 + NYC (different city)." % a.imgsz,
        "clahe_baseline = stock yolo11s with CLAHE (+gamma lift on dark frames) preprocessing, no training.",
        f"Test set is small: {len(var)} variant clips x 93 frames (frames within a clip are highly correlated); "
        f"held-out clean seeds: {len([r for r in per_clip if r['cond']=='clean'])}. Treat per-condition numbers (1-5 clips) as indicative.",
        "Real-clip row (if present) has no ground truth: more detections/frame is suggestive, not proof (could include FPs).",
    ]
    doc = {"results": results, "notes": notes, "per_clip": per_clip,
           "train": {k: v for k, v in train_log.items() if k != "results_csv"},
           "eval_seconds": round(time.time() - t0, 1), "device": a.device}
    if "spore" not in methods:
        print("spore weights missing -> base/clahe only, not writing results/fix.json"); print(json.dumps(results, indent=1)); return
    get = lambda st, k: next((r[k] for r in results if r["set"] == st and r["metric"].startswith("vehicle recall")), None)
    hv, cl = "held-out synthetic variants (all conditions)", "held-out CLEAN seeds (daytime regression check)"
    ok = get(hv, "after") is not None and get(hv, "after") > get(hv, "before") and get(cl, "after") >= get(cl, "before") - 0.05
    doc["gate"] = {"passed": bool(ok), "rule": "after > before on held-out variants AND clean-seed recall drop <= 5 pts"}
    notes.append("VERDICT (run 2, SGD lr 0.001, freeze=10, 10 ep, 2.4k imgs): NO reliable win. Held-out i24 camera p1c3 recall "
                 "0.161 -> 0.205 but CLAHE alone gets 0.202; NYC physics variants (all NYC Cosmos variants failed edge-SSIM 0.45-0.50) and clean seeds regress, real-clip detections/frame drop. "
                 "Run 1 (AdamW lr 0.002, labels from yolo11n@640) collapsed confidences (clean recall 0.89 -> 0.02); kept as "
                 "results/fix/yolo11s_spore_run1_collapsed.pt. Likely causes: tiny, highly correlated train set (6 seeds), "
                 "pseudo-labels miss vehicles the model learns to suppress, no hard-negative/label-quality control. Not a demo claim.")
    results.append({"metric": "notes", "set": "-", "before": None, "after": None, "notes": notes, "gate": doc["gate"]})
    (ROOT / "results" / ("fix.json" if ok else "fix_wip.json")).write_text(json.dumps(results, indent=2))
    print("GATE", ok)
    (OUT / "fix_detail.json").write_text(json.dumps(doc, indent=2))
    for r in results:
        if r["metric"] == "notes":
            continue
        print(f"{r['metric'][:34]:34s} | {r['set'][:58]:58s} n={r['n_clips']:2d}  before {r['before']:.3f}  clahe {r['clahe_baseline']:.3f}  after {r['after']:.3f}")


if __name__ == "__main__":
    main()
