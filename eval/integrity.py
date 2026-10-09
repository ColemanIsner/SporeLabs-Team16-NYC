"""Per-vehicle integrity: did the GENERATOR keep each seed vehicle, or blur / merge / erase it?

For every seed vehicle box (pseudo-GT) we compare the seed patch with the variant patch at the
same location using a contrast-normalized gradient correlation. Normalizing each patch first means
fog / night (lower contrast, same shape) still scores high, while a car the generator smeared,
fused with its neighbour or removed scores low.

  intact vehicle  = patch_score >= THRESH
  vehicles_intact = intact / all seed vehicles            (generator quality, per clip)
  recall_intact   = detected-and-intact / intact          (detector quality on cars that ARE there)

A miss only counts as a blind spot if the vehicle is intact. Writes `integrity` into
results/evals/<clip_id>.json and gates fidelity.pass on vehicles_intact.

  eval/.venv/bin/python eval/integrity.py            # all variants in data/synthetic/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fidelity import EVALS, ROOT, load_json, merge_eval, read_frames, recompute_failure
from yolo_eval import IOU_MATCH, VEHICLES, iou_matrix

THRESH = 0.35
MIN_INTACT = 0.7
BOXES = ROOT / "results" / "boxes"


def _grad(p):
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-3)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def patch_score(a, b):
    if a.size == 0 or b.size == 0 or min(a.shape[:2]) < 6:
        return None
    ga, gb = _grad(a).ravel(), _grad(b).ravel()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float((ga @ gb) / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-6))


def run(clip_id: str, seed_id: str, src: str, seed_src: str, cond: dict, step: int = 3) -> dict | None:
    sb, vb = load_json(BOXES / f"{seed_id}.json", None), load_json(BOXES / f"{clip_id}.json", None)
    if not sb or not vb:
        return None
    sf, vf = read_frames(ROOT / seed_src), read_frames(ROOT / src)
    n = min(len(sf), len(vf), len(sb["frames"]), len(vb["frames"]))
    total = intact = intact_hit = 0
    scores = []
    for i in range(0, n, step):
        gt = [b for b in sb["frames"][i] if b[5] in VEHICLES]
        pr = [b for b in vb["frames"][i] if b[5] in VEHICLES]
        m = iou_matrix(gt, pr) if gt and pr else None
        H, W = vf[i].shape[:2]
        for gi, b in enumerate(gt):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            pad = 4
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
            s = patch_score(sf[i][y1:y2, x1:x2], vf[i][y1:y2, x1:x2])
            if s is None:
                continue
            total += 1
            scores.append(s)
            if s >= THRESH:
                intact += 1
                if m is not None and m[gi].max() >= IOU_MATCH:
                    intact_hit += 1
    if not total:
        return None
    out = {"vehicles_intact": round(intact / total, 4),
           "recall_intact": round(intact_hit / intact, 4) if intact else None,
           "patch_score_median": round(float(np.median(scores)), 4),
           "n_vehicle_patches": total, "thresh": THRESH, "min_intact": MIN_INTACT}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "synthetic" / "manifest.json"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    rows = load_json(args.manifest, [])
    seeds = {r["seed_id"]: r["src"] for r in load_json(ROOT / "data" / "seeds" / "manifest.json", [])}
    for r in rows:
        cid = r["clip_id"]
        if args.only and args.only not in cid:
            continue
        if r["seed_id"] not in seeds:
            continue
        res = run(cid, r["seed_id"], r["src"], seeds[r["seed_id"]], r.get("condition") or {})
        if res is None:
            print(f"[skip] {cid}: no boxes yet")
            continue
        ev = merge_eval(EVALS / f"{cid}.json", cid, r["seed_id"], r.get("condition"), {"integrity": res})
        print(f"[ok] {cid}: intact={res['vehicles_intact']} recall_intact={res['recall_intact']} "
              f"median={res['patch_score_median']} fail={ev.get('failure')}", flush=True)


if __name__ == "__main__":
    main()
