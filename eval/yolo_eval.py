#!/usr/bin/env python3
"""YOLO detection under test (seed boxes = pseudo ground truth).

Usage:
  eval/.venv/bin/python eval/yolo_eval.py data/synthetic/X__fog.mp4 --seed data/seeds/X.mp4 [--out results/evals/X__fog.json]

- Backend (--backend / env SPORE_YOLO_BACKEND):
    hosted (default): the VSS ingest Detector, YOLO11s at $YOLO_URL (default http://166.19.38.112:8002),
      POST /v1/infer {"video_base64", "filename", "include_frames": true}, auth Bearer $GPU_BEARER_TOKEN
      (repo-root .env). Server-side conf >= 0.4 (not settable), every decoded frame returned
      (frames[i] = {frame_index, shape [h,w], detections [{label, confidence, bbox [x1,y1,x2,y2] px}]}).
      Labels are mapped to COCO ids; cache "model" = "hosted-yolo11s".
    local: ultralytics yolo11n, conf >= 0.4 (fallback).
  Classes kept: person/bicycle/car/motorcycle/bus/truck.
- Boxes per frame -> results/boxes/<clip_id>.json (cached; reused if newer than the mp4).
- mean_count = mean vehicles per frame (vehicles = bicycle/car/motorcycle/bus/truck).
- recall_vs_seed = per frame, fraction of seed vehicle boxes matched (greedy, IoU >= 0.5,
  class-agnostic within vehicles) by a variant vehicle box; mean over frames that have >=1
  seed box. recall_vs_seed_all is the same including person.
Merges {"yolo": {...}} into the eval JSON, keeping other keys.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from fidelity import EVALS, ROOT, clip_info, load_json, merge_eval, read_frames, rel

BOXES = ROOT / "results" / "boxes"
LOCAL_MODEL = "yolo11n"
HOSTED_MODEL = "hosted-yolo11s"
BACKEND = os.environ.get("SPORE_YOLO_BACKEND", "hosted")
MODEL_NAME = HOSTED_MODEL if BACKEND == "hosted" else LOCAL_MODEL
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", Path(__file__).resolve().parent / ".venv" / "yolo11n.pt"))
CONF, IOU_MATCH = 0.4, 0.5
PERSON = {0}
VEHICLES = {1, 2, 3, 5, 7}
CLASSES = sorted(PERSON | VEHICLES)
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        _MODEL = YOLO(str(WEIGHTS) if WEIGHTS.exists() else LOCAL_MODEL + ".pt")
        if not WEIGHTS.exists():  # ultralytics downloads into cwd; move to cache location
            dl = Path.cwd() / (LOCAL_MODEL + ".pt")
            if dl.exists():
                os.replace(dl, WEIGHTS)
    return _MODEL


def set_backend(name):
    global BACKEND, MODEL_NAME
    assert name in ("hosted", "local"), name
    BACKEND, MODEL_NAME = name, (HOSTED_MODEL if name == "hosted" else LOCAL_MODEL)


_HOSTED = None


def _hosted_cfg():
    global _HOSTED
    if _HOSTED is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        tok = os.environ.get("GPU_BEARER_TOKEN", "").strip()
        if not tok:
            raise SystemExit("GPU_BEARER_TOKEN missing (set it in repo-root .env)")
        _HOSTED = (os.environ.get("YOLO_URL", "http://166.19.38.112:8002").rstrip("/"),
                   {"Authorization": f"Bearer {tok}"})
    return _HOSTED


def hosted_infer(src, retries=4):
    """POST the mp4 to the hosted VSS YOLO11s -> (frames [[x1,y1,x2,y2,conf,cls],...] per frame, frame_indices, raw meta)."""
    import base64
    import requests
    url, hdr = _hosted_cfg()
    body = {"video_base64": base64.b64encode(Path(src).read_bytes()).decode(), "filename": Path(src).name,
            "include_frames": True}
    err = None
    for k in range(retries):
        try:
            r = requests.post(url + "/v1/infer", headers=hdr, json=body, timeout=180)
            if r.status_code in (401, 403):
                raise SystemExit(f"hosted YOLO auth failed ({r.status_code}); check GPU_BEARER_TOKEN")
            r.raise_for_status()
            d = r.json()
            if not (d.get("ok") and d.get("perception_ok")) or "frames" not in d:
                raise RuntimeError(f"hosted YOLO not ok: { {k2: d.get(k2) for k2 in ('ok', 'perception_ok', 'error')} }")
            break
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - retry transient network/server errors
            err = e
            time.sleep(1.5 * (k + 1))
    else:
        raise RuntimeError(f"hosted YOLO failed after {retries} tries: {err!r}")
    frames = sorted(d["frames"], key=lambda f: f["frame_index"])
    idx = [int(f["frame_index"]) for f in frames]
    dets = []
    for f in frames:
        row = []
        for x in f.get("detections") or []:
            lab = x.get("label")
            cls = COCO.index(lab) if lab in COCO else None
            if cls is None or cls not in CLASSES:
                continue
            row.append([round(float(v), 1) for v in x["bbox"][:4]] + [round(float(x["confidence"]), 3), cls])
        dets.append(row)
    return dets, idx, {"frame_count": (d.get("perception_json") or {}).get("frame_count"),
                       "object_counts": d.get("object_counts")}


def detect(src, clip_id, device="cpu", force=False) -> dict:
    """-> {"clip_id","model","conf","frames":[[[x1,y1,x2,y2,conf,cls],...],...],"seconds"} (cached)."""
    out = BOXES / f"{clip_id}.json"
    if not force and out.exists() and out.stat().st_mtime >= Path(src).stat().st_mtime:
        d = load_json(out, None)
        if d and d.get("model") == MODEL_NAME and d.get("conf") == CONF and d.get("src") == rel(src):
            return d
    t0 = time.time()
    if BACKEND == "hosted":
        dets, idx, meta = hosted_infer(src)
        d = {"clip_id": clip_id, "src": rel(src), "model": MODEL_NAME, "conf": CONF, "classes": CLASSES,
             "n_frames": len(dets), "frame_indices": idx, "contiguous": idx == list(range(len(idx))),
             "seconds": round(time.time() - t0, 2), "device": "hosted", "server": meta, "frames": dets}
        BOXES.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, out)
        return d
    frames = read_frames(src)
    m = model()
    dets = []
    for i in range(0, len(frames), 16):
        for r in m.predict(frames[i:i + 16], conf=CONF, classes=CLASSES, device=device, verbose=False, imgsz=640):
            b = r.boxes
            xyxy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
            dets.append([[round(float(v), 1) for v in xyxy[k]] + [round(float(cf[k]), 3), int(cl[k])]
                         for k in range(len(cf))])
    d = {"clip_id": clip_id, "src": rel(src), "model": MODEL_NAME, "conf": CONF, "classes": CLASSES,
         "n_frames": len(dets), "seconds": round(time.time() - t0, 2), "device": device, "frames": dets}
    BOXES.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, out)
    return d


def iou_matrix(a, b):
    a, b = np.asarray(a, float)[:, None, :4], np.asarray(b, float)[None, :, :4]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def frame_recall(gt, pred):
    if not gt:
        return None
    if not pred:
        return 0.0
    m = iou_matrix(gt, pred)
    hit, used = 0, set()
    for gi, pi in sorted(((g, p) for g in range(m.shape[0]) for p in range(m.shape[1]) if m[g, p] >= IOU_MATCH),
                         key=lambda t: -m[t]):
        if gi in used or ("p", pi) in used:
            continue
        used.add(gi), used.add(("p", pi))
        hit += 1
    return hit / len(gt)


def by_index(d):
    """{frame_index: boxes} for a box cache (frame_indices if the backend sampled frames, else 0..n-1)."""
    fr = d["frames"]
    return dict(zip(d.get("frame_indices") or range(len(fr)), fr))


def recall(seed_frames, var_frames, classes):
    if isinstance(seed_frames, dict):  # {frame_index: boxes}: only frames both backends actually saw
        common = sorted(set(seed_frames) & set(var_frames))
        seed_frames, var_frames = [seed_frames[i] for i in common], [var_frames[i] for i in common]
    n = min(len(seed_frames), len(var_frames))
    vals = [frame_recall([b for b in seed_frames[i] if b[5] in classes], [b for b in var_frames[i] if b[5] in classes])
            for i in range(n)]
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def run(clip, seed, out=None, device="cpu", force=False) -> dict:
    clip_id, seed_id, cond = clip_info(clip, seed)
    same = Path(clip).resolve() == Path(seed).resolve()
    seed_clip_id = clip_id if same else (clip_info(seed, seed)[0])
    var = detect(clip, clip_id, device, force)
    sd = var if same else detect(seed, seed_clip_id, device, force)
    vf, sf = var["frames"], sd["frames"]
    vi, si = by_index(var), by_index(sd)
    count = lambda fr, cls: round(float(np.mean([sum(b[5] in cls for b in f) for f in fr])), 3) if fr else 0.0
    yolo = {
        "model": MODEL_NAME, "conf": CONF,
        "mean_count": count(vf, VEHICLES), "mean_count_person": count(vf, PERSON),
        "seed_mean_count": count(sf, VEHICLES),
        "recall_vs_seed": recall(si, vi, VEHICLES), "recall_vs_seed_all": recall(si, vi, VEHICLES | PERSON),
        "frames": len(set(si) & set(vi)), "backend": BACKEND, "seconds": var["seconds"], "device": var.get("device"),
        "boxes": rel(BOXES / f"{clip_id}.json"),
    }
    merge_eval(Path(out) if out else EVALS / f"{clip_id}.json", clip_id, seed_id, cond, {"yolo": yolo})
    return yolo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="eval JSON (default results/evals/<clip_id>.json)")
    ap.add_argument("--device", default="cpu", help="cpu | mps | 0 (cuda)")
    ap.add_argument("--force", action="store_true", help="ignore cached boxes")
    ap.add_argument("--backend", choices=["hosted", "local"], default=BACKEND,
                    help="hosted VSS YOLO11s (default; env SPORE_YOLO_BACKEND) or local ultralytics yolo11n")
    a = ap.parse_args()
    set_backend(a.backend)
    print(json.dumps(run(a.clip, a.seed, a.out, a.device, a.force)))


if __name__ == "__main__":
    main()
