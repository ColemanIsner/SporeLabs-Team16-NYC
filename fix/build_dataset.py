#!/usr/bin/env python3
"""Build a YOLO-format dataset from Cosmos-Transfer synthetic variants with FREE labels.

Labels = base YOLO (yolo11n, conf>=0.4) vehicle boxes on the CLEAN seed (results/boxes/<seed_id>.json),
copied onto the same frame index of each structure-preserving variant. Only variants with
fidelity.pass == true (results/evals/<clip_id>.json) are used.

Split by CAMERA (no leakage):
  train = i24 p1c1 + p1c2 variants + their clean seeds (p1c2_08 held out as val for best.pt selection)
  test  = i24 p1c3 variants + nyc variants (held-out camera / city) -- NOT written as images, evaluated by fix/evaluate.py

Usage: eval/.venv/bin/python fix/build_dataset.py [--stride 3] [--out fix/dataset]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VEHICLES = {1, 2, 3, 5, 7}
CONF = 0.4
COCO_NAMES = None  # filled from ultralytics at train time; we keep COCO ids (nc=80) for a drop-in model


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def camera(seed_id: str) -> str:
    # i24_scene1_p1c1_00 -> i24_p1c1 ; nyc_002_00 -> nyc_002
    parts = seed_id.split("_")
    return f"i24_{parts[2]}" if seed_id.startswith("i24") else "_".join(parts[:2])


def split_of(seed_id: str) -> str:
    cam = camera(seed_id)
    if cam in ("i24_p1c1", "i24_p1c2"):
        return "val" if seed_id == "i24_scene1_p1c2_08" else "train"
    return "test"


def seed_labels(seed_id):
    d = load(ROOT / "results" / "boxes" / f"{seed_id}.json")
    if not d:
        return None
    return [[b for b in f if b[5] in VEHICLES and b[4] >= CONF] for f in d["frames"]]


_FCACHE = ROOT / "fix" / "fidelity_cache.json"


def fidelity_fallback(v):
    """results/evals not written yet -> compute edge_ssim ourselves (eval/fidelity.edge_ssim), cached in fix/."""
    cache = load(_FCACHE, {})
    if v["clip_id"] in cache:
        return cache[v["clip_id"]]
    sys.path.insert(0, str(ROOT / "eval"))
    from fidelity import edge_ssim
    seed_src = ROOT / "data" / "seeds" / f"{v['seed_id']}.mp4"
    try:
        fid = edge_ssim(ROOT / v["src"], seed_src)
        fid = {"edge_ssim": fid["edge_ssim"], "pass": bool(fid["pass"]), "source": "fix/fallback"}
    except Exception as e:  # e.g. mp4 still being written
        return {"error": str(e)}
    cache[v["clip_id"]] = fid
    _FCACHE.write_text(json.dumps(cache, indent=1))
    return fid


def clips():
    """-> list of dict(clip_id, seed_id, src, cond, kind) for passing variants + clean seeds."""
    out = []
    seeds = load(ROOT / "data" / "seeds" / "manifest.json", [])
    for s in seeds:
        out.append({"clip_id": s["seed_id"], "seed_id": s["seed_id"], "src": s["src"], "cond": "clean", "kind": "seed"})
    for v in load(ROOT / "data" / "synthetic" / "manifest.json", []):
        c = v.get("condition") or {}
        if c.get("control_weight") != 0.5:
            continue
        ev = load(ROOT / "results" / "evals" / f"{v['clip_id']}.json", {})
        fid = ev.get("fidelity") or fidelity_fallback(v)
        cond = f"{c.get('weather')}_{c.get('time')}_{c.get('intensity')}"
        rec = {"clip_id": v["clip_id"], "seed_id": v["seed_id"], "src": v["src"], "cond": cond, "kind": "variant",
               "edge_ssim": fid.get("edge_ssim"), "fidelity_pass": fid.get("pass")}
        if not (ROOT / v["src"]).exists():
            continue
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "fix" / "dataset")
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    stats = {"train": {}, "val": {}, "test": {}}
    skipped, test_clips = [], []
    for c in clips():
        sp = split_of(c["seed_id"])
        if c["kind"] == "variant" and c.get("fidelity_pass") is not True:
            skipped.append({"clip_id": c["clip_id"], "edge_ssim": c.get("edge_ssim"), "fidelity_pass": c.get("fidelity_pass")})
            continue
        labels = seed_labels(c["seed_id"])
        if labels is None:
            skipped.append({"clip_id": c["clip_id"], "why": "no seed boxes"})
            continue
        if sp == "test":
            test_clips.append(c)
            stats["test"][c["cond"]] = stats["test"].get(c["cond"], 0) + 1
            continue
        img_dir, lbl_dir = a.out / "images" / sp, a.out / "labels" / sp
        img_dir.mkdir(parents=True, exist_ok=True), lbl_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / c["src"]))
        i, n = 0, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.stride == 0 and i < len(labels):
                h, w = f.shape[:2]
                name = f"{c['clip_id']}__f{i:03d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                lines = []
                for x1, y1, x2, y2, _, cl in labels[i]:
                    lines.append(f"{int(cl)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n += 1
            i += 1
        cap.release()
        stats[sp][c["cond"]] = stats[sp].get(c["cond"], 0) + n
    meta = {"stride": a.stride, "images_per_split_by_condition": stats, "test_clips": test_clips, "skipped": skipped,
            "label_source": "yolo11n conf>=0.4 vehicle boxes on clean seed, same frame index"}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    (a.out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n")  # names/nc added at train time
    print(json.dumps({k: v for k, v in meta.items() if k != "test_clips"} | {"n_test_clips": len(test_clips)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
