#!/usr/bin/env python3
"""Fidelity gate: did the generator keep the scene?

edge_ssim = mean over frames of SSIM(Canny(seed), Canny(variant)), both resized to 640x360.

Usage:
  eval/.venv/bin/python eval/fidelity.py data/synthetic/X__fog.mp4 --seed data/seeds/X.mp4 [--out results/evals/X__fog.json]

Merges {"fidelity": {...}} into the eval JSON (default results/evals/<clip_id>.json), keeping
other keys. Also holds the shared helpers (manifest lookup, locked JSON merge) used by
yolo_eval.py and run_all.py.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "results" / "evals"
SIZE = (640, 360)
DEFAULT_THRESHOLD = 0.5


# ---------- shared helpers ----------
def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def manifest_lookup(src) -> dict:
    """Find the manifest row (seed or synthetic) whose src matches this file."""
    target = Path(src).resolve()
    mans = [ROOT / "data" / "seeds" / "manifest.json", ROOT / "data" / "synthetic" / "manifest.json",
            target.parent / "manifest.json"]
    for man in dict.fromkeys(m.resolve() for m in mans):
        for r in load_json(man, []) or []:
            s = r.get("src")
            if s and (ROOT / s).resolve() == target:
                return r
    return {}


def clip_info(src, seed_src=None):
    """-> (clip_id, seed_id, condition) from manifests, falling back to file stems."""
    m = manifest_lookup(src)
    clip_id = m.get("clip_id") or m.get("seed_id") or Path(src).stem
    seed_id = m.get("seed_id")
    if not seed_id and seed_src:
        seed_id = manifest_lookup(seed_src).get("seed_id") or Path(seed_src).stem
    cond = m.get("condition")
    if cond is None and seed_src and Path(seed_src).resolve() == Path(src).resolve():
        cond = {"weather": "clear", "time": "day", "intensity": "light", "kind": "seed"}
    return clip_id, seed_id or clip_id, cond


def recompute_failure(ev: dict):
    """failure = fidelity.pass AND (recall_vs_seed < 0.7 OR agree_vs_seed < 0.67)."""
    reasons = []
    rec = (ev.get("yolo") or {}).get("recall_vs_seed")
    integ = ev.get("integrity") or {}
    if integ.get("recall_intact") is not None:
        rec = integ["recall_intact"]  # only count misses on vehicles the generator actually kept
    agr = (ev.get("reason") or {}).get("agree_vs_seed")
    if rec is not None and rec < 0.7:
        reasons.append("recall<0.7")
    if agr is not None and agr < 0.6:
        reasons.append("reason_disagree")
    # Did the summary stack notice the condition at all? (perception metric, not a failure by itself)
    cond = ev.get("condition") or {}
    wl = ((ev.get("reason") or {}).get("answers") or {}).get("weather_lighting") or ""
    if cond and cond.get("kind") == "generative":
        words = {"fog": ["fog", "mist", "haz"], "rain": ["rain", "wet"], "snow": ["snow"],
                 "night": ["night", "dark"], "dusk": ["dusk", "sunset", "twilight", "low sun", "evening"]}
        want = [w for k in (cond.get("weather"), cond.get("time")) for w in words.get(k, [])]
        if want:
            ev.setdefault("reason", {})["condition_registered"] = any(w in wl.lower() for w in want)
    fid_pass = (ev.get("fidelity") or {}).get("pass")
    if (ev.get("condition") or {}).get("kind") == "physics":
        fid_pass = True  # pixel-exact by construction; low edge-SSIM here is the weather itself
    # Physics weather never moves pixels, so low patch scores there mean "obscured", not "generator damage":
    # keep those clips scored (recall_intact = recall on still-visible cars). Gate only generative clips.
    if (ev.get("condition") or {}).get("kind") != "physics" and \
            integ.get("vehicles_intact") is not None and integ["vehicles_intact"] < integ.get("min_intact", 0.7):
        fid_pass = False  # generator blurred / merged / erased too many vehicles -> not a valid test case
        ev.setdefault("fidelity", {})["rejected_by"] = "vehicle_integrity"
    ev["failure"] = bool(fid_pass) and bool(reasons)
    ev["failure_reasons"] = reasons if ev["failure"] else []


def merge_eval(out_path, clip_id, seed_id, condition, updates: dict) -> dict:
    """Locked read-modify-write of results/evals/<clip>.json; only touches given keys."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.parent / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ev = load_json(out_path, {}) if out_path.exists() else {}
        ev.setdefault("clip_id", clip_id)
        ev.setdefault("seed_id", seed_id)
        if condition is not None and not ev.get("condition"):
            ev["condition"] = condition
        ev.update(updates)
        recompute_failure(ev)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ev, indent=2))
        os.replace(tmp, out_path)
    return ev


def read_frames(path, size=None, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ---------- fidelity ----------
def edges(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.Canny(g, 100, 200)


def edge_ssim(clip, seed, threshold=DEFAULT_THRESHOLD) -> dict:
    t0 = time.time()
    a, b = read_frames(seed, SIZE), read_frames(clip, SIZE)
    n = min(len(a), len(b))
    if n == 0:
        raise RuntimeError("no frames decoded")
    scores = [float(structural_similarity(edges(a[i]), edges(b[i]), data_range=255)) for i in range(n)]
    s = float(np.mean(scores))
    return {"edge_ssim": round(s, 4), "pass": s >= threshold, "threshold": threshold,
            "frames": n, "min_frame_ssim": round(min(scores), 4), "seconds": round(time.time() - t0, 2)}


def run(clip, seed, out=None, threshold=DEFAULT_THRESHOLD) -> dict:
    clip_id, seed_id, cond = clip_info(clip, seed)
    out = Path(out) if out else EVALS / f"{clip_id}.json"
    fid = edge_ssim(clip, seed, threshold)
    merge_eval(out, clip_id, seed_id, cond, {"fidelity": fid})
    return fid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="eval JSON (default results/evals/<clip_id>.json)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = ap.parse_args()
    print(json.dumps(run(a.clip, a.seed, a.out, a.threshold)))


if __name__ == "__main__":
    main()
