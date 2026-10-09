#!/usr/bin/env python3
"""Run fidelity + YOLO for every seed and synthetic clip -> results/evals/<clip_id>.json.

Usage:
  eval/.venv/bin/python eval/run_all.py [--force] [--device cpu|mps] [--only SUBSTR]
      [--seeds data/seeds/manifest.json] [--synthetic data/synthetic/manifest.json] [--reason]

--reason: afterwards run eval/reason_eval.py (Cosmos3-Reason Q&A) with the same --only/--force/manifests.

- Seeds are compared to themselves (edge_ssim 1.0, recall 1.0; their boxes become pseudo-GT).
- Clips rated "bad" in results/reviews.json ({repo-relative path: {rating, note}}) are skipped;
  if they already have an eval JSON it gets "excluded": true.
- Idempotent: a clip is skipped when its eval JSON already has fidelity+yolo and is newer than
  both the clip and its seed (use --force to recompute). YOLO boxes are cached in results/boxes/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity  # noqa: E402
import yolo_eval  # noqa: E402
from fidelity import EVALS, ROOT, load_json, merge_eval, rel  # noqa: E402

REVIEWS = ROOT / "results" / "reviews.json"


def resolve(src):
    p = Path(src)
    return p if p.is_absolute() else ROOT / p


def is_bad(reviews, path):
    r = reviews.get(rel(path)) or reviews.get(str(path))
    return isinstance(r, dict) and str(r.get("rating", "")).lower() == "bad"


def up_to_date(ev_path, *srcs):
    ev = load_json(ev_path, None)
    if not ev or "fidelity" not in ev or "yolo" not in ev:
        return False
    t = ev_path.stat().st_mtime
    return all(t >= Path(s).stat().st_mtime for s in srcs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=ROOT / "data" / "seeds" / "manifest.json")
    ap.add_argument("--synthetic", type=Path, default=ROOT / "data" / "synthetic" / "manifest.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="only clip_ids containing this substring")
    ap.add_argument("--reviews", type=Path, default=REVIEWS)
    ap.add_argument("--threshold", type=float, default=fidelity.DEFAULT_THRESHOLD)
    ap.add_argument("--reason", action="store_true", help="then run reason_eval.py (Cosmos3-Reason Q&A)")
    ap.add_argument("--backend", choices=["hosted", "local"], default=yolo_eval.BACKEND,
                    help="YOLO under test: hosted VSS YOLO11s (default; env SPORE_YOLO_BACKEND) or local yolo11n")
    ap.add_argument("--workers", type=int, default=6, help="parallel hosted YOLO requests")
    a = ap.parse_args()
    yolo_eval.set_backend(a.backend)

    reviews = load_json(a.reviews, {}) or {}
    seeds = load_json(a.seeds, []) or []
    synth = load_json(a.synthetic, []) or []
    seed_src = {s["seed_id"]: resolve(s["src"]) for s in seeds if s.get("seed_id") and s.get("src")}

    jobs = []  # (clip_id, clip_path, seed_path)
    for s in seeds:
        if s.get("seed_id") and s.get("src"):
            jobs.append((s["seed_id"], resolve(s["src"]), resolve(s["src"])))
    for v in synth:
        if not (v.get("clip_id") and v.get("src")):
            continue
        sp = seed_src.get(v.get("seed_id")) or ROOT / "data" / "seeds" / f"{v.get('seed_id')}.mp4"
        jobs.append((v["clip_id"], resolve(v["src"]), sp))

    stats = {"done": 0, "cached": 0, "bad": 0, "missing": 0, "error": 0}
    t_all = time.time()
    if a.backend == "hosted":  # prefetch boxes in parallel; the sequential loop below then hits the cache
        from concurrent.futures import ThreadPoolExecutor
        todo = {}
        for clip_id, clip, seed in jobs:
            if (a.only and a.only not in clip_id) or not clip.exists() or not seed.exists():
                continue
            if is_bad(reviews, clip) or (clip != seed and is_bad(reviews, seed)):
                continue
            if not a.force and up_to_date(EVALS / f"{clip_id}.json", clip, seed):
                continue
            todo[clip_id] = clip
            sid = fidelity.clip_info(seed, seed)[0]
            todo.setdefault(sid, seed)

        def fetch(item):
            cid, path = item
            try:
                return cid, yolo_eval.detect(path, cid, force=a.force)["seconds"], None
            except BaseException as e:  # noqa: BLE001
                return cid, None, e

        if todo:
            t0 = time.time()
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(fetch, todo.items()))
            secs = [r[1] for r in res if r[1] is not None]
            for cid, _, e in res:
                if e is not None:
                    print(f"[error]   hosted YOLO {cid}: {e!r}")
            print(f"[hosted]  {len(secs)}/{len(todo)} clips in {time.time() - t0:.1f}s wall "
                  f"({(time.time() - t0) / max(len(todo), 1):.2f}s/clip effective, "
                  f"{sum(secs) / max(len(secs), 1):.2f}s/clip per request, {a.workers} workers)", flush=True)
            a.force_boxes = False
    for clip_id, clip, seed in jobs:
        if a.only and a.only not in clip_id:
            continue
        ev_path = EVALS / f"{clip_id}.json"
        if not clip.exists() or not seed.exists():
            print(f"[missing] {clip_id}: {rel(clip) if not clip.exists() else rel(seed)}")
            stats["missing"] += 1
            continue
        if is_bad(reviews, clip) or (clip != seed and is_bad(reviews, seed)):
            print(f"[bad]     {clip_id} (review rating bad) -> skipped")
            if ev_path.exists():
                merge_eval(ev_path, clip_id, None, None, {"excluded": True})
            stats["bad"] += 1
            continue
        if not a.force and up_to_date(ev_path, clip, seed):
            if load_json(ev_path, {}).get("excluded"):
                merge_eval(ev_path, clip_id, None, None, {"excluded": False})
            stats["cached"] += 1
            continue
        try:
            t0 = time.time()
            fid = fidelity.run(clip, seed, ev_path, a.threshold)
            yo = yolo_eval.run(clip, seed, ev_path, a.device, getattr(a, "force_boxes", a.force))
            merge_eval(ev_path, clip_id, None, None, {"excluded": False})
            print(f"[ok]      {clip_id}: edge_ssim={fid['edge_ssim']} count={yo['mean_count']} "
                  f"recall={yo['recall_vs_seed']} ({time.time() - t0:.1f}s)")
            stats["done"] += 1
        except Exception as e:  # keep going; one broken clip must not stop the batch
            print(f"[error]   {clip_id}: {e!r}")
            stats["error"] += 1
    print(json.dumps({**stats, "seconds": round(time.time() - t_all, 1)}), flush=True)
    if a.reason:
        import subprocess
        cmd = [sys.executable, str(Path(__file__).with_name("reason_eval.py")), "--seeds", str(a.seeds),
               "--synthetic", str(a.synthetic), "--reviews", str(a.reviews)]
        cmd += (["--only", a.only] if a.only else []) + (["--force"] if a.force else [])
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
