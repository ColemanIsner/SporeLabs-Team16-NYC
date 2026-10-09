"""Upload Spore synthetic variants into VSS so the provided stack (Detector / Reasoner / Embedder)
processes them like any other footage, then poll until indexed and record VSS's own row.

  python3 vss/upload_synthetic.py                 # upload new non-bad clips, poll until indexed
  python3 vss/upload_synthetic.py --no-upload     # only poll / refresh rows for already-uploaded clips
  python3 vss/upload_synthetic.py --include-bad   # also upload clips rated "bad" in results/reviews.json

State: results/vss_uploads.json  {clip_id: {object_key, uploaded_at, fields, indexed, vss_row, ...}}
Idempotent: clips already in vss_uploads.json are never re-uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402

ROOT = client.ROOT
MANIFEST = ROOT / "data" / "synthetic" / "manifest.json"
SEEDS = ROOT / "data" / "seeds" / "manifest.json"
REVIEWS = ROOT / "results" / "reviews.json"
STATE = ROOT / "results" / "vss_uploads.json"
CAMERA = "spore_synthetic"


def load(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def save_state(st):
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(STATE)


def bad_clips(baseline: dict) -> set[str]:
    """clip_ids rated "bad". Entries identical to the 12:42 baseline were ratings of the old
    control_weight=1.0 set (moved to data/synthetic_cw1/) and do not apply to new clips at the same path."""
    out = set()
    for k, v in load(REVIEWS, {}).items():
        if (v or {}).get("rating") == "bad" and baseline.get(k) != v:
            out.add(v.get("clip_id") or Path(k).stem)
    return out


def control_weight(row: dict) -> float:
    for k in ("control_weight", "cw"):
        if row.get(k) is not None:
            return float(row[k])
    if isinstance(row.get("params"), dict) and row["params"].get("control_weight") is not None:
        return float(row["params"]["control_weight"])
    return 0.5  # everything in data/synthetic/ after the 13:2x regen is cw0.5


def scenario_for(dataset: str) -> str:
    d = dataset.lower()
    if d.startswith("i24") or d.startswith("nyc") or d.startswith("sf_"):
        return "traffic"
    if d.startswith("pie"):
        return "live_driving"
    return "general"


def fields_for(row: dict, seeds: dict) -> dict:
    seed = seeds.get(row["seed_id"], {})
    dataset = seed.get("dataset", row["seed_id"].split("_")[0])
    c = row["condition"]
    tags = [row["seed_id"], c.get("weather"), c.get("time"), c.get("intensity"), "spore",
            f"cw{control_weight(row):g}"]
    return {"is_public": "false", "camera_id": CAMERA, "location": dataset, "capture_type": "synthetic",
            "tags": ",".join(t for t in tags if t), "scenario": scenario_for(dataset)}


def mine_spore_chunks() -> list[dict]:
    out, off = [], 0
    while True:
        ch = client._get(f"/api/v1/videos/explore?scope=mine&limit=48&offset={off}").get("chunks", [])
        out += ch
        if len(ch) < 48:
            break
        off += 48
    return [c for c in out if c.get("camera_id") == CAMERA or any(t == "spore" for t in (c.get("tags") or []))
            or "__" in (c.get("filename") or "")]


def vss_row(chunk: dict) -> dict:
    segs = []
    for s in chunk.get("timeline") or []:
        segs.append({k: s.get(k) for k in ("segment_number", "segment_start_sec", "segment_end_sec", "source",
                                            "reasoning_content")})
    return {"original_video": chunk.get("original_video"), "filename": chunk.get("filename"),
            "camera_id": chunk.get("camera_id"), "location": chunk.get("location"),
            "capture_type": chunk.get("capture_type"), "tags": chunk.get("tags"),
            "upload_timestamp": chunk.get("upload_timestamp"), "total_segments": chunk.get("total_segments"),
            "reasoning_content": chunk.get("reasoning_content"), "segments": segs}


def match(st: dict) -> int:
    chunks = mine_spore_chunks()
    n = 0
    for cid, rec in st.items():
        if cid.startswith("_"):
            continue
        cid = rec.get("clip_id", cid)
        key_name = Path(rec.get("object_key") or "").name
        for c in chunks:
            fn = c.get("filename") or ""
            if key_name and (fn == key_name or Path(c.get("original_video", "")).name == key_name):
                rec["vss_row"] = vss_row(c)
                if not rec.get("indexed"):
                    rec["indexed"] = True
                    rec["indexed_at"] = time.time()
                    rec["index_latency_s"] = round(rec["indexed_at"] - rec.get("uploaded_at", rec["indexed_at"]), 1)
                n += 1
                break
    return n


def upload_new(args, st, seeds, exts, max_mb) -> int:
    rows = [r for r in load(MANIFEST, []) if r.get("seed_id") and r.get("condition")]
    if not args.include_pixel:
        rows = [r for r in rows if r["condition"].get("kind", "generative") != "pixel"]
    bad = set() if args.include_bad else bad_clips(st.get("_meta", {}).get("stale_reviews_baseline", {}))
    only = set(args.only.split(",")) if args.only else None
    n = 0
    for r in rows:
        cid = r["clip_id"]
        if (only and cid not in only) or cid in st:
            continue
        if cid in bad:
            if cid not in _SKIPPED:
                print("skip (rated bad)", cid)
                _SKIPPED.add(cid)
            continue
        p = ROOT / r["src"]
        if not p.exists() or p.suffix not in exts or p.stat().st_size > max_mb * 1e6 \
                or time.time() - p.stat().st_mtime < 10:
            continue  # missing / still being written / not allowed
        f = fields_for(r, seeds)
        t0 = time.time()
        res = client.upload(p, f)
        if not res.get("success") and "capture_type" in json.dumps(res):
            f["capture_type"] = "traffic"
            res = client.upload(p, f)
        print("upload", cid, round(time.time() - t0, 1), "s", res.get("object_key") or res, flush=True)
        if res.get("success"):
            st[cid] = {"clip_id": cid, "seed_id": r["seed_id"], "condition": r["condition"],
                       "object_key": res.get("object_key"), "uploaded_at": time.time(),
                       "upload_seconds": round(time.time() - t0, 1), "fields": f, "indexed": False,
                       "control_weight": control_weight(r), "src": r["src"]}
            save_state(st)
            n += 1
    return n


_SKIPPED: set = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--include-pixel", action="store_true")
    ap.add_argument("--watch", action="store_true", help="keep re-reading the manifest and uploading new clips")
    ap.add_argument("--timeout", type=int, default=900, help="max seconds to poll / watch")
    ap.add_argument("--only", default=None, help="comma-separated clip_ids")
    args = ap.parse_args()

    cfg = client._get("/api/v1/config").get("app", {})
    max_mb = cfg.get("max_upload_size_mb", 25)
    exts = cfg.get("allowed_video_extensions", [".mp4"])
    st = load(STATE, {})

    t_end = time.time() + args.timeout
    while True:
        if not args.no_upload:
            upload_new(args, st, load_seeds(), exts, max_mb)
        n = match(st)
        save_state(st)
        pending = [c for c, r in st.items() if not c.startswith("_") and not r.get("indexed")]
        total = len([c for c in st if not c.startswith("_")])
        print(f"{time.strftime('%H:%M:%S')} indexed {n}/{total}; pending {len(pending)}", flush=True)
        if time.time() > t_end or (not pending and not args.watch):
            break
        time.sleep(20)


def load_seeds() -> dict:
    return {s["seed_id"]: s for s in load(SEEDS, [])}


if __name__ == "__main__":
    main()
