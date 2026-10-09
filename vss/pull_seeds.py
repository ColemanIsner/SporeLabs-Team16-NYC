"""Download seed clips from VSS and normalize them for Cosmos Transfer.

Seeds: 6 s, 16 fps, 1280x720, 93 frames -> data/seeds/<seed_id>.mp4 + data/seeds/manifest.json

  python3 vss/pull_seeds.py --camera i24_cam-1 --chunks 0,4,8 --start 2
  python3 vss/pull_seeds.py --camera nyc_streets_cam-1 --chunks 0 --prefix nyc
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402

ROOT = client.ROOT
SEEDS = ROOT / "data" / "seeds"
RAW = ROOT / "data" / "raw"


def upsert(entry: dict):
    man = SEEDS / "manifest.json"
    rows = json.loads(man.read_text()) if man.exists() else []
    rows = [r for r in rows if r["seed_id"] != entry["seed_id"]] + [entry]
    man.write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--chunks", default="0,4,8", help="chunk indexes within each sub-camera")
    ap.add_argument("--start", type=float, default=2.0, help="seconds into the chunk to start the 6 s seed")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    want = {int(x) for x in args.chunks.split(",")}

    chunks = [c for c in client.explore_all() if c.get("camera_id") == args.camera]
    SEEDS.mkdir(parents=True, exist_ok=True)
    for c in sorted(chunks, key=lambda c: c["filename"]):
        m = re.search(r"(scene\d+_p\d+c\d+|[A-Za-z0-9]+)_chunk_(\d+)", c["filename"])
        sub, idx = (m.group(1), int(m.group(2))) if m else ("x", 0)
        if idx not in want:
            continue
        prefix = args.prefix or args.camera.split("_cam")[0]
        seed_id = f"{prefix}_{sub}_{idx:02d}".replace("-", "_")
        raw = client.download(c["original_video"], RAW / c["filename"])
        out = SEEDS / f"{seed_id}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(args.start), "-i", str(raw),
                        "-vf", "fps=16,scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                        "-frames:v", "93", "-an", "-pix_fmt", "yuv420p", str(out)], check=True)
        upsert({"seed_id": seed_id, "dataset": args.camera, "vss_id": c["original_video"],
                "src": str(out.relative_to(ROOT)), "start_s": args.start, "end_s": args.start + 93 / 16,
                "fps": 16, "width": 1280, "height": 720,
                "notes": (c.get("reasoning_content") or "")[:300]})
        print("seed", seed_id, out.stat().st_size // 1024, "KB", flush=True)


if __name__ == "__main__":
    main()
