#!/usr/bin/env python3
"""Cheap pixel-level conditions via ffmpeg (no GPU, stdlib only).

Usage:
  python3 gen/degrade.py --seed data/seeds/X.mp4 --cond glare --out data/synthetic/
  python3 gen/degrade.py --seed data/seeds/X.mp4 --cond all --out data/synthetic/

Writes <out>/<seed_id>__<cond>.mp4 (93 frames, 1280x720, 16 fps, h264) and upserts an
entry (kind "pixel") into <out>/manifest.json under a file lock. Idempotent: an existing
output is reused (manifest still upserted) unless --force.
"""
import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, NFRAMES = 1280, 720, 16, 93
NORM = f"fps={FPS},scale={W}:{H}:flags=bicubic,setsar=1"


def _lens_dirt_alpha(rng_seed=7, n=9):
    """geq alpha expression for a fixed set of smudges (rendered once to a PNG mask)."""
    rng = random.Random(rng_seed)
    spots = []
    for _ in range(n):
        x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        r = rng.uniform(0.04, 0.10)
        a = rng.uniform(0.35, 0.65)
        spots.append(f"{a:.2f}*exp(-((X/W-{x:.3f})*(X/W-{x:.3f})*3.16+(Y/H-{y:.3f})*(Y/H-{y:.3f}))/{2*r*r:.5f})")
    return f"255*min(0.85,{'+'.join(spots)})"


# condition name -> (filter chain applied after normalization, extra encoder args, condition dict)
CONDS = {
    "glare": (
        "format=yuv420p,geq=lum='min(255,lum(X,Y)*0.85+235*exp(-((X-W*0.68)*(X-W*0.68)+(Y-H*0.22)*(Y-H*0.22))/(2*(W*0.16)*(W*0.16))))'"
        ":cb='cb(X,Y)':cr='cr(X,Y)',eq=contrast=0.8:brightness=0.06",
        [], {"weather": "clear", "time": "day", "intensity": "heavy"}),
    "motion_blur": ("tmix=frames=4,avgblur=sizeX=14:sizeY=1", [],
                    {"weather": "clear", "time": "day", "intensity": "heavy"}),
    "low_bitrate": ("scale=320:180:flags=area,scale=1280:720:flags=neighbor", ["-crf", "44"],
                    {"weather": "clear", "time": "day", "intensity": "heavy"}),
    "lens_dirt": ("gblur=sigma=1.2", [], {"weather": "clear", "time": "day", "intensity": "heavy"}),  # + mask overlay
    "fog_pixel": (
        "format=yuv420p,geq=lum='lum(X,Y)*(0.30+0.40*Y/H)+190*(0.70-0.40*Y/H)'"
        ":cb='128+(cb(X,Y)-128)*0.4':cr='128+(cr(X,Y)-128)*0.4',gblur=sigma=1.5",
        [], {"weather": "fog", "time": "day", "intensity": "heavy"}),
    "night_pixel": ("eq=brightness=-0.12:gamma=0.6:saturation=0.45,colorbalance=bs=0.12:bm=0.06,noise=alls=14:allf=t",
                    [], {"weather": "clear", "time": "night", "intensity": "heavy"}),
}


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def seed_id_for(seed: Path) -> str:
    man = ROOT / "data" / "seeds" / "manifest.json"
    if man.exists():
        try:
            for r in json.loads(man.read_text()):
                if Path(r.get("src", "")).resolve() == seed.resolve() or (ROOT / r.get("src", "")).resolve() == seed.resolve():
                    return r["seed_id"]
        except Exception:
            pass
    return seed.stem


def upsert_manifest(man: Path, entry: dict):
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(str(man) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = []
        if man.exists() and man.stat().st_size:
            rows = json.loads(man.read_text())
        rows = [r for r in rows if r.get("clip_id") != entry["clip_id"]] + [entry]
        tmp = man.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=2))
        os.replace(tmp, man)


def degrade(seed: Path, cond: str, out_dir: Path, force=False) -> dict:
    vf, enc, cdict = CONDS[cond]
    sid = seed_id_for(seed)
    clip_id = f"{sid}__{cond}"
    out = out_dir / f"{clip_id}.mp4"
    t0 = time.time()
    if force or not out.exists():
        tmp = out.with_name(out.stem + ".tmp.mp4")
        if cond == "lens_dirt":
            mask = out.with_name(out.stem + ".mask.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}",
                            "-vf", f"format=rgba,geq=r=58:g=48:b=38:a='{_lens_dirt_alpha()}'", "-frames:v", "1",
                            str(mask)], check=True)
            graph = ["-loop", "1", "-i", str(mask), "-filter_complex",
                     f"[0:v]{NORM}[b];[b][1:v]overlay=shortest=1,{vf},format=yuv420p[v]", "-map", "[v]"]
        else:
            mask, graph = None, ["-vf", f"{NORM},{vf},format=yuv420p"]
        cmd = (["ffmpeg", "-y", "-loglevel", "error", "-i", str(seed), "-an"] + graph +
               ["-frames:v", str(NFRAMES), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast"] +
               (enc or ["-crf", "20"]) + [str(tmp)])
        try:
            subprocess.run(cmd, check=True)
        finally:
            if mask is not None:
                mask.unlink(missing_ok=True)
        os.replace(tmp, out)
        secs = round(time.time() - t0, 2)
    else:
        secs = None
    entry = {
        "clip_id": clip_id, "seed_id": sid, "src": rel(out),
        "condition": {**cdict, "kind": "pixel", "pixel": cond},
        "prompt": None, "generator": f"ffmpeg/{cond}", "gpu": None,
    }
    if secs is not None:
        entry["gen_seconds"] = secs
    else:  # keep previous gen_seconds if present
        man = out_dir / "manifest.json"
        if man.exists():
            for r in json.loads(man.read_text() or "[]"):
                if r.get("clip_id") == clip_id and "gen_seconds" in r:
                    entry["gen_seconds"] = r["gen_seconds"]
    upsert_manifest(out_dir / "manifest.json", entry)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--cond", required=True, help="|".join(CONDS) + "|all (comma-separated ok)")
    ap.add_argument("--out", default=ROOT / "data" / "synthetic", type=Path)
    ap.add_argument("--force", action="store_true", help="regenerate even if output exists")
    a = ap.parse_args()
    conds = list(CONDS) if a.cond == "all" else a.cond.split(",")
    bad = [c for c in conds if c not in CONDS]
    if bad:
        sys.exit(f"unknown cond(s) {bad}; choose from {list(CONDS)}")
    if not a.seed.exists():
        sys.exit(f"seed not found: {a.seed}")
    a.out.mkdir(parents=True, exist_ok=True)
    for c in conds:
        e = degrade(a.seed, c, a.out, a.force)
        print(json.dumps({"clip_id": e["clip_id"], "src": e["src"], "gen_seconds": e.get("gen_seconds")}))


if __name__ == "__main__":
    main()
