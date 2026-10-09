"""BEFORE snapshot for the VSS re-ingest prompt fix (read-only; does not re-ingest).

    python3 fix/reingest_before.py              # -> results/fix_before.json, results/fix/before_frames*.jpg

1. Picks the 10 latest complete neighborhood_cam-1 chunks (fix/reingest_common.targets).
2. Records their per-segment VSS captions (reasoning_content from Explore timeline).
3. Runs the fog / night / rain queries from vss/coverage.py through /api/v1/search,
   metadata_filters camera_id=neighborhood_cam-1, top-10, with scores.
4. Downloads every neighborhood_cam-1 chunk (client.download), extracts 3 frames each
   (10%/50%/90%), builds contact sheets: results/fix/before_frames.jpg (targets) and
   results/fix/before_frames_others.jpg (non-target chunks, needed because search hits
   land outside the target set). Ground-truth labels are read from fix/visual_labels.json,
   written by a human/agent looking at those sheets -- not from captions.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reingest_common import (BEFORE, CAMERA, CUSTOM_PROMPT, QUERIES, QUERY_IDS, ROOT, SET, camera_chunks,
                             caption_rate, client, dump, load_labels, precision, search_hits, seg_captions,
                             targets)

VID = ROOT / "data" / "fix_frames" / "videos"
FRM = ROOT / "data" / "fix_frames" / "frames"
SHEETS = ROOT / "results" / "fix"
FRACS = (0.1, 0.5, 0.9)


def short(c: dict) -> str:
    return Path(c["filename"]).stem.split("_neighborhood_")[-1]  # e.g. 20260902_chunk_0012


def frames_for(c: dict) -> list[Path]:
    v = VID / c["filename"]
    if not v.exists() or v.stat().st_size < 10_000:
        client.download(c["original_video"], v)
    dur = float(c.get("chunk_duration_sec") or 0) or float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
        capture_output=True, text=True).stdout.strip() or 5)
    out = []
    for i, f in enumerate(FRACS):
        p = FRM / f"{short(c)}_{i}.jpg"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * f:.2f}", "-i", str(v), "-frames:v", "1",
                            "-vf", f"scale=480:-2,drawtext=text='{short(c)} f{i}':x=6:y=6:fontsize=20:"
                                   "fontcolor=yellow:box=1:boxcolor=black@0.6", str(p)], check=False)
        if p.exists():
            out.append(p)
    return out


def sheet(chunks: list[dict], frames: dict, out: Path) -> None:
    """Rows = chunks, cols = 3 frames."""
    rows = []
    tmp = out.parent / "_rows"
    tmp.mkdir(parents=True, exist_ok=True)
    for c in chunks:
        fs = frames.get(c["original_video"]) or []
        if len(fs) != 3:
            continue
        r = tmp / f"{short(c)}.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "error", *sum([["-i", str(f)] for f in fs], []),
                        "-filter_complex", "[0][1][2]hstack=inputs=3", str(r)], check=True)
        rows.append(r)
    if not rows:
        return
    subprocess.run(["ffmpeg", "-y", "-v", "error", *sum([["-i", str(r)] for r in rows], []),
                    "-filter_complex", f"{''.join(f'[{i}]' for i in range(len(rows)))}vstack=inputs={len(rows)}",
                    "-q:v", "4", str(out)], check=True)
    for r in rows:
        r.unlink()
    tmp.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    chunks = client.explore_all()
    cam = camera_chunks(chunks)
    tgts = targets(chunks)
    tset = {c["original_video"] for c in tgts}
    others = sorted([c for c in cam if c["original_video"] not in tset], key=lambda c: c["filename"])

    frames = {}
    if not a.no_frames:
        with ThreadPoolExecutor(6) as ex:
            for c, fs in zip(cam, ex.map(frames_for, cam)):
                frames[c["original_video"]] = fs
        sheet(sorted(tgts, key=lambda c: c["filename"]), frames, SHEETS / "before_frames.jpg")
        half = len(others) // 2
        sheet(others[:half], frames, SHEETS / "before_frames_others_a.jpg")
        sheet(others[half:], frames, SHEETS / "before_frames_others_b.jpg")

    labels = load_labels()
    searches = {q: search_hits(q) for q in QUERY_IDS}
    out = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "set": SET, "camera_id": CAMERA,
        "selection": "10 latest complete chunks (every segment 1..total_segments) by (upload_timestamp, chunk_index)",
        "custom_prompt": CUSTOM_PROMPT,
        "targets": [{
            "original_video": c["original_video"], "filename": c["filename"], "stream_id": c.get("stream_id"),
            "chunk_index": c.get("chunk_index"), "total_segments": c.get("total_segments"),
            "upload_timestamp": c.get("upload_timestamp"),
            "visual_label": labels.get(c["original_video"]),
            "segments": seg_captions(c),
        } for c in tgts],
        "n_target_segments": sum(len(c["timeline"]) for c in tgts),
        "labels_note": "labels by visual inspection of frames (3 frames/chunk at 10/50/90% of the chunk; "
                       "results/fix/before_frames.jpg for targets, before_frames_others_{a,b}.jpg for the other "
                       "neighborhood_cam-1 chunks); never derived from VSS captions. Source: fix/visual_labels.json",
        "visual_labels_all_camera_chunks": labels,
        "queries": {q: {"query": QUERIES[q], "endpoint": "POST /api/v1/search",
                        "metadata_filters": {"camera_id": CAMERA}, "top_k": 10,
                        "hits": [dict(h, in_target_set=h["original_video"] in tset) for h in searches[q]],
                        "precision_at_10": precision(searches[q], q, labels) if labels else None}
                    for q in QUERY_IDS},
        "captions_naming_condition": caption_rate(tgts, labels) if labels else None,
        "seconds": round(time.time() - t0, 1),
    }
    dump(BEFORE, out)
    print(f"targets ({len(tgts)}, {out['n_target_segments']} segments):")
    for t in out["targets"]:
        print(f"  {t['filename']}  idx={t['chunk_index']}  segs={len(t['segments'])}  label={t['visual_label']}")
    for q in QUERY_IDS:
        print(f"  {q:<6} P@10={out['queries'][q]['precision_at_10']}")
    print("captions naming condition:", out["captions_naming_condition"])
    print(f"-> {BEFORE} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
