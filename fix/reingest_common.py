"""Shared bits for the VSS re-ingest prompt fix (fix/reingest_{before,fire,after}.py).

Target: camera_id neighborhood_cam-1, the 10 latest COMPLETE chunks (timeline has every
segment 1..total_segments). Ground-truth condition labels come from visual inspection of
extracted frames (fix/visual_labels.json), never from VSS captions.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vss"))
import client  # noqa: E402
from coverage import CONDITIONS, matches  # noqa: E402

CAMERA = "neighborhood_cam-1"
N_TARGET = 10
LABELS = ROOT / "fix" / "visual_labels.json"
JOBS = ROOT / "results" / "fix_jobs.json"
SET = "real archive, neighborhood_cam-1, n=10 chunks"

# Target sets. "latest" = the approved spec (10 latest complete chunks = 20260902 stream chunks 12..21,
# all visually night, captions already say night). "nightfog" = alternative with headroom: 20260902 stream
# chunks 2..11, visually night + fog/mist, whose captions name the fog in 3/60 segments.
TARGET_SETS = ("latest", "nightfog")


def before_path(which: str = "latest") -> Path:
    return ROOT / "results" / ("fix_before.json" if which == "latest" else f"fix_before_{which}.json")


BEFORE = before_path("latest")

CUSTOM_PROMPT = (
    "First state the weather (clear, rain, fog, snow), lighting (day, dusk, night, glare) and "
    "visibility (good, reduced, poor), and whether the road is wet. Then describe the scene: road "
    "type, every vehicle you can clearly see with its type and color, people, and any notable events "
    "such as stops, lane changes or near-misses. Give a vehicle count of only clearly visible vehicles."
)

# Same phrasings as vss/coverage.py
QUERY_IDS = ["fog", "night", "rain"]
QUERIES = {c["id"]: c["query"] for c in CONDITIONS if c["id"] in QUERY_IDS}
COND = {c["id"]: c for c in CONDITIONS}


def complete(c: dict) -> bool:
    segs = sorted(t["segment_number"] for t in c.get("timeline") or [])
    return bool(segs) and segs == list(range(1, int(c.get("total_segments") or 0) + 1))


def camera_chunks(chunks: list[dict] | None = None) -> list[dict]:
    chunks = chunks if chunks is not None else client.explore_all()
    return [c for c in chunks if c.get("camera_id") == CAMERA]


def latest_key(c: dict):
    # latest = most recent upload session, then highest chunk index (filename timestamp as tiebreak)
    return (c.get("upload_timestamp") or "", c.get("chunk_index") if c.get("chunk_index") is not None else -1,
            c.get("filename") or "")


def targets(chunks: list[dict] | None = None, which: str = "latest") -> list[dict]:
    cam = sorted([c for c in camera_chunks(chunks) if complete(c)], key=latest_key, reverse=True)
    if which == "latest":
        return cam[:N_TARGET]
    if which == "nightfog":
        return [c for c in cam if "_neighborhood_20260902_chunk_" in c["filename"]
                and 2 <= int(c["filename"].rsplit("_chunk_", 1)[1][:4]) <= 11]
    raise ValueError(which)


def seg_captions(c: dict) -> list[dict]:
    return [{"segment_number": t["segment_number"], "source": t["source"],
             "caption": t.get("reasoning_content") or ""}
            for t in sorted(c["timeline"], key=lambda t: t["segment_number"])]


def search_hits(qid: str, k: int = 10) -> list[dict]:
    r = client.search(QUERIES[qid], k=k, metadata_filters={"camera_id": CAMERA})
    if not r.get("results") and "detail" in r:
        raise RuntimeError(f"search error: {str(r)[:300]}")
    return [{"rank": i + 1, "original_video": h.get("original_video"), "filename": h.get("filename"),
             "segment_number": h.get("segment_number"), "score": round(float(h.get("similarity_score") or 0), 4),
             "caption": (h.get("reasoning_content") or "")[:240]}
            for i, h in enumerate(r.get("results", [])[:k])]


def load_labels() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def chunk_conditions(label: dict) -> set[str]:
    """Set of {fog, night, rain, clear} the chunk visually shows."""
    return set(label.get("conditions") or [])


def precision(hits: list[dict], qid: str, labels: dict) -> dict:
    rel = unl = 0
    for h in hits:
        lab = labels.get(h["original_video"])
        if lab is None:
            unl += 1
        elif qid in chunk_conditions(lab):
            rel += 1
    n = len(hits)
    return {"relevant": rel, "n": n, "unlabeled": unl, "p": round(rel / n, 3) if n else None}


def caption_names(caption: str, cond: str) -> bool:
    if cond == "clear":
        return bool(re.search(r"\bclear\b", (caption or "").lower()))
    return matches(COND[cond], caption)


def caption_rate(tgts: list[dict], labels: dict) -> dict:
    """Fraction of target segments whose caption names (one of) the chunk's visually-labeled condition(s)."""
    hit = tot = 0
    for c in tgts:
        conds = chunk_conditions(labels.get(c["original_video"], {}))
        if not conds:
            continue
        for s in seg_captions(c):
            tot += 1
            hit += all(caption_names(s["caption"], x) for x in conds)
    return {"hit": hit, "n": tot, "pct": round(100 * hit / tot, 1) if tot else None}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
