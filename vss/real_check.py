"""Search the REAL VSS archive for adverse-condition footage and pull short clips for confirmation.

Two independent signals per condition:
  1. VSS hybrid search  (POST /api/v1/search, several phrasings, all real cameras)
  2. keyword scan of every indexed segment caption (reasoning_content) from explore_all()
A clip is "both" when a search hit's own caption also matches the condition keywords (non-negated).

Re-runnable / growing corpus: chunks already seen are remembered in results/real_check.json (`_seen`);
each run scans only newly indexed chunks for new candidates, appends picks, and logs a timestamped run.

  python3 vss/real_check.py                 # default: up to 2 new picks per condition per run
  python3 vss/real_check.py --per-cond 3 --no-download
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402

ROOT = client.ROOT
OUT = ROOT / "results" / "real_check.json"
STATE = ROOT / "data" / "real_check" / "state.json"
DIR = ROOT / "data" / "real_check"
RAW = DIR / "raw"
SYNTH_CAM = "spore_synthetic"
INDOOR = {"sdg_warehouse_cam-2", "smartspace_cam-1"}

CONDITIONS = {
    "night": {"condition": {"weather": "clear", "time": "night", "intensity": "heavy", "kind": "real"},
              "queries": ["street at night with headlights and streetlights", "highway traffic at night",
                          "dark nighttime road illuminated by street lamps"],
              "kw": [r"\bnight(?:time)?\b", r"\bdarkness\b", r"\bafter dark\b", r"\bdark sky\b"]},
    "rain": {"condition": {"weather": "rain", "time": "day", "intensity": "heavy", "kind": "real"},
             "queries": ["rainy street with wet road and reflections", "traffic in heavy rain",
                         "wet pavement, raindrops, umbrellas"],
             "kw": [r"\brain(?:y|ing|fall|drops?)?\b", r"\bwet\b", r"\bumbrellas?\b", r"\bpuddles?\b", r"\bdrizzle\b"]},
    "fog": {"condition": {"weather": "fog", "time": "day", "intensity": "heavy", "kind": "real"},
            "queries": ["highway in dense fog with low visibility", "foggy misty road", "hazy overcast low visibility"],
            "kw": [r"\bfog(?:gy)?\b", r"\bmist(?:y)?\b", r"\bhaz(?:e|y)\b", r"low visibility", r"reduced visibility"]},
    "snow": {"condition": {"weather": "snow", "time": "day", "intensity": "heavy", "kind": "real"},
             "queries": ["snowy street with snow on the ground", "driving in snowfall", "winter road covered in snow"],
             "kw": [r"\bsnow(?:y|ing|fall)?\b", r"\bslush\b", r"\bicy\b", r"\bwintry\b", r"\bsleet\b"]},
    "dusk": {"condition": {"weather": "clear", "time": "dusk", "intensity": "light", "kind": "real"},
             "queries": ["street at dusk with low sun", "sunset golden hour traffic", "twilight road with fading light"],
             "kw": [r"\bdusk\b", r"\bsunset\b", r"\btwilight\b", r"golden hour", r"\blow sun\b", r"\bsunrise\b", r"\bdawn\b"]},
    "glare": {"condition": {"weather": "clear", "time": "day", "intensity": "heavy", "kind": "real", "glare": True},
              "queries": ["strong sun glare into the camera", "lens flare from bright sunlight", "headlight glare blinding"],
              "kw": [r"\bglare\b", r"lens flare", r"\bblinding\b", r"\bwashed out\b", r"\boverexposed\b",
                     r"sun (?:is )?(?:shining )?directly"]},
}
NEG = re.compile(r"\b(no|not|without|absence of|free of|nor|neither|lack of|isn't|aren't|dry)\b[^.]{0,40}$", re.I)


def load(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def kw_hits(text: str, pats: list[str]) -> list[str]:
    """Keyword matches that are not negated within the same sentence prefix (e.g. 'no rain', 'without fog')."""
    hits = []
    for pat in pats:
        for m in re.finditer(pat, text or "", re.I):
            pre = (text[max(0, m.start() - 60):m.start()]).split(".")[-1]
            if NEG.search(pre):
                continue
            hits.append(m.group(0).lower())
    return hits


def segments_from_explore(chunks: list[dict]) -> list[dict]:
    segs = []
    for c in chunks:
        if c.get("camera_id") == SYNTH_CAM:
            continue
        tl = c.get("timeline") or [{"segment_number": c.get("best_segment_number"),
                                    "segment_start_sec": c.get("best_match_start_sec", 0),
                                    "segment_end_sec": c.get("best_match_end_sec", 5),
                                    "source": c.get("preview_source"),
                                    "reasoning_content": c.get("reasoning_content")}]
        for s in tl:
            segs.append({"original_video": c["original_video"], "camera_id": c.get("camera_id"),
                         "location": c.get("location"), "chunk_duration_sec": c.get("chunk_duration_sec"),
                         "upload_timestamp": c.get("upload_timestamp"), "source": s.get("source"),
                         "segment_start_sec": s.get("segment_start_sec") or 0.0,
                         "segment_end_sec": s.get("segment_end_sec"), "caption": s.get("reasoning_content") or ""})
    return segs


def trim(raw: Path, out: Path, start: float, dur_total: float | None):
    start = max(0.0, min(start, (dur_total or start + 6) - 93 / 16))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(raw),
                    "-vf", "fps=16,scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                    "-frames:v", "93", "-an", "-pix_fmt", "yuv420p", str(out)], check=True)
    return start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cond", type=int, default=2, help="new picks per condition per run")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="on re-runs also pick from already-seen chunks")
    ap.add_argument("--min-sim", type=float, default=0.0)
    args = ap.parse_args()
    t0 = time.time()
    DIR.mkdir(parents=True, exist_ok=True)

    out = load(OUT, [])
    state = load(STATE, {"seen_chunks": [], "runs": []})
    seen = set(state["seen_chunks"])
    picked = {c["vss_id"] + "#" + str(c.get("segment_start_sec")) for e in out for c in e.get("clips", [])}
    picked_videos = {c["vss_id"] for e in out for c in e.get("clips", [])}

    chunks = client.explore_all()
    real_chunks = [c for c in chunks if c.get("camera_id") != SYNTH_CAM]
    new_chunks = [c for c in real_chunks if c["original_video"] not in seen]
    segs = segments_from_explore(real_chunks)
    seg_by_src = {s["source"]: s for s in segs}
    cams = Counter(c.get("camera_id") for c in real_chunks)
    new_cams = Counter(c.get("camera_id") for c in new_chunks)
    print(f"archive: {len(real_chunks)} real chunks, {len(segs)} segments; new since last run: {len(new_chunks)} "
          f"{dict(new_cams)}", flush=True)

    run_log = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "real_chunks": len(real_chunks), "segments": len(segs),
               "new_chunks": len(new_chunks), "new_by_camera": dict(new_cams), "picks": {}}
    by_name = {e.get("condition_name"): e for e in out}

    for name, spec in CONDITIONS.items():
        # 1) keyword scan of all captions
        kw_segs = []
        for s in segs:
            h = kw_hits(s["caption"], spec["kw"])
            if h:
                kw_segs.append((s, h))
        kw_by_cam = Counter(s["camera_id"] for s, _ in kw_segs)
        # 2) VSS search, real cameras only
        cands: dict[str, dict] = {}
        search_log = []
        for q in spec["queries"]:
            ts = time.time()
            res = client.search(q, k=30, llm_top_n=1, min_similarity=args.min_sim)
            rows = [r for r in res.get("results", []) if r.get("camera_id", "") != SYNTH_CAM
                    and SYNTH_CAM not in json.dumps(r.get("camera_id", ""))]
            search_log.append({"query": q, "n": len(rows), "ms": round((time.time() - ts) * 1000),
                               "top_cameras": dict(Counter(r.get("camera_id") for r in rows[:10]))})
            for rank, r in enumerate(rows):
                src = r.get("source")
                cap = r.get("reasoning_content") or seg_by_src.get(src, {}).get("caption", "")
                h = kw_hits(cap, spec["kw"])
                key = src
                c = cands.setdefault(key, {"source": src, "original_video": r.get("original_video"),
                                           "camera_id": r.get("camera_id"), "caption": cap,
                                           "segment_start_sec": r.get("segment_start_sec") or 0.0,
                                           "chunk_duration_sec": seg_by_src.get(src, {}).get("chunk_duration_sec"),
                                           "best_sim": 0.0, "queries": [], "kw": h, "best_rank": rank})
                c["best_sim"] = max(c["best_sim"], float(r.get("similarity_score") or 0))
                c["best_rank"] = min(c["best_rank"], rank)
                c["queries"].append(q)
        for s, h in kw_segs:
            c = cands.setdefault(s["source"], {"source": s["source"], "original_video": s["original_video"],
                                               "camera_id": s["camera_id"], "caption": s["caption"],
                                               "segment_start_sec": s["segment_start_sec"],
                                               "chunk_duration_sec": s["chunk_duration_sec"], "best_sim": 0.0,
                                               "queries": [], "kw": h, "best_rank": 999})
        for c in cands.values():
            c["signal"] = "both" if c["queries"] and c["kw"] else ("search_only" if c["queries"] else "keyword_only")
            c["score"] = (2 if c["signal"] == "both" else 1 if c["signal"] == "keyword_only" else 0) \
                + min(len(set(c["kw"])), 3) * 0.3 + c["best_sim"]
            c["is_new"] = c["original_video"] not in seen
        # pick: VSS's own caption must support the condition; indoor cameras excluded; strong = search+caption
        # agree, or >=2 distinct condition keywords. Prefer new chunks, then strong, then score.
        confirmed_kw = [c for c in cands.values() if c["kw"] and c["camera_id"] not in INDOOR]
        for c in confirmed_kw:
            c["strong"] = c["signal"] == "both" or len(set(c["kw"])) >= 2
        ranked = sorted(confirmed_kw, key=lambda c: (-c["is_new"], -c["strong"], -c["score"]))
        if seen and not args.backfill:  # re-run: only newly indexed chunks can add picks (growing corpus)
            ranked = [c for c in ranked if c["is_new"]]
        entry = by_name.get(name) or {"condition_name": name, "condition": spec["condition"],
                                      "vss_query": spec["queries"][0], "clips": [], "confirmed": None}
        entry["queries"] = spec["queries"]
        entry["search_log"] = search_log
        entry["archive_caption_matches"] = {"segments": len(kw_segs), "by_camera": dict(kw_by_cam)}
        entry["candidates"] = {"both": sum(c["signal"] == "both" for c in cands.values()),
                               "keyword_only": sum(c["signal"] == "keyword_only" for c in cands.values()),
                               "search_only": sum(c["signal"] == "search_only" for c in cands.values())}
        top_search_only = sorted([c for c in cands.values() if c["signal"] == "search_only"],
                                 key=lambda c: -c["best_sim"])[:3]
        entry["search_only_examples"] = [{"vss_id": c["original_video"], "camera_id": c["camera_id"],
                                          "similarity": round(c["best_sim"], 3), "caption": c["caption"][:300]}
                                         for c in top_search_only]
        new_picks, used_cams = [], Counter(c.get("camera_id") for c in entry["clips"])
        for c in ranked:
            if len(new_picks) >= args.per_cond:
                break
            k = c["original_video"] + "#" + str(c["segment_start_sec"])
            if k in picked or c["original_video"] in picked_videos:
                continue
            if new_picks and not c["strong"] and any(o["strong"] for o in ranked
                                                     if o["original_video"] not in picked_videos):
                continue
            if used_cams[c["camera_id"]] >= 1 and any(o["camera_id"] not in used_cams and o["strong"] >= c["strong"]
                                                       for o in ranked if o["original_video"] not in picked_videos):
                continue  # prefer another camera first
            new_picks.append(c)
            picked.add(k)
            picked_videos.add(c["original_video"])
            used_cams[c["camera_id"]] += 1
        for c in new_picks:
            stem = Path(c["original_video"]).stem
            clip_id = f"real_{name}_{c['camera_id']}_{stem[-24:]}_{int(c['segment_start_sec'])}".replace("-", "_")
            rec = {"clip_id": clip_id, "vss_id": c["original_video"], "segment_source": c["source"],
                   "camera_id": c["camera_id"], "segment_start_sec": c["segment_start_sec"],
                   "vss_query": c["queries"][0] if c["queries"] else None, "signal": c["signal"],
                   "similarity": round(c["best_sim"], 3), "keyword_hits": sorted(set(c["kw"])),
                   "strong": c["strong"], "condition_guess": name, "vss_caption": c["caption"], "human_note": "",
                   "added_at": run_log["ts"], "from_new_chunk": c["is_new"]}
            if not args.no_download:
                raw = RAW / Path(c["original_video"]).name
                if not raw.exists():
                    client.download(c["original_video"], raw)
                outp = DIR / f"{clip_id}.mp4"
                try:
                    rec["trim_start_s"] = trim(raw, outp, float(c["segment_start_sec"]), c.get("chunk_duration_sec"))
                    rec["src"] = str(outp.relative_to(ROOT))
                except subprocess.CalledProcessError as e:
                    rec["download_error"] = str(e)
            entry["clips"].append(rec)
        entry["found"] = bool(entry["clips"])
        if not entry["clips"]:
            entry["note"] = (f"No real {name} footage found: 0 non-negated '{name}' keywords in "
                             f"{len(segs)} VSS captions across {len(cams)} real cameras, and search hits' captions "
                             f"do not describe {name}. Synthetic is the only way to test this condition.")
        else:
            entry.pop("note", None)
        by_name[name] = entry
        run_log["picks"][name] = [c["camera_id"] for c in new_picks]
        print(f"{name:6s} caption-matches={len(kw_segs):4d} {dict(kw_by_cam)} | cands={entry['candidates']} "
              f"| new picks={[(c['camera_id'], round(c['best_sim'], 2), c['signal']) for c in new_picks]}", flush=True)

    state["seen_chunks"] = sorted(seen | {c["original_video"] for c in real_chunks})
    run_log["seconds"] = round(time.time() - t0, 1)
    state["runs"].append(run_log)
    STATE.write_text(json.dumps(state, indent=2))
    result = [by_name[n] for n in CONDITIONS if n in by_name] + \
        [e for k, e in by_name.items() if k not in CONDITIONS]
    for e in result:
        e["runs"] = [{"ts": r["ts"], "new_chunks": r["new_chunks"], "picks": r["picks"].get(e["condition_name"], [])}
                     for r in state["runs"]]
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)} in {run_log['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
