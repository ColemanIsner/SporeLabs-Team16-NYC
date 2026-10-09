"""Stage 1 — SEARCH TO FIND THE GAP.

Uses the provided VSS stack two ways over the real archive:
  1. /api/v1/search with a plain-English condition query per condition (top-k hits + scores).
  2. explore_all() -> scan every indexed clip's VSS reasoning_content caption for condition keywords.

Coverage per condition = how many of the indexed clips actually show that condition
(caption match), split into all clips vs highway/traffic clips, plus whether the search
top hits really are that condition. Empty rows are the gap.

    python3 vss/coverage.py            # -> results/coverage.json
    python3 vss/coverage.py --k 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vss"))
import client  # noqa: E402

OUT = ROOT / "results" / "coverage.json"

# id -> (search query, caption regexes that must ALL match, arm hint for the generator)
CONDITIONS = [
    {"id": "clear_day", "label": "Clear daytime", "query": "highway in clear daytime sunshine",
     "patterns": [r"\b(daytime|daylight|sunny|sunlit|bright(ly)? lit|clear sky|clear weather|clear,? (bright|sunny))\b"],
     "arm": None},
    {"id": "night", "label": "Highway at night", "query": "highway at night",
     "patterns": [r"\b(night|nighttime|night-time|after dark|darkness)\b"],
     "arm": {"weather": "clear", "time": "night", "intensity": "light"}},
    {"id": "rain", "label": "Heavy rain", "query": "heavy rain on a wet road",
     "patterns": [r"\b(rain|raining|rainy|rainfall|downpour|drizzle|wet (road|pavement|surface|asphalt))\b"],
     "arm": {"weather": "rain", "time": "day", "intensity": "heavy"}},
    {"id": "fog", "label": "Dense fog", "query": "dense fog with low visibility",
     "patterns": [r"\b(fog|foggy|mist|misty|haze|hazy|low visibility)\b"],
     "arm": {"weather": "fog", "time": "day", "intensity": "heavy"}},
    {"id": "snow", "label": "Snow", "query": "snow covered road in a snowstorm",
     "patterns": [r"\b(snow|snowy|snowing|snowfall|blizzard|sleet)\b"],
     "arm": {"weather": "snow", "time": "day", "intensity": "heavy"}},
    {"id": "glare", "label": "Low sun glare", "query": "low sun glare into the camera",
     "patterns": [r"\b(glare|sun glare|low sun|setting sun|sunset|sunrise|lens flare)\b"],
     "arm": {"kind": "pixel", "pixel": "glare", "effect": "glare"}},
    {"id": "night_rain", "label": "Night + rain", "query": "highway at night in heavy rain",
     "patterns": [r"\b(night|nighttime|night-time|darkness)\b",
                  r"\b(rain|raining|rainy|downpour|drizzle|wet (road|pavement|surface|asphalt))\b"],
     "arm": {"weather": "rain", "time": "night", "intensity": "heavy"}},
]

NEG = re.compile(r"\b(no|not|without|absence of|free of|isn't|is not|are no)\s+(\w+\s+){0,2}$")
HIGHWAY = re.compile(r"\b(highway|freeway|interstate|expressway|motorway|multi-lane|lanes of traffic)\b")


def _hit(pattern: str, text: str) -> bool:
    for m in re.finditer(pattern, text):
        if not NEG.search(text[max(0, m.start() - 30):m.start()]):
            return True
    return False


def matches(cond: dict, caption: str) -> bool:
    t = (caption or "").lower()
    return bool(t) and all(_hit(p, t) for p in cond["patterns"])


def is_highway(clip: dict) -> bool:
    return clip.get("capture_type") == "traffic" or str(clip.get("camera_id", "")).startswith("i24") \
        or bool(HIGHWAY.search((clip.get("reasoning_content") or "").lower()))


def run_search(cond: dict, k: int) -> dict:
    t0 = time.time()
    try:
        res = client.search(cond["query"], k=k).get("results", [])
    except Exception as e:  # never die on one query
        return {"error": str(e)[:200], "hits": [], "seconds": round(time.time() - t0, 2)}
    hits = []
    for r in res:
        cap = r.get("reasoning_content") or ""
        hits.append({
            "filename": r.get("filename"), "camera_id": r.get("camera_id"),
            "capture_type": r.get("capture_type"), "score": round(float(r.get("similarity_score") or 0), 4),
            "caption_matches": matches(cond, cap), "highway": is_highway(r),
            "caption": cap[:220], "source": r.get("source"),
        })
    return {"hits": hits, "seconds": round(time.time() - t0, 2)}


def coverage(k: int = 10) -> dict:
    t0 = time.time()
    clips = client.explore_all()
    hw = [c for c in clips if is_highway(c)]
    with ThreadPoolExecutor(4) as ex:
        searches = list(ex.map(lambda c: run_search(c, k), CONDITIONS))
    rows = []
    for cond, s in zip(CONDITIONS, searches):
        m_all = [c for c in clips if matches(cond, c.get("reasoning_content"))]
        m_hw = [c for c in m_all if is_highway(c)]
        hits = s["hits"]
        rel = [h for h in hits if h["caption_matches"]]
        rows.append({
            "id": cond["id"], "label": cond["label"], "vss_query": cond["query"],
            "endpoint": "POST /api/v1/search", "top_k": k, "arm": cond["arm"],
            "n_clips": len(m_all), "n_highway": len(m_hw),
            "frac_clips": round(len(m_all) / max(1, len(clips)), 4),
            "frac_highway": round(len(m_hw) / max(1, len(hw)), 4),
            "cameras": sorted({c.get("camera_id") for c in m_all if c.get("camera_id")}),
            "search_top_score": max((h["score"] for h in hits), default=None),
            "search_relevant": len(rel), "search_relevant_highway": sum(1 for h in rel if h["highway"]),
            "search_precision": round(len(rel) / len(hits), 3) if hits else None,
            "top_hits": hits[:5], "search_seconds": s.get("seconds"), "error": s.get("error"),
            "gap": cond["arm"] is not None and len(m_hw) <= max(1, int(0.02 * len(hw))),
        })
    by_cam: dict = {}
    for c in clips:
        by_cam[c.get("camera_id") or "?"] = by_cam.get(c.get("camera_id") or "?", 0) + 1
    return {
        "updated": time.time(), "seconds": round(time.time() - t0, 1),
        "n_indexed": len(clips), "n_highway": len(hw), "cameras": by_cam,
        "method": "VSS /api/v1/search per condition query + keyword scan of VSS reasoning_content captions from /api/v1/videos/explore",
        "conditions": rows,
        "gaps": [r["id"] for r in rows if r["gap"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    cov = coverage(a.k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(cov, indent=2))
    tmp.replace(a.out)
    print(f"indexed={cov['n_indexed']} highway={cov['n_highway']} ({cov['seconds']}s)")
    for r in cov["conditions"]:
        print(f"  {r['label']:<18} all={r['n_clips']:>3} highway={r['n_highway']:>3} "
              f"search rel={r['search_relevant']}/{len(r['top_hits']) and r['top_k']} top={r['search_top_score']}"
              f"{'  <-- GAP' if r['gap'] else ''}")


if __name__ == "__main__":
    main()
