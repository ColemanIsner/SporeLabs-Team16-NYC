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

# Indoor-only conditions (warehouses don't get rain): matched on captions like the weather ones.
INDOOR_CONDITIONS = [
    {"id": "low_light", "label": "Dim / low light", "query": "dim poorly lit warehouse",
     "patterns": [r"\b(dim|dimly|poorly lit|low light|low-light|dark(ened)? (area|aisle|room)|shadowy)\b"], "arm": None},
    {"id": "occlusion", "label": "Heavy occlusion", "query": "people and objects blocking the view",
     "patterns": [r"\b(occlud\w*|obstruct\w*|blocking the view|partially (hidden|blocked|obscured)|crowded)\b"], "arm": None},
]

# Scene types: camera_id prefix first, then capture_type / caption keywords.
SCENES = [
    {"id": "highway", "label": "Highway / traffic cam", "cams": ["i24"],
     "conditions": ["night", "rain", "fog", "snow", "glare", "night_rain"]},
    {"id": "dashcam", "label": "Dashcam / driving", "cams": ["pie_"],
     "conditions": ["night", "rain", "fog", "snow", "glare", "night_rain"]},
    {"id": "city_street", "label": "City street cam", "cams": ["nyc_streets", "sf_streets"],
     "conditions": ["night", "rain", "snow", "night_rain"]},
    {"id": "residential", "label": "Residential / neighborhood", "cams": ["neighborhood"],
     "conditions": ["night", "rain", "snow"]},
    {"id": "bike", "label": "Bike / egocentric", "cams": ["nyc_bike"],
     "conditions": ["night", "rain", "snow", "night_rain"]},
    {"id": "indoor", "label": "Warehouse / indoor", "cams": ["sdg_warehouse", "smartspace"],
     "conditions": ["low_light", "occlusion"]},
]


DUSK = {"id": "dusk", "label": "Dusk / dawn", "patterns": [r"\b(dusk|dawn|twilight|sunset|sunrise|golden hour)\b"]}
FLAT_CONDS = ["night", "rain", "fog", "snow", "glare", "dusk"]


def inventory_list(clips: list[dict]) -> list[dict]:
    """Flat per-scene inventory for results/inventory.json (fixed condition keys)."""
    conds = {c["id"]: c for c in CONDITIONS + [DUSK]}
    out = []
    for sc in SCENES:
        cl = [c for c in clips if scene_of(c) == sc["id"]]
        out.append({"scene_type": sc["id"], "label": sc["label"], "n_clips": len(cl),
                    "cameras": sorted({c.get("camera_id") for c in cl if c.get("camera_id")}),
                    "conditions": {k: sum(1 for c in cl if matches(conds[k], c.get("reasoning_content"))) for k in FLAT_CONDS}})
    return out


def scene_of(clip: dict) -> str:
    cam = str(clip.get("camera_id") or "")
    for sc in SCENES:
        if any(cam.startswith(p) for p in sc["cams"]):
            return sc["id"]
    ct = clip.get("capture_type")
    cap = (clip.get("reasoning_content") or "").lower()
    if ct in ("warehouse", "crowds") or "warehouse" in cap or "indoor" in cap:
        return "indoor"
    if ct == "traffic" or HIGHWAY.search(cap):
        return "highway"
    if "dashboard" in cap or "mounted on a moving vehicle" in cap:
        return "dashcam"
    return "city_street"


def inventory(clips: list[dict]) -> dict:
    conds = {c["id"]: c for c in CONDITIONS + INDOOR_CONDITIONS}
    by_scene: dict = {}
    for c in clips:
        by_scene.setdefault(scene_of(c), []).append(c)
    scenes = []
    for sc in SCENES:
        cl = by_scene.get(sc["id"], [])
        cams: dict = {}
        for c in cl:
            cams[c.get("camera_id") or "?"] = cams.get(c.get("camera_id") or "?", 0) + 1
        grid = []
        for cid in ["clear_day"] + sc["conditions"]:
            cond = conds[cid]
            n = sum(1 for c in cl if matches(cond, c.get("reasoning_content")))
            grid.append({"id": cid, "label": cond["label"], "n": n,
                         "frac": round(n / max(1, len(cl)), 3), "relevant": cid != "clear_day",
                         "empty": cid != "clear_day" and n == 0})
        scenes.append({"id": sc["id"], "label": sc["label"], "n_clips": len(cl), "cameras": cams,
                       "conditions": grid, "n_empty": sum(1 for g in grid if g["empty"])})
    return {"scenes": scenes, "n_scenes": sum(1 for s in scenes if s["n_clips"]),
            "method": "scene type from camera_id prefix (fallback: capture_type + caption); condition = keyword match on the VSS caption"}


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
        "inventory": inventory(clips),
        "inventory_list": inventory_list(clips),
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
    inv = Path(a.out).parent / "inventory.json"
    inv.with_suffix(".tmp").write_text(json.dumps(cov["inventory_list"], indent=2))
    inv.with_suffix(".tmp").replace(inv)
    print(f"indexed={cov['n_indexed']} highway={cov['n_highway']} ({cov['seconds']}s)")
    for r in cov["conditions"]:
        print(f"  {r['label']:<18} all={r['n_clips']:>3} highway={r['n_highway']:>3} "
              f"search rel={r['search_relevant']}/{len(r['top_hits']) and r['top_k']} top={r['search_top_score']}"
              f"{'  <-- GAP' if r['gap'] else ''}")
    for sc in cov["inventory"]["scenes"]:
        print(f"  [{sc['label']}] {sc['n_clips']} clips: " + ", ".join(f"{g['id']}={g['n']}" for g in sc["conditions"]))


if __name__ == "__main__":
    main()
