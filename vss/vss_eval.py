"""Evaluate the provided VSS stack (its own Reasoner captions + hybrid search) on our uploaded synthetic variants.

Per indexed variant (from results/vss_uploads.json):
  caption_condition_hit     does VSS's caption mention the generated condition (weather and, if not day, time)?
  seed_caption_consistency  keyword overlap (vehicle types, scene/lane terms, counts) between the variant caption
                            and the VSS caption of its clean seed — does the scene description survive the condition?
  seed_query_rank           rank of the variant when searching (camera_id=spore_synthetic) with the seed caption's
                            first sentence
Set level: condition queries -> precision@5 over the uploaded synthetic set using our known labels.

  eval/.venv/bin/python vss/vss_eval.py          (needs eval/fidelity.py's merge_eval; falls back to no merge)
Writes results/vss_eval.json and merges a `vss` block into results/evals/<clip_id>.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402
from real_check import kw_hits  # noqa: E402  (negation-aware keyword matcher)

ROOT = client.ROOT
UPLOADS = ROOT / "results" / "vss_uploads.json"
SEEDS = ROOT / "data" / "seeds" / "manifest.json"
OUT = ROOT / "results" / "vss_eval.json"
CAM = "spore_synthetic"

try:
    sys.path.insert(0, str(ROOT / "eval"))
    from fidelity import merge_eval  # type: ignore
except Exception as e:  # noqa: BLE001
    merge_eval = None
    print("merge_eval unavailable:", e)

COND_KW = {
    "fog": [r"\bfog(?:gy)?\b", r"\bmist(?:y)?\b", r"\bhaz(?:e|y)\b", r"(?:low|reduced|poor|limited) visibility",
            r"\bobscur\w*"],
    "rain": [r"\brain(?:y|ing|fall|drops?)?\b", r"\bwet\b", r"\bpuddles?\b", r"\bspray\b", r"\bdownpour\b",
             r"\bstorm\w*"],
    "snow": [r"\bsnow(?:y|ing|fall|flakes?)?\b", r"\bslush\b", r"\bicy\b", r"\bwint(?:ry|er)\b", r"\bsleet\b"],
    "night": [r"\bnight(?:time)?\b", r"\bdarkness\b", r"\bdark (?:sky|conditions|road|scene|environment)\b", r"\bafter dark\b", r"\bheadlights? (?:illuminat|on\b|glow)"],
    "dusk": [r"\bdusk\b", r"\bsunset\b", r"\btwilight\b", r"\bevening\b", r"\bdim(?:ly)?\b", r"\blow sun\b",
             r"golden hour", r"\bfading light\b"],
}
CLEAR_KW = [r"\bclear(?: weather| skies| sky| day)?\b", r"\bdry\b", r"\bbright(?:ly)? (?:sun|daylight)\w*", r"\bsunny\b"]
VEH = ["car", "truck", "bus", "van", "suv", "sedan", "pickup", "semi", "tractor-trailer", "trailer", "motorcycle",
       "bicycle", "cyclist", "pedestrian", "taxi"]
SCENE = ["highway", "interstate", "median", "barrier", "gantry", "overpass", "lane markings", "multi-lane",
         "intersection", "crosswalk", "sidewalk", "traffic light", "both directions", "concrete"]
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COND_QUERIES = [
    {"query": "highway in dense fog", "strict": {"weather": "fog"}, "lenient": {"weather": "fog"}},
    {"query": "traffic at night in heavy rain", "strict": {"weather": "rain", "time": "night"},
     "lenient_any": [{"weather": "rain"}, {"time": "night"}]},
    {"query": "snowy street at dusk", "strict": {"weather": "snow", "time": "dusk"}, "lenient": {"weather": "snow"}},
]


def load(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def terms(text: str) -> dict:
    t = (text or "").lower()
    veh = {v for v in VEH if re.search(rf"\b{re.escape(v)}s?\b", t)}
    scene = {s for s in SCENE if s in t}
    lanes = set()
    for m in re.finditer(r"\b(\d+|" + "|".join(NUMW) + r")[- ]lanes?\b", t):
        lanes.add(str(NUMW.get(m.group(1), m.group(1))))
    return {"vehicles": veh, "scene": scene, "lanes": lanes}


def jacc(a: set, b: set):
    return None if not a and not b else round(len(a & b) / len(a | b), 3)


def match_cond(cond: dict, want: dict) -> bool:
    return all(cond.get(k) == v for k, v in want.items())


def seed_captions() -> dict:
    """VSS caption for each seed: the segments overlapping the seed trim window (start_s..end_s)."""
    seeds = {s["seed_id"]: s for s in load(SEEDS, [])}
    by_vid = {c["original_video"]: c for c in client.explore_all()}
    out = {}
    for sid, s in seeds.items():
        c = by_vid.get(s.get("vss_id"))
        if not c:
            continue
        segs = [x for x in c.get("timeline") or []
                if (x.get("segment_end_sec") or 0) > s.get("start_s", 0) and (x.get("segment_start_sec") or 0) < s.get("end_s", 6)]
        out[sid] = {"caption": " ".join(x.get("reasoning_content") or "" for x in segs) or c.get("reasoning_content", ""),
                    "first_segment_caption": (segs[0].get("reasoning_content") if segs else c.get("reasoning_content")) or ""}
    return out


def first_sentence(t: str) -> str:
    return re.split(r"(?<=[.!?])\s", (t or "").strip(), maxsplit=1)[0][:300]


def main():
    t0 = time.time()
    ups = {k: v for k, v in load(UPLOADS, {}).items() if not k.startswith("_")}
    idx = {k: v for k, v in ups.items() if v.get("indexed") and v.get("vss_row")}
    by_video = {v["vss_row"]["original_video"]: v for v in idx.values()}
    seeds = seed_captions()
    print(f"uploads={len(ups)} indexed={len(idx)} seeds_with_caption={len(seeds)}", flush=True)

    per_clip = []
    for key, rec in idx.items():
        cond = rec["condition"]
        row = rec["vss_row"]
        cap = row.get("reasoning_content") or ""
        all_cap = " ".join(s.get("reasoning_content") or "" for s in row.get("segments") or []) or cap
        w, tm = cond.get("weather"), cond.get("time")
        need = [x for x in (w if w != "clear" else None, tm if tm not in (None, "day") else None) if x]
        hits = {x: sorted(set(kw_hits(all_cap, COND_KW[x]))) for x in need}
        cond_hit = all(hits[x] for x in need) if need else None
        says_clear = sorted(set(kw_hits(all_cap, CLEAR_KW)))
        seed = seeds.get(rec["seed_id"], {})
        ts, tv = terms(seed.get("caption", "")), terms(all_cap)
        cons = {k: jacc(ts[k], tv[k]) for k in ts}
        vals = [v for v in cons.values() if v is not None]
        # retrieval with seed's own caption
        q = first_sentence(seed.get("first_segment_caption", ""))
        rank = None
        if q:
            r = client.search(q, k=100, llm_top_n=1, metadata_filters={"camera_id": CAM})
            vids = []
            for x in r.get("results", []):
                if x.get("original_video") not in vids:
                    vids.append(x.get("original_video"))
            rank = vids.index(row["original_video"]) + 1 if row["original_video"] in vids else None
        res = {"clip_id": rec["clip_id"], "seed_id": rec["seed_id"], "condition": cond,
               "control_weight": rec.get("control_weight"), "vss_original_video": row.get("original_video"),
               "vss_caption": cap, "vss_tags": row.get("tags"), "n_segments": len(row.get("segments") or []),
               "index_latency_s": rec.get("index_latency_s"),
               "caption_condition_hit": cond_hit, "condition_keyword_hits": hits, "caption_says_clear_or_dry": says_clear,
               "seed_caption": seed.get("caption", "")[:600],
               "seed_caption_consistency": round(sum(vals) / len(vals), 3) if vals else None,
               "consistency_parts": cons,
               "seed_terms": {k: sorted(v) for k, v in ts.items()}, "variant_terms": {k: sorted(v) for k, v in tv.items()},
               "seed_query": q, "seed_query_rank": rank, "synthetic_set_size": len(idx)}
        per_clip.append(res)
        print(f"{rec['clip_id']:50s} cond_hit={cond_hit} hits={hits} clear={says_clear} cons={res['seed_caption_consistency']} rank={rank}",
              flush=True)
        if merge_eval is not None:
            clip_id = rec["clip_id"]
            merge_eval(ROOT / "results" / "evals" / f"{clip_id}.json", clip_id, rec["seed_id"], cond,
                       {"vss": {k: res[k] for k in ("vss_original_video", "vss_caption", "caption_condition_hit",
                                                     "condition_keyword_hits", "caption_says_clear_or_dry",
                                                     "seed_caption_consistency", "seed_query_rank",
                                                     "synthetic_set_size", "index_latency_s")}})

    # condition queries over the synthetic set
    cq = []
    for spec in COND_QUERIES:
        r = client.search(spec["query"], k=50, llm_top_n=1, metadata_filters={"camera_id": CAM})
        vids = []
        for x in r.get("results", []):
            v = x.get("original_video")
            if v in by_video and v not in vids:
                vids.append(v)
        top = [by_video[v] for v in vids[:5]]

        def rel(rec, mode):
            c = rec["condition"]
            if mode == "strict":
                return match_cond(c, spec["strict"])
            if "lenient_any" in spec:
                return any(match_cond(c, w) for w in spec["lenient_any"])
            return match_cond(c, spec["lenient"])

        n_rel = sum(rel(v, "strict") for v in idx.values())
        n_rel_len = sum(rel(v, "lenient") for v in idx.values())
        k = len(top)
        cq.append({"query": spec["query"], "k": 5, "returned": k,
                   "precision_at_5_strict": round(sum(rel(v, "strict") for v in top) / k, 3) if k else None,
                   "precision_at_5_lenient": round(sum(rel(v, "lenient") for v in top) / k, 3) if k else None,
                   "relevant_in_set_strict": n_rel, "relevant_in_set_lenient": n_rel_len,
                   "random_baseline_strict": round(n_rel / len(idx), 3) if idx else None,
                   "top5": [{"clip_id": v["clip_id"], "condition": {kk: v["condition"].get(kk) for kk in ("weather", "time", "intensity")}}
                            for v in top]})
        print(f"Q '{spec['query']}': P@5 strict={cq[-1]['precision_at_5_strict']} lenient={cq[-1]['precision_at_5_lenient']} "
              f"(baseline {cq[-1]['random_baseline_strict']})", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    by_cond = {}
    for c in per_clip:
        key = f"{c['condition'].get('weather')}/{c['condition'].get('time')}"
        by_cond.setdefault(key, []).append(c)
    summary = {
        "n_indexed": len(per_clip), "n_uploaded": len(ups),
        "caption_condition_hit_rate": mean([float(c["caption_condition_hit"]) for c in per_clip if c["caption_condition_hit"] is not None]),
        "caption_says_clear_rate": mean([float(bool(c["caption_says_clear_or_dry"])) for c in per_clip]),
        "mean_seed_caption_consistency": mean([c["seed_caption_consistency"] for c in per_clip]),
        "seed_query_top1_rate": mean([float(c["seed_query_rank"] == 1) for c in per_clip]),
        "seed_query_top5_rate": mean([float(c["seed_query_rank"] is not None and c["seed_query_rank"] <= 5) for c in per_clip]),
        "by_condition": {k: {"n": len(v),
                             "caption_condition_hit_rate": mean([float(c["caption_condition_hit"]) for c in v if c["caption_condition_hit"] is not None]),
                             "mean_consistency": mean([c["seed_caption_consistency"] for c in v])}
                         for k, v in sorted(by_cond.items())},
        "condition_query_mean_p_at_5_strict": mean([q["precision_at_5_strict"] for q in cq]),
    }
    OUT.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "camera_id": CAM,
                               "seconds": round(time.time() - t0, 1), "summary": summary,
                               "condition_queries": cq, "clips": per_clip}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
