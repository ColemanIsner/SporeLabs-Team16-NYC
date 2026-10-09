"""Stage 2 — REPORT & EXPLAIN THE GAP (relevance-ranked).

Reads results/coverage.json (VSS archive inventory: scene types x conditions) + what the stack already
does on synthetic data (results/failure_map.json) and ranks candidate gaps by

    relevance(scene, condition)   only conditions that scene realistically faces
                                  (highway: rain/fog/snow/night/glare; city street: rain/night/snow;
                                   warehouse: lighting/occlusion, never rain)
  x emptiness                     how few real clips of that scene show that condition
  x failure                       how badly the stack already fails on it (unknown = worth a look)
  / tested                        diminishing returns for conditions we've already sampled a lot
  x repeat penalty                don't fill the same condition twice in a row
  x fillable                      we have real seeds of that scene type to grow it from

Then asks the W&B Inference LLM (Weave-traced via loop/llm.py chat()) to write the gap report and a
one-line "why this gap matters" per top candidate. Deterministic fallback when there's no key.

    loop/.venv/bin/python loop/report.py      # -> results/gap_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "loop"))

import tracing  # noqa: E402

tracing.init("sporelabs-hackathon")
import llm  # noqa: E402

R = ROOT / "results"
OUT = R / "gap_report.json"

# generator arm per condition (Cosmos/physics); None = no generator yet
ARMS = {
    "night": {"weather": "clear", "time": "night", "intensity": "heavy"},
    "rain": {"weather": "rain", "time": "day", "intensity": "heavy"},
    "fog": {"weather": "fog", "time": "day", "intensity": "heavy"},
    "snow": {"weather": "snow", "time": "day", "intensity": "heavy"},
    "glare": {"kind": "pixel", "pixel": "glare", "effect": "glare"},
    "night_rain": {"weather": "rain", "time": "night", "intensity": "heavy"},
    "low_light": None,
    "occlusion": None,
}
# which failure_map arms tell us how the stack does on each condition
COSMOS_ARMS = {"night": ["clear_night_light", "clear_night_heavy"], "night_rain": ["rain_night_heavy"],
               "fog": ["fog_day_heavy"], "snow": ["snow_day_heavy"], "rain": ["rain_day_heavy"]}
# scene -> seed_id prefixes we can grow from
SEED_PREFIX = {"highway": ["i24"], "city_street": ["nyc"]}
RISK = {
    "night": "night-time vehicle and pedestrian queries",
    "rain": "wet-road incidents and spray-obscured vehicles",
    "fog": "low-visibility pileup and stopped-vehicle searches",
    "snow": "winter-weather incidents and slow traffic",
    "glare": "sun-glare washout at dawn/dusk",
    "night_rain": "the worst-visibility near-miss searches",
    "low_light": "dim-aisle forklift/worker safety events",
    "occlusion": "people hidden behind racks and pallets",
}


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def stack_on(cid: str, fm: dict) -> dict:
    """How the stack does on condition `cid` from failure_map arms (physics preferred, Cosmos fallback)."""
    arms = fm.get("arms") or []
    phys = [a for a in arms if (a.get("condition") or {}).get("kind") == "physics"
            and (a.get("condition") or {}).get("physics") == cid and a.get("n")]
    if cid == "night_rain":
        phys = []
    src = "physics" if phys else "cosmos"
    if not phys:
        phys = [a for a in arms if a.get("arm") in COSMOS_ARMS.get(cid, []) and a.get("n")]
    n = sum(a["n"] for a in phys)
    if not n:
        return {"n": 0, "failure_rate": None, "mean_recall": None, "source": None}
    fr = sum((a.get("failure_rate") or 0) * a["n"] for a in phys) / n
    rec = [a.get("mean_recall") for a in phys if a.get("mean_recall") is not None]
    return {"n": n, "failure_rate": round(fr, 3), "mean_recall": round(min(rec), 3) if rec else None, "source": src}


def _seeds_for(scene: str) -> int:
    seeds = load(ROOT / "data" / "seeds" / "manifest.json", []) or []
    pre = SEED_PREFIX.get(scene, [])
    return sum(1 for s in seeds if any(s.get("seed_id", "").startswith(p) for p in pre))


def _cond_of(gid: str | None) -> str:
    return (gid or "").split(":")[-1]


def candidates(cov: dict, fm: dict, history: list[str]) -> list[dict]:
    recent = [_cond_of(h) for h in history[-2:]]
    out = []
    for sc in (cov.get("inventory") or {}).get("scenes", []):
        if not sc.get("n_clips"):
            continue
        n_seeds = _seeds_for(sc["id"])
        for g in sc["conditions"]:
            if not g.get("relevant"):
                continue
            cid = g["id"]
            st = stack_on(cid, fm)
            empt = 1.0 - min(1.0, g["n"] / max(3.0, 0.05 * sc["n_clips"]))
            fail = st["failure_rate"] if st["failure_rate"] is not None else 0.6
            fillable = bool(n_seeds) and ARMS.get(cid) is not None
            pr = empt * (0.25 + fail) / (1 + st["n"] / 12)
            pr *= 0.25 if cid in recent else 1.0
            pr *= 1.0 if fillable else 0.5
            if st["failure_rate"] is None:
                fail_txt = "the stack has never been tested on it"
            else:
                fail_txt = (f"the stack already fails {round(st['failure_rate'] * 100)}% of synthetic tests"
                            + (f" (recall down to {round(st['mean_recall'] * 100)}%)" if st["mean_recall"] is not None else ""))
            why = (f"{sc['label']} footage faces {g['label'].lower()}, the archive has {g['n']} of "
                   f"{sc['n_clips']} such clips, and {fail_txt}; it blocks {RISK.get(cid, 'these searches')}.")
            out.append({"id": f"{sc['id']}:{cid}", "scene": sc["id"], "scene_type": sc["id"], "scene_label": sc["label"],
                        "condition": cid, "label": g["label"], "arm": ARMS.get(cid),
                        "real_clips": g["n"], "scene_clips": sc["n_clips"], "emptiness": round(empt, 3),
                        "synthetic_n": st["n"], "failure_rate": st["failure_rate"], "mean_recall": st["mean_recall"],
                        "stack_source": st["source"], "fillable": fillable, "seeds": n_seeds,
                        "repeat": cid in recent, "priority": round(pr, 3), "why": why})
    out.sort(key=lambda c: -c["priority"])
    for i, c in enumerate(out):
        c["rank"] = i + 1
        c["score"] = c["priority"]
    return out


def _fallback(cands: list[dict], cov: dict, why: str) -> dict:
    top = cands[0] if cands else None
    inv = (cov.get("inventory") or {}).get("scenes", [])
    empty = [f"{s['label']}: {', '.join(g['label'].lower() for g in s['conditions'] if g.get('empty'))}"
             for s in inv if s.get("n_empty")]
    gap = (f"The agent inventoried {cov.get('n_indexed', '?')} real clips across {len(inv)} scene types. "
           f"Every scene is almost all clear daytime; the empty cells are {'; '.join(empty[:3])}.")
    ev = [c["why"] for c in cands[:4]]
    risk = ("Searches for these conditions silently return nothing, and the detector has never been validated "
            "there: where we can synthesize them, recall collapses.")
    rec = (f"Fill {top['scene_label'].lower()} x {top['label'].lower()} next: highest relevance-weighted priority "
           f"({top['priority']}), and we have {top['seeds']} real seeds of that scene to grow it from." if top else "Nothing to fill.")
    return {"gap": gap, "evidence": ev, "risk": risk, "recommendation": rec,
            "next_id": top["id"] if top else None, "source": "deterministic", "note": why}


@tracing.traced
def make_report(history: list[str] | None = None) -> dict:
    history = history or []
    cov = load(R / "coverage.json", {}) or {}
    fm = load(R / "failure_map.json", {}) or {}
    fix = load(R / "fix.json")
    cands = candidates(cov, fm, history)
    ctx = {
        "inventory": [{"scene": s["id"], "label": s["label"], "n_clips": s["n_clips"], "cameras": list(s["cameras"]),
                       "conditions": {g["id"]: g["n"] for g in s["conditions"]}}
                      for s in (cov.get("inventory") or {}).get("scenes", [])],
        "n_indexed": cov.get("n_indexed"),
        "vss_search": [{k: r.get(k) for k in ("id", "vss_query", "search_relevant", "search_top_score")}
                       for r in cov.get("conditions", [])],
        "candidates_ranked": [{k: c[k] for k in ("id", "scene_label", "label", "real_clips", "scene_clips",
                                                 "synthetic_n", "failure_rate", "mean_recall", "fillable",
                                                 "repeat", "priority")} for c in cands[:10]],
        "fix": fix if isinstance(fix, list) else None,
        "recently_filled": history[-3:],
    }
    t0 = time.time()
    rep, why = None, ""
    if not llm._api_key():
        why = "No WANDB_API_KEY."
    else:
        msgs = [
            {"role": "system", "content": (
                "You are the analyst agent in a data-gap loop for a video AI stack (VAST VSS search + YOLO "
                "detector + Cosmos Reason captions). The agent has inventoried the real archive by scene type "
                "and, per scene, counted clips per condition. Rank gaps by RELEVANCE: a gap only matters if that "
                "scene realistically faces the condition (highways: rain, fog, snow, night, glare; city streets: "
                "rain, night, snow; warehouses/indoor: lighting and occlusion, never rain). Weight by how empty the "
                "archive is for that scene x condition and by how badly the stack already fails on it (failure_rate, "
                "mean_recall from synthetic tests; null = untested, worth exploring). Prefer fillable candidates "
                "(we have real seeds of that scene) and avoid repeating recently_filled conditions. The "
                "candidates_ranked list already applies this formula; follow it unless you have a clear reason. "
                "Plain English, cite numbers, never mention a demo. Reply JSON only: "
                '{"gap": "<1-2 sentences: what the archive is missing, by scene>", '
                '"evidence": ["<3-5 short bullets with numbers>"], '
                '"risk": "<1-2 sentences: what that leads to in practice>", '
                '"recommendation": "<1-2 sentences: which gap to fill next and why>", '
                '"next_id": "<one candidate id>", '
                '"why": {"<candidate id>": "<one plain-English line on the real-world consequence of this gap, with one number; no priority scores>", ... for the top 5}}')},
            {"role": "user", "content": json.dumps(ctx, default=str)},
        ]
        try:
            text = llm.chat(msgs, json_mode=True)
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            ids = {c["id"] for c in cands}
            if data.get("next_id") not in ids:
                data["next_id"] = cands[0]["id"] if cands else None
            if isinstance(data.get("evidence"), str):
                data["evidence"] = [data["evidence"]]
            if all(isinstance(data.get(k), (str, list)) for k in ("gap", "risk", "recommendation")):
                rep = {k: data.get(k) for k in ("gap", "evidence", "risk", "recommendation", "next_id")}
                for c in cands:
                    w = (data.get("why") or {}).get(c["id"]) if isinstance(data.get("why"), dict) else None
                    if isinstance(w, str) and w.strip():
                        c["why_llm"] = w.strip()
                        c["why_rule"] = c["why"]
                        c["why"] = w.strip()
                rep["source"] = "llm"
                rep["model"] = llm.os.environ.get("SPORE_LLM_MODEL", "").strip() or llm.DEFAULT_MODEL
            else:
                why = "Model JSON missing fields."
        except Exception as ex:
            why = f"W&B Inference call failed: {str(ex)[:160]}"
    if rep is None:
        rep = _fallback(cands, cov, why)
    nxt = next((c for c in cands if c["id"] == rep["next_id"]), None)
    rep.update({
        "next_condition": nxt["condition"] if nxt else None, "next_scene": nxt["scene"] if nxt else None,
        "next_arm": nxt["arm"] if nxt else None,
        "next_label": f"{nxt['label']} · {nxt['scene_label']}" if nxt else None,
        "next_why": (nxt.get("why_llm") or nxt["why"]) if nxt else None,
        "candidates": cands, "llm_seconds": round(time.time() - t0, 2), "updated": time.time(),
        "weave": tracing.enabled(), "history": history[-3:],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(OUT)
    return rep


if __name__ == "__main__":
    r = make_report()
    print(json.dumps({k: r[k] for k in ("gap", "risk", "recommendation", "next_id", "next_why", "source")}, indent=2))
    for c in r["candidates"][:6]:
        print(f"  {c['rank']}. {c['id']:<24} prio={c['priority']} {c.get('why_llm') or c['why']}")
