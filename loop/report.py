"""Stage 2 — REPORT & EXPLAIN THE GAP.

Reads results/coverage.json (real-archive coverage from VSS) + what the synthetic evals say
(results/failure_map.json, results/evals/*.json, results/fix.json, results/vss_eval.json,
results/real_check.json) and asks the W&B Inference LLM (Weave-traced via loop/llm.py chat())
for a short gap report. Deterministic fallback when there's no key / the call fails.

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


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def arm_key(c: dict | None) -> str | None:
    if not c:
        return None
    if c.get("kind") == "pixel":
        return "pixel:" + str(c.get("pixel") or c.get("effect"))
    return f"{c.get('weather')}_{c.get('time')}_{c.get('intensity')}"


def _synthetic_stats() -> dict:
    """Per arm (weather_time_intensity / pixel:x / physics kind) stats from every eval file."""
    out: dict = {}
    for f in (R / "evals").glob("*.json"):
        ev = load(f, {}) or {}
        c = ev.get("condition") or {}
        if not c or ev.get("excluded") or ev.get("clip_id") == ev.get("seed_id"):
            continue
        k = arm_key(c)
        s = out.setdefault(k, {"n": 0, "fail": 0, "recall": [], "agree": [], "kinds": set()})
        s["kinds"].add(c.get("kind") or "?")
        if (ev.get("fidelity") or {}).get("pass") is False:
            continue
        s["n"] += 1
        s["fail"] += int(bool(ev.get("failure")))
        r = (ev.get("yolo") or {}).get("recall_vs_seed")
        if r is not None:
            s["recall"].append(r)
    res = {}
    for k, s in out.items():
        res[k] = {"n": s["n"], "failure_rate": round(s["fail"] / s["n"], 3) if s["n"] else None,
                  "mean_recall": round(sum(s["recall"]) / len(s["recall"]), 3) if s["recall"] else None,
                  "generators": sorted(s["kinds"])}
    return res


def candidates(cov: dict, syn: dict, history: list[str]) -> list[dict]:
    hw = max(1, cov.get("n_highway") or 1)
    out = []
    for row in cov.get("conditions", []):
        if not row.get("arm"):
            continue
        k = arm_key(row["arm"])
        st = syn.get(k, {})
        n, fr = st.get("n", 0), st.get("failure_rate")
        gap = 1.0 - min(1.0, (row.get("n_highway") or 0) / max(3.0, 0.04 * hw))  # 1 = nothing real
        fail = fr if fr is not None else 0.6  # untested = worth a look
        score = round(gap * (0.25 + fail) / (1 + n / 6) * (0.3 if row["id"] in history[-2:] else 1.0), 3)
        out.append({"id": row["id"], "label": row["label"], "arm": row["arm"], "arm_key": k,
                    "real_clips_all": row.get("n_clips"), "real_clips_highway": row.get("n_highway"),
                    "vss_query": row.get("vss_query"), "vss_search_relevant": row.get("search_relevant"),
                    "vss_search_top_score": row.get("search_top_score"),
                    "synthetic_n": n, "synthetic_failure_rate": fr, "synthetic_mean_recall": st.get("mean_recall"),
                    "gap_score": round(gap, 3), "priority": score})
    out.sort(key=lambda c: -c["priority"])
    return out


def _fallback(cands: list[dict], cov: dict, fm: dict, fix, why: str) -> dict:
    top = cands[0] if cands else None
    empty = [c for c in cands if not c["real_clips_highway"]]
    worst = sorted([a for a in fm.get("arms", []) if a.get("n")], key=lambda a: -(a.get("failure_rate") or 0))[:2]
    gap = (f"Of {cov.get('n_indexed', '?')} indexed real clips ({cov.get('n_highway', '?')} highway/traffic), "
           f"{', '.join(c['label'] for c in empty) or 'no condition'} have zero highway examples.")
    ev = [f"VSS search \"{c['vss_query']}\": {c['vss_search_relevant']}/10 top hits actually show it; "
          f"caption scan finds {c['real_clips_highway']} highway clips." for c in cands[:3]]
    ev += [f"Synthetic {arm_key(a['condition'])}: detector recall {round((a.get('mean_recall') or 0) * 100)}% "
           f"vs clean seed, failure rate {round((a.get('failure_rate') or 0) * 100)}% (n={a['n']})." for a in worst]
    if fix:
        f0 = fix[0] if isinstance(fix, list) and fix else fix if isinstance(fix, dict) else None
        if isinstance(f0, dict) and "before" in f0:
            ev.append(f"Fix measured: {f0.get('metric')} {f0.get('before')} -> {f0.get('after')}.")
    risk = ("A 'near-miss at night in the rain' search would silently return nothing, and the detector has never "
            "been validated on these conditions: where we can synthesize them, recall collapses.")
    rec = (f"Fill {top['label']} next: no real coverage and the highest priority score "
           f"({top['priority']}). Generate variants from the clean highway seeds, then measure." if top else
           "Nothing to fill.")
    return {"gap": gap, "evidence": ev, "risk": risk, "recommendation": rec,
            "next_condition": top["id"] if top else None, "source": "deterministic", "note": why}


@tracing.traced
def make_report(history: list[str] | None = None) -> dict:
    history = history or []
    cov = load(R / "coverage.json", {}) or {}
    fm = load(R / "failure_map.json", {}) or {}
    fix = load(R / "fix.json")
    vss_eval = load(R / "vss_eval.json")
    real = load(R / "real_check.json")
    syn = _synthetic_stats()
    cands = candidates(cov, syn, history)
    ctx = {
        "real_archive": {"n_indexed": cov.get("n_indexed"), "n_highway": cov.get("n_highway"),
                         "conditions": [{k: r.get(k) for k in ("id", "label", "vss_query", "n_clips", "n_highway",
                                                                 "search_relevant", "search_top_score")}
                                        for r in cov.get("conditions", [])]},
        "synthetic_failure_map": [{"arm": a.get("arm"), "n": a.get("n"), "failure_rate": a.get("failure_rate"),
                                   "mean_recall": a.get("mean_recall")} for a in fm.get("arms", [])][:12],
        "candidates": [{k: c[k] for k in ("id", "label", "real_clips_highway", "synthetic_n",
                                         "synthetic_failure_rate", "synthetic_mean_recall", "priority")} for c in cands],
        "fix": fix if isinstance(fix, (list, dict)) else None,
        "vss_eval": vss_eval if isinstance(vss_eval, (list, dict)) and len(json.dumps(vss_eval)) < 4000 else None,
        "real_check_summary": ([{"condition": r.get("condition"), "confirmed": r.get("confirmed"),
                                 "n": len(r.get("clips") or [])} for r in real[:8]]
                               if isinstance(real, list) else None),
        "already_filled_recently": history[-3:],
    }
    t0 = time.time()
    rep = None
    why = ""
    if not llm._api_key():
        why = "No WANDB_API_KEY."
    else:
        msgs = [
            {"role": "system", "content": (
                "You are the analyst in a data-gap loop for a traffic video AI stack (VAST VSS search + YOLO + "
                "Cosmos Reason). The real archive was searched per condition; synthetic variants were generated "
                "from clean real seeds and measured (recall vs the clean seed). Write a SHORT, concrete gap report "
                "for an engineering audience. Plain English, no hedging, cite numbers, never mention a demo or presentation. Reply JSON only: "
                '{"gap": "<1-2 sentences: what real coverage is missing>", '
                '"evidence": ["<3-5 short bullets with numbers>"], '
                '"risk": "<1-2 sentences: what that leads to in practice, e.g. searches that silently return '
                'nothing, detector blind spots>", '
                '"recommendation": "<1-2 sentences: which gap to fill next and why>", '
                '"next_condition": "<one candidate id from candidates>"}. Prefer the highest-priority candidate '
                "unless you have a clear reason; avoid repeating already_filled_recently.")},
            {"role": "user", "content": json.dumps(ctx, default=str)},
        ]
        try:
            text = llm.chat(msgs, json_mode=True)
            s, e = text.find("{"), text.rfind("}")
            data = json.loads(text[s:e + 1])
            ids = {c["id"] for c in cands}
            if data.get("next_condition") not in ids:
                data["next_condition"] = cands[0]["id"] if cands else None
            if isinstance(data.get("evidence"), str):
                data["evidence"] = [data["evidence"]]
            if all(isinstance(data.get(k), (str, list)) for k in ("gap", "risk", "recommendation")):
                rep = {k: data.get(k) for k in ("gap", "evidence", "risk", "recommendation", "next_condition")}
                rep["source"] = "llm"
                rep["model"] = llm.os.environ.get("SPORE_LLM_MODEL", "").strip() or llm.DEFAULT_MODEL
            else:
                why = "Model JSON missing fields."
        except Exception as ex:
            why = f"W&B Inference call failed: {str(ex)[:160]}"
    if rep is None:
        rep = _fallback(cands, cov, fm, fix, why)
    nxt = next((c for c in cands if c["id"] == rep["next_condition"]), None)
    rep.update({"next_arm": nxt["arm"] if nxt else None, "next_label": nxt["label"] if nxt else None,
                "candidates": cands, "llm_seconds": round(time.time() - t0, 2), "updated": time.time(),
                "weave": tracing.enabled()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(OUT)
    return rep


if __name__ == "__main__":
    r = make_report()
    print(json.dumps({k: r[k] for k in ("gap", "evidence", "risk", "recommendation", "next_condition", "source")}, indent=2))
