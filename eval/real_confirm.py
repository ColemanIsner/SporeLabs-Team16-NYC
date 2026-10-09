#!/usr/bin/env python3
"""Real-archive confirmation: hosted YOLO11s vs Cosmos3-Reason vehicle count on real condition clips.

Usage: eval/.venv/bin/python eval/real_confirm.py
Per clip in results/real_check.json: hosted VSS YOLO11s vehicles/frame (mean over all frames) and
Cosmos3-Reason vehicle_count (same prompt as reason_eval.py). Evidence -> results/real_check_evidence/.
Per condition: confirmed = true if any clip shows YOLO mean_count < 50% of Reason's count (the synthetic
blind-spot signature), false if Reason sees vehicles but YOLO keeps up, null if Reason sees no vehicles.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_eval  # noqa: E402
from fidelity import ROOT, load_json  # noqa: E402
from reason_eval import Reasoner  # noqa: E402

RC = ROOT / "results" / "real_check.json"
EVD = ROOT / "results" / "real_check_evidence"
RATIO = 0.5


def one(c):
    p = ROOT / c["src"]
    out = EVD / f"{c['clip_id']}.json"
    d = load_json(out, None)
    if not d:
        frames, idx, meta = yolo_eval.hosted_infer(p)
        r = Reasoner().ask(p)
        d = {"clip_id": c["clip_id"], "model": yolo_eval.HOSTED_MODEL, "n_frames": len(frames),
             "frames": frames, "server": meta, "reason": r}
        out.write_text(json.dumps(d))
    fr = d["frames"]
    veh = [sum(b[5] in yolo_eval.VEHICLES for b in f) for f in fr]
    ans = (d["reason"] or {}).get("answers") or {}
    return c["clip_id"], {
        "yolo_model": "hosted-yolo11s", "yolo_mean_count": round(sum(veh) / max(len(veh), 1), 3),
        "yolo_max_count": max(veh) if veh else 0, "yolo_frames": len(fr),
        "reason_vehicle_count": ans.get("vehicle_count"), "reason_weather_lighting": ans.get("weather_lighting"),
        "reason_summary": (d["reason"] or {}).get("summary", "")[:240],
        "caption_excerpt": (c.get("vss_caption") or "")[:220],
        "evidence": str(out.relative_to(ROOT)),
    }


def main():
    rc = load_json(RC, [])
    EVD.mkdir(parents=True, exist_ok=True)
    clips = [c for e in rc for c in e.get("clips", []) if c.get("src") and (ROOT / c["src"]).exists()]
    with ThreadPoolExecutor(6) as ex:
        res = dict(ex.map(one, clips))
    for e in rc:
        if not e.get("clips"):
            continue  # e.g. snow: no real footage exists; leave as is
        verdicts = []
        for c in e["clips"]:
            ev = res.get(c["clip_id"])
            if not ev:
                continue
            rcnt, y = ev["reason_vehicle_count"], ev["yolo_mean_count"]
            if not rcnt:
                v, why = None, "Reason sees no vehicles"
            else:
                v = y < RATIO * rcnt
                why = f"YOLO {y}/frame vs Reason {rcnt} ({y / rcnt:.0%})"
            c["confirmation"] = {**ev, "blind_spot": v, "why": why}
            verdicts.append(v)
        known = [v for v in verdicts if v is not None]
        e["confirmed"] = (any(known) if known else None)
        e["confirm_strength"] = (None if not known else "consistent" if len(set(known)) == 1 else
                                 f"mixed ({sum(known)}/{len(known)} clips show it)")
        e["confirm_rule"] = (f"true if hosted YOLO11s vehicles/frame < {RATIO:.0%} of Cosmos3-Reason vehicle_count on "
                             "any real clip of this condition (same blind-spot signature as synthetic); false if "
                             "Reason sees vehicles and YOLO keeps up; null if Reason sees no vehicles")
        e["confirm_note"] = "; ".join(f"{c['clip_id']}: {c['confirmation']['why']}" for c in e["clips"] if "confirmation" in c)
        print(f"{e['condition_name']:6s} confirmed={e['confirmed']} [{e['confirm_strength']}]  {e['confirm_note']}")
    tmp = RC.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rc, indent=2))
    tmp.replace(RC)


if __name__ == "__main__":
    main()
