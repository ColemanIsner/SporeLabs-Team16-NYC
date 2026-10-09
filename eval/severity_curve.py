"""Detector breaking point: mean vehicle recall vs weather severity, per condition.

Reads physics-layer variants (condition.kind == "physics", which carry `severity`) from
results/evals and writes results/severity_curve.json:
  {"curves": {"fog": [{"severity": 0.0, "recall": 1.0, "n": 11}, {"severity": 0.4, ...}, ...]},
   "breaking_point": {"fog": 0.55, ...}}   # first severity where mean recall < 0.5 (linear interp)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "results" / "evals"
OUT = ROOT / "results" / "severity_curve.json"


def main():
    acc = defaultdict(lambda: defaultdict(list))
    per_scene = defaultdict(lambda: defaultdict(list))
    for f in EVALS.glob("*.json"):
        try:
            ev = json.loads(f.read_text())
        except Exception:
            continue
        c = ev.get("condition") or {}
        if c.get("kind") != "physics" or c.get("severity") is None or ev.get("excluded"):
            continue
        integ = ev.get("integrity") or {}
        r = integ.get("recall_intact")
        if r is None:
            r = (ev.get("yolo") or {}).get("recall_vs_seed")
        if r is None:
            continue
        kind = c.get("effect") or c.get("weather")
        sev = round(float(c["severity"]), 2)
        acc[kind][sev].append(r)
        per_scene[(ev.get("seed_id") or "").split("_")[0]][f"{kind}@{sev}"].append(r)
    curves, bp = {}, {}
    for kind, by in acc.items():
        pts = [{"severity": 0.0, "recall": 1.0, "n": None}]
        pts += [{"severity": s, "recall": round(sum(v) / len(v), 4), "n": len(v)} for s, v in sorted(by.items())]
        curves[kind] = pts
        for a, b in zip(pts, pts[1:]):
            if a["recall"] >= 0.5 > b["recall"]:
                t = (a["recall"] - 0.5) / max(a["recall"] - b["recall"], 1e-9)
                bp[kind] = round(a["severity"] + t * (b["severity"] - a["severity"]), 2)
                break
    scenes = {sc: {k: round(sum(v) / len(v), 4) for k, v in d.items()} for sc, d in per_scene.items()}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"curves": curves, "breaking_point": bp, "by_scene": scenes,
                               "metric": "mean YOLO recall on intact seed vehicles"}, indent=2))
    tmp.replace(OUT)
    print(json.dumps({"breaking_point": bp, "n": {k: sum(len(v) for v in by.values()) for k, by in acc.items()}}))


if __name__ == "__main__":
    main()
