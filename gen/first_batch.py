"""Gate batch: I-24 vs NYC street seeds under a few generative conditions (parallel on Modal)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gen"), str(ROOT / "loop")]

import modal  # noqa: E402
from degrade import upsert_manifest  # noqa: E402
from prompts import prompt_for  # noqa: E402

STREET = ("A fixed street camera looks down a busy New York City avenue, with the same buildings, "
          "crosswalks, parked cars, moving traffic, and pedestrians in the same places. {look}")
LOOKS = {
    ("fog", "day", "heavy"): "Dense grey fog fills the street, visibility is about 40 meters, distant buildings fade out, and headlights glow softly.",
    ("rain", "night", "heavy"): "It is night in heavy rain: wet black asphalt mirrors streetlights and headlights, raindrops streak the air, and contrast is low.",
    ("snow", "dusk", "light"): "It is dusk with light snow falling, a thin layer of snow on the sidewalks, and warm streetlights turning on.",
    ("snow", "day", "heavy"): "A heavy snowstorm: thick falling snow fills the air, snow covers the road and sidewalks, and visibility is poor.",
    ("clear", "day", "light"): "It is a clear, ordinary sunny day with normal contrast and dry pavement.",
    ("clear", "night", "light"): "It is a clear night: dark sky, streetlights and headlights are the only light, and deep shadows hide details.",
}

import json as _json, os as _os
_SEEDS = [r["seed_id"] for r in _json.load(open(ROOT / "data" / "seeds" / "manifest.json"))]
I24 = [s for s in _SEEDS if s.startswith("i24")]
NYC = [s for s in _SEEDS if s.startswith("nyc")]
import os as _o2
CONDS = [tuple(x.split("_")) for x in _o2.environ["SPORE_CONDS"].split(",")] if _o2.environ.get("SPORE_CONDS") else [("fog", "day", "heavy"), ("rain", "night", "heavy"), ("snow", "day", "heavy"), ("clear", "night", "light")]
CW = float(_os.environ.get("SPORE_CW", "0.5"))


def main():
    fn = modal.Function.from_name("cosmos-transfer25-edge", "transfer")
    jobs = []
    for s in I24:
        for w, t, i in CONDS:
            c = {"weather": w, "time": t, "intensity": i, "kind": "generative", "control_weight": CW}
            jobs.append((s, c, prompt_for("highway", c)))
    for s in NYC:
        for w, t, i in CONDS:
            c = {"weather": w, "time": t, "intensity": i, "kind": "generative", "control_weight": CW}
            jobs.append((s, c, STREET.format(look=LOOKS[(w, t, i)])))
    t0 = time.time()
    calls = []
    for s, c, p in jobs:
        name = f"{s}__{c['weather']}_{c['time']}_{c['intensity']}__cw{CW}"
        calls.append((s, c, p, name, fn.spawn((ROOT / "data" / "seeds" / f"{s}.mp4").read_bytes(), p, name, 1, CW)))
    print(f"spawned {len(calls)} jobs", flush=True)
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, c, p, name, call in calls:
        try:
            mp4 = call.get(timeout=60 * 30)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)
            continue
        out = out_dir / f"{name}.mp4"
        out.write_bytes(mp4)
        upsert_manifest(out_dir / "manifest.json", {
            "clip_id": name, "seed_id": s, "src": str(out.relative_to(ROOT)), "condition": c, "prompt": p,
            "generator": "cosmos-transfer2.5-2b/edge-distilled", "gen_seconds": round(time.time() - t0, 1),
            "gpu": "H100 (Modal)"})
        print(f"OK {name} at {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
