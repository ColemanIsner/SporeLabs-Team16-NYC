"""Sweep non-distilled Cosmos-Transfer2.5 multi-control configs for strong weather.

    eval/.venv/bin/python gen/sweep_full.py   (after `modal deploy gen/modal_transfer.py`)
Writes data/sweep_full/<cond>_<cfg>.mp4 and timings.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from prompts import NEGATIVE, prompt_for  # noqa: E402

SEED = Path("data/seeds/i24_scene1_p1c2_04.mp4")
OUT = Path("data/sweep_full")

LEAD = {
    "fog": ("Thick white fog fills the whole scene, visibility 30 m, vehicles fade into grey-white haze "
            "a few car lengths away, the far road and sky are completely white.",
            {"weather": "fog", "time": "day", "intensity": "heavy"},
            "clear sky, sharp distant details, high contrast, sunny"),
    "rainnight": ("Torrential rain at night, dense rain streaks across the whole frame, flooded wet asphalt "
                  "with long glare reflections of headlights and streetlights, dark sky.",
                  {"weather": "rain", "time": "night", "intensity": "heavy"},
                  "daylight, sunny, dry road, blue sky"),
    "snow": ("Blizzard, heavy falling snow fills the air, snow-covered lanes and shoulders, white snow "
             "on barriers and vehicle roofs, low visibility.",
             {"weather": "snow", "time": "day", "intensity": "heavy"},
             "dry asphalt, green grass, summer, clear sky"),
}

CONFIGS = {
    "A_depth07": {"depth": {"control_weight": 0.7}},
    "B_depth05_edge03": {"depth": {"control_weight": 0.5}, "edge": {"control_weight": 0.3}},
    "C_seg06_edge02": {"seg": {"control_weight": 0.6, "control_prompt": "car truck road lane barrier"},
                       "edge": {"control_weight": 0.2}},
    "D_edge04": {"edge": {"control_weight": 0.4}},
}


def main():
    only = sys.argv[1:]  # optional filter of cond names
    OUT.mkdir(parents=True, exist_ok=True)
    f = modal.Function.from_name("cosmos-transfer25-edge", "transfer_full")
    vid = SEED.read_bytes()
    calls = []
    for cond, (lead, c, neg_extra) in LEAD.items():
        if only and cond not in only:
            continue
        prompt = lead + " " + prompt_for("highway", c) + " " + lead
        neg = NEGATIVE + ", " + neg_extra
        for cfg, controls in CONFIGS.items():
            name = f"{cond}_{cfg}"
            calls.append((name, prompt, time.time(), f.spawn(vid, prompt, name, controls, 35, 7, 1, neg)))
            print("spawned", name, flush=True)
    timings = {}
    for name, prompt, t0, fc in calls:
        try:
            data = fc.get(timeout=3600)
            (OUT / f"{name}.mp4").write_bytes(data)
            timings[name] = round(time.time() - t0, 1)
            print(f"OK {name} {timings[name]}s", flush=True)
        except Exception as e:  # noqa: BLE001
            timings[name] = f"ERROR: {type(e).__name__}: {e}"
            print(f"FAIL {name}: {e!r}", flush=True)
        (OUT / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
