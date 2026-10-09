"""Control-weight sweep: how much does edge control suppress the condition prompt?

Writes data/sweep/<seed>__<cond>__cw<w>.mp4 (kept out of data/synthetic so the loop ignores it).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gen")]

import modal  # noqa: E402
from first_batch import LOOKS, STREET  # noqa: E402
from prompts import prompt_for  # noqa: E402

JOBS = [
    ("i24_scene1_p1c2_04", "highway", ("fog", "day", "heavy")),
    ("i24_scene1_p1c2_04", "highway", ("rain", "night", "heavy")),
    ("nyc_002_00", "street", ("fog", "day", "heavy")),
    ("nyc_002_00", "street", ("rain", "night", "heavy")),
]
WEIGHTS = [0.3, 0.5, 0.7]


def main():
    fn = modal.Function.from_name("cosmos-transfer25-edge", "transfer")
    out_dir = ROOT / "data" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    for seed, scene, (w, t, i) in JOBS:
        c = {"weather": w, "time": t, "intensity": i}
        prompt = prompt_for("highway", c) if scene == "highway" else STREET.format(look=LOOKS[(w, t, i)])
        for cw in WEIGHTS:
            name = f"{seed}__{w}_{t}_{i}__cw{cw}"
            calls.append((name, fn.spawn((ROOT / "data" / "seeds" / f"{seed}.mp4").read_bytes(), prompt, name,
                                         1, cw)))
    t0 = time.time()
    print(f"spawned {len(calls)}", flush=True)
    for name, call in calls:
        try:
            (out_dir / f"{name}.mp4").write_bytes(call.get(timeout=1800))
            print(f"OK {name} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)


if __name__ == "__main__":
    main()
