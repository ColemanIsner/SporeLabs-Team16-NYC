"""Spore failure-hunting loop.

Each round:
  1. Read every results/evals/*.json and aggregate per-condition outcomes -> results/failure_map.json
  2. Thompson-sample a Beta(failures+1, passes+1) per untried-enough arm; pick the top `--batch` arms
  3. Pair each arm with a seed it hasn't been run on, generate variants
       generative arms -> Cosmos Transfer on Modal (parallel, one GPU container per clip)
       pixel arms      -> gen/degrade.py (local ffmpeg)
  4. Run eval/run_all.py (fidelity + YOLO; skips clips rated "bad" in the review tool)
  5. Log the round (+ LLM rationale via W&B Inference) to results/loop_log.json and Weave

Usage:
  python loop/run_loop.py --rounds 4 --batch 6 --scene highway
  python loop/run_loop.py --aggregate-only        # just rebuild failure_map.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen"))
sys.path.insert(0, str(ROOT / "loop"))

from prompts import CONDITIONS, prompt_for  # noqa: E402
from degrade import upsert_manifest  # noqa: E402

try:
    from tracing import init as tracing_init, traced
    from llm import propose_next_condition
except Exception:  # loop must run even if W&B deps are missing
    def tracing_init(*a, **k):
        return False

    def traced(f):
        return f

    def propose_next_condition(failure_map, conditions):
        return {"condition": None, "rationale": "LLM unavailable"}

SEEDS = ROOT / "data" / "seeds" / "manifest.json"
SYN_DIR = ROOT / "data" / "synthetic"
SYN = SYN_DIR / "manifest.json"
EVALS = ROOT / "results" / "evals"
FAILURE_MAP = ROOT / "results" / "failure_map.json"
LOOP_LOG = ROOT / "results" / "loop_log.json"
REVIEWS = ROOT / "results" / "reviews.json"
EVAL_PY = ROOT / "eval" / ".venv" / "bin" / "python"
PIXEL_ARMS = ["glare", "motion_blur", "low_bitrate", "lens_dirt"]
CONTROL = {"weather": "clear", "time": "day", "intensity": "light"}


def load(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def arm_key(cond: dict) -> str:
    if cond.get("kind") == "pixel":
        return "pixel:" + (cond.get("pixel") or cond.get("effect") or "?")
    if cond.get("kind") in ("physics", "hybrid"):
        return f"{cond['kind']}:{cond.get('effect') or cond.get('weather')}@{cond.get('severity')}"
    return f"{cond.get('weather')}_{cond.get('time')}_{cond.get('intensity')}"


def all_arms() -> list[dict]:
    gen = [dict(c, kind="generative") for c in CONDITIONS
           if {k: c.get(k) for k in CONTROL} != CONTROL]
    pix = [{"kind": "pixel", "pixel": p, "effect": p} for p in PIXEL_ARMS]
    return gen + pix


@traced
def aggregate() -> dict:
    stats = defaultdict(lambda: {"n": 0, "fail": 0, "recall": [], "agree": [], "fid_reject": 0})
    conds = {}
    evs = [load(f, {}) for f in EVALS.glob("*.json")]

    def _r(ev):
        r = (ev.get("integrity") or {}).get("recall_intact")
        return r if r is not None else (ev.get("yolo") or {}).get("recall_vs_seed")

    # Per-seed Cosmos control: if Transfer can't even reproduce a clear day on this seed
    # (small/far vehicles smeared), its Cosmos condition clips are generator noise -> excluded.
    seed_ctrl = {ev.get("seed_id"): _r(ev) for ev in evs
                 if (ev.get("condition") or {}).get("kind") == "generative"
                 and arm_key(ev["condition"]) == "clear_day_light" and _r(ev) is not None}
    excluded_seeds = sorted(sd for sd, r in seed_ctrl.items() if r < 0.5)
    for ev in evs:
        cond = ev.get("condition") or {}
        if not cond or ev.get("excluded") or ev.get("clip_id") == ev.get("seed_id"):
            continue
        if cond.get("kind") == "generative" and ev.get("seed_id") in excluded_seeds:
            continue
        k = arm_key(cond)
        conds[k] = cond
        s = stats[k]
        if cond.get("kind") != "physics" and not (ev.get("fidelity") or {}).get("pass", True):
            s["fid_reject"] += 1
            continue
        s["n"] += 1
        s["fail"] += int(bool(ev.get("failure")))
        r = (ev.get("integrity") or {}).get("recall_intact")
        if r is None:
            r = (ev.get("yolo") or {}).get("recall_vs_seed")
        a = (ev.get("reason") or {}).get("agree_vs_seed")
        if r is not None:
            s["recall"].append(r)
        if a is not None:
            s["agree"].append(a)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    arms = [{"condition": conds[k], "arm": k, "n": s["n"],
             "failure_rate": round(s["fail"] / s["n"], 3) if s["n"] else None,
             "mean_recall": mean(s["recall"]), "mean_agree": mean(s["agree"]),
             "fidelity_rejects": s["fid_reject"], "n_gated": s["fid_reject"],
             "scored": s["n"] > 0} for k, s in stats.items()]
    # Attribute Cosmos drops to the condition, not the generator: divide by the clear-day Cosmos control.
    ctrl = next((a for a in arms if a["arm"] == "clear_day_light"), None)
    for a in arms:
        a["kind"] = conds[a["arm"]].get("kind", "generative")
        if a["kind"] == "generative" and ctrl and ctrl["mean_recall"]:
            a["control_recall"] = ctrl["mean_recall"]
            if a["mean_recall"] is not None:
                a["recall_rel_control"] = round(min(1.0, a["mean_recall"] / ctrl["mean_recall"]), 3)
    arms.sort(key=lambda a: -(a["failure_rate"] or 0))
    prev = load(FAILURE_MAP, {})
    fm = {"cosmos_control_by_seed": seed_ctrl, "cosmos_excluded_seeds": excluded_seeds,
          "cosmos_exclusion_note": "Cosmos Transfer drifts on far/small vehicles: seeds whose clear-day Cosmos control recall < 0.5 are excluded from Cosmos arms (physics arms keep all seeds).",
          "arms": arms, "budget_used": sum(a["n"] + a["fidelity_rejects"] for a in arms),
          "rounds": prev.get("rounds", 0), "updated": time.time()}
    FAILURE_MAP.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_MAP.write_text(json.dumps(fm, indent=2))
    return fm


def choose_arms(fm: dict, batch: int, rng: random.Random, include_pixel: bool) -> list[dict]:
    by_key = {a["arm"]: a for a in fm["arms"]}
    scored = []
    for cond in all_arms():
        if cond["kind"] == "pixel" and not include_pixel:
            continue
        a = by_key.get(arm_key(cond), {})
        n, fr = a.get("n", 0), a.get("failure_rate") or 0
        fails = round(fr * n)
        # Thompson sample on failure probability: we *want* to find failures
        scored.append((rng.betavariate(fails + 1, n - fails + 1), cond))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:batch]]


def done_pairs() -> set[tuple[str, str]]:
    return {(e["seed_id"], arm_key(e["condition"])) for e in load(SYN, []) if e.get("seed_id")}


def bad_seeds() -> set[str]:
    return {Path(k).stem for k, v in load(REVIEWS, {}).items() if v.get("rating") == "bad" and "/seeds/" in k}


def pair_seeds(arms: list[dict], seeds: list[dict], rng: random.Random) -> list[tuple[dict, dict]]:
    done = done_pairs()
    bad = bad_seeds()
    pool = [s for s in seeds if s["seed_id"] not in bad]
    jobs = []
    for cond in arms:
        cands = [s for s in pool if (s["seed_id"], arm_key(cond)) not in done]
        if cands:
            jobs.append((rng.choice(cands), cond))
    return jobs


def review_notes() -> str:
    notes = [f"{k}: {v.get('note')}" for k, v in load(REVIEWS, {}).items() if v.get("note")]
    return "\n".join(notes[-10:])


@traced
def generate(jobs: list[tuple[dict, dict]], scene: str) -> list[dict]:
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    pix = [(s, c) for s, c in jobs if c["kind"] == "pixel"]
    gen = [(s, c) for s, c in jobs if c["kind"] == "generative"]
    for s, c in pix:
        subprocess.run([sys.executable, str(ROOT / "gen" / "degrade.py"), "--seed", str(ROOT / s["src"]),
                        "--cond", c["pixel"], "--out", str(SYN_DIR)], check=False)
        made.append({"seed_id": s["seed_id"], "condition": c})
    if gen:
        import modal
        fn = modal.Function.from_name("cosmos-transfer25-edge", "transfer")
        args = []
        for s, c in gen:
            name = f"{s['seed_id']}__{arm_key(c)}"
            prompt = prompt_for(scene, c)
            args.append((s, c, name, prompt))
        t0 = time.time()
        calls = [(a, fn.spawn((ROOT / a[0]["src"]).read_bytes(), a[3], a[2])) for a in args]
        for (s, c, name, prompt), call in calls:
            try:
                mp4 = call.get(timeout=60 * 30)
            except Exception as e:
                print(f"[gen] {name} failed: {e}", flush=True)
                continue
            out = SYN_DIR / f"{name}.mp4"
            out.write_bytes(mp4)
            entry = {"clip_id": name, "seed_id": s["seed_id"], "src": str(out.relative_to(ROOT)),
                     "condition": c, "prompt": prompt, "generator": "cosmos-transfer2.5-2b/edge-distilled",
                     "gen_seconds": round(time.time() - t0, 1), "gpu": "H100 (Modal)"}
            upsert_manifest(SYN, entry)
            made.append(entry)
            print(f"[gen] {name} ok", flush=True)
    return made


@traced
def evaluate():
    py = str(EVAL_PY) if EVAL_PY.exists() else sys.executable
    subprocess.run([py, str(ROOT / "eval" / "run_all.py")], check=False)


def log_round(entry: dict):
    log = load(LOOP_LOG, [])
    log.append(entry)
    LOOP_LOG.write_text(json.dumps(log, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--scene", default="highway", choices=["highway", "warehouse"])
    ap.add_argument("--no-pixel", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    tracing_init("sporelabs-hackathon")
    rng = random.Random(args.seed)
    if args.aggregate_only:
        print(json.dumps(aggregate(), indent=2))
        return
    seeds = [s for s in load(SEEDS, []) if (ROOT / s["src"]).exists()]
    if not seeds:
        sys.exit("no seeds in data/seeds/manifest.json")

    evaluate()  # make sure seeds have baseline evals
    for r in range(args.rounds):
        fm = aggregate()
        arms = choose_arms(fm, args.batch, rng, include_pixel=not args.no_pixel)
        jobs = pair_seeds(arms, seeds, rng)
        if not jobs:
            print("nothing left to try")
            break
        rationale = propose_next_condition(fm, [arm_key(c) for c in arms])
        print(f"[round {r}] arms: {[arm_key(c) for _, c in jobs]}", flush=True)
        made = generate(jobs, args.scene)
        evaluate()
        fm = aggregate()
        fm["rounds"] = fm.get("rounds", 0) + 1
        FAILURE_MAP.write_text(json.dumps(fm, indent=2))
        log_round({"round": fm["rounds"], "t": time.time(), "arms": [arm_key(c) for _, c in jobs],
                   "made": len(made), "rationale": rationale, "reviewer_notes": review_notes(),
                   "top_failures": fm["arms"][:3]})


if __name__ == "__main__":
    main()
