"""Spore: the 5-stage gap loop, one CLI, every stage Weave-traced.

  1 SEARCH   vss/coverage.py   -> results/coverage.json      (VSS /search + caption scan of the real archive)
  2 REPORT   loop/report.py    -> results/gap_report.json    (W&B Inference LLM explains the gap, picks next)
  3 FILL     gen/weather.py (physics, seconds) and/or Cosmos Transfer on Modal -> data/synthetic/
  4 MEASURE  eval/run_all.py --reason, eval/render_all.py, vss/vss_eval.py (if present), failure_map
  5 LOOP     append the iteration to results/loop_log.json, re-search the archive, go again

  loop/.venv/bin/python loop/spore.py --iterations 2 --k 3            # full loop
  loop/.venv/bin/python loop/spore.py --stage 1                       # one stage, live demo
  loop/.venv/bin/python loop/spore.py --stage 3 --k 2 --gen physics   # fill the reported gap only
  loop/.venv/bin/python loop/spore.py --gen cosmos,physics            # also spawn Cosmos Transfer (~5 min)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("loop", "vss", "gen"):
    sys.path.insert(0, str(ROOT / p))

import tracing  # noqa: E402

tracing.init("sporelabs-hackathon")
traced = tracing.traced

import coverage as cov_mod  # noqa: E402  (vss/coverage.py)
import report as report_mod  # noqa: E402  (loop/report.py)
import run_loop  # noqa: E402

R = ROOT / "results"
LOG = R / "loop_log.json"
SEEDS = ROOT / "data" / "seeds" / "manifest.json"
SYN_DIR = ROOT / "data" / "synthetic"
SYN = SYN_DIR / "manifest.json"
EVAL_PY = ROOT / "eval" / ".venv" / "bin" / "python"
PY = str(EVAL_PY) if EVAL_PY.exists() else sys.executable

# coverage condition id -> physics layers (applied in order) for gen/weather.py
PHYSICS = {"night": ["night"], "rain": ["rain"], "fog": ["fog"], "snow": ["snow"], "glare": ["glare"],
           "night_rain": ["night", "rain"]}


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)


class Log:
    """One loop_log.json entry, rewritten after every stage so the UI fills in live."""

    def __init__(self, iteration: int, mode: str):
        self.entry = {"kind": "spore", "iteration": iteration, "mode": mode, "started": time.time(),
                      "status": "running", "stages": {}}
        self.idx = None
        self.flush()

    def stage(self, n: int, name: str, **kw):
        self.entry["stages"][str(n)] = {"name": name, "t0": time.time(), "status": "running", **kw}
        self.entry["running_stage"] = n
        self.flush()

    def done(self, n: int, **kw):
        s = self.entry["stages"][str(n)]
        s.update(kw, t1=time.time(), status="done", seconds=round(time.time() - s["t0"], 1))
        self.entry["running_stage"] = None
        self.flush()

    def finish(self, **kw):
        self.entry.update(kw, status="done", finished=time.time())
        self.flush()

    def flush(self):
        log = load(LOG, [])
        if not isinstance(log, list):
            log = []
        if self.idx is None:
            log.append(self.entry)
            self.idx = len(log) - 1
        else:
            log[self.idx] = self.entry
        _write(LOG, log)


def _next_iteration() -> int:
    log = load(LOG, []) or []
    return 1 + max([e.get("iteration", 0) for e in log if isinstance(e, dict) and e.get("kind") == "spore"] or [0])


def _history() -> list[str]:
    return [e.get("gap_id") for e in (load(LOG, []) or []) if isinstance(e, dict) and e.get("gap_id")]


# ---------------------------------------------------------------- stages
@traced
def stage_search(k: int = 10) -> dict:
    c = cov_mod.coverage(k)
    _write(R / "coverage.json", c)
    return {"n_indexed": c["n_indexed"], "n_highway": c["n_highway"], "gaps": c["gaps"],
            "per_condition": {r["id"]: r["n_highway"] for r in c["conditions"]}}


@traced
def stage_report() -> dict:
    return report_mod.make_report(_history())


def _pick_seeds(cid: str, k: int) -> list[dict]:
    bad = run_loop.bad_seeds()
    made = {(e.get("seed_id"), e.get("clip_id")) for e in load(SYN, []) or []}
    seeds = [s for s in load(SEEDS, []) or [] if (ROOT / s["src"]).exists() and s["seed_id"] not in bad
             and s.get("dataset", s["seed_id"]).startswith("i24")]
    fresh = [s for s in seeds if (s["seed_id"], f"{s['seed_id']}__phys_{cid}") not in made]
    return (fresh or seeds)[:k]


@traced
def fill_physics(cid: str, seeds: list[dict], severity: float = 0.8) -> list[str]:
    layers = PHYSICS.get(cid)
    if not layers:
        return []
    tmpd = SYN_DIR / "_tmp"
    tmpd.mkdir(parents=True, exist_ok=True)
    out_ids = []
    for s in seeds:
        clip_id = f"{s['seed_id']}__phys_{cid}"
        cur = ROOT / s["src"]
        base_cond = None
        for i, kind in enumerate(layers):
            last = i == len(layers) - 1
            out = (SYN_DIR / f"{clip_id}.mp4") if last else (tmpd / f"{clip_id}__{kind}.mp4")
            cmd = [PY, str(ROOT / "gen" / "weather.py"), "--in", str(cur), "--kind", kind,
                   "--severity", str(severity), "--out", str(out)]
            if last:
                cmd += ["--manifest", str(SYN), "--seed-id", s["seed_id"], "--base", "seed"]
                if base_cond:
                    cmd += ["--base-condition", json.dumps(base_cond)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[fill] {clip_id} {kind} failed: {r.stderr[-300:]}", flush=True)
                break
            if kind == "night":
                base_cond = {"weather": "clear", "time": "night", "intensity": "heavy"}
            cur = out
        else:
            out_ids.append(clip_id)
            print(f"[fill] physics {clip_id} ok", flush=True)
    return out_ids


@traced
def fill_cosmos(arm: dict, seeds: list[dict]) -> list[str]:
    if not arm or arm.get("kind") == "pixel":
        return []
    cond = dict(arm, kind="generative")
    done = run_loop.done_pairs()
    jobs = [(s, cond) for s in seeds if (s["seed_id"], run_loop.arm_key(cond)) not in done]
    if not jobs:
        return []
    made = run_loop.generate(jobs, "highway")
    return [m["clip_id"] for m in made if m.get("clip_id")]


@traced
def stage_fill(rep: dict, k: int, gens: list[str]) -> dict:
    cid, arm = rep.get("next_condition"), rep.get("next_arm")
    if not cid:
        return {"condition": None, "clips": []}
    seeds = _pick_seeds(cid, k)
    clips: list[str] = []
    if "physics" in gens:
        clips += fill_physics(cid, seeds)
    if "cosmos" in gens:
        try:
            clips += fill_cosmos(arm, seeds)
        except Exception as e:
            print(f"[fill] cosmos failed: {e}", flush=True)
    return {"condition": cid, "label": rep.get("next_label"), "arm": arm, "seeds": [s["seed_id"] for s in seeds],
            "clips": clips, "generators": gens}


def _run(cmd: list[str], name: str, timeout: int = 1800) -> dict:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        tail = (r.stdout + r.stderr)[-400:]
    except Exception as e:
        ok, tail = False, str(e)[:400]
    print(f"[measure] {name}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
    return {"step": name, "ok": ok, "seconds": round(time.time() - t0, 1), "tail": tail if not ok else ""}


@traced
def stage_measure(clips: list[str], reason: bool = True, render: bool = True) -> dict:
    steps = []
    run_all = [PY, str(ROOT / "eval" / "run_all.py")] + (["--reason"] if reason else [])
    steps.append(_run(run_all, "eval/run_all.py" + (" --reason" if reason else "")))
    if render and (ROOT / "eval" / "render_all.py").exists():
        steps.append(_run([PY, str(ROOT / "eval" / "render_all.py")], "eval/render_all.py"))
    if (ROOT / "vss" / "vss_eval.py").exists():
        steps.append(_run([sys.executable, str(ROOT / "vss" / "vss_eval.py")], "vss/vss_eval.py", timeout=900))
    fm = run_loop.aggregate()
    rows = [load(R / "evals" / f"{c}.json", {}) or {} for c in clips]
    rec = [(e.get("yolo") or {}).get("recall_vs_seed") for e in rows]
    rec = [x for x in rec if x is not None]
    fails = [bool(e.get("failure")) for e in rows if e]
    return {"steps": steps, "n_clips": len(clips), "n_evaluated": len(fails),
            "mean_recall": round(sum(rec) / len(rec), 3) if rec else None,
            "failure_rate": round(sum(fails) / len(fails), 3) if fails else None,
            "per_clip": [{"clip_id": c, "recall": (e.get("yolo") or {}).get("recall_vs_seed"),
                          "agree": (e.get("reason") or {}).get("agree_vs_seed"),
                          "fidelity": (e.get("fidelity") or {}).get("edge_ssim"), "failure": e.get("failure")}
                         for c, e in zip(clips, rows)],
            "top_failures": [{"arm": a.get("arm"), "failure_rate": a.get("failure_rate"), "n": a.get("n"),
                              "mean_recall": a.get("mean_recall")} for a in fm.get("arms", [])[:4]]}


# ---------------------------------------------------------------- driver
@traced
def iteration(it: int, k: int, gens: list[str], reason: bool, render: bool) -> dict:
    log = Log(it, "loop")
    log.stage(1, "search", headline="Search the real archive for each condition")
    s1 = stage_search()
    log.done(1, **s1)
    log.stage(2, "report", headline="Explain the gap (W&B Inference)")
    rep = stage_report()
    log.done(2, gap=rep.get("gap"), next_condition=rep.get("next_condition"), next_label=rep.get("next_label"),
             recommendation=rep.get("recommendation"), source=rep.get("source"))
    log.entry["gap_id"] = rep.get("next_condition")
    log.stage(3, "fill", headline=f"Generate {rep.get('next_label')} clips")
    s3 = stage_fill(rep, k, gens)
    log.done(3, **s3)
    log.stage(4, "measure", headline="Measure the stack on the new clips")
    s4 = stage_measure(s3["clips"], reason, render)
    log.done(4, **{k_: v for k_, v in s4.items() if k_ != "steps"}, steps=s4["steps"])
    log.stage(5, "continue", headline="Re-search the archive, pick the next gap")
    s5 = stage_search()  # newly indexed real clips (incl. uploaded synthetic) are picked up here
    log.done(5, **s5)
    log.finish(summary=f"{rep.get('next_label')}: {s1['per_condition'].get(rep.get('next_condition'), '?')} real highway "
                       f"clips -> filled {len(s3['clips'])} -> recall {s4.get('mean_recall')}")
    return log.entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--k", type=int, default=3, help="clips to generate per iteration")
    ap.add_argument("--gen", default="physics", help="comma list: physics,cosmos")
    ap.add_argument("--no-reason", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    gens = [g.strip() for g in a.gen.split(",") if g.strip()]
    reason, render = not a.no_reason, not a.no_render

    if a.stage:
        it = _next_iteration()
        if a.stage == 5:
            print(json.dumps(iteration(it, a.k, gens, reason, render), indent=2, default=str)[:3000])
            return
        log = Log(it, f"stage{a.stage}")
        names = {1: "search", 2: "report", 3: "fill", 4: "measure"}
        log.stage(a.stage, names[a.stage])
        if a.stage == 1:
            out = stage_search()
        elif a.stage == 2:
            rep = stage_report()
            log.entry["gap_id"] = rep.get("next_condition")
            out = {k: rep.get(k) for k in ("gap", "risk", "recommendation", "next_condition", "next_label", "source")}
        elif a.stage == 3:
            out = stage_fill(load(R / "gap_report.json", {}) or stage_report(), a.k, gens)
        else:
            last = [e for e in load(LOG, []) or [] if isinstance(e, dict) and (e.get("stages") or {}).get("3")]
            clips = last[-1]["stages"]["3"].get("clips", []) if last else []
            out = stage_measure(clips, reason, render)
        log.done(a.stage, **{k: v for k, v in out.items() if k != "candidates"})
        log.finish()
        print(json.dumps(out, indent=2, default=str)[:3000])
        return

    for _ in range(a.iterations):
        e = iteration(_next_iteration(), a.k, gens, reason, render)
        print(f"[spore] iteration {e['iteration']}: {e.get('summary')}", flush=True)


if __name__ == "__main__":
    main()
