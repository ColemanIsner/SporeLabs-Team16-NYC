#!/usr/bin/env python3
"""Cosmos3-Reason Q&A (fixed question set) for every seed + synthetic clip -> results/evals.

Usage:
  eval/.venv/bin/python eval/reason_eval.py [--only SUBSTR] [--force] [--workers 4]
      [--seeds data/seeds/manifest.json] [--synthetic data/synthetic/manifest.json]
  eval/.venv/bin/python eval/reason_eval.py --clip path.mp4      # one-off: print answers, write nothing

Endpoint: hosted NVIDIA Cosmos3-Reason (OpenAI-compatible chat). Env (repo-root .env is loaded):
  COSMOS3_REASON_URL  (default http://166.19.38.112:8001)
  GPU_BEARER_TOKEN    (required; never printed)
  COSMOS3_REASON_MODEL (optional; else GET /v1/models -> data[0].id)
Video goes in as an OpenAI-style {"type":"video_url"} data URI; if the server rejects it we fall back
to N evenly sampled JPEG frames as image_url parts (--frames, default 8).

Merges {"reason": {summary, answers, agree_vs_seed, ...}} into results/evals/<clip_id>.json via
fidelity.merge_eval (locked; failure/failure_reasons recomputed). agree_vs_seed = fraction of
{vehicle_count (equal within +-1), lane_change, stopped_vehicle} equal to the seed's answers
(questions the seed could not answer are skipped). Seeds vs themselves = 1.0.
Idempotent: the model is only called when reason.answers is missing or the clip file changed
(--force to re-ask); agree_vs_seed is always recomputed from cached answers.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import EVALS, ROOT, load_json, merge_eval, rel  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

REVIEWS = ROOT / "results" / "reviews.json"
DEFAULT_URL = "http://166.19.38.112:8001"
STRUCT_KEYS = ("vehicle_count", "lane_change", "stopped_vehicle")

# Fixed question set for Cosmos Reason
QUESTIONS = [
    "How many vehicles are visible? (integer)",
    "Does any vehicle change lanes? (yes/no)",
    "Is any vehicle stopped or nearly stopped? (yes/no)",
    "Describe the weather and lighting. (free text)",
]
PROMPT = (
    "Watch this traffic video and answer these questions:\n"
    + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(QUESTIONS))
    + "\n\nRespond with ONLY a JSON object (no markdown) with exactly these keys:\n"
    '{"vehicle_count": <integer: vehicles visible>, "lane_change": <true|false>, '
    '"stopped_vehicle": <true|false>, "weather_lighting": "<short description>", '
    '"summary": "<1-2 sentence description of the scene>"}'
)


# ---------------- parsing / scoring (pure, unit-testable) ----------------
def _yesno(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s.startswith(("yes", "true")):
        return True
    if s.startswith(("no", "false")):
        return False
    return None


def _int(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(round(v))
    m = re.search(r"-?\d+", str(v))
    return int(m.group()) if m else None


def parse_answer(text: str) -> dict:
    """Model text -> {"answers": {...}, "summary", "parsed_json"}. JSON first (inside <answer> if
    present, think blocks stripped), then per-field regex."""
    text = text or ""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    body = m.group(1) if m else re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    body = re.sub(r"```(?:json)?", "", body)
    obj = None
    for src in (body, text):
        cands = re.findall(r"\{.*\}", src, re.S) + list(reversed(re.findall(r"\{[^{}]*\}", src, re.S)))
        for c in cands:
            for fix in (c, c.replace("'", '"').replace("True", "true").replace("False", "false")):
                try:
                    o = json.loads(fix)
                except Exception:
                    continue
                if isinstance(o, dict) and any(k in o for k in (*STRUCT_KEYS, "summary")):
                    obj = o
                    break
            if obj:
                break
        if obj:
            break
    obj = obj or {}

    def rx(pat):
        mm = re.search(pat, body, re.I | re.S)
        return mm.group(1) if mm else None

    vc = obj.get("vehicle_count", rx(r"vehicle_count\"?\s*[:=]\s*\"?(\d+)") or rx(r"(\d+)\s+(?:vehicles?|cars?)"))
    lc = obj.get("lane_change", rx(r"lane_change\"?\s*[:=]\s*\"?(\w+)") or rx(r"change[s]?\s+lanes?\??\s*[:\-]?\s*(yes|no)"))
    sv = obj.get("stopped_vehicle", rx(r"stopped_vehicle\"?\s*[:=]\s*\"?(\w+)") or rx(r"stopped[^\n]*?\b(yes|no)\b"))
    wl = obj.get("weather_lighting") or rx(r"weather_lighting\"?\s*[:=]\s*\"([^\"]+)") or ""
    summ = obj.get("summary") or rx(r"summary\"?\s*[:=]\s*\"([^\"]+)") or ""
    if not summ:
        summ = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()[:300]
    return {"answers": {"vehicle_count": _int(vc), "lane_change": _yesno(lc), "stopped_vehicle": _yesno(sv),
                        "weather_lighting": str(wl).strip()},
            "summary": str(summ).strip(), "parsed_json": bool(obj)}


def agree(ans: dict, seed: dict):
    """Fraction of the 3 structured answers equal to the seed's (vehicle_count within max(2, 30%) -- Cosmos3 Nano counts are noisy)."""
    n = hits = 0
    for k in STRUCT_KEYS:
        s, v = (seed or {}).get(k), (ans or {}).get(k)
        if s is None:
            continue
        n += 1
        if v is None:
            continue
        hits += (abs(int(v) - int(s)) <= max(2, 0.3 * int(s))) if k == "vehicle_count" else (bool(v) == bool(s))
    return round(hits / n, 4) if n else None


# ---------------- endpoint client ----------------
class Reasoner:
    def __init__(self, n_frames=8, timeout=300):
        self.url = os.environ.get("COSMOS3_REASON_URL", DEFAULT_URL).rstrip("/")
        tok = os.environ.get("GPU_BEARER_TOKEN", "").strip()
        if not tok:
            raise SystemExit("GPU_BEARER_TOKEN missing (set it in repo-root .env)")
        self.h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
        self.n_frames, self.timeout = n_frames, timeout
        self.model = os.environ.get("COSMOS3_REASON_MODEL") or self._model_id()
        self.mode = os.environ.get("COSMOS3_REASON_MODE", "video")  # video | frames (auto-downgrades)
        self._lock = threading.Lock()

    def _model_id(self):
        r = requests.get(f"{self.url}/v1/models", headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def _frames(self, path):
        import cv2

        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        parts = []
        for i in range(self.n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * n / self.n_frames))
            ok, f = cap.read()
            if not ok:
                continue
            f = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}})
        cap.release()
        return parts

    def _post(self, content):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 2048, "temperature": 0}
        return requests.post(f"{self.url}/v1/chat/completions", headers=self.h, json=body, timeout=self.timeout)

    def ask(self, path) -> dict:
        t0 = time.time()
        path = Path(path)
        mode = self.mode
        r = None
        if mode == "video":
            b64 = base64.b64encode(path.read_bytes()).decode()
            r = self._post([{"type": "text", "text": PROMPT},
                            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}}])
            if r.status_code in (400, 404, 413, 415, 422, 500):
                print(f"[reason] video_url rejected ({r.status_code}: {r.text[:200]!r}); using frames", flush=True)
                with self._lock:
                    self.mode = mode = "frames"
        if mode == "frames":
            r = self._post([{"type": "text", "text": PROMPT + f"\n(The video is given as {self.n_frames} frames in order.)"},
                            *self._frames(path)])
        if r.status_code in (401, 403):
            raise RuntimeError(f"auth failed ({r.status_code}); check GPU_BEARER_TOKEN")
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or ""
        if not text.strip():
            text = msg.get("reasoning_content") or ""
        out = parse_answer(text)
        out.update({"model": self.model, "input": mode, "raw": text[:2000], "seconds": round(time.time() - t0, 2)})
        return out


# ---------------- batch over manifests ----------------
def resolve(src):
    p = Path(src)
    return p if p.is_absolute() else ROOT / p


def is_bad(reviews, path):
    r = reviews.get(rel(path)) or reviews.get(str(path))
    return isinstance(r, dict) and str(r.get("rating", "")).lower() == "bad"


def cached(ev_path, clip):
    rs = (load_json(ev_path, {}) or {}).get("reason") or {}
    if rs.get("answers") and rs.get("src_mtime") == round(clip.stat().st_mtime, 3):
        return rs
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=ROOT / "data" / "seeds" / "manifest.json")
    ap.add_argument("--synthetic", type=Path, default=ROOT / "data" / "synthetic" / "manifest.json")
    ap.add_argument("--reviews", type=Path, default=REVIEWS)
    ap.add_argument("--only", help="only clip_ids containing this substring (their seeds are asked too)")
    ap.add_argument("--force", action="store_true", help="re-ask the model even if cached")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8, help="frames for the image fallback")
    ap.add_argument("--clip", type=Path, help="one-off: ask about this mp4, print, write nothing")
    a = ap.parse_args()

    if a.clip:
        r = Reasoner(a.frames).ask(a.clip)
        print(json.dumps(r, indent=2))
        return

    reviews = load_json(a.reviews, {}) or {}
    seeds = [s for s in load_json(a.seeds, []) or [] if s.get("seed_id") and s.get("src")]
    synth = [v for v in load_json(a.synthetic, []) or [] if v.get("clip_id") and v.get("src")]
    seed_src = {s["seed_id"]: resolve(s["src"]) for s in seeds}

    # (clip_id, seed_id, clip_path, condition, is_seed)
    variants = [(v["clip_id"], v.get("seed_id"), resolve(v["src"]), v.get("condition"), False) for v in synth
                if not a.only or a.only in v["clip_id"]]
    need_seeds = {v[1] for v in variants}
    seed_jobs = [(s["seed_id"], s["seed_id"], seed_src[s["seed_id"]],
                  {"weather": "clear", "time": "day", "intensity": "light", "kind": "seed"}, True) for s in seeds
                 if not a.only or a.only in s["seed_id"] or s["seed_id"] in need_seeds]

    stats = {"asked": 0, "cached": 0, "bad": 0, "missing": 0, "error": 0}
    secs = []
    client = None
    seed_answers = {}

    def process(job):
        nonlocal client
        clip_id, seed_id, clip, cond, is_seed = job
        ev_path = EVALS / f"{clip_id}.json"
        if not clip.exists():
            print(f"[missing] {clip_id}: {rel(clip)}")
            stats["missing"] += 1
            return
        sp = seed_src.get(seed_id)
        if is_bad(reviews, clip) or (sp and not is_seed and is_bad(reviews, sp)):
            print(f"[bad]     {clip_id} -> skipped")
            stats["bad"] += 1
            return
        try:
            rs = None if a.force else cached(ev_path, clip)
            hit = bool(rs)
            if hit:
                stats["cached"] += 1
            else:
                if client is None:
                    client = Reasoner(a.frames)
                r = client.ask(clip)
                rs = {"summary": r["summary"], "answers": r["answers"], "model": r["model"], "input": r["input"],
                      "parsed_json": r["parsed_json"], "seconds": r["seconds"], "raw": r["raw"],
                      "src_mtime": round(clip.stat().st_mtime, 3)}
                secs.append(r["seconds"])
                stats["asked"] += 1
            if is_seed:
                seed_answers[clip_id] = rs["answers"]
                rs["agree_vs_seed"] = 1.0
            else:
                sa = seed_answers.get(seed_id)
                rs["agree_vs_seed"] = agree(rs["answers"], sa) if sa else None
            merge_eval(ev_path, clip_id, seed_id, cond, {"reason": rs})
            print(f"[ok]      {clip_id}: {json.dumps(rs['answers'])} agree={rs['agree_vs_seed']} "
                  f"({'cached' if hit else str(rs['seconds']) + 's'})", flush=True)
        except SystemExit:
            raise
        except Exception as e:  # one broken clip must not stop the batch
            print(f"[error]   {clip_id}: {e!r}", flush=True)
            stats["error"] += 1

    t0 = time.time()
    # Seeds first (variants need their answers); first job alone so model id / video mode get settled.
    for batch in (seed_jobs[:1], seed_jobs[1:], variants):
        if batch:
            with ThreadPoolExecutor(max(1, a.workers)) as ex:
                list(ex.map(process, batch))
    out = {**stats, "seconds": round(time.time() - t0, 1)}
    if secs:
        out["mean_seconds_per_ask"] = round(sum(secs) / len(secs), 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
