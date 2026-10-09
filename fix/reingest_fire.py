"""Fire the VSS re-ingest with the condition-first custom prompt (WRITE step -- needs user confirmation).

Default is a DRY RUN: prints the exact request(s) it would send (token redacted) and validates the
fields against GET /api/v1/metadata/ingest-config. Nothing is posted without --go.

    python3 fix/reingest_fire.py                      # dry run, approved set (10 latest complete chunks)
    python3 fix/reingest_fire.py --go                 # FIRE: one POST per chunk (original_video, chunk_count 1)
    python3 fix/reingest_fire.py --go --mode stream   # FIRE: one POST {stream_id, chunk_count: 10} (latest set only)
    python3 fix/reingest_fire.py --set nightfog ...   # alternative target set (20260902 chunks 2..11)
    python3 fix/reingest_fire.py --poll               # re-poll jobs already in results/fix_jobs.json

API (.cursor/skills/ingest/reingest-{videos,chunk}/SKILL.md):
  POST /api/v1/dashboard/reingest  {original_video|stream_id, chunk_count, [custom_prompt|scenario],
                                    [camera_id, capture_type, location]}  -> job_id, selected_chunks, copied_segments
  GET  /api/v1/dashboard/reingest/<job_id>  -> status, completed_chunks/total_chunks, indexed_segments/total_segments
Prompt mode: custom_prompt (overrides scenario; scenario omitted). Metadata: kept (camera_id/capture_type/location omitted).
Polling: every 4 s, hard cutoff 10 min per job (each HTTP call also has a 60 s curl timeout).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from reingest_common import CAMERA, CUSTOM_PROMPT, JOBS, TARGET_SETS, client, dump, targets

ENDPOINT = "/api/v1/dashboard/reingest"
ALLOWED = {"stream_id", "original_video", "chunk_count", "camera_id", "capture_type", "location", "scenario",
           "custom_prompt"}
POLL_S, CUTOFF_S = 4, 600


def build(tgts: list[dict], mode: str) -> list[dict]:
    if mode == "stream":
        sids = {c.get("stream_id") for c in tgts}
        if len(sids) != 1 or not next(iter(sids)):
            sys.exit(f"stream mode needs all targets in one stream, got {sids}")
        return [{"stream_id": next(iter(sids)), "chunk_count": len(tgts), "custom_prompt": CUSTOM_PROMPT}]
    return [{"original_video": c["original_video"], "chunk_count": 1, "custom_prompt": CUSTOM_PROMPT} for c in tgts]


def validate(bodies: list[dict], tgts: list[dict], mode: str, which: str) -> list[str]:
    cfg = client._get("/api/v1/metadata/ingest-config")
    errs = []
    mx = int(cfg.get("custom_prompt_max_length") or 800)
    scen = {s["value"] for s in cfg.get("analysis_scenarios", [])}
    caps = {s["value"] for s in cfg.get("capture_types", [])}
    for b in bodies:
        if set(b) - ALLOWED:
            errs.append(f"unknown fields {set(b) - ALLOWED}")
        if ("stream_id" in b) == ("original_video" in b):
            errs.append("exactly one of stream_id / original_video required")
        if not 1 <= b["chunk_count"] <= 100:
            errs.append(f"chunk_count {b['chunk_count']} out of 1..100")
        if "original_video" in b and b["chunk_count"] != 1:
            errs.append("original_video form must use chunk_count 1")
        cp = b.get("custom_prompt", "")
        if not cp or len(cp) > mx:
            errs.append(f"custom_prompt length {len(cp)} not in 1..{mx}")
        if "scenario" in b and b["scenario"] not in scen:
            errs.append(f"scenario {b['scenario']} not in {sorted(scen)}")
        if "capture_type" in b and b["capture_type"] not in caps:
            errs.append(f"capture_type {b['capture_type']} not in {sorted(caps)}")
    # targets must still be indexed + complete on Explore
    if len(tgts) != 10:
        errs.append(f"expected 10 targets, got {len(tgts)}")
    if mode == "stream":
        if which != "latest":
            errs.append("stream mode = 'latest N chunks of the session'; only valid for --set latest")
        sid = bodies[0]["stream_id"]
        sess = sorted([c for c in client.explore_all() if c.get("stream_id") == sid],
                      key=lambda c: c.get("chunk_index") or 0, reverse=True)[:len(tgts)]
        if {c["original_video"] for c in sess} != {c["original_video"] for c in tgts}:
            errs.append("latest-N chunks of the stream != target set")
    print(f"ingest-config: custom_prompt_max_length={mx}; scenarios={sorted(scen)}; "
          f"filterable={[f['key'] for f in cfg.get('filterable_fields', [])]}")
    return errs


def curl_preview(body: dict) -> str:
    return (f"curl -s -X POST '{client.BASE}{ENDPOINT}' -H 'Authorization: Bearer <REDACTED>' "
            f"-H 'Content-Type: application/json' -d '{json.dumps(body)}'")


def status(job_id: str) -> dict:
    r = subprocess.run(["curl", "-s", "-m", "60", f"{client.BASE}{ENDPOINT}/{job_id}",
                        "-H", f"Authorization: Bearer {client.token()}"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"status": "poll_error", "raw": r.stdout[:200]}


def poll(rec: dict) -> dict:
    jobs = [j for j in rec["jobs"] if j.get("job_id")]
    t0 = time.time()
    while True:
        pending = [j for j in jobs if j.get("status") not in ("completed", "failed", "error")]
        if not pending:
            break
        for j in pending:
            if time.time() - j.setdefault("_poll_start", time.time()) > CUTOFF_S:
                j["status"] = j.get("status") or "unknown"
                j["cutoff"] = True
                continue
            s = status(j["job_id"])
            j["last"] = s
            j["status"] = s.get("status") or s.get("detail") or "unknown"
        live = [j for j in jobs if not j.get("cutoff") and j.get("status") not in ("completed", "failed", "error")]
        cc = sum(int((j.get("last") or {}).get("completed_chunks") or 0) for j in jobs)
        tc = sum(int((j.get("last") or {}).get("total_chunks") or 0) for j in jobs)
        si = sum(int((j.get("last") or {}).get("indexed_segments") or 0) for j in jobs)
        ts = sum(int((j.get("last") or {}).get("total_segments") or 0) for j in jobs)
        print(f"[{time.time() - t0:5.0f}s] Re-ingest: {cc}/{tc} chunks, {si}/{ts} clips; "
              f"statuses={sorted({str(j['status']) for j in jobs})}", flush=True)
        dump(JOBS, rec)
        if not live:
            break
        time.sleep(POLL_S)
    for j in jobs:
        j.pop("_poll_start", None)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["all_completed"] = all(j.get("status") == "completed" for j in jobs) and bool(jobs)
    dump(JOBS, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually POST the re-ingest (default: dry run)")
    ap.add_argument("--mode", choices=("chunk", "stream"), default="chunk")
    ap.add_argument("--set", default="latest", choices=TARGET_SETS)
    ap.add_argument("--poll", action="store_true", help="only re-poll jobs in results/fix_jobs.json")
    a = ap.parse_args()

    if a.poll:
        rec = poll(json.loads(JOBS.read_text()))
        print("all_completed:", rec["all_completed"])
        return

    tgts = targets(which=a.set)
    bodies = build(tgts, a.mode)
    print(f"camera={CAMERA} set={a.set} mode={a.mode} targets={len(tgts)} "
          f"segments={sum(len(c['timeline']) for c in tgts)} requests={len(bodies)}")
    for c in tgts:
        print(f"  {c['filename']}  stream={c.get('stream_id')} idx={c.get('chunk_index')} segs={len(c['timeline'])}")
    errs = validate(bodies, tgts, a.mode, a.set)
    print(f"\n{len(bodies)} request(s) that {'WILL' if a.go else 'WOULD'} be sent:")
    for b in bodies:
        print(curl_preview(b))
    if errs:
        print("\nVALIDATION FAILED:", *errs, sep="\n  ")
        sys.exit(1)
    print("\nvalidation OK (fields vs ingest-config + skill API; targets indexed and complete)")
    if not a.go:
        print("DRY RUN -- nothing sent. Re-run with --go to fire.")
        return

    rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "set": a.set, "mode": a.mode, "camera_id": CAMERA,
           "custom_prompt": CUSTOM_PROMPT, "targets": [c["original_video"] for c in tgts], "jobs": []}
    for b in bodies:
        r = client._post(ENDPOINT, b)
        rec["jobs"].append({"request": b, "job_id": r.get("job_id"), "selected_chunks": r.get("selected_chunks"),
                            "copied_segments": r.get("copied_segments"), "status": r.get("status"),
                            "response": {k: v for k, v in r.items() if k not in ("selected_chunks",)}})
        print(f"fired {b.get('original_video') or b.get('stream_id')} -> job {r.get('job_id')} {str(r)[:160]}")
        dump(JOBS, rec)
    rec = poll(rec)
    print("all_completed:", rec["all_completed"], "->", JOBS)


if __name__ == "__main__":
    main()
