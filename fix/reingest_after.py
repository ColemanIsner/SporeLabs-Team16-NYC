"""AFTER measurement for the VSS re-ingest prompt fix. Run only after fix/reingest_fire.py --go completed.

    python3 fix/reingest_after.py            # -> results/fix.json (+ results/fix_after_detail.json)

Re-fetches the target chunks' captions from Explore and reruns the fog / night / rain queries
(same phrasings as vss/coverage.py, camera_id=neighborhood_cam-1, top-10). Relevance = visual GT
labels (fix/visual_labels.json), never captions. Compares against the BEFORE snapshot for the same
target set (results/fix_before*.json).
"""
from __future__ import annotations

import argparse
import json
import time

from reingest_common import (JOBS, QUERIES, QUERY_IDS, ROOT, SET, before_path, caption_rate, client, dump,
                             load_labels, precision, search_hits, seg_captions)

OUT = ROOT / "results" / "fix.json"
DETAIL = ROOT / "results" / "fix_after_detail.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="defaults to the set recorded in results/fix_jobs.json")
    ap.add_argument("--force", action="store_true", help="run even if fix_jobs.json does not say all_completed")
    a = ap.parse_args()
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    if not a.force and not jobs.get("all_completed"):
        raise SystemExit("results/fix_jobs.json does not show all jobs completed; use --force to measure anyway")
    which = a.set or jobs.get("set") or "latest"
    before = json.loads(before_path(which).read_text())
    labels = load_labels()

    chunks = client.explore_all()
    by_ov = {c["original_video"]: c for c in chunks}
    tset = [t["original_video"] for t in before["targets"]]
    missing = [ov for ov in tset if ov not in by_ov]
    tgts = [by_ov[ov] for ov in tset if ov in by_ov]
    searches = {q: search_hits(q) for q in QUERY_IDS}

    # before caption rate recomputed from the stored captions with the same function
    b_tgts = [{"original_video": t["original_video"],
               "timeline": [{"segment_number": s["segment_number"], "source": s["source"],
                             "reasoning_content": s["caption"]} for s in t["segments"]]} for t in before["targets"]]
    cap_b, cap_a = caption_rate(b_tgts, labels), caption_rate(tgts, labels)
    n = len(tset)
    set_label = SET if which == "latest" else f"{SET} ({which}: 20260902 chunks 2..11)"
    rows = []
    for q in QUERY_IDS:
        pb = before["queries"][q]["precision_at_10"]
        pa = precision(searches[q], q, labels)
        rows.append({"metric": f"precision@10, {q} query", "before": pb["p"], "after": pa["p"], "set": set_label,
                     "note": f"query '{QUERIES[q]}', /api/v1/search top-10 filtered camera_id=neighborhood_cam-1; "
                             f"relevant = visual GT label (fix/visual_labels.json), not captions; "
                             f"relevant {pb['relevant']}/{pb['n']} -> {pa['relevant']}/{pa['n']}"})
    rows.append({"metric": "captions naming the condition (%)", "before": cap_b["pct"], "after": cap_a["pct"],
                 "set": set_label,
                 "note": f"target segments whose VSS caption names the visually-labeled condition(s) "
                         f"({cap_b['hit']}/{cap_b['n']} -> {cap_a['hit']}/{cap_a['n']}); re-ingested with the "
                         f"condition-first custom prompt; labels by visual inspection of frames"
                         + (f"; MISSING after re-ingest: {len(missing)} chunks" if missing else "")})
    dump(OUT, rows)
    dump(DETAIL, {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "set": which, "n_targets": n, "missing": missing,
        "targets": [{"original_video": c["original_video"], "filename": c["filename"],
                     "visual_label": labels.get(c["original_video"]), "segments": seg_captions(c)} for c in tgts],
        "queries": {q: {"query": QUERIES[q], "hits": searches[q]} for q in QUERY_IDS},
        "jobs": jobs.get("jobs"),
    })
    for r in rows:
        print(f"  {r['metric']:<36} {r['before']} -> {r['after']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
