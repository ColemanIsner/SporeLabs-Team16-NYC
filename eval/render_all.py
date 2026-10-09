#!/usr/bin/env python3
"""Render AHA overlays for every evaluated variant, then write results/overlays/index.json.

Usage:
  eval/.venv/bin/python eval/render_all.py [--manifest data/synthetic/manifest.json] [--evals-dir results/evals]
      [--out-dir results/overlays] [--only SUBSTR] [--force] [--jobs 4]

Renders each manifest row whose eval JSON has a `yolo` block (skips existing outputs unless --force).
index.json = only clips with fidelity.pass and integrity.vehicles_intact >= 0.7, sorted by lowest
integrity.recall_intact (detector misses on cars the generator actually kept).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fidelity import load_json
from integrity import MIN_INTACT
from render_overlay import EVALS, OUT, SYN_MANIFEST, render


def _job(args):
    clip_id, manifest, evals_dir, out_dir, force = args
    return render(clip_id, manifest, evals_dir, out_dir, force, quiet=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SYN_MANIFEST)
    ap.add_argument("--evals-dir", type=Path, default=EVALS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    rows = load_json(a.manifest, []) or []
    todo = []
    for r in rows:
        cid = r.get("clip_id")
        if not cid or a.only not in cid:
            continue
        ev = load_json(a.evals_dir / f"{cid}.json", {})
        if not ev.get("yolo") or ev.get("excluded"):
            continue
        todo.append((cid, a.manifest, a.evals_dir, a.out_dir, a.force))
    print(f"{len(todo)} variants with yolo evals")

    metas = []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(_job, t): t[0] for t in todo}
        for f in as_completed(futs):
            cid = futs[f]
            try:
                m = f.result()
                metas.append(m)
                print(f"{'skip' if m.get('skipped') else 'ok  '} {cid}  recall={m.get('recall_vs_seed')}  "
                      f"poster_missed={m.get('poster_missed')}/{m.get('poster_seed_count')}")
            except BaseException as e:  # never stop on one clip (SystemExit included)
                print(f"FAIL {cid}: {e}")

    # include previously rendered overlays from this out dir that are still in the manifest
    have = {m["clip_id"] for m in metas}
    ids = {r.get("clip_id") for r in rows}
    for p in Path(a.out_dir).glob("*.meta.json"):
        m = load_json(p, None)
        if m and m.get("clip_id") in ids and m["clip_id"] not in have:
            metas.append(m)

    # only trustworthy blind spots: scene kept (fidelity) AND cars kept (integrity); worst detector first
    index = []
    for m in metas:
        m = {k: v for k, v in m.items() if k not in ("skipped", "sig")}
        if m.get("fidelity_pass") is not True or (m.get("vehicles_intact") or 0) < MIN_INTACT \
                or m.get("recall_intact") is None:
            continue
        m["recall_drop"] = round(1 - m["recall_intact"], 4)
        index.append(m)
    index.sort(key=lambda m: m["recall_intact"])
    out = Path(a.out_dir) / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.json.tmp")
    tmp.write_text(json.dumps(index, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out} ({len(index)} entries)")
    if index:
        print(f"top: {index[0]['clip_id']} recall_intact={index[0]['recall_intact']} "
              f"vehicles_intact={index[0]['vehicles_intact']} -> {index[0]['mp4']}")


if __name__ == "__main__":
    main()
