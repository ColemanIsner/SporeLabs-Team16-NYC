"""Freeze the Story demo into one static page for sporelabs.dev/research/spore.

Records the Story's VSS searches (vss/ask.py), gathers the results JSON it reads,
transcodes its clips to small H.264 files named media/spore-*.mp4, writes
ui/snapshot/snapshot.json, builds the UI with VITE_SNAPSHOT=1 and inlines the
bundle into one HTML file.

    python3 tools/snapshot_story.py --out <dir>        # writes <dir>/spore-research.html + <dir>/media/
    python3 tools/snapshot_story.py --out <dir> --reuse-asks   # skip re-querying VSS
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import urllib.parse
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
SEED = "i24_scene1_p1c2_00"
HERO = f"{SEED}__phys_fog_s04"
JSON_PATHS = [
    "results/coverage.json", "results/inventory.json", "results/gap_report.json",
    "results/severity_curve.json", "results/headline.json", "results/fix.json",
    "results/loop_log.json", f"results/evals/{SEED}.json", f"results/evals/{HERO}.json",
    "results/train_samples/train_samples.json",
]
# Story.tsx VARIANTS + hero overlay: (repo path, output name, width)
CLIPS = [
    (f"data/seeds/{SEED}.mp4", "spore-real", 960),
    (f"data/synthetic/{SEED}__phys_fog_s07.mp4", "spore-fog", 960),
    (f"data/synthetic/{SEED}__phys_rain_s07.mp4", "spore-rain", 960),
    (f"data/synthetic/{SEED}__phys_snow_s07.mp4", "spore-snow", 960),
    (f"data/synthetic/{SEED}__clear_night_light__cw0.5.mp4", "spore-night", 960),
    (f"results/overlays/{HERO}.mp4", "spore-hero", 1440),
]
GAP_QUERIES = ["highway in clear daytime", "highway at night", "highway in heavy rain",
               "highway in dense fog", "highway covered in snow"]
SUGGEST = ["construction zone at night", "pedestrian crossing in the rain", "truck stopped on the shoulder"]
LOGOS = ["vast-data.svg", "nvidia.svg", "coreweave.svg", "spacex.svg", "wandb-dark.svg"]
PAGE_URL = "https://sporelabs.dev/research/spore"


def transcode(src: Path, dst: Path, width: int) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
                    "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)


def ask(q: str) -> dict:
    out = subprocess.run(["python3", str(ROOT / "vss" / "ask.py"), q], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def fetch_segment(source: str, dst: Path, width: int = 640) -> None:
    tok = (ROOT / ".vss_token").read_text().strip()
    url = ("https://team-16-vss.thecosmoslabs.com/api/v1/videos/stream?source="
           + urllib.parse.quote(source, safe="") + "&token=" + tok)
    raw = dst.with_suffix(".raw.mp4")
    subprocess.run(["curl", "-sf", "-m", "90", "-o", str(raw), url], check=True)
    transcode(raw, dst, width)
    raw.unlink()


def inline_bundle(dist: Path) -> str:
    html = (dist / "index.html").read_text()
    def js(m: re.Match) -> str:
        code = (dist / m.group(1)).read_text().replace("</script", "<\\/script")
        return f"<script type=\"module\">{code}</script>"
    def css(m: re.Match) -> str:
        return f"<style>{(dist / m.group(1)).read_text()}</style>"
    html = re.sub(r'<script type="module" crossorigin src="\./([^"]+)"></script>', js, html)
    html = re.sub(r'<link rel="stylesheet" crossorigin href="\./([^"]+)">', css, html)
    assert "./assets/" not in html, "unresolved asset reference left in the bundle"
    return html


def _cut(v, n=360):
    """Shorten long strings inside a trace so excerpts stay readable."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "…"
    if isinstance(v, list):
        return [_cut(x, n) for x in v]
    if isinstance(v, dict):
        return {k: _cut(x, n) for k, x in v.items()}
    return v


def traces(asks: dict) -> dict:
    """Real records from the run, trimmed, for the write-up's "how it works" panels."""
    J = lambda p: json.loads((ROOT / p).read_text())
    inv = J("results/inventory.json")
    syn = J("data/synthetic/manifest.json")
    ev = J(f"results/evals/{HERO}.json")
    fix = J("results/fix/fix_detail.json")
    train = J("results/fix/train_log.json")
    meta = ROOT / "fix/dataset/meta.json"
    loop = J("results/loop_log.json")
    gap = J("results/gap_report.json")
    q = "highway in dense fog"
    vss = {k: asks[q][k] for k in ("query", "endpoint", "seconds", "top_k", "terms")}
    vss["hits"] = [_cut({k: h.get(k) for k in ("camera_id", "score", "caption", "matched", "shows_it")}, 160)
                   for h in asks[q]["hits"][:2]]
    vss["note"] = f"{asks[q].get('n_synthetic_dropped', 0)} hits from our own synthetic clips removed"
    csv = train.get("results_csv", "").strip().splitlines()
    return {
        "look": inv[:3],
        "find": vss,
        "decide": _cut({k: gap.get(k) for k in ("model", "llm_seconds", "gap", "evidence", "risk", "recommendation",
                                                  "next_id", "next_why")}, 600),
        "grow": {"seed": _cut(J("data/seeds/manifest.json")[0], 200),
                 "weather_layer": next(v for v in syn if v["clip_id"].endswith("phys_fog_s04")),
                 "cosmos_transfer": next(v for v in syn if "fog_day_heavy__cw0.5" in v["clip_id"])},
        "test": _cut({k: ev.get(k) for k in ("clip_id", "fidelity", "yolo", "reason", "integrity", "failure",
                                             "failure_reasons")}, 400),
        "fix": {"dataset": json.loads(meta.read_text()).get("images_per_split_by_condition") if meta.exists() else None,
                "train": _cut(fix.get("train"), 300), "gate": fix.get("gate"),
                "train_log_csv": "\n".join(csv[:4] + (["…"] + csv[-1:] if len(csv) > 4 else []))},
        "repeat": _cut(loop[-1], 300),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse-asks", action="store_true")
    a = ap.parse_args()
    media_dir = a.out / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = UI / "snapshot"
    snap_dir.mkdir(exist_ok=True)
    asks_file = snap_dir / "asks.json"

    snap: dict = {"recorded": date.today().isoformat(), "json": {}, "media": {}, "ask": {}}
    for p in JSON_PATHS:
        f = ROOT / p
        snap["json"][p] = json.loads(f.read_text()) if f.exists() else None
    for src, name, width in CLIPS:
        transcode(ROOT / src, media_dir / f"{name}.mp4", width)
        snap["media"][src] = f"/media/{name}.mp4"
    # Fix · training data step: the sample frames named in train_samples.json (tools/train_samples.py).
    ts = snap["json"].get("results/train_samples/train_samples.json") or {}
    for k, sm in enumerate(ts.get("samples", []), 1):
        out = media_dir / f"spore-train-{k}.jpg"
        out.write_bytes((ROOT / sm["img"]).read_bytes())
        snap["media"][sm["img"]] = f"/media/{out.name}"
    for logo in LOGOS:
        b64 = base64.b64encode((UI / "public" / "logos" / logo).read_bytes()).decode()
        snap["media"][f"logos/{logo}"] = f"data:image/svg+xml;base64,{b64}"

    asks = json.loads(asks_file.read_text()) if a.reuse_asks and asks_file.exists() else {
        q: ask(q) for q in GAP_QUERIES + SUGGEST}
    asks_file.write_text(json.dumps(asks, indent=1))
    # Late on 2026-10-09 our own synthetic clips were indexed in VSS (camera spore_synthetic). The page shows
    # the real archive only, so drop those hits from every recorded search.
    real = {}
    for q, r in asks.items():
        hits = [h for h in r["hits"] if h.get("camera_id") != "spore_synthetic"]
        real[q] = {**r, "hits": hits, "top_k": len(hits), "n_shows_it": sum(h["shows_it"] for h in hits),
                   "n_synthetic_dropped": len(r["hits"]) - len(hits)}
    asks = real
    snap["ask"] = asks
    # The Ask step plays up to three hits per suggestion: those it shows first, then the rest.
    n = 0
    for q in SUGGEST:
        hits = asks[q]["hits"]
        shown = ([h for h in hits if h["shows_it"]] + [h for h in hits if not h["shows_it"]])[:3]
        for h in shown:
            src = h.get("source")
            if not src or src in snap["media"]:
                continue
            n += 1
            name = f"spore-vss-{n}"
            fetch_segment(src, media_dir / f"{name}.mp4")
            snap["media"][src] = f"/media/{name}.mp4"

    snap["traces"] = traces(asks)
    (snap_dir / "snapshot.json").write_text(json.dumps(snap))
    subprocess.run(["npx", "vite", "build"], cwd=UI, check=True,
                   env={**__import__("os").environ, "VITE_SNAPSHOT": "1"})
    html = inline_bundle(UI / "dist")
    head = (f'<link rel="canonical" href="{PAGE_URL}" />\n'
            '<meta name="description" content="Spore finds where a video AI is weak, grows synthetic footage '
            'to cover it, and measures what breaks. Built at the VAST Builders Challenge NYC." />\n'
            '<meta property="og:title" content="Spore: find where your video AI is weak, grow the data to fix it" />\n'
            f'<meta property="og:url" content="{PAGE_URL}" />\n')
    html = html.replace("<title>Spore · SporeLabs</title>",
                        "<title>Spore: find where your video AI is weak | SporeLabs Research</title>\n" + head)
    (a.out / "spore-research.html").write_text(html)
    total = sum(f.stat().st_size for f in media_dir.glob("spore-*.mp4"))
    print(f"page {len(html) / 1e6:.2f} MB, media {total / 1e6:.1f} MB in {media_dir}")


if __name__ == "__main__":
    main()
