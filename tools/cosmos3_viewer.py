#!/usr/bin/env python3
"""Side-by-side viewer: real seed vs physics layer vs Cosmos Transfer 2.5 vs new Cosmos3 clips.

    python3 tools/cosmos3_viewer.py      # writes data/synthetic_cosmos3/index.html; re-run as clips land
    python3 tools/review/server.py       # then open http://localhost:8765/file/data/synthetic_cosmos3/index.html

One row per (seed, weather). Cosmos3 clips are picked up from data/synthetic_cosmos3/ (batch) and
data/synthetic_cosmos3/sweep/ (settings sweep), so every variant shows up as its own column.
Clicking any video plays/pauses the whole row in sync.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic_cosmos3"
OUT = OUT_DIR / "index.html"
WEATHER_ORDER = ["fog", "rain", "snow", "night-rain"]


def rel(p: Path) -> str:
    # Served by tools/review/server.py, which maps /file/<repo path> to the file (with Range support).
    return "/file/" + p.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    seeds = {s["seed_id"]: ROOT / s["src"] for s in json.loads((ROOT / "data/seeds/manifest.json").read_text())}
    manifest = json.loads((ROOT / "data/synthetic/manifest.json").read_text())

    # (seed, weather) -> {column label: path}
    rows: dict[tuple[str, str], dict[str, Path]] = {}

    def add(seed: str, weather: str, label: str, path: Path) -> None:
        if path.exists():
            rows.setdefault((seed, weather), {})[label] = path

    for c in sorted(OUT_DIR.glob("*.mp4")) + sorted((OUT_DIR / "sweep").glob("*.mp4")):
        m = re.match(r"(.+?)__c3_?(\w+)$", c.stem)
        if not m:
            continue
        seed, tag = m.groups()
        if tag.startswith("fog_") and "_day_" not in tag and "_night_" not in tag:  # sweep: c3fog_<variant>
            weather, label = "fog", f"Cosmos3 · {tag[4:]}"
        else:
            w, t, _ = (tag.split("_") + ["", "", ""])[:3]
            weather = "night-rain" if (w == "nightrain" or (w == "rain" and t == "night")) else w
            label = "Cosmos3" if c.parent == OUT_DIR else f"Cosmos3 · {tag}"
        add(seed, weather, label, c)

    for clip in manifest:
        cond = clip["condition"]
        w = "night-rain" if (cond["weather"] == "rain" and cond.get("time") == "night") else cond["weather"]
        if cond.get("kind") == "physics" and cond.get("severity") == 1.0 and cond.get("physics") in ("fog", "rain", "snow"):
            add(clip["seed_id"], w, "Physics layer (sev 1.0)", ROOT / clip["src"])
        elif cond.get("kind") == "generative" and cond.get("intensity") == "heavy" and cond.get("time") in ("day", "night"):
            if cond.get("time") == "night" and w != "night-rain":
                continue
            add(clip["seed_id"], w, "Cosmos Transfer 2.5 (old)", ROOT / clip["src"])

    rows = {k: v for k, v in rows.items() if any(l.startswith("Cosmos3") for l in v)}
    order = lambda k: (WEATHER_ORDER.index(k[1]) if k[1] in WEATHER_ORDER else 9, k[0])

    sections = []
    for seed, weather in sorted(rows, key=order):
        cols = rows[(seed, weather)]
        cells = [("Real seed (clear)", seeds[seed])] if seed in seeds else []
        cells += [(l, cols[l]) for l in ("Physics layer (sev 1.0)", "Cosmos Transfer 2.5 (old)") if l in cols]
        cells += sorted((l, p) for l, p in cols.items() if l.startswith("Cosmos3"))
        tiles = "".join(
            f'<figure class="{"new" if l.startswith("Cosmos3") else ""}"><video src="{html.escape(rel(p))}" muted loop '
            f'playsinline preload="metadata"></video><figcaption>{html.escape(l)}</figcaption></figure>'
            for l, p in cells
        )
        sections.append(f'<section data-w="{weather}"><h2>{html.escape(weather)} <span>{html.escape(seed)}</span></h2>'
                        f'<div class="row">{tiles}</div></section>')

    weathers = sorted({w for _, w in rows}, key=lambda w: WEATHER_ORDER.index(w) if w in WEATHER_ORDER else 9)
    chips = "".join(f'<button data-f="{w}">{w}</button>' for w in weathers)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmos3 Clip Review</title><style>
:root{{--bg:#0f1115;--fg:#e8e8ea;--mut:#9aa0aa;--card:#181b21;--acc:#76b900}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:16px}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;z-index:2}}
h1{{font-size:18px;margin:0 0 8px}} p{{color:var(--mut);margin:0 0 10px}}
button{{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:6px 10px;margin:0 6px 6px 0;cursor:pointer}}
button.on{{border-color:var(--acc);color:var(--acc)}}
section{{background:var(--card);border-radius:10px;padding:12px;margin:0 0 14px}}
h2{{font-size:15px;margin:0 0 8px;text-transform:capitalize}} h2 span{{color:var(--mut);font-weight:400;text-transform:none;margin-left:8px}}
.row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
figure{{margin:0}} video{{width:100%;border-radius:6px;background:#000;cursor:pointer;display:block}}
figure.new video{{outline:2px solid var(--acc)}} figcaption{{color:var(--mut);font-size:12px;margin-top:4px}}
figure.new figcaption{{color:var(--acc)}}
</style></head><body>
<header><h1>Cosmos3 clip review</h1>
<p>Green outline = new Cosmos3-Nano clips. Click any video to play or pause its whole row in sync. Check two things: is the weather actually visible, and are the vehicles still where they are in the real seed?</p>
<button data-f="all" class="on">all</button>{chips}<button id="playall">play all visible</button></header>
{''.join(sections) or '<p>No Cosmos3 clips yet. Re-run tools/cosmos3_viewer.py.</p>'}
<script>
document.querySelectorAll('.row').forEach(r=>r.addEventListener('click',e=>{{if(e.target.tagName!=='VIDEO')return;
 const vs=[...r.querySelectorAll('video')];const play=vs[0].paused;vs.forEach(v=>{{v.currentTime=0;play?v.play():v.pause();}});}}));
document.querySelectorAll('button[data-f]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button[data-f]').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('section').forEach(s=>s.style.display=(b.dataset.f==='all'||s.dataset.w===b.dataset.f)?'':'none');}});
document.getElementById('playall').onclick=()=>document.querySelectorAll('section').forEach(s=>{{if(s.style.display==='none')return;
 s.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play();}});}});
</script></body></html>""")
    n = sum(1 for v in rows.values() for l in v if l.startswith("Cosmos3"))
    print(f"wrote {OUT} ({len(rows)} rows, {n} Cosmos3 clips)")


if __name__ == "__main__":
    main()
