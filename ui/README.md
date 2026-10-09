# Spore UI (demo)

Single-page demo for Spore by SporeLabs, five live stages: 1 Search (results/coverage.json) → 2 Report (results/gap_report.json) → 3 Fill (data/synthetic/manifest.json, "just filled" from loop_log) → 4 Measure (results/overlays/index.json[0] + failure_map + fix.json) → 5 Loop (results/loop_log.json). Driven by `loop/.venv/bin/python loop/spore.py` (`--stage N` for one stage).

```bash
cd ui && npm i && npm run dev
# real data:  http://localhost:5173/
# fixtures:   http://localhost:5173/?fixtures=1
```

- Reads repo-root `data/` and `results/` live (served by a small middleware in `vite.config.ts`, with HTTP Range for video). Missing JSON shows an empty state.
- Polls `data/synthetic/manifest.json`, `results/evals/*.json`, `results/failure_map.json`, `results/real_check.json`, `results/fix.json` every 3 s, so the page fills in while the loop runs.
- `?fixtures=1` uses `ui/fixtures/` (fake JSON matching SPEC + tiny ffmpeg placeholder clips in `fixtures/media/`). In fixtures mode the failure map replays arm by arm so the live fill-in can be rehearsed.
- Variant grid videos are clock-synced to the seed (pause / restart buttons in the Grow header).
- Fonts load from Google Fonts; falls back to system serif/mono if offline.
