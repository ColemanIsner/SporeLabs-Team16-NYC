# Spore by SporeLabs

Take real clips from the VAST index, grow structure-preserving synthetic variants (fog, night, rain, glare…) with Cosmos Transfer, hunt for the conditions where the provided video stack fails, then check that those failures show up in the real archive, and fix them.

Transfer keeps the seed's edges and geometry, so the stack's answer on the clean seed (vehicle boxes, counts, events, summary) is the expected answer for every variant. Each synthetic clip is a test case with a known answer. A fidelity gate separates a bad generation from a real blind spot. A bandit spends the GPU budget on the conditions that break the stack. VSS then pulls real archive clips in that condition. A condition-aware prompt is the fix, re-measured on held-out real clips.

## Architecture

```
seed
  -> Cosmos Transfer on Modal   (gen/; text prompts from gen/prompts.py)
  -> ffmpeg pixel arms          (gen/degrade.py: glare, motion_blur, low_bitrate, lens_dirt)
  -> fidelity gate              (eval/: edge-map SSIM; a miss counts only if this passes)
  -> YOLO + Cosmos Reason       (systems under test)
  -> bandit loop                (loop/: Thompson sampling over the condition grid)
  -> VSS real-archive confirmation
  -> fix                        (condition-aware Reason prompt; YOLO fine-tune is stretch)
```

W&B Weave traces every gen and eval step. Footage review (`tools/review`) feeds notes back into generation and eval: bad-rated clips are excluded.

## Sponsor tools

| Sponsor | Use |
|---|---|
| VAST (VSS / VastDB) | Seed retrieval, real-archive confirmation search, ingest of synthetic clips |
| NVIDIA | Cosmos Transfer 2.5 (generate), Cosmos Reason (summaries / Q&A under test), YOLO (detection under test) |
| W&B | Weave traces of every gen/eval step + eval table |
| CoreWeave | W&B Inference (serverless, runs on CoreWeave) powers the loop agent's LLM calls (condition prompts, failure write-ups). The VSS stack itself also runs on CoreWeave GPUs. No raw GPU offered → Transfer runs on Modal |
| SpaceXAI (Cursor) | Cursor agents (incl. Grok fast) build parts of the system in parallel |

## Repo layout

```
SPEC.md                    this file
tasks/                     per-agent task lists
gen/                       Cosmos Transfer on Modal (Claude)          -> writes data/synthetic/
gen/degrade.py             ffmpeg cheap conditions (Grok)              -> writes data/synthetic/
vss/                       VSS client: search, download, ingest (Cursor)
eval/                      YOLO, fidelity, Cosmos Reason Q&A, scoring (Claude + Grok)
loop/                      bandit orchestrator + Weave (Claude)
ui/                        demo UI, reads results/*.json + data/ (Grok)
data/seeds/                seed mp4s + manifest.json
data/synthetic/            variant mp4s + manifest.json
data/real_check/           real archive clips pulled for confirmation
results/                   evals + failure map + real check (JSON)
```

`gen/prompts.py` holds the 24-arm condition grid (`weather × time × intensity`) and `prompt_for(scene, condition)`.

## How to run

### Prompts

```bash
python3 gen/prompts.py
```

Prints `NEGATIVE` plus a Transfer prompt for every condition on both scenes (`highway`, `warehouse`). Pass `prompt_for(...)` as the `--prompt` to Modal.

### Generate (Cosmos Transfer)

Details and one-time Modal / Hugging Face setup: [gen/README.md](gen/README.md).

```bash
modal run gen/modal_transfer.py::smoke
modal run gen/modal_transfer.py --video path.mp4 --prompt "..." --out out.mp4
modal run gen/modal_transfer.py::batch --jobs jobs.json --out-dir data/synthetic
```

Outputs land in `data/synthetic/` as 93-frame, 16 fps, 1280×720 mp4s, with `manifest.json`.

### Generate (cheap pixel arms, no GPU)

```bash
python3 gen/degrade.py --seed data/seeds/X.mp4 --cond all --out data/synthetic/
```

Arms: `glare`, `motion_blur`, `low_bitrate`, `lens_dirt`, plus `fog_pixel` and `night_pixel`.

### Eval

Details: [eval/README.md](eval/README.md).

```bash
eval/.venv/bin/python eval/fidelity.py data/synthetic/X__fog.mp4 --seed data/seeds/X.mp4
eval/.venv/bin/python eval/yolo_eval.py data/synthetic/X__fog.mp4 --seed data/seeds/X.mp4
```

Each command merges into `results/evals/<clip_id>.json`. Box caches go to `results/boxes/`.

### Loop, real check, fix

`loop/` runs the bandit: read evals, pick the next arm, write `results/failure_map.json`. `vss/` searches the real archive and writes `results/real_check.json`. The fix re-measures held-out real clips into `results/fix.json`. See those directories' READMEs as they land.

### Demo UI

Details: [ui/README.md](ui/README.md).

```bash
cd ui && npm install && npm run dev
```

Serves the grid at http://localhost:5173 and reads `data/` and `results/` from the repo root.

### Footage review

```bash
python3 tools/review/server.py
```

http://localhost:8765. Ratings and notes land in `results/reviews.json`. Generation and eval must read it.

## Inputs and outputs

Seeds (`data/seeds/manifest.json`): about 6 s, 16 fps, 93 frames, 1280×720.

```json
[{"seed_id": "i24_c1_007", "dataset": "i24_cam-1", "vss_id": "<id from VSS>", "src": "data/seeds/i24_c1_007.mp4",
  "start_s": 0, "end_s": 6, "fps": 16, "width": 1280, "height": 720, "notes": "dense traffic, 1 lane change"}]
```

Variants (`data/synthetic/manifest.json`). `kind` is `generative` (Cosmos) or `pixel` (ffmpeg). Seeds themselves appear with condition `clear/day/light`.

```json
[{"clip_id": "i24_c1_007__fog_night_heavy", "seed_id": "i24_c1_007", "src": "data/synthetic/i24_c1_007__fog_night_heavy.mp4",
  "condition": {"weather": "fog", "time": "night", "intensity": "heavy", "kind": "generative"},
  "prompt": "...", "generator": "cosmos-transfer2.5-2b/edge-distilled", "gen_seconds": 41.2, "gpu": "H100"}]
```

Per-clip eval (`results/evals/<clip_id>.json`):

```json
{"clip_id": "...", "seed_id": "...", "condition": {...},
 "fidelity": {"edge_ssim": 0.71, "pass": true},
 "yolo": {"model": "yolo11n", "mean_count": 5.2, "recall_vs_seed": 0.63, "boxes": "results/boxes/<clip_id>.json"},
 "reason": {"summary": "...", "answers": {"vehicle_count": 4, "lane_change": false, "stopped_vehicle": false}, "agree_vs_seed": 0.33},
 "failure": true, "failure_reasons": ["recall<0.7", "reason_disagree"]}
```

Failure map (`results/failure_map.json`):

```json
{"arms": [{"condition": {}, "n": 3, "failure_rate": 0.67, "mean_recall": 0.52, "mean_agree": 0.44}], "budget_used": 30}
```

Real check (`results/real_check.json`):

```json
[{"condition": {}, "vss_query": "highway at night in heavy rain", "clips": [{"vss_id": "...", "src": "...", "yolo_mean_count": 0, "reason_answers": {}, "human_note": ""}], "confirmed": true}]
```

Cosmos Reason question set (seed and variants):

1. How many vehicles are visible? (integer)
2. Does any vehicle change lanes? (yes/no)
3. Is any vehicle stopped or nearly stopped? (yes/no)
4. Describe the weather and lighting. (free text, used to check the condition "took")

### Metric definitions

- `edge_ssim`: SSIM between Canny edge maps of seed and variant, averaged over frames. `pass` if ≥ threshold (calibrate on first batch, start 0.5).
- `recall_vs_seed`: per frame, fraction of seed YOLO boxes (conf≥0.4, vehicle classes) matched by a variant box with IoU≥0.5; mean over frames. Seed boxes are pseudo-GT.
- `agree_vs_seed`: fraction of the fixed question set where the variant answer equals the seed answer.
- `failure` = fidelity.pass AND (recall_vs_seed < 0.7 OR agree_vs_seed < 0.67).

## Results

<!-- TODO(results): replace the TODOs below with measured numbers before submit. Do not invent them. -->

- Seeds evaluated: **TODO**
- Variants generated (generative / pixel): **TODO** / **TODO**
- Clips evaluated (`budget_used`): **TODO**
- Top failure arm and failure rate: **TODO**
- Mean recall / mean agree on that arm: **TODO** / **TODO**
- Real-archive conditions confirmed: **TODO**
- Fix, before → after on held-out real clips: **TODO**
