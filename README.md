**DEMO VIDEO: <<add link>>**

# Spore by SporeLabs

**Find where your video AI is weak. Grow the data to fix it.**

Video AI fails quietly on the conditions your cameras don't see often. Spore helps video model operators find gaps in their evaluation data, grow synthetic data to cover them, and build new evals and benchmarks for training better models.

![Spore demo: the six-step loop](docs/img/spore_story.png)

We stress-test the exact models inside your VSS pipeline: the hosted YOLO11s (the VSS Detector) and the hosted Cosmos3 Nano Reasoner (the VSS Reasoner). VSS is where Spore searches the real archive, and it is also where the grown footage lands: we pushed 61 synthetic clips through the VSS ingest pipeline, and VSS search now returns them for the conditions the real cameras never saw (see *Synthetic clips in VSS* below).

Every number below comes from a JSON file in `results/`, and the file is named next to it.

## Try it yourself

The full pipeline is **not a one-click demo**. Producing the numbers below took about **3.5 hours of continuous compute** on sponsor infrastructure (12:05 to 15:40 on build day), plus the time to pull seeds and index the archive. It also needs event credentials (VSS login, hosted GPU token) that stop working after the event. So we split it into what you can run in a minute and what we ran for you:

| You want to... | Do this | Needs |
|---|---|---|
| See the whole story | Watch the demo video (link at top) | nothing |
| Click through the demo UI on our real results | `cd ui && npm install && npm run dev` → http://localhost:5173 | Node only. Every JSON result and every overlay still is committed. The large mp4s (3 GB of seeds and synthetic clips) are not, so the video tiles stay blank |
| Click through the UI fully offline, videos included | same, then open http://localhost:5173/?fixtures=1 | Node only. Uses placeholder data in `ui/fixtures/` |
| Check any number in this README | open the JSON file named next to it in `results/` | nothing |
| Re-run a phase | see *Pipeline phases* below and *Reproduce* | event credentials, Modal account |

## Pipeline phases

These are the six steps in the demo, in order. Each phase writes files that the next phase reads, so you can inspect every hand-off in `results/` and `data/`. Runtimes are wall-clock from our build-day logs.

| # | Phase | What it does | Sponsor | Code | Writes | Read by next phase | Runtime |
|---|---|---|---|---|---|---|---|
| 1 | Look | Reads every clip indexed in VSS with its Cosmos Reason caption | **VAST** (VSS) | `vss/coverage.py` | `results/inventory.json` | phase 2 | minutes |
| 2 | Find weak spots | One plain-English VSS search per condition, plus a keyword scan of all captions | **VAST** (VSS) | `vss/coverage.py` | `results/coverage.json` | phase 3 | minutes |
| 3 | Decide | An LLM ranks which gaps matter and picks the next condition. Every call is traced | **CoreWeave / W&B** (W&B Inference, Weave) | `loop/report.py`, `loop/llm.py`, `loop/tracing.py` | `results/gap_report.json` | phase 4 | ~15 s per report |
| 4a | Grow data: physics | Per-pixel fog, rain and snow overlays at 3 severities on 11 real seeds pulled from VSS. Cars never move, so seed labels stay exact | **VAST** (seeds from VSS) | `gen/weather.py` | `data/synthetic/` (99 clips) | phase 5 | ~7 min (CPU) |
| 4b | Grow data: generative | Cosmos Transfer 2.5 relights the same seeds (night, night + rain, fog, snow, plus a clear-day control) | **NVIDIA** (Cosmos Transfer 2.5), run on Modal H100 | `gen/modal_transfer.py`, `gen/first_batch.py`, `gen/sweep_full.py` | `data/synthetic/` (55 clips) | phase 5 | ~30 min (~37 s diffusion per clip plus cold starts) |
| 5 | Test | Hosted YOLO11s (VSS Detector) and Cosmos3 Nano Reasoner (VSS Reasoner) on every clip, then the edge-SSIM and vehicle-integrity gates, the severity curve and the overlay renders | **NVIDIA** (Cosmos3 Reasoner, YOLO11s) on **CoreWeave** GPUs | `eval/run_all.py`, `eval/integrity.py`, `eval/severity_curve.py`, `eval/render_all.py` | `results/evals/`, `severity_curve.json`, `failure_map.json`, `overlays/` | phase 6 and the loop | ~3 h: ran continuously as new clips landed, ~7 s per clip per model |
| 5b | Check against real footage | Searches VSS for real clips of each condition and runs the same two models on them | **VAST** (VSS), **NVIDIA** on **CoreWeave** | `vss/real_check.py`, `eval/real_confirm.py` | `results/real_check.json` | the loop | minutes |
| 4c | Index grown clips in VSS | Uploads grown clips through the VSS upload API; VAST's DataEngine pipeline segments, detects, captions, embeds and indexes them, so VSS search returns them | **VAST** (VSS, DataEngine, VastDB) | `vss/upload_synthetic.py` | `results/vss_uploads.json` | search | minutes per batch |
| 6 | Fix | Builds an auto-labeled training set from the grown clips and fine-tunes YOLO11s on it; measures VSS caption quality as the baseline for a weather-aware re-ingest prompt | Modal L40S; **VAST** (VSS) | `fix/build_dataset.py`, `fix/train_modal.py`, `fix/evaluate.py`, `fix/reingest_before.py` | `results/fix/fix_detail.json`, `results/fix_before.json` | the loop | 136 s training, plus dataset build and eval |
| ↺ | Loop | Picks the next weak spot, re-searches the archive, and runs phases 2 to 5 again | all of the above | `loop/spore.py` | `results/loop_log.json` | phase 2 | one iteration ≈ phases 2 to 5 for one condition |

Cursor agents (including Grok 4.7, **SpaceXAI / Cursor**) ran many of these phases in parallel during the build. They built the code; they are not part of the pipeline.

## Results in one screen

- **The gap is real.** There are 612 real clips indexed; 30 are from the I-24 highway camera, and all 30 are clear daylight, verified by eye (`results/verify/i24_all_chunks.jpg`). The highway camera has **0** night, rain, fog, snow, glare, or night-rain clips. Snow has **0 clips on any camera**. (`results/coverage.json`)
- **The detector breaks early.** Recall on intact vehicles falls below 50% at severity **0.34 for snow, 0.35 for fog, 0.60 for rain**. (`results/severity_curve.json`)
- **The Detector and Reasoner disagree.** In the worst physics condition (fog at severity 1.0), YOLO11s finds **0.34 vehicles/frame**, down from 7.77 on the clear seeds. Cosmos Reason still counts **7.6 vehicles**, down from 15.1. YOLO finds under 0.5 vehicles/frame on 7 of 11 clips, while Reason reports vehicles on 10 of 11. (computed from `results/evals/*__phys_fog_s10.json` and the seed evals)
- **Real footage shows the same signature.** On real glare, dusk and rain clips from the archive, YOLO falls well short of Cosmos Reason, as it does on the grown clips. No real snow footage exists anywhere, so grown footage is the only way to test it. (`results/real_check.json`)
- **Grown footage is a sharper test than training set.** Every grown frame comes auto-labeled (1,858 frames, zero hand-drawn boxes). A 10-epoch fine-tune on it showed that at this scale the footage is most valuable as a stress test; a CLAHE contrast step gave the best recall on held-out variants. (`results/fix/fix_detail.json`)

## Stages in the code

```
1 Search    VSS /api/v1/search + explore captions over 612 real clips   -> results/coverage.json
2 Report    W&B Inference LLM gap report, Weave-traced                    -> results/gap_report.json
3 Fill      gen/weather.py physics fog/rain/snow, sev 0.4/0.7/1.0         -> data/synthetic/ (99 clips)
            + NVIDIA Cosmos Transfer 2.5 relighting on Modal H100            (44 clips + 11 clear-day controls)
4 Measure   hosted YOLO11s (VSS Detector) + Cosmos3 Nano Reasoner (VSS Reasoner)
            eval/integrity.py vehicle gate, severity breaking point     -> results/evals/, severity_curve.json, failure_map.json
5 Loop      loop/spore.py picks the next gap, re-searches the archive    -> results/loop_log.json, results/real_check.json
            for newly indexed real clips
```

### 1. Search: what has the archive never seen?

`vss/coverage.py` sends one plain-English query per condition to `POST /api/v1/search`. It then keyword-scans the VSS caption (`reasoning_content` from `/api/v1/videos/explore`) of every indexed clip. Source: `results/coverage.json` (13 cameras).

| Condition | Clips, any camera | Highway clips |
|---|---|---|
| Clear daytime | 101 | 30 |
| Night | 32 | **0** |
| Heavy rain | 8 | **0** |
| Dense fog | 4 | **0** |
| Low-sun glare | 3 | **0** |
| Snow | **0** | **0** |
| Night + rain | **0** | **0** |
| **Total indexed** | **612** | **30** |

### 2. Report: explain the gap (W&B Inference + Weave)

`loop/report.py` gives the coverage and eval numbers to a W&B Inference LLM (`deepseek-ai/DeepSeek-V4-Pro-0813`, 15.3 s). The LLM writes the gap report, ranks 25 scene × condition candidates, and picks the next one. The LLM call and every loop stage in `loop/spore.py` are traced in Weave: [wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave](https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave).

From `results/gap_report.json`:

> "Fill highway:glare next because it is the top-ranked fillable candidate, has 0 real highway clips and no synthetic coverage, and glare is a realistic highway condition that can hide vehicles from both search and detection."

Its top picks: highway glare (0.85), highway night (0.50), highway night + rain (0.50). Fog and snow rank lower because they already have 33 synthetic tests each. The demo walks through fog, the condition where the detector fails hardest; the loop then filled the LLM's top pick, glare, in its next iteration (`results/loop_log.json`).

### 3. Fill: generate the missing footage

| Generator | What | Clips | Source |
|---|---|---|---|
| Physics weather layer (`gen/weather.py`) | fog, rain, snow at severity 0.4, 0.7, and 1.0 on 11 real seeds (9 I-24 highway, 2 NYC) | 99 (+4 rain at 0.8 and 2 glare at 0.8 from loop iterations) | `data/synthetic/manifest.json` |
| NVIDIA Cosmos Transfer 2.5 (edge-distilled 2B, Modal H100) | night, night + rain, fog, snow relighting | 44 | same |
| Cosmos clear-day control | same generator, "nothing changes" prompt | 11 | same |
| **Total** | | **160** | |

The physics layer is the headline result because it only adds per-pixel overlays and never warps the image. Vehicles stay at exactly the same pixels, so the seed's labels remain exact ground truth.

### 4. Measure: what breaks in your pipeline

Breaking point is the severity at which mean recall on intact vehicles drops below 50%. Source: `results/severity_curve.json`, 11 clips per cell.

| Layer | sev 0.4 | sev 0.7 | sev 1.0 | Breaking point |
|---|---|---|---|---|
| Snow | 0.41 | 0.28 | 0.21 | **0.34** |
| Fog | 0.43 | 0.30 | 0.06 | **0.35** |
| Rain | 0.71 | 0.39 | 0.14 | **0.60** |

The highway camera is far more fragile than the NYC street camera. At fog 0.4, I-24 recall is 0.315 while NYC recall is 0.957 (`by_scene`, `results/severity_curve.json`; NYC is only 2 of the 11 seeds).

**Reasoner vs Detector**, computed from `results/evals/*.json` with 11 clips per row:

| Condition | YOLO11s vehicles/frame | Cosmos Reason vehicle_count |
|---|---|---|
| Clear seeds | 7.77 | 15.1 |
| Snow 0.7 (all 11 pass the integrity gate) | 2.00 (26% of seed) | 12.5 (83% of seed) |
| Fog 1.0 (worst condition) | 0.34 (4% of seed) | 7.6 (51% of seed) |

Reason's absolute count is about twice YOLO's even on clean seeds, so compare the drop relative to each model's own seed count.

The overlay shown by default in the UI is `i24_scene1_p1c3_04__phys_snow_s07`. On its poster frame, **YOLO sees 1 of 12** seed vehicles, and every vehicle is intact (vehicles_intact 1.0). Cosmos Reason on the same clip says "15 vehicles, snowy, overcast". (`results/overlays/i24_scene1_p1c3_04__phys_snow_s07.meta.json`)

**Cosmos arms are measured against a Cosmos control.** Every Cosmos run has a matching clear-day control ("nothing changes" prompt) on the same seed, so the effect of the condition is separated from the effect of re-rendering. Seeds where the control itself scores below 0.5 recall (far, small vehicles) are excluded from the Cosmos arms; the physics arms keep all 11 seeds. Source: `results/failure_map.json`, n = 6 per arm.

| Cosmos arm | Mean recall | Relative to control |
|---|---|---|
| Clear day (control) | 0.807 | 1.00 |
| Night + heavy rain | 0.480 | **0.60** |
| Clear night | 0.694 | 0.86 |
| Heavy fog | 0.740 | 0.92 |
| Heavy snow | 0.769 | 0.95 |

Cosmos Transfer covers what an overlay cannot, like turning day into night; night with heavy rain is its strongest finding. The physics layer is the headline for fog, rain and snow because its pixels never move, so its labels are exact.

### 5. Loop: keep searching

`loop/spore.py` runs Search → Report → Fill → Measure → Continue, with every stage traced in Weave. It logs each iteration to `results/loop_log.json`. Four iterations ran on the day; the latest took the LLM's top pick, "Low sun glare · Highway", filled 2 clips, and found YOLO holds up there (recall 0.76). Re-searching the archive found the same 612 real clips and **0 new real chunks** (`results/real_check.json` → `runs`), so the loop keeps filling the open gaps with grown footage.

## Synthetic clips in VSS

`vss/upload_synthetic.py` uploads grown clips through the VSS upload API (camera `spore_synthetic`), so VAST's own DataEngine pipeline segments them, runs the Detector, captions them with Cosmos Reason, embeds them and writes them to VastDB like any other footage. We uploaded 61 clips (44 Cosmos Transfer, 17 weather layer). VSS indexed them and its captions describe the grown conditions ("a multi-lane highway at night", "during heavy rain", "dense fog significantly reducing visibility").

The result: the searches that returned nothing real now return grown footage. In the recorded searches (`ui/snapshot/asks.json`), "highway at night", "highway in heavy rain" and "highway covered in snow" each return 10 of 10 hits from Spore's clips, and "highway in dense fog" returns 7 of 10; 25 distinct grown clips appear across the searches. Coverage counts and the real-archive tables in this README exclude `spore_synthetic`, so they describe real footage only.

## Real-archive check

`vss/real_check.py` searched the real archive for each condition and pulled 2 clips per condition (10 total). `eval/real_confirm.py` ran hosted YOLO11s and Cosmos Reason on them. A condition counts as "confirmed" if YOLO finds fewer than 50% of Reason's vehicle count on any clip, which is the same signature we see on synthetic clips. Source: `results/real_check.json`.

| Condition | Confirmed | Evidence (YOLO vehicles/frame vs Reason count) |
|---|---|---|
| Glare | yes, 1 of 2 clips | **0.505 vs 2 (25%)**: the strongest case. The other clip: 8.796 vs 6 |
| Dusk | yes, 1 of 2 | 0.441 vs 1 (44%); 1.57 vs 2 (78%) |
| Rain | yes, 1 of 2 (borderline) | 0.989 vs 2 (49%); 4.0 vs 1 (400%). Rain is barely visible in either clip |
| Night | no | 6.204 vs 8 (78%); on the other clip, an IR camera, Reason sees no vehicles |
| Fog | **no** | YOLO beats Reason on real fog: 4.495 vs 2 and 4.505 vs 2. Both clips are adjacent chunks from one camera |
| Snow | n/a | **No real snow footage exists**: 0 snow mentions in 3,537 VSS captions across 13 cameras |

Glare is the clearest real-world match: YOLO finds a quarter of what Reason counts. The archive holds very few real clips of these conditions (2 pulled per condition), which is the gap Spore exists to fill. For snow, grown footage is the only way to test at all.

## Fix: fine-tune experiment

We fine-tuned YOLO11s on about 2.4k synthetic frames (Cosmos + physics, with labels taken from the seeds). Training took 136 s for 10 epochs on a Modal L40S. We evaluated it on held-out cameras. This eval used local yolo11s weights, because the hosted endpoint can't swap weights. Source: `results/fix/fix_detail.json`, same numbers as `results/fix_wip.json`.

| Held-out set | Stock | Fine-tuned | CLAHE only (no training) |
|---|---|---|---|
| 60 synthetic variants, recall | 0.318 | 0.310 | **0.363 (+4.5 pts)** |
| Held-out camera i24 p1c3 (42 clips), recall | 0.161 | 0.205 | 0.202 |
| Clean seeds (5), recall | 1.000 | 0.763 | 0.956 |
| 10 real check clips, detections/frame (no ground truth) | 3.54 | 2.31 | 3.50 |

The ship gate was "beats stock on held-out variants AND loses at most 5 pts on clean seeds"; the CLAHE step comes closest. **At this scale, grown footage is most valuable as a stress test, and it already comes labeled for larger training runs.**

**VSS re-ingest prompt fix.** `fix/reingest_*.py` re-ingests real archive clips through VSS with a condition-aware `custom_prompt` that makes the Reasoner state weather, lighting and visibility first. The baseline is measured on 10 real `neighborhood_cam-1` chunks (55 segments) labeled by eye: today's captions name the visible condition in 44 of 55 segments (80%), with search results per condition in `results/fix_before.json`.

## Run the demo

```bash
cd ui && npm install && npm run dev   # → http://localhost:5173
```

- 10-step walkthrough (→ / space to advance, ← back). Old dashboard: http://localhost:5173/?full=1
- Live VSS steps ("Ask the archive", highway searches) need team credentials in a repo-root `.env` (copy of `/config/team-16.config` from the lab VM; see *Environment* below) and run server-side in the Vite dev server.
- Ask the archive from the terminal: `python3 vss/ask.py "highway in dense fog"`
- Footage review tool: `python3 tools/review/server.py` → http://localhost:8765

## Sponsor tools

| Sponsor | How Spore uses it |
|---|---|
| VAST | VSS `/api/v1/search`, explore and captions to find the gap and pull real seed and real-condition clips; VSS upload + DataEngine ingest for 61 grown clips, now searchable; caption-quality baseline for a re-ingest prompt |
| NVIDIA | Cosmos Transfer 2.5 on a Modal H100 (gap fill); hosted Cosmos3 Nano Reasoner and YOLO11s (the models under test) |
| CoreWeave / Weights & Biases | Serverless W&B Inference for the gap-report agent; Weave traces on the LLM call and every loop stage. The hosted VSS models run on CoreWeave GPUs |
| SpaceXAI / Cursor | Built with Cursor agents, including Grok 4.7 running parallel tasks |

## Reproduce

### Environment (`.env` at repo root, names only)

| Variable | Used by |
|---|---|
| `USERNAME`, `PASSWORD` | VSS login (`vss/client.py`) |
| `GPU_BEARER_TOKEN` | hosted YOLO11s and Cosmos3 Reasoner (`eval/yolo_eval.py`, `eval/reason_eval.py`) |
| `WANDB_API_KEY` | W&B Inference + Weave (`loop/llm.py`, `loop/tracing.py`) |
| optional: `WANDB_ENTITY`, `WANDB_PROJECT`, `SPORE_LLM_MODEL`, `YOLO_URL`, `COSMOS3_REASON_URL`, `SPORE_YOLO_BACKEND` | overrides |

Modal (Cosmos Transfer, fine-tune) authenticates with `modal token new`, not `.env`.

### Per-stage commands

```bash
# Whole loop, or one stage (1 search, 2 report, 3 fill, 4 measure, 5 loop)
loop/.venv/bin/python loop/spore.py --iterations 1 --k 2
loop/.venv/bin/python loop/spore.py --stage N

# 1 Search
python3 vss/coverage.py                       # -> results/coverage.json
python3 vss/real_check.py                     # -> results/real_check.json + data/real_check/*.mp4

# 2 Report
loop/.venv/bin/python loop/report.py          # -> results/gap_report.json (Weave-traced)

# 3 Fill: physics grid (99 clips = 11 seeds x fog/rain/snow x 0.4/0.7/1.0)
for S in $(python3 -c "import json;print(' '.join(s['seed_id'] for s in json.load(open('data/seeds/manifest.json'))))"); do
  for K in fog rain snow; do for V in 0.4 0.7 1.0; do
    python3 gen/weather.py --in data/seeds/$S.mp4 --kind $K --severity $V \
      --out data/synthetic/${S}__phys_${K}_s$(printf %02d $(python3 -c "print(int($V*10))")).mp4 \
      --manifest data/synthetic/manifest.json --seed-id $S
  done; done
done
# 3 Fill: Cosmos Transfer 2.5 on Modal (4 conditions + clear-day control, control weight 0.5)
modal deploy gen/modal_transfer.py
SPORE_CONDS=fog_day_heavy,rain_night_heavy,snow_day_heavy,clear_night_light,clear_day_light python3 gen/first_batch.py

# 4 Measure
eval/.venv/bin/python eval/run_all.py --reason   # hosted YOLO11s + Cosmos3 Reasoner + edge-SSIM -> results/evals/
eval/.venv/bin/python eval/integrity.py          # vehicle-integrity gate -> integrity block in results/evals/
eval/.venv/bin/python eval/severity_curve.py     # -> results/severity_curve.json
eval/.venv/bin/python eval/render_all.py         # -> results/overlays/*.mp4|jpg + index.json
eval/.venv/bin/python eval/real_confirm.py       # YOLO vs Reason on real clips -> results/real_check.json

# Demo UI
cd ui && npm run dev                              # http://localhost:5173
```

Venv setup: [eval/README.md](eval/README.md), [gen/README.md](gen/README.md), [loop/README.md](loop/README.md).

### Inputs and outputs

| Path | Contents |
|---|---|
| `data/seeds/manifest.json` | 11 real seeds pulled from VSS (9 I-24 highway, 2 NYC), ~6 s, 16 fps, 93 frames, 1280×720 |
| `data/synthetic/manifest.json` | 160 variants: `kind` = `physics` (weather layer, with `severity`) or `generative` (Cosmos, with `control_weight`) |
| `results/coverage.json` | per-condition clip counts (any camera / highway) + VSS search top hits |
| `results/gap_report.json` | LLM gap report, candidates, next condition |
| `results/evals/<clip_id>.json` | `fidelity`, `yolo`, `reason`, `integrity`, `failure` per clip |
| `results/severity_curve.json` | recall vs severity per layer, breaking points, per-scene split |
| `results/failure_map.json` | per-arm n, mean recall, failure rate, `recall_rel_control` for Cosmos arms |
| `results/overlays/` | seed vs variant overlay videos and stills, `index.json` (integrity-gated) |
| `results/real_check.json` | real-archive clips per condition, YOLO vs Reason, confirmed flag, run log |
| `results/loop_log.json` | per-iteration stage log |
| `results/fix/fix_detail.json` | fine-tune vs stock vs CLAHE, per clip |
| `results/vss_uploads.json` | synthetic clips uploaded to VSS, object keys and index status |
| `results/fix_before.json` | VSS caption and search baseline for the re-ingest prompt |

### Metric definitions

- **Seed pseudo-ground truth:** hosted YOLO11s vehicle boxes (conf 0.4) on the clean seed, copied to the same frame of each variant. We measure consistency with the clean-day detector, not absolute accuracy.
- **`edge_ssim`:** mean over frames of SSIM(Canny(seed), Canny(variant)). The clip passes at ≥ 0.5 (`eval/fidelity.py`).
- **`vehicles_intact`:** the fraction of seed vehicles whose seed and variant patches have a contrast-normalized gradient correlation ≥ 0.35, i.e. the generator kept the car. A clip is a valid test case only if this is ≥ 0.7 (`eval/integrity.py`).
- **`recall_intact`:** detected intact vehicles divided by all intact vehicles (IoU ≥ 0.5). A miss counts only on a car that is actually there.
- **Breaking point:** the first severity at which mean `recall_intact` falls below 0.5, linearly interpolated (`eval/severity_curve.py`).
- **`recall_rel_control`:** a Cosmos arm's mean recall divided by the Cosmos clear-day control's recall (0.807, seeds with control recall ≥ 0.5).

## Repo layout

```
vss/      VSS client: coverage search, real-archive check, synthetic upload
loop/     loop/spore.py 5-stage loop, report.py (W&B Inference), Weave tracing
gen/      weather.py physics layer, Cosmos Transfer 2.5 on Modal
eval/     hosted YOLO11s + Cosmos3 Reasoner, fidelity, integrity, severity curve, overlays
fix/      YOLO fine-tune experiment, VSS re-ingest prompt baseline
ui/       demo UI (reads data/ and results/)
results/  all JSON outputs
```
