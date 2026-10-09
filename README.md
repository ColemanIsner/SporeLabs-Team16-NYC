**DEMO VIDEO: <<add link>>**

# Spore by SporeLabs

**Spore by SporeLabs: Other tools find what your video archive is missing. Spore grows the data to fill it, and shows where your AI goes blind.**

Spore is a video agent that audits your archive: it searches VSS for what your cameras have never seen, generates that missing footage, and measures what breaks in your own pipeline.

We stress-test the exact models inside your VSS pipeline: the hosted YOLO11s (the VSS Detector) and the hosted Cosmos3 Nano Reasoner (the VSS Reasoner). Synthetic clips are never uploaded to or indexed in VSS. VSS is used to search the real archive.

Every number below comes from a JSON file in `results/`, and the file is named next to it.

## Results in one screen

- **The gap is real.** There are 612 real clips indexed; 30 are from the I-24 highway cameras, and all 30 are clear daylight, verified by eye (`results/verify/i24_all_chunks.jpg`). Highway cameras have **0** night, rain, fog, snow, glare, or night-rain clips. Snow has **0 clips on any camera**. (`results/coverage.json`)
- **The detector breaks early.** Recall on intact vehicles falls below 50% at severity **0.34 for snow, 0.35 for fog, 0.60 for rain**. (`results/severity_curve.json`)
- **The Detector and Reasoner disagree.** In the worst physics condition (fog at severity 1.0), YOLO11s finds **0.34 vehicles/frame**, down from 7.77 on the clear seeds. Cosmos Reason still counts **7.6 vehicles**, down from 15.1. YOLO finds under 0.5 vehicles/frame on 7 of 11 clips, while Reason reports vehicles on 10 of 11. (computed from `results/evals/*__phys_fog_s10.json` and the seed evals)
- **Real footage gives weak, mixed confirmation.** Glare is the strongest case. Fog is **not** confirmed. No real snow footage exists. (`results/real_check.json`)
- **Fine-tuning gave a negative result.** YOLO fine-tuned on our synthetic clips did not beat a free CLAHE contrast trick and hurt real clips. (`results/fix/fix_detail.json`)

## The 5 stages

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
| Clear daytime | 101 | 23 |
| Night | 32 | 5 |
| Heavy rain | 8 | **0** |
| Dense fog | 4 | **0** |
| Low-sun glare | 3 | **0** |
| Snow | **0** | **0** |
| Night + rain | **0** | **0** |
| **Total indexed** | **612** | **127** |

### 2. Report: explain the gap (W&B Inference + Weave)

`loop/report.py` gives the coverage and eval numbers to a W&B Inference LLM (`deepseek-ai/DeepSeek-V4-Pro-0813`, 11.31 s). The LLM writes the gap report and picks the next condition. The call is traced in Weave: [wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave](https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave).

From `results/gap_report.json`:

> "Real archive coverage is missing highway scenes in heavy rain: 0 of 8 rain clips are highway, while synthetic heavy-rain validation fails completely." Next condition: **rain**.

This report was generated at 13:00, before we switched the evals to the hosted YOLO11s. Its archive counts still hold. Its synthetic recall figures (for example 0.148) come from the earlier local yolo11n run. Run `loop/spore.py --stage 2` to refresh them.

### 3. Fill: generate the missing footage

| Generator | What | Clips | Source |
|---|---|---|---|
| Physics weather layer (`gen/weather.py`) | fog, rain, snow at severity 0.4, 0.7, and 1.0 on 11 real seeds (9 I-24 highway, 2 NYC) | 99 (+4 rain at 0.8 from a loop iteration) | `data/synthetic/manifest.json` |
| NVIDIA Cosmos Transfer 2.5 (edge-distilled 2B, Modal H100) | night, night + rain, fog, snow relighting | 44 | same |
| Cosmos clear-day control | same generator, "nothing changes" prompt | 11 | same |

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

**Cosmos arms are reported relative to the control.** The Cosmos clear-day control, which should change nothing, already has recall **0.584**. So the Cosmos generator alone costs about 40% of recall, and we report Cosmos condition arms as `recall_rel_control`. Source: `results/failure_map.json`, n = 9 per arm; the 2 NYC seeds failed the edge-SSIM gate.

| Cosmos arm | Mean recall | Relative to control |
|---|---|---|
| Clear day (control) | 0.584 | 1.00 |
| Night + heavy rain | 0.405 | 0.69 |
| Clear night | 0.479 | 0.82 |
| Heavy fog | 0.508 | 0.87 |
| Heavy snow | 0.556 | 0.95 |

That is why the physics layer is the headline. Its pixels never move, so its labels are exact.

There is one quirk in `failure_map.json`. It lists `failure_rate` 0.0 for fog 0.7, fog 1.0 and rain 1.0. That is because the vehicle-integrity gate marks most of those clips as invalid test cases (vehicles_intact < 0.7: heavy fog and rain wash out the vehicle patches). Their recall still collapses (`mean_recall` 0.20, 0.078 and 0.058).

### 5. Loop: keep searching

`loop/spore.py` runs Search → Report → Fill → Measure → Continue. It logs each iteration to `results/loop_log.json`. The logged loop iteration reads: "Heavy rain: 0 real highway clips -> filled 2 -> recall 0.044". (That recall predates the switch to hosted YOLO.) The re-search found the same 612 indexed clips. The second `vss/real_check.py` run found **0 new chunks** (`results/real_check.json` → `runs`). The loop is ready for newly indexed footage, but none has arrived yet.

## Real-archive confirmation (honest: weak and mixed)

`vss/real_check.py` searched the real archive for each condition and pulled 2 clips per condition (10 total). `eval/real_confirm.py` ran hosted YOLO11s and Cosmos Reason on them. A condition counts as "confirmed" if YOLO finds fewer than 50% of Reason's vehicle count on any clip, which is the same signature we see on synthetic clips. Source: `results/real_check.json`.

| Condition | Confirmed | Evidence (YOLO vehicles/frame vs Reason count) |
|---|---|---|
| Glare | yes, 1 of 2 clips | **0.505 vs 2 (25%)**: the strongest case. The other clip: 8.796 vs 6 |
| Dusk | yes, 1 of 2 | 0.441 vs 1 (44%); 1.57 vs 2 (78%) |
| Rain | yes, 1 of 2 (borderline) | 0.989 vs 2 (49%); 4.0 vs 1 (400%). Rain is barely visible in either clip |
| Night | no | 6.204 vs 8 (78%); on the other clip, an IR camera, Reason sees no vehicles |
| Fog | **no** | YOLO beats Reason on real fog: 4.495 vs 2 and 4.505 vs 2. Both clips are adjacent chunks from one camera |
| Snow | n/a | **No real snow footage exists**: 0 snow mentions in 3,537 VSS captions across 13 cameras |

The sample is tiny and the confirmations are weak. We do not claim that real footage proves the synthetic findings. For snow, synthetic footage is the only way to test at all.

## Fix attempt: a negative result

We fine-tuned YOLO11s on about 2.4k synthetic frames (Cosmos + physics, with labels taken from the seeds). Training took 136 s for 10 epochs on a Modal L40S. We evaluated it on held-out cameras. This eval used local yolo11s weights, because the hosted endpoint can't swap weights. Source: `results/fix/fix_detail.json`, same numbers as `results/fix_wip.json`.

| Held-out set | Stock | Fine-tuned | CLAHE only (no training) |
|---|---|---|---|
| 60 synthetic variants, recall | 0.318 | 0.310 | **0.363 (+4.5 pts)** |
| Held-out camera i24 p1c3 (42 clips), recall | 0.161 | 0.205 | 0.202 |
| Clean seeds (5), recall | 1.000 | 0.763 | 0.956 |
| 10 real check clips, detections/frame (no ground truth) | 3.54 | 2.31 | 3.50 |

The fine-tune failed its gate ("after > before on held-out variants AND clean-seed recall drop <= 5 pts"). **At this scale, synthetic footage is more valuable as a stress test than as training data.** `results/fix.json` was never written.

**Next step:** a VSS re-ingest prompt fix. Re-ingest real archive clips with a condition-aware `custom_prompt` that makes the Reasoner state weather, lighting and visibility first. Then measure search precision on held-out real clips. This is designed but not measured.

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
| VAST | VSS `/api/v1/search`, explore, and captions: finding the gap (stage 1) and finding real-condition clips for confirmation |
| NVIDIA | Cosmos Transfer 2.5 on a Modal H100 (gap fill); hosted Cosmos3 Nano Reasoner and YOLO11s (the models under test) |
| CoreWeave / Weights & Biases | Serverless W&B Inference for the gap-report agent, plus Weave traces. The hosted models run on CoreWeave GPUs |
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
| `data/synthetic/manifest.json` | 158 variants: `kind` = `physics` (weather layer, with `severity`) or `generative` (Cosmos, with `control_weight`) |
| `results/coverage.json` | per-condition clip counts (any camera / highway) + VSS search top hits |
| `results/gap_report.json` | LLM gap report, candidates, next condition |
| `results/evals/<clip_id>.json` | `fidelity`, `yolo`, `reason`, `integrity`, `failure` per clip |
| `results/severity_curve.json` | recall vs severity per layer, breaking points, per-scene split |
| `results/failure_map.json` | per-arm n, mean recall, failure rate, `recall_rel_control` for Cosmos arms |
| `results/overlays/` | seed vs variant overlay videos and stills, `index.json` (integrity-gated) |
| `results/real_check.json` | real-archive clips per condition, YOLO vs Reason, confirmed flag, run log |
| `results/loop_log.json` | per-iteration stage log |
| `results/fix/fix_detail.json` | fine-tune vs stock vs CLAHE, per clip |

### Metric definitions

- **Seed pseudo-ground truth:** hosted YOLO11s vehicle boxes (conf 0.4) on the clean seed, copied to the same frame of each variant. We measure consistency with the clean-day detector, not absolute accuracy.
- **`edge_ssim`:** mean over frames of SSIM(Canny(seed), Canny(variant)). The clip passes at ≥ 0.5 (`eval/fidelity.py`).
- **`vehicles_intact`:** the fraction of seed vehicles whose seed and variant patches have a contrast-normalized gradient correlation ≥ 0.35, i.e. the generator kept the car. A clip is a valid test case only if this is ≥ 0.7 (`eval/integrity.py`).
- **`recall_intact`:** detected intact vehicles divided by all intact vehicles (IoU ≥ 0.5). A miss counts only on a car that is actually there.
- **Breaking point:** the first severity at which mean `recall_intact` falls below 0.5, linearly interpolated (`eval/severity_curve.py`).
- **`recall_rel_control`:** a Cosmos arm's mean recall divided by the Cosmos clear-day control's recall (0.584).

## Repo layout

```
vss/      VSS client: coverage search, real-archive check (no synthetic ingest)
loop/     loop/spore.py 5-stage loop, report.py (W&B Inference), Weave tracing
gen/      weather.py physics layer, Cosmos Transfer 2.5 on Modal
eval/     hosted YOLO11s + Cosmos3 Reasoner, fidelity, integrity, severity curve, overlays
fix/      YOLO fine-tune experiment (negative result)
ui/       demo UI (reads data/ and results/)
results/  all JSON outputs
```
