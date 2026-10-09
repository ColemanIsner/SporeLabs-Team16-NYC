# eval/ — pixel conditions, fidelity gate, YOLO

## Setup (once)
```bash
uv venv --python 3.12 eval/.venv
VIRTUAL_ENV=eval/.venv uv pip install -r eval/requirements.txt
```
yolo11n weights auto-download on first run to `eval/.venv/yolo11n.pt` (git-ignored by the venv). Override with `YOLO_WEIGHTS=/path/yolo11n.pt`.

## Pixel conditions (no GPU, stdlib python + ffmpeg)
```bash
python3 gen/degrade.py --seed data/seeds/X.mp4 --cond glare --out data/synthetic/
python3 gen/degrade.py --seed data/seeds/X.mp4 --cond all            # all 6
```
Conditions: `glare motion_blur low_bitrate lens_dirt fog_pixel night_pixel`. Output `data/synthetic/<seed_id>__<cond>.mp4` (93 frames, 1280x720, 16 fps) and an upserted row in `data/synthetic/manifest.json` (`condition.kind = "pixel"`, `condition.pixel = <cond>`; file-locked read-modify-write). Existing outputs are reused unless `--force`. ~0.2–4 s per clip.

## Per clip
```bash
P=eval/.venv/bin/python
$P eval/fidelity.py  data/synthetic/X__fog_pixel.mp4 --seed data/seeds/X.mp4   # -> fidelity
$P eval/yolo_eval.py data/synthetic/X__fog_pixel.mp4 --seed data/seeds/X.mp4   # -> yolo (+ results/boxes/<clip_id>.json)
```
Both merge into `results/evals/<clip_id>.json` (`--out` to override) under a lock, keeping keys written by other tools, and recompute `failure` / `failure_reasons` per SPEC from whatever is present (`fidelity.pass`, `yolo.recall_vs_seed`, `reason.agree_vs_seed`).

- `fidelity.edge_ssim`: mean over frames of SSIM(Canny(seed), Canny(variant)), both resized to 640x360 (gray, 5x5 Gaussian, Canny 100/200). `pass` = ≥ `--threshold` (default 0.5).
- `yolo`: yolo11n, conf ≥ 0.4, classes person/bicycle/car/motorcycle/bus/truck. `mean_count` = vehicles per frame; `recall_vs_seed` = per frame fraction of seed vehicle boxes matched (greedy, IoU ≥ 0.5, class-agnostic among vehicles) by variant vehicle boxes, mean over frames with ≥1 seed box (`null` if the seed has no vehicles). `recall_vs_seed_all` includes person. Boxes cached in `results/boxes/<clip_id>.json` as `[x1,y1,x2,y2,conf,cls]` per frame; `--force` recomputes. `--device mps` for Apple GPU.

## Everything
```bash
eval/.venv/bin/python eval/run_all.py            # seeds (vs themselves) then every synthetic clip
eval/.venv/bin/python eval/run_all.py --force    # recompute all
eval/.venv/bin/python eval/run_all.py --only fog # subset by clip_id substring
```
Reads `data/seeds/manifest.json` + `data/synthetic/manifest.json`, skips clips rated `bad` in `results/reviews.json` (and marks an existing eval `"excluded": true`), skips clips whose eval already has fidelity+yolo and is newer than clip and seed. Never stops on one broken clip.

## Cosmos3-Reason Q&A (`reason`)
Hosted NVIDIA Cosmos3-Reason NIM (OpenAI-compatible chat; model id from `/v1/models`, currently `nvidia/cosmos3-nano-reasoner`). Needs `GPU_BEARER_TOKEN` in repo-root `.env` (loaded via python-dotenv; `COSMOS3_REASON_URL` defaults to `http://166.19.38.112:8001`).
```bash
P=eval/.venv/bin/python
$P eval/reason_eval.py                      # all seeds + synthetic clips (seeds first), 4 parallel requests
$P eval/reason_eval.py --only fog --force   # subset / re-ask
$P eval/reason_eval.py --clip X.mp4         # one-off, prints answers, writes nothing
$P eval/run_all.py --reason                 # fidelity + YOLO, then reason_eval (same --only/--force)
```
- Whole mp4 sent as `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}` (accepted by this NIM). If a server ever rejects it, auto-fallback to 8 sampled JPEG frames as `image_url` parts (`--frames`).
- Prompt = SPEC's 4 fixed questions, asks for a JSON object `{vehicle_count, lane_change, stopped_vehicle, weather_lighting, summary}`; temperature 0. Parser: JSON (inside `<answer>`/code fences/after `<think>` ok) then per-field regex.
- Writes `reason: {summary, answers, agree_vs_seed, model, input, parsed_json, seconds, raw, src_mtime}`. `agree_vs_seed` = fraction of {vehicle_count (within ±1), lane_change, stopped_vehicle} equal to the seed's; questions the seed left unanswered are skipped; seeds = 1.0. Merged with `fidelity.merge_eval`, so `failure` includes `reason_disagree` (agree < 0.67).
- Idempotent: re-asks only if `reason.answers` is missing, the clip's mtime changed, or `--force`; agreement is always recomputed from cached answers. Skips `bad`-rated clips.
- Timing: ~2.2–2.6 s per 93-frame 1280x720 clip (no cold start, hosted); answers were identical across two `--force` runs (temp 0).

## Timings (M-series Mac, CPU)
- YOLO yolo11n, 93 frames 1280x720: ~5.8 s inference, ~7 s wall per clip incl. startup (seed boxes cached, so a variant costs one pass).
- Fidelity: ~1 s per clip. run_all: ~4.5–5 s per variant.

## Calibration notes
Edge-SSIM is high on low-texture/empty scenes (mostly black edge maps agree). On a busy test scene: lens_dirt 0.91, glare 0.83, night_pixel 0.80, low_bitrate 0.75, fog_pixel 0.72, motion_blur 0.58. Recalibrate the 0.5 threshold on the first Cosmos batch.
