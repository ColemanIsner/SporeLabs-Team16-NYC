# gen/ — Cosmos-Transfer2.5-2B (distilled edge) on Modal

`gen/modal_transfer.py` runs NVIDIA Cosmos-Transfer2.5-2B, using the **distilled edge** model (4 steps) for video-to-video, on a Modal H100.

## Inputs / outputs
- **Input:** any video plus a text prompt. Before inference, ffmpeg converts it to **16 fps, 1280x720 (scaled up, then center-cropped), exactly 93 frames**. If the clip is shorter than 93 frames at 16 fps (about 5.8 s), the last frame is repeated to fill the gap. The model generates the edge (canny) control video itself.
- **Output:** an mp4 with 93 frames at 16 fps, 720p (about 5.8 s).
- Guardrails are turned off (`--disable-guardrails`).

## One-time setup
```bash
modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)
```
Your HF account must accept the licenses for these gated repos: `nvidia/Cosmos-Transfer2.5-2B` (distilled weights) and `nvidia/Cosmos-Predict2.5-2B` (the Wan2.1 VAE, `tokenizer.pth`). The text encoder, `nvidia/Cosmos-Reason1-7B`, is not gated. Note: an upstream bug causes the distilled config registry to also download `nvidia/Cosmos-Predict2.5-2B/robot/action-cond/...` while loading. It is only downloaded once, because it is cached in the volume. Checkpoints are cached in the Modal Volume `cosmos-transfer-cache` (`HF_HOME=/cache/hf`).

## Run
```bash
# smoke test on the repo's robot example -> data/synthetic/smoke_robot.mp4
modal run gen/modal_transfer.py::smoke
modal run gen/modal_transfer.py::smoke --check-only     # image/GPU/imports/--help only

# single clip
modal run gen/modal_transfer.py --video path.mp4 --prompt "..." --out out.mp4

# batch: jobs.json = [{"video": "a.mp4", "prompt": "...", "name": "a"}, ...]
# runs in parallel (one GPU container per clip) and writes <out-dir>/<name>.mp4
modal run gen/modal_transfer.py::batch --jobs jobs.json --out-dir data/synthetic
```

From Python, after `modal deploy gen/modal_transfer.py`:
```python
import modal
f = modal.Function.from_name("cosmos-transfer25-edge", "transfer")
mp4_bytes = f.remote(open("in.mp4", "rb").read(), "prompt", "name")
```

Set `TRANSFER_GPU=B200` to use a different GPU. Only H100 has been tested.

## Implementation notes
- Image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, with the repo cloned at a pinned commit and set up with `uv sync --frozen --python 3.10 --extra=cu128`. The cu128 flash-attn wheel only exists for cp310, so the repo's default Python 3.13 fails to install. Torch is 2.7.0+cu128.
- `COSMOS_EXPERIMENTAL_CHECKPOINTS=1` must be set, or `--model=edge/distilled` is not registered.
- Each call loads the model in a new subprocess. Batch jobs fan out with one container per clip.
