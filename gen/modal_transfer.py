"""Cosmos-Transfer2.5-2B (distilled edge) video-to-video on Modal.

Single clip:
    modal run gen/modal_transfer.py --video in.mp4 --prompt "..." --out out.mp4
Batch (JSON list of {"video","prompt","name"}), parallel fan-out:
    modal run gen/modal_transfer.py::batch --jobs jobs.json --out-dir data/synthetic
Smoke test on the repo's robot example:
    modal run gen/modal_transfer.py::smoke
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO_URL = "https://github.com/nvidia-cosmos/cosmos-transfer2.5"
REPO_COMMIT = "2ff49d0717af02057ae79bc75c00fbff9da1b4e7"
WORKDIR = "/workspace"
VENV_PY = f"{WORKDIR}/.venv/bin/python"
CACHE = "/cache"
GPU = os.environ.get("TRANSFER_GPU", "H100")

NUM_FRAMES = 93
FPS = 16
W, H = 1280, 720

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("curl", "ffmpeg", "git", "git-lfs", "libx11-dev", "libgl1", "libglib2.0-0", "wget", "tree")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.8.12/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
        "git lfs install",
        f"GIT_LFS_SKIP_SMUDGE=1 git clone {REPO_URL} {WORKDIR}"
        f" && cd {WORKDIR} && git checkout {REPO_COMMIT}"
        " && git lfs pull --include='assets/robot_example/**'",
    )
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_INSTALL_DIR": "/opt/uv-python"})
    .run_commands(f"cd {WORKDIR} && echo 3.10 > .python-version && uv sync --frozen --python 3.10 --extra=cu128 && echo cu128 > .cuda-name")
    .env(
        {
            "HF_HOME": f"{CACHE}/hf",
            "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1",  # required for edge/distilled to be registered
            "TORCH_HOME": f"{CACHE}/torch",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor",
        }
    )
)

app = modal.App("cosmos-transfer25-edge", image=image)
cache_vol = modal.Volume.from_name("cosmos-transfer-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _preprocess(src: Path, dst: Path) -> None:
    """Resample to 16fps, center-crop/scale to 1280x720, exactly 93 frames (pad by cloning last frame)."""
    vf = (
        f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"tpad=stop_mode=clone:stop={NUM_FRAMES},setsar=1"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-frames:v", str(NUM_FRAMES),
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(dst)],
        check=True,
    )


def _run_inference(spec: dict, outdir: Path, extra_args: list[str] | None = None) -> None:
    spec_path = outdir / f"{spec['name']}_spec.json"
    spec_path.write_text(json.dumps(spec))
    cmd = [VENV_PY, "examples/inference.py", "-i", str(spec_path), "-o", str(outdir),
           "--model=edge/distilled", "--disable-guardrails", *(extra_args or [])]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=WORKDIR, check=True)


@app.function(gpu=GPU, volumes={CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 40)
def transfer(video_bytes: bytes, prompt: str, name: str = "clip", seed: int = 1,
             control_weight: float = 1.0, preprocess: bool = True) -> bytes:
    """Video bytes + prompt -> generated mp4 bytes (93 frames, 16fps, 720p)."""
    t0 = time.time()
    work = Path("/tmp/job") / name
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw.mp4"
    raw.write_bytes(video_bytes)
    inp = work / "input.mp4"
    if preprocess:
        _preprocess(raw, inp)
    else:
        inp = raw
    outdir = work / "out"
    outdir.mkdir(exist_ok=True)
    spec = {
        "name": name,
        "prompt": prompt,
        "video_path": str(inp),
        "guidance": 3,
        "num_steps": 4,
        "seed": seed,
        "edge": {"control_weight": control_weight},  # control video auto-computed (canny)
    }
    try:
        _run_inference(spec, outdir)
    finally:
        cache_vol.commit()
    out = outdir / f"{name}.mp4"
    if not out.exists():
        raise RuntimeError(f"no output; files: {list(outdir.iterdir())}")
    print(f"[transfer] {name} done in {time.time() - t0:.1f}s on {GPU}", flush=True)
    return out.read_bytes()


@app.function(gpu=GPU, volumes={CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 60)
def transfer_full(video_bytes: bytes, prompt: str, name: str = "clip", controls: dict | None = None,
                  num_steps: int = 35, guidance: int = 7, seed: int = 1,
                  negative_prompt: str | None = None, preprocess: bool = True) -> bytes:
    """Non-distilled Cosmos-Transfer2.5-2B, multi-control.

    controls: e.g. {"depth": {"control_weight": 0.6}, "edge": {"control_weight": 0.3}}.
    Missing control_path -> auto-generated (edge=canny, depth=VideoDepthAnything, seg=GroundingDINO+SAM2,
    vis=blur). Weights are normalized upstream to sum <= 1. guidance is 0..7.
    """
    t0 = time.time()
    controls = controls or {"edge": {"control_weight": 1.0}}
    work = Path("/tmp/job") / name
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw.mp4"
    raw.write_bytes(video_bytes)
    inp = work / "input.mp4"
    if preprocess:
        _preprocess(raw, inp)
    else:
        inp = raw
    outdir = work / "out"
    outdir.mkdir(exist_ok=True)
    spec = {"name": name, "prompt": prompt, "video_path": str(inp), "guidance": int(guidance),
            "num_steps": int(num_steps), "seed": seed}
    if negative_prompt:
        spec["negative_prompt"] = negative_prompt
    for k, v in controls.items():
        spec[k] = dict(v)
    extra = []
    if len(controls) == 1:  # single-control base model variant (edge/depth/seg/vis)
        extra = [f"--model={next(iter(controls))}"]
    spec_path = outdir / f"{name}_spec.json"
    spec_path.write_text(json.dumps(spec))
    cmd = [VENV_PY, "examples/inference.py", "-i", str(spec_path), "-o", str(outdir), "--disable-guardrails", *extra]
    print("RUN:", " ".join(cmd), json.dumps(spec), flush=True)
    try:
        subprocess.run(cmd, cwd=WORKDIR, check=True)
    finally:
        cache_vol.commit()
    out = outdir / f"{name}.mp4"
    if not out.exists():
        raise RuntimeError(f"no output; files: {list(outdir.iterdir())}")
    print(f"[transfer_full] {name} done in {time.time() - t0:.1f}s on {GPU}", flush=True)
    return out.read_bytes()


@app.function(gpu=GPU, volumes={CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 10)
def check_env() -> str:
    """Validate the image: GPU, torch, imports, CLI --help."""
    r = subprocess.run(
        [VENV_PY, "-c", "import torch,cosmos_transfer2;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
        cwd=WORKDIR, capture_output=True, text=True)
    h = subprocess.run([VENV_PY, "examples/inference.py", "--help"], cwd=WORKDIR, capture_output=True, text=True)
    return f"IMPORT rc={r.returncode}\n{r.stdout}{r.stderr[-3000:]}\nHELP rc={h.returncode}\n{h.stdout[-6000:]}{h.stderr[-3000:]}"


@app.local_entrypoint()
def main(video: str, prompt: str, out: str = "out.mp4", name: str = "clip", seed: int = 1):
    t0 = time.time()
    data = transfer.remote(Path(video).read_bytes(), prompt, name, seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    print(f"wrote {out} ({len(data)} bytes) in {time.time() - t0:.1f}s")


@app.local_entrypoint()
def batch(jobs: str, out_dir: str = "data/synthetic"):
    """jobs: JSON file with a list of {"video": path, "prompt": str, "name": str[, "seed": int]}."""
    items = json.loads(Path(jobs).read_text())
    base = Path(jobs).parent
    args = []
    for it in items:
        vp = Path(it["video"])
        if not vp.is_absolute() and not vp.exists():
            vp = base / vp
        args.append((vp.read_bytes(), it["prompt"], it["name"], it.get("seed", 1)))
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for it, res in zip(items, transfer.starmap(args, return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"FAILED {it['name']}: {res}")
            continue
        (od / f"{it['name']}.mp4").write_bytes(res)
        print(f"wrote {od / (it['name'] + '.mp4')}")
    print(f"batch of {len(items)} done in {time.time() - t0:.1f}s")


@app.function(volumes={CACHE: cache_vol}, timeout=60 * 5)
def _example_asset() -> tuple[bytes, str]:
    return (Path(WORKDIR, "assets/robot_example/robot_input.mp4").read_bytes(),
            Path(WORKDIR, "assets/robot_example/robot_prompt.txt").read_text().strip())


@app.local_entrypoint()
def smoke(out: str = "data/synthetic/smoke_robot.mp4", check_only: bool = False):
    print(check_env.remote())
    if check_only:
        return
    video, prompt = _example_asset.remote()
    t0 = time.time()
    data = transfer.remote(video, prompt, "robot_edge_smoke")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    print(f"SMOKE OK: wrote {out} ({len(data)} bytes) in {time.time() - t0:.1f}s on {GPU}")
