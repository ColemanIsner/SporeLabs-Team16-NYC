"""NVIDIA Cosmos3-Nano Transfer (edge-controlled video-to-video) on Modal, served by vLLM-Omni.

No hosted Cosmos3 generation API exists (build.nvidia.com / the hackathon GPU box only serve the
Reasoner), so we self-host the released `vllm/vllm-omni:cosmos3` image and call its
`/v1/videos/sync` endpoint from inside the container.

Smoke (one seed, heavy fog):
    modal run gen/modal_cosmos3.py::smoke
Batch (JSON list of {"video","prompt","name"}), one container per GPU slot:
    modal run gen/modal_cosmos3.py::batch --jobs jobs.json --out-dir data/synthetic
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

MODEL = os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano")
GPU = os.environ.get("COSMOS3_GPU", "H100")
HF_CACHE = "/root/.cache/huggingface"
PORT = 8000
NUM_FRAMES, FPS, W, H = 93, 16, 1280, 720

NEGATIVE = (
    "cartoon, illustration, painting, blurry, distorted, low quality, jittery, deformed, extra vehicles, "
    "missing vehicles, changed lane markings, new camera angle, text, watermark, warped geometry"
)

image = (
    modal.Image.from_registry("vllm/vllm-omni:cosmos3")
    .entrypoint([])
    .run_commands("which ffmpeg || (apt-get update && apt-get install -y ffmpeg) || pip install imageio-ffmpeg")
    # OpenCV's thread pool can deadlock inside the forked API server; keep it single-threaded.
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0", "OPENCV_FOR_THREADS_NUM": "1",
          # torch/OpenMP intra-op threads deadlock in vLLM-Omni's forked API server on the first transfer
          # request (hangs after frame decode, then 504 at 600 s); single-threaded OpenMP avoids it.
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          # baked in so the container's re-import sees the same volume config as the local side
          "COSMOS3_LOCAL_CACHE": os.environ.get("COSMOS3_LOCAL_CACHE", "0")})
)

app = modal.App("cosmos3-nano-transfer", image=image)
hf_vol = modal.Volume.from_name("cosmos3-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


EDGE_PRESETS = {"low": (50, 100), "medium": (100, 200), "high": (200, 300), "very_high": (300, 400)}


def _edge_video(src: Path, dst: Path, preset: str = "medium") -> Path:
    """Same canny control vLLM-Omni computes (RGB frames, preset thresholds), written as an mp4."""
    import cv2

    cv2.setNumThreads(1)
    lo, hi = EDGE_PRESETS[preset]
    cap = cv2.VideoCapture(str(src))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "8", str(dst)], stdin=subprocess.PIPE)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ff.stdin.write(cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), lo, hi).tobytes())
    ff.stdin.close()
    ff.wait()
    return dst


# Extra CPU/RAM so video decode and prompt prep never starve.
# COSMOS3_LOCAL_CACHE=1: no shared volume; each container downloads its own weights (~2 min) so containers
# never contend on HF cache locks in the shared volume.
LOCAL_CACHE = os.environ.get("COSMOS3_LOCAL_CACHE") == "1"


@app.cls(gpu=GPU, cpu=16, memory=65536, volumes={} if LOCAL_CACHE else {HF_CACHE: hf_vol}, secrets=[hf_secret],
         timeout=60 * 60,
         scaledown_window=300,
         max_containers=int(os.environ.get("COSMOS3_MAX_CONTAINERS", "6")))
class Cosmos3:
    @modal.enter()
    def start(self):
        import requests

        t0 = time.time()
        # Weights are already in the shared volume; offline mode keeps the server from taking HF file
        # locks on it (suspected cause of all-but-one containers hanging on their first transfer request).
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} if os.environ.get("COSMOS3_OFFLINE") == "1" else None
        self.proc = subprocess.Popen(
            ["vllm", "serve", MODEL, "--omni", "--model-class-name", "Cosmos3OmniDiffusersPipeline",
             "--no-guardrails", "--vae-use-tiling", "--allowed-local-media-path", "/",
             "--host", "127.0.0.1", "--port", str(PORT), "--init-timeout", "1800"],
            env=env,
        )
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(5)
        print(f"[cosmos3] server ready in {time.time() - t0:.0f}s on {GPU}", flush=True)

    @modal.method()
    def transfer(self, video_bytes: bytes, prompt: str, name: str = "clip", seed: int = 1,
                 steps: int = 35, controls: dict | None = None, extra_overrides: dict | None = None) -> dict:
        import requests

        t0 = time.time()
        work = Path("/tmp/job") / name
        work.mkdir(parents=True, exist_ok=True)
        raw, inp = work / "raw.mp4", work / "input.mp4"
        raw.write_bytes(video_bytes)
        vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"tpad=stop_mode=clone:stop={NUM_FRAMES},setsar=1")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", vf,
                        "-frames:v", str(NUM_FRAMES), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "16", str(inp)], check=True)
        controls = dict(controls or {"edge": True})
        if controls.get("edge") is True or (isinstance(controls.get("edge"), dict) and "control_path" not in controls["edge"]):
            # Precompute the canny control here instead of in the server: on-the-fly edge generation inside
            # vLLM-Omni intermittently hangs before denoising (and then 504s at 600 s).
            hint = controls["edge"] if isinstance(controls["edge"], dict) else {}
            controls["edge"] = {**hint, "control_path": str(_edge_video(inp, work / "edge.mp4",
                                                                        hint.get("preset_edge_threshold", "medium")))}
            controls["edge"].pop("preset_edge_threshold", None)
        extra = {**controls, "max_frames": NUM_FRAMES, "resolution": "720",
                 "num_video_frames_per_chunk": NUM_FRAMES, **(extra_overrides or {})}
        with inp.open("rb") as f:
            r = requests.post(
                f"http://127.0.0.1:{PORT}/v1/videos/sync",
                headers={"Accept": "video/mp4"},
                data={"model": MODEL, "prompt": prompt, "negative_prompt": NEGATIVE,
                      "size": f"{W}x{H}", "num_frames": str(NUM_FRAMES), "fps": str(FPS),
                      "num_inference_steps": str(steps), "seed": str(seed),
                      "extra_params": json.dumps(extra)},
                files={"input_reference": ("input.mp4", f, "video/mp4")},
                timeout=60 * 50,
            )
        if not r.ok:
            raise RuntimeError(f"{name}: HTTP {r.status_code}: {r.text[:1000]}")
        secs = time.time() - t0
        print(f"[cosmos3] {name} done in {secs:.1f}s", flush=True)
        return {"name": name, "mp4": r.content, "gen_seconds": round(secs, 1)}


# Defaults (guidance 3.0, control_guidance 1.5, emphasized edges) reproduce the clear seed with no
# weather; these come from the sweep below and are overwritten once a winner is picked.
TUNED_EXTRA: dict = {"guidance_scale": 7.0, "control_guidance": 1.0, "emphasize_control_in_prompt": False}

FOG_PROMPT = (
    "A locked-off elevated traffic camera looks along a multi-lane interstate in dense fog. Thick grey fog "
    "fills the air and drops visibility to about 50 m: the far lanes, trees and buildings fade into a flat "
    "white-grey wall, the sky is uniformly overcast with no sun and no shadows, colors are washed out and low "
    "contrast, and headlights glow as soft halos. The same cars and trucks hold their lanes."
)


def _run_jobs(jobs: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in jobs if not (out_dir / f"{j['name']}.mp4").exists()]
    print(f"{len(jobs)} jobs to run", flush=True)
    payloads = [(Path(j["video"]).read_bytes(), j["prompt"], j["name"], 1, 35, j.get("controls"),
                 j.get("extra", TUNED_EXTRA)) for j in jobs]
    results = []
    for res in Cosmos3().transfer.starmap(payloads, return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            print("FAILED:", res, flush=True)
            continue
        (out_dir / f"{res['name']}.mp4").write_bytes(res["mp4"])
        print(f"wrote {out_dir / (res['name'] + '.mp4')} ({res['gen_seconds']}s)", flush=True)
        results.append({"name": res["name"], "gen_seconds": res["gen_seconds"]})
    return results


@app.local_entrypoint()
def smoke(seed: str = "data/seeds/i24_scene1_p1c1_00.mp4", out_dir: str = "data/synthetic_cosmos3"):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompts import prompt_for

    cond = {"weather": "fog", "time": "day", "intensity": "heavy"}
    name = f"{Path(seed).stem}__c3_fog_day_heavy"
    print(_run_jobs([{"video": seed, "prompt": prompt_for("highway", cond), "name": name}], Path(out_dir)))


@app.local_entrypoint()
def sweep(seed: str = "data/seeds/i24_scene1_p1c1_00.mp4", out_dir: str = "data/synthetic_cosmos3/sweep"):
    """Fog settings sweep on one seed; each variant runs in its own container."""
    stem = Path(seed).stem
    variants = {
        "g7_cg1": ({"edge": True}, {"guidance_scale": 7.0, "control_guidance": 1.0, "emphasize_control_in_prompt": False}),
        "g7_cg05": ({"edge": True}, {"guidance_scale": 7.0, "control_guidance": 0.5, "emphasize_control_in_prompt": False}),
        "g7_cg1_hi": ({"edge": {"preset_edge_threshold": "very_high"}},
                      {"guidance_scale": 7.0, "control_guidance": 1.0, "emphasize_control_in_prompt": False}),
        "g10_cg05_hi": ({"edge": {"preset_edge_threshold": "very_high"}},
                        {"guidance_scale": 10.0, "control_guidance": 0.5, "emphasize_control_in_prompt": False}),
        "g7_cg1_iv": ({"edge": True}, {"guidance_scale": 7.0, "control_guidance": 1.5,
                                       "control_guidance_interval": [0.0, 0.5], "emphasize_control_in_prompt": False}),
    }
    jobs = [{"video": seed, "prompt": FOG_PROMPT, "name": f"{stem}__c3fog_{k}", "controls": c, "extra": e}
            for k, (c, e) in variants.items()]
    print(_run_jobs(jobs, Path(out_dir)))


@app.local_entrypoint()
def batch(jobs: str, out_dir: str = "data/synthetic_cosmos3"):
    res = _run_jobs(json.loads(Path(jobs).read_text()), Path(out_dir))
    (Path(out_dir) / "_timings.json").write_text(json.dumps(res, indent=1))
