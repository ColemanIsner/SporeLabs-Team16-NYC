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
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App("cosmos3-nano-transfer", image=image)
hf_vol = modal.Volume.from_name("cosmos3-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


@app.cls(gpu=GPU, volumes={HF_CACHE: hf_vol}, secrets=[hf_secret], timeout=60 * 60, scaledown_window=300,
         max_containers=int(os.environ.get("COSMOS3_MAX_CONTAINERS", "6")))
class Cosmos3:
    @modal.enter()
    def start(self):
        import requests

        t0 = time.time()
        self.proc = subprocess.Popen(
            ["vllm", "serve", MODEL, "--omni", "--model-class-name", "Cosmos3OmniDiffusersPipeline",
             "--no-guardrails", "--vae-use-tiling", "--allowed-local-media-path", "/",
             "--host", "127.0.0.1", "--port", str(PORT), "--init-timeout", "1800"],
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
        hf_vol.commit()
        print(f"[cosmos3] server ready in {time.time() - t0:.0f}s on {GPU}", flush=True)

    @modal.method()
    def transfer(self, video_bytes: bytes, prompt: str, name: str = "clip", seed: int = 1,
                 steps: int = 35, controls: dict | None = None) -> dict:
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
        extra = {**(controls or {"edge": True}), "max_frames": NUM_FRAMES, "resolution": "720",
                 "num_video_frames_per_chunk": NUM_FRAMES}
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


def _run_jobs(jobs: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(Path(j["video"]).read_bytes(), j["prompt"], j["name"]) for j in jobs]
    results = []
    for res in Cosmos3().transfer.starmap(payloads, return_exceptions=True):
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
def batch(jobs: str, out_dir: str = "data/synthetic_cosmos3"):
    res = _run_jobs(json.loads(Path(jobs).read_text()), Path(out_dir))
    (Path(out_dir) / "_timings.json").write_text(json.dumps(res, indent=1))
