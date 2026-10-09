"""Visual check that the I-24 archive really has no bad weather: 1 frame from every indexed i24 chunk."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import client
ROOT = client.ROOT
out = ROOT / "results" / "verify"; tmp = out / "i24_frames"; tmp.mkdir(parents=True, exist_ok=True)
chunks = sorted([c for c in client.explore_all() if c.get("camera_id") == "i24_cam-1"], key=lambda c: c["filename"])
rows = []
for c in chunks:
    seg = c.get("preview_source") or c["original_video"]
    mp4 = tmp / (Path(c["filename"]).stem + ".mp4"); jpg = mp4.with_suffix(".jpg")
    if not jpg.exists():
        client.download(seg, mp4)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "1", "-i", str(mp4), "-frames:v", "1",
                        "-vf", "scale=320:180", str(jpg)], check=False)
        mp4.unlink(missing_ok=True)
    rows.append({"filename": c["filename"], "frame": str(jpg.relative_to(ROOT)),
                 "caption_excerpt": (c.get("reasoning_content") or "")[:160]})
imgs = [r["frame"] for r in rows if (ROOT / r["frame"]).exists()]
args = sum([["-i", str(ROOT / f)] for f in imgs], [])
n = len(imgs); cols = 6
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args, "-filter_complex",
                "".join(f"[{i}:v]" for i in range(n)) + f"xstack=inputs={n}:layout=" +
                "|".join(f"{(i % cols) * 320}_{(i // cols) * 180}" for i in range(n)) + ":fill=black",
                str(out / "i24_all_chunks.jpg")], check=False)
json.dump({"n_chunks": len(rows), "sheet": "results/verify/i24_all_chunks.jpg", "chunks": rows},
          open(out / "i24_verify.json", "w"), indent=2)
print(len(rows), "chunks ->", out / "i24_all_chunks.jpg")
