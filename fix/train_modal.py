"""Fine-tune ultralytics yolo11s on the Spore synthetic dataset (free labels) on a Modal GPU.

    eval/.venv/bin/python fix/build_dataset.py
    eval/.venv/bin/modal run fix/train_modal.py [--epochs 20 --imgsz 960]
-> results/fix/yolo11s_spore.pt (+ results/fix/train_log.json, results/fix/yolo11s_base.pt)
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import modal

GPU = os.environ.get("FIX_GPU", "L40S")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.3.200", "opencv-python-headless")
)
app = modal.App("spore-fix-yolo", image=image)
vol = modal.Volume.from_name("spore-fix-data", create_if_missing=True)


@app.function(gpu=GPU, volumes={"/data": vol}, timeout=60 * 30, cpu=8, memory=32768)
def train(epochs: int = 20, imgsz: int = 960, batch: int = 16) -> dict:
    from ultralytics import YOLO

    t0 = time.time()
    vol.reload()
    work = Path("/tmp/ds")
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile("/data/dataset.zip") as z:
        z.extractall(work)
    os.chdir("/tmp")
    model = YOLO("yolo11s.pt")
    names = model.names  # keep COCO ids, nc=80 -> drop-in replacement
    (work / "data.yaml").write_text(
        f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()))
    t1 = time.time()
    model.train(data=str(work / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, device=0, workers=8,
                lr0=float(os.environ.get("FIX_LR", "0.001")), optimizer="SGD", freeze=10, warmup_epochs=1, cos_lr=True, close_mosaic=3, patience=100,
                project="/tmp/runs", name="spore", exist_ok=True, plots=False, verbose=False)
    train_s = time.time() - t1
    run = Path("/tmp/runs/spore")
    res = (run / "results.csv").read_text() if (run / "results.csv").exists() else ""
    return {"best": (run / "weights" / "best.pt").read_bytes(), "base": Path("/tmp/yolo11s.pt").read_bytes(),
            "results_csv": res, "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
            "gpu": GPU, "epochs": epochs, "imgsz": imgsz, "batch": batch}


@app.local_entrypoint()
def main(epochs: int = 20, imgsz: int = 960, dataset: str = "fix/dataset"):
    root = Path(__file__).resolve().parent.parent
    ds = root / dataset
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in ds.rglob("*"):
            if p.is_file() and p.suffix in (".jpg", ".txt"):
                z.write(p, p.relative_to(ds))
    zp = Path(os.environ.get("TMPDIR", "/tmp")) / "spore_fix_dataset.zip"
    zp.write_bytes(buf.getvalue())
    t0 = time.time()
    with vol.batch_upload(force=True) as b:
        b.put_file(str(zp), "/dataset.zip")
    print(f"uploaded {zp.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
    out = train.remote(epochs=epochs, imgsz=imgsz)
    od = root / "results" / "fix"
    od.mkdir(parents=True, exist_ok=True)
    (od / "yolo11s_spore.pt").write_bytes(out.pop("best"))
    base = out.pop("base")
    if not (od / "yolo11s_base.pt").exists():
        (od / "yolo11s_base.pt").write_bytes(base)
    meta = json.loads((ds / "meta.json").read_text())
    out["dataset"] = {k: meta[k] for k in ("stride", "images_per_split_by_condition", "label_source")}
    (od / "train_log.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "results_csv"}, indent=2))
    print(out["results_csv"][-1500:])
