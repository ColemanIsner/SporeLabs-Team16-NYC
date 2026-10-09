#!/usr/bin/env python3
"""Export a sample of the Fix-step training set (fix/dataset, built by fix/build_dataset.py) for the Story UI.

Writes results/train_samples/*.jpg (one seed frame + its weather-layer variants, same frame index)
and results/train_samples/train_samples.json (per-sample YOLO boxes read from the real label files, a label-file
excerpt, and counts over every weather-layer frame in the dataset).

Only weather-layer ("phys_*") variants are shown: they are pixel-aligned with the seed, so the seed's
boxes are exact. Cosmos Transfer variants re-frame the scene, so their copied boxes do not line up.

Usage: eval/.venv/bin/python tools/train_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "fix" / "dataset"
OUT = ROOT / "results" / "train_samples"
SEED, FRAME = "i24_scene1_p1c1_00", "f048"
SHOW = [  # (condition tag in the file name, label)
    ("", "Clear day"),
    ("phys_fog_s04", "Fog · light"),
    ("phys_fog_s10", "Fog · heavy"),
    ("phys_rain_s07", "Rain"),
    ("phys_snow_s07", "Snow"),
    ("phys_snow_s10", "Snow · heavy"),
]
COCO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def cond_of(stem: str) -> str:
    # i24_scene1_p1c1_00__phys_fog_s04__f048 -> phys_fog_s04 ; seed frame -> clean
    parts = stem.split("__")
    return "clean" if len(parts) == 2 else parts[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for tag, label in SHOW:
        stem = f"{SEED}__{tag}__{FRAME}" if tag else f"{SEED}__{FRAME}"
        img = cv2.imread(str(DS / "images" / "train" / f"{stem}.jpg"))
        txt = (DS / "labels" / "train" / f"{stem}.txt").read_text()
        cv2.imwrite(str(OUT / f"{stem}.jpg"), cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        boxes = [[int(p[0])] + [float(v) for v in p[1:]] for p in (l.split() for l in txt.splitlines()) if p]
        samples.append({"cond": tag or "clean", "label": label, "img": f"results/train_samples/{stem}.jpg",
                        "label_file": f"{stem}.txt", "boxes": boxes})

    ref = samples[1]
    ref_txt = (DS / "labels" / "train" / ref["label_file"]).read_text().splitlines()
    identical = all(s["boxes"] == samples[0]["boxes"] for s in samples)

    # Counts over every weather-layer frame (and the clear seeds they were grown from) in train + val.
    frames = boxes = 0
    conds, seeds, classes = set(), set(), {}
    seed_frames = seed_boxes = 0
    for split in ("train", "val"):
        for f in (DS / "labels" / split).glob("*.txt"):
            c = cond_of(f.stem)
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            if c == "clean":
                seed_frames += 1
                seed_boxes += len(lines)
                continue
            if not c.startswith("phys_"):
                continue
            frames += 1
            boxes += len(lines)
            conds.add(c)
            seeds.add(f.stem.split("__")[0])
            for l in lines:
                n = COCO.get(int(l.split()[0]), l.split()[0])
                classes[n] = classes.get(n, 0) + 1

    meta = json.loads((DS / "meta.json").read_text())
    out = {
        "seed": SEED, "frame": int(FRAME[1:]),
        "samples": samples, "boxes_identical_across_samples": identical,
        "excerpt": {"file": f"labels/train/{ref['label_file']}", "lines": ref_txt[:6], "total_lines": len(ref_txt)},
        "weather_layer": {"frames": frames, "boxes": boxes, "conditions": len(conds), "seed_clips": len(seeds),
                          "by_class": dict(sorted(classes.items(), key=lambda kv: -kv[1]))},
        "clear_seed": {"frames": seed_frames, "boxes": seed_boxes},
        "hand_labeled_boxes": 0,
        "label_source": meta.get("label_source"),
        "source": "fix/dataset (fix/build_dataset.py)",
    }
    (OUT / "train_samples.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
