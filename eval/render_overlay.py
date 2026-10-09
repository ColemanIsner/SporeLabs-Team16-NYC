#!/usr/bin/env python3
"""Side-by-side "AHA" overlay: real seed vs synthetic variant, YOLO boxes + ghost misses.

Usage:
  eval/.venv/bin/python eval/render_overlay.py <clip_id> [--manifest data/synthetic/manifest.json]
      [--evals-dir results/evals] [--out-dir results/overlays] [--force]

Output (1920x540, 16 fps, h264 yuv420p):
  results/overlays/<clip_id>.mp4       left = REAL seed (green YOLO vehicle boxes, live count)
                                        right = SYNTHETIC variant: green = seed vehicle still detected
                                        (IoU>=0.5, greedy, same as yolo_eval), RED dashed = seed vehicle
                                        YOLO missed ("ghost"), running recall %.
                                        Bottom strip: Cosmos Reason answer for each side.
  results/overlays/<clip_id>.jpg       poster = frame with the most misses
  results/overlays/<clip_id>.meta.json numbers used by render_all's index.json
Boxes come from results/boxes/<clip_id>.json (computed with yolo_eval.detect if missing).
"""
import argparse
import json
import os
import subprocess
import uuid
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import ROOT, load_json, rel  # noqa: E402
from integrity import THRESH, patch_score  # noqa: E402

RENDER_VERSION = 3

BOXES = ROOT / "results" / "boxes"
EVALS = ROOT / "results" / "evals"
OUT = ROOT / "results" / "overlays"
SYN_MANIFEST = ROOT / "data" / "synthetic" / "manifest.json"
SEED_MANIFEST = ROOT / "data" / "seeds" / "manifest.json"
VEHICLES = {1, 2, 3, 5, 7}
IOU_MATCH = 0.5
PW, PH = 960, 540
FPS = 16

GREEN = (60, 230, 90)    # BGR
RED = (40, 40, 255)
WHITE = (255, 255, 255)

# ---------- fonts ----------
_FONT_CANDIDATES = {
    "black": ["/System/Library/Fonts/Supplemental/Arial Black.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "reg": ["/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}
_FONTS = {}


def font(kind, size):
    key = (kind, size)
    if key not in _FONTS:
        f = None
        for p in _FONT_CANDIDATES[kind]:
            if Path(p).exists():
                f = ImageFont.truetype(p, size)
                break
        _FONTS[key] = f or ImageFont.load_default(size=size)
    return _FONTS[key]


# ---------- data ----------
def manifest_row(clip_id, manifest):
    for r in load_json(manifest, []) or []:
        if r.get("clip_id") == clip_id:
            return r
    return None


def seed_src(seed_id):
    for r in load_json(SEED_MANIFEST, []) or []:
        if r.get("seed_id") == seed_id and r.get("src"):
            return ROOT / r["src"]
    return ROOT / "data" / "seeds" / f"{seed_id}.mp4"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_boxes(clip_id, src, ev):
    p = resolve((ev or {}).get("yolo", {}).get("boxes") or BOXES / f"{clip_id}.json")
    d = load_json(p, None)
    if not d or not d.get("frames"):
        from yolo_eval import detect  # lazy: pulls in ultralytics
        d = detect(src, clip_id)
    return [[b for b in f if int(b[5]) in VEHICLES] for f in d["frames"]]


def read_frames(src, n=None):
    cap = cv2.VideoCapture(str(src))
    out = []
    while n is None or len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def match(gt, pred):
    """Greedy IoU>=0.5 matching (same rule as yolo_eval.frame_recall). -> {gt_idx: pred_idx}."""
    if not gt or not pred:
        return {}
    a = np.asarray(gt, float)[:, None, :4]
    b = np.asarray(pred, float)[None, :, :4]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    m = inter / np.maximum(area(a) + area(b) - inter, 1e-9)
    pairs = sorted(((g, p) for g in range(m.shape[0]) for p in range(m.shape[1]) if m[g, p] >= IOU_MATCH),
                   key=lambda t: -m[t])
    out, used_p = {}, set()
    for g, p in pairs:
        if g in out or p in used_p:
            continue
        out[g] = p
        used_p.add(p)
    return out


def intact_flags(seed_f, var_f, sb, pad=4):
    """Same per-vehicle patch test as eval/integrity.py. True = generator kept the car,
    False = blurred/merged/erased (gen artifact), None = patch too small to judge (treated as artifact)."""
    H, W = var_f.shape[:2]
    out = []
    for b in sb:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
        sc = patch_score(seed_f[y1:y2, x1:x2], var_f[y1:y2, x1:x2])
        out.append(None if sc is None else sc >= THRESH)
    return out


def dotted_rect(img, p1, p2, color, th=1, step=6):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2 + 1, step):
        cv2.circle(img, (x, y1), th, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y2), th, color, -1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, step):
        cv2.circle(img, (x1, y), th, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2, y), th, color, -1, cv2.LINE_AA)


def cond_label(cond):
    cond = cond or {}
    if cond.get("kind") == "pixel" or cond.get("pixel"):
        return str(cond.get("pixel") or "pixel").replace("_", " ").upper()
    w, t, i = cond.get("weather"), cond.get("time"), cond.get("intensity")
    parts = []
    if w and w != "clear":
        parts.append(f"{i} {w}" if i == "heavy" else w)
    if t:
        parts.append({"day": "daylight"}.get(t, t))
    return " · ".join(parts).upper() or "VARIANT"


def reason_text(ev):
    a = ((ev or {}).get("reason") or {}).get("answers") or {}
    if not a:
        return None
    wl = (a.get("weather_lighting") or "").strip().rstrip(".")
    vc = a.get("vehicle_count")
    s = f"{vc} vehicles" if vc is not None else ""
    if wl:
        s += f"{'  ·  ' if s else ''}“{wl}”"
    return s


# ---------- drawing ----------
def scale_box(b, sx, sy):
    return int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)


def dashed_rect(img, p1, p2, color, th=3, dash=10, gap=7):
    (x1, y1), (x2, y2) = p1, p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = max(abs(bx - ax), abs(by - ay))
        if L == 0:
            continue
        s = 0
        while s < L:
            e = min(s + dash, L)
            cv2.line(img, (int(ax + (bx - ax) * s / L), int(ay + (by - ay) * s / L)),
                     (int(ax + (bx - ax) * e / L), int(ay + (by - ay) * e / L)), color, th, cv2.LINE_AA)
            s += dash + gap


def solid_rect(img, p1, p2, color, th=3):
    cv2.rectangle(img, p1, p2, (0, 0, 0), th + 2, cv2.LINE_AA)  # dark outline for contrast
    cv2.rectangle(img, p1, p2, color, th, cv2.LINE_AA)


def darken(img, y0, y1, alpha=0.62):
    img[y0:y1] = (img[y0:y1].astype(np.float32) * (1 - alpha)).astype(np.uint8)


TOP_H, BOT_H = 92, 58


def recall_color(r):
    """RGB (for PIL): green >=0.7, amber >=0.4, red below."""
    return (90, 230, 110) if r >= 0.7 else (255, 190, 40) if r >= 0.4 else (255, 70, 70)


def compose(seed_f, var_f, sb, vb, m, info, run_hit, run_tot, ok):
    sh, sw = seed_f.shape[:2]
    left = cv2.resize(seed_f, (PW, PH), interpolation=cv2.INTER_AREA)
    right = cv2.resize(var_f, (PW, PH), interpolation=cv2.INTER_AREA)
    sx, sy = PW / sw, PH / sh
    vh, vw = var_f.shape[:2]
    vsx, vsy = PW / vw, PH / vh

    for b in sb:
        x1, y1, x2, y2 = scale_box(b, sx, sy)
        solid_rect(left, (x1, y1), (x2, y2), GREEN)
    ghosts, artifacts = [], []
    for gi, b in enumerate(sb):
        if not ok[gi]:
            x1, y1, x2, y2 = scale_box(b, sx, sy)
            dotted_rect(right, (x1, y1), (x2, y2), (190, 190, 190), th=1)
            artifacts.append((x1, y1, x2, y2))
        elif gi in m:
            x1, y1, x2, y2 = scale_box(vb[m[gi]], vsx, vsy)
            solid_rect(right, (x1, y1), (x2, y2), GREEN)
        else:
            x1, y1, x2, y2 = scale_box(b, sx, sy)
            # faint red fill so ghosts pop even when small
            roi = right[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            if roi.size:
                roi[:] = (roi.astype(np.float32) * 0.7 + np.array(RED, np.float32) * 0.3).astype(np.uint8)
            dashed_rect(right, (x1, y1), (x2, y2), (0, 0, 0), th=6)
            dashed_rect(right, (x1, y1), (x2, y2), RED, th=3)
            ghosts.append((x1, y1, x2, y2))

    canvas = np.concatenate([left, right], axis=1)
    darken(canvas, 0, TOP_H)
    darken(canvas, PH - BOT_H, PH)
    cv2.line(canvas, (PW, 0), (PW, PH), (20, 20, 20), 6)
    cv2.line(canvas, (PW, 0), (PW, PH), (235, 235, 235), 2)

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    n_all = len(sb)
    n = sum(1 for v in ok if v)
    k = sum(1 for gi in m if ok[gi])
    n_art = n_all - n
    for (x1, y1, x2, y2) in artifacts:
        if x2 - x1 >= 70 and y1 > TOP_H + 16 and y1 < PH - BOT_H - 16:
            f = font("bold", 11)
            d.text((PW + max(x1, 0) + 1, y1 - 14), "gen artifact", font=f, fill=(200, 200, 200),
                   stroke_width=2, stroke_fill=(0, 0, 0))

    def pill(x, y, text, fill, fg=(255, 255, 255), size=17):
        f = font("black", size)
        tw = d.textlength(text, font=f)
        d.rounded_rectangle((x, y, x + tw + 20, y + size + 12), radius=6, fill=fill)
        d.text((x + 10, y + 5), text, font=f, fill=fg)
        return x + tw + 20

    def shadow_text(xy, text, f, fill, anchor="la"):
        x, y = xy
        d.text((x + 2, y + 2), text, font=f, fill=(0, 0, 0), anchor=anchor)
        d.text((x, y), text, font=f, fill=fill, anchor=anchor)

    # ghost tags
    for (x1, y1, x2, y2) in ghosts:
        if x2 - x1 >= 34 and y2 > TOP_H + 18 and y1 < PH - BOT_H - 18:
            f = font("black", 12)
            ty = y1 - 16 if y1 >= TOP_H + 16 else max(y1, TOP_H) + 2
            tw = d.textlength("MISSED", font=f) + 8
            tx = PW + min(max(x1, 0), PW - 4 - int(tw))
            d.rectangle((tx, ty, tx + tw, ty + 15), fill=(230, 30, 30))
            d.text((tx + 4, ty + 1), "MISSED", font=f, fill=(255, 255, 255))

    # LEFT header
    pill(18, 12, "REAL  ·  " + info["seed_cond"], (30, 150, 60))
    shadow_text((18, 46), f"YOLO sees {n_all} vehicle{'s' if n_all != 1 else ''}", font("black", 32), (255, 255, 255))

    # RIGHT header
    pill(PW + 18, 12, "SYNTHETIC  ·  " + info["var_cond"], (200, 40, 40))
    big = font("black", 32)
    shadow_text((PW + 18, 46), f"YOLO sees {k} of {n}", big, (255, 255, 255))
    if n - k > 0:
        x = PW + 18 + d.textlength(f"YOLO sees {k} of {n}", font=big) + 16
        shadow_text((x, 56), f"{n - k} missed", font("black", 22), (255, 80, 80))
    if n_art:
        shadow_text((PW + 18 + d.textlength("SYNTHETIC  ·  " + info["var_cond"], font=font("black", 17)) + 34, 18),
                    f"+{n_art} gen artifact{'s' if n_art != 1 else ''} (not counted)", font("bold", 15), (190, 190, 190))
    rr = run_hit / run_tot if run_tot else 1.0
    shadow_text((2 * PW - 18, 8), f"{rr * 100:.0f}%", font("black", 50), recall_color(rr), anchor="ra")
    shadow_text((2 * PW - 18, 66), "RUNNING RECALL", font("bold", 14), (220, 220, 220), anchor="ra")

    # bottom: Cosmos Reason
    for x0, txt in ((0, info["seed_reason"]), (PW, info["var_reason"])):
        lab = "Cosmos Reason:"
        fl, ft = font("bold", 18), font("bold", 24)
        y = PH - BOT_H + 16
        d.text((x0 + 18, y + 4), lab, font=fl, fill=(150, 200, 255))
        tx = x0 + 18 + d.textlength(lab, font=fl) + 10
        txt = txt or "(no answer yet)"
        maxw = x0 + PW - 18 - tx
        while d.textlength(txt, font=ft) > maxw and len(txt) > 4:
            txt = txt[:-2].rstrip() + "…" if not txt.endswith("…") else txt[:-2] + "…"
        d.text((tx, y), txt, font=ft, fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---------- main ----------
def render(clip_id, manifest=SYN_MANIFEST, evals_dir=EVALS, out_dir=OUT, force=False, quiet=False):
    out_dir = Path(out_dir)
    mp4, jpg, meta_p = out_dir / f"{clip_id}.mp4", out_dir / f"{clip_id}.jpg", out_dir / f"{clip_id}.meta.json"
    row = manifest_row(clip_id, manifest)
    if not row:
        raise SystemExit(f"{clip_id} not in {manifest}")
    seed_id = row["seed_id"]
    var_src, s_src = resolve(row["src"]), seed_src(seed_id)
    ev = load_json(Path(evals_dir) / f"{clip_id}.json", {})
    sev = load_json(Path(evals_dir) / f"{seed_id}.json", None) or load_json(EVALS / f"{seed_id}.json", {})
    sig = json.dumps([RENDER_VERSION, (ev.get("reason") or {}).get("answers"), (sev.get("reason") or {}).get("answers"),
                      ev.get("integrity"), ev.get("fidelity"), (ev.get("yolo") or {}).get("recall_vs_seed")],
                     sort_keys=True, default=str)
    old = load_json(meta_p, None)
    if not force and old and old.get("sig") == sig and mp4.exists() and jpg.exists():
        return old | {"skipped": True}

    sboxes = load_boxes(seed_id, s_src, sev)
    vboxes = load_boxes(clip_id, var_src, ev)
    n = min(len(sboxes), len(vboxes))
    sframes = read_frames(s_src, n)
    vframes = read_frames(var_src, n)
    n = min(n, len(sframes), len(vframes))

    info = {"seed_cond": "CLEAR DAYLIGHT", "var_cond": cond_label(row.get("condition") or ev.get("condition")),
            "seed_reason": reason_text(sev), "var_reason": reason_text(ev)}

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    tmp = out_dir / f"{clip_id}{tag}.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * PW}x{PH}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    run_hit = run_tot = 0
    best = (-1, -1, 0, None)  # (misses, n, frame_idx, image)
    per_frame = []
    for i in range(n):
        sb, vb = sboxes[i], vboxes[i]
        m = match(sb, vb)
        ok = intact_flags(sframes[i], vframes[i], sb)
        n_ok = sum(1 for v in ok if v)
        hit = sum(1 for gi in m if ok[gi])
        run_hit += hit
        run_tot += n_ok
        img = compose(sframes[i], vframes[i], sb, vb, m, info, run_hit, run_tot, ok)
        ff.stdin.write(img.tobytes())
        miss = n_ok - hit
        per_frame.append([n_ok, hit, len(sb)])
        if (miss, n_ok) > best[:2]:
            best = (miss, n_ok, i, img)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {clip_id}")
    os.replace(tmp, mp4)
    if best[3] is not None:
        okj, buf = cv2.imencode(".jpg", best[3], [cv2.IMWRITE_JPEG_QUALITY, 92])
        tj = out_dir / f"{clip_id}{tag}.jpg"
        tj.write_bytes(buf.tobytes())
        os.replace(tj, jpg)

    fr = [h / t for t, h, _ in per_frame if t]
    integ = ev.get("integrity") or {}
    yolo, fid = ev.get("yolo") or {}, ev.get("fidelity") or {}
    meta = {
        "clip_id": clip_id, "seed_id": seed_id, "condition": row.get("condition") or ev.get("condition"),
        "label": info["var_cond"], "mp4": rel(mp4), "jpg": rel(jpg), "frames": n,
        "recall_vs_seed": yolo.get("recall_vs_seed", round(float(np.mean(fr)), 4) if fr else None),
        "pooled_recall": round(run_hit / run_tot, 4) if run_tot else None,
        "recall_intact": integ.get("recall_intact"), "vehicles_intact": integ.get("vehicles_intact"),
        "patch_score_median": integ.get("patch_score_median"),
        "render_recall_intact": round(float(np.mean(fr)), 4) if fr else None,
        "seed_mean_count": round(float(np.mean([a for _, _, a in per_frame])), 3) if per_frame else 0,
        "seed_mean_intact": round(float(np.mean([t for t, _, _ in per_frame])), 3) if per_frame else 0,
        "mean_detected": round(float(np.mean([h for _, h, _ in per_frame])), 3) if per_frame else 0,
        "poster_frame": best[2], "poster_missed": max(best[0], 0), "poster_seed_count": max(best[1], 0),
        "fidelity_pass": fid.get("pass"), "edge_ssim": fid.get("edge_ssim"),
        "reason_seed": info["seed_reason"], "reason_variant": info["var_reason"],
        "failure": ev.get("failure"), "failure_reasons": ev.get("failure_reasons"),
    }
    meta["sig"] = sig
    tm = out_dir / f"{clip_id}{tag}.json"
    tm.write_text(json.dumps(meta, indent=2))
    os.replace(tm, meta_p)
    if not quiet:
        print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_id")
    ap.add_argument("--manifest", type=Path, default=SYN_MANIFEST)
    ap.add_argument("--evals-dir", type=Path, default=EVALS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    render(a.clip_id, a.manifest, a.evals_dir, a.out_dir, a.force)


if __name__ == "__main__":
    main()
