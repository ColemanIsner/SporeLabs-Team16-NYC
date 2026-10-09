#!/usr/bin/env python3
"""Physically-motivated, geometry-preserving weather layer with a SEVERITY knob (0..1).

Every effect is a per-pixel / additive overlay: no warps, so vehicles stay at exactly the
same pixel positions and the seed's labels remain exact ground truth.

  fog   : depth-aware Koschmieder haze I = J*t + A*(1-t), t = exp(-beta*d), drifting patches
  rain  : multi-layer motion-blurred streaks, wet-road darkening + sheen, overcast grade, lens drops
  snow  : parallax flakes (3 depth layers, swaying), accumulation on low-texture/low-sat areas
  night : gamma + blue shift + sensor grain, headlight/taillight blooms on bright blobs
  glare : low sun bloom + veiling glare + anamorphic streak + flare ghosts

API:  apply(frames_bgr, kind, severity, seed) -> frames_bgr
CLI:  gen/weather.py --in X.mp4 --kind fog --severity 0.8 --out Y.mp4
          [--manifest data/synthetic/manifest.json --seed-id S --base-condition '{...}']
      gen/weather.py --sweep   # test sweep + contact sheets into data/sweep_weather/
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
W, H, FPS, NFRAMES = 1280, 720, 16, 93
KINDS = ["fog", "rain", "snow", "night", "glare"]


# ---------------------------------------------------------------- helpers
def _depth(h, w, frame=None):
    """Cheap depth proxy in [0,1]: elevated traffic cam -> higher in frame = farther."""
    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]  # 1 at top
    d = 0.08 + 0.92 * y ** 1.35
    d = np.repeat(d, w, axis=1)
    if frame is not None:  # bright, smooth regions near the top (sky/haze) read farther
        lum = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 25).astype(np.float32) / 255
        d = np.clip(d + 0.15 * (lum - lum.mean()) * y, 0, 1)
    return d


def _noise_field(rng, h, w, scale, octaves=3):
    """Smooth value noise in [0,1] (sum of upsampled random grids)."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(scale * 2 ** o))
        g = rng.random((max(2, h // (w // s)), s)).astype(np.float32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    acc /= tot
    return np.clip((acc - acc.min()) / (np.ptp(acc) + 1e-6), 0, 1)


def _desat(img, k):
    g = img.mean(axis=2, keepdims=True)
    return img * (1 - k) + g * k


def _to_u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- fog
def fog(frames, s, rng):
    h, w = frames[0].shape[:2]
    d = _depth(h, w, frames[0])
    beta = 0.4 + 6.5 * s ** 1.1  # s=1: top t~exp(-6.9)=0.001 (gone), bottom t~0.6
    # drifting patchy density (wind), larger than frame so we can slide it
    big = _noise_field(rng, h, w + 400, 6)
    A = np.array([205, 203, 198], np.float32)  # BGR light grey, slightly cool
    out = []
    for i, f in enumerate(frames):
        off = int(i * 2.5) % 400
        patch = big[:, off:off + w]
        dd = d * (0.75 + 0.5 * patch)
        t = np.exp(-beta * dd)[..., None]
        J = f.astype(np.float32)
        sig = 0.5 + 3.0 * s
        Jb = cv2.GaussianBlur(J, (0, 0), sig)  # forward scatter: far = blurrier
        J = Jb * (1 - t) + J * t
        J = _desat(J, 0.5 * s)
        I = J * t + A * (1 - t)
        I = (I - 128) * (1 - 0.25 * s) + 128 + 10 * s  # contrast loss
        out.append(_to_u8(I))
    return out


# ---------------------------------------------------------------- particles
class _Particles:
    def __init__(self, rng, n, w, h, speed, angle_deg, length, thick, alpha, sway=0.0):
        self.rng, self.w, self.h = rng, w, h
        self.x = rng.uniform(-100, w + 100, n).astype(np.float32)
        self.y = rng.uniform(-100, h + 100, n).astype(np.float32)
        a = np.deg2rad(angle_deg)
        sp = speed * rng.uniform(0.8, 1.2, n)
        self.vx, self.vy = (np.sin(a) * sp).astype(np.float32), (np.cos(a) * sp).astype(np.float32)
        self.len = (length * rng.uniform(0.6, 1.4, n)).astype(np.float32)
        self.thick, self.alpha, self.sway = thick, alpha, sway
        self.a = rng.uniform(0.4, 1.0, n).astype(np.float32)
        self.phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)

    def step(self, i):
        self.x += self.vx
        self.y += self.vy
        if self.sway:
            self.x += self.sway * np.sin(0.25 * i + self.phase)
        wrap = self.y > self.h + 100
        self.y[wrap] -= self.h + 200
        self.x = (self.x + 100) % (self.w + 200) - 100


def _draw_streaks(layer, p, scale=1):
    nx, ny = p.vx / (np.hypot(p.vx, p.vy) + 1e-6), p.vy / (np.hypot(p.vx, p.vy) + 1e-6)
    x0, y0 = p.x, p.y
    x1, y1 = p.x - nx * p.len, p.y - ny * p.len
    pts = np.stack([x0, y0, x1, y1], 1) * scale
    vals = p.a * p.alpha
    for (a, b, c, d), v in zip(pts.astype(np.int32), vals):
        cv2.line(layer, (a, b), (c, d), float(v), p.thick, cv2.LINE_AA)


def _draw_flakes(layer, p, r):
    for x, y, a, rr in zip(p.x.astype(np.int32), p.y.astype(np.int32), p.a * p.alpha, p.len):
        cv2.circle(layer, (int(x), int(y)), max(1, int(rr * r)), float(a), -1, cv2.LINE_AA)


# ---------------------------------------------------------------- rain
def rain(frames, s, rng):
    h, w = frames[0].shape[:2]
    d = _depth(h, w)
    ang = rng.uniform(8, 18)
    layers = [  # far (fine, dim, many) -> near (long, bright, few)
        _Particles(rng, int(2600 * s), w, h, 38, ang, 22, 1, 0.22),
        _Particles(rng, int(1100 * s), w, h, 60, ang, 45, 1, 0.38),
        _Particles(rng, int(260 * s), w, h, 95, ang, 90, 2, 0.45),
    ]
    # lens droplets: fixed-ish, slowly sliding, refract (blur) what's behind
    nd = int(round(10 * s ** 1.5))
    drops = [[rng.uniform(0, w), rng.uniform(0, h), rng.uniform(18, 55), rng.uniform(0.2, 1.2)] for _ in range(nd)]
    road = (1 - d)[..., None]  # nearer = more visible wet road
    grey = np.array([150, 148, 145], np.float32)
    out = []
    for i, f in enumerate(frames):
        J = f.astype(np.float32)
        # overcast grade: darker, desaturated, cooler, lifted blacks, slight haze w/ depth
        J = _desat(J, 0.45 * s)
        J = J * (1 - 0.30 * s) + 10 * s
        J[..., 0] *= 1 + 0.06 * s
        t = np.exp(-(1.8 * s) * d)[..., None]
        J = J * t + grey * (1 - t)
        # wet road: darken mid-tones, add specular sheen = vertically smeared highlights
        lum = J.mean(axis=2, keepdims=True)
        wet = 0.35 * s * road * (lum < 200)
        J = J * (1 - wet)
        hi = np.clip(f.astype(np.float32) - 170, 0, None)
        sheen = cv2.GaussianBlur(hi, (0, 0), sigmaX=2, sigmaY=14)
        J += 0.9 * s * sheen * road
        J = cv2.GaussianBlur(J, (0, 0), 0.4 + 1.0 * s)
        # streaks (rendered at full res, then motion-blurred a touch)
        streak = np.zeros((h, w), np.float32)
        for L in layers:
            L.step(i)
            _draw_streaks(streak, L)
        streak = cv2.GaussianBlur(streak, (0, 0), 0.8)
        J = J + (215 - J) * np.clip(streak, 0, 0.85)[..., None]
        # lens droplets
        if drops:
            blur = cv2.GaussianBlur(J, (0, 0), 9)
            flip = cv2.flip(blur, 0)
            m = np.zeros((h, w), np.float32)
            for dr in drops:
                dr[1] += dr[3]
                if dr[1] > h + 60:
                    dr[1] = -60
                cv2.ellipse(m, (int(dr[0]), int(dr[1])), (int(dr[2]), int(dr[2] * 1.15)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
            J = J * (1 - m) + (0.6 * blur + 0.4 * flip + 12) * m
        out.append(_to_u8(J))
    return out


# ---------------------------------------------------------------- snow
def snow(frames, s, rng):
    h, w = frames[0].shape[:2]
    d = _depth(h, w)
    layers = [
        _Particles(rng, int(3500 * s), w, h, 2.2, 6, 1.2, 1, 0.7, sway=0.6),
        _Particles(rng, int(1400 * s), w, h, 4.5, 8, 2.8, 1, 0.9, sway=1.2),
        _Particles(rng, int(260 * s), w, h, 9.0, 10, 6.5, 1, 1.0, sway=2.0),
    ]
    sigmas = [0.5, 0.9, 2.0]  # near flakes defocused
    grain = _noise_field(rng, h, w, 80, octaves=2)
    white = np.array([238, 236, 232], np.float32)
    sky = np.array([212, 208, 204], np.float32)
    out = []
    for i, f in enumerate(frames):
        J = f.astype(np.float32)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255
        tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (0, 0), 3)
        # accumulation: low texture & (low sat road OR any vegetation-ish flat area), patchy
        smooth = np.clip(1 - tex / 18, 0, 1)
        acc = smooth * np.clip(1.15 - sat * 1.3, 0, 1) * (0.55 + 0.6 * grain)
        acc = np.clip(acc * (0.95 * s ** 0.8), 0, 0.92)[..., None]
        J = _desat(J, 0.5 * s)
        J = J * (1 - acc) + white * acc
        # snowfall haze + contrast loss + cool tint
        t = np.exp(-(2.6 * s) * d)[..., None]
        J = J * t + sky * (1 - t)
        J = (J - 128) * (1 - 0.22 * s) + 128 + 8 * s
        J[..., 0] += 6 * s
        J[..., 2] -= 4 * s
        flakes = np.zeros((h, w), np.float32)
        for L, sg in zip(layers, sigmas):
            L.step(i)
            lay = np.zeros((h, w), np.float32)
            _draw_flakes(lay, L, 1.0)
            flakes += cv2.GaussianBlur(lay, (0, 0), sg)
        a = np.clip(flakes * 1.3, 0, 0.97)[..., None]
        J = J * (1 - a) + 250 * a
        out.append(_to_u8(J))
    return out


# ---------------------------------------------------------------- night
def night(frames, s, rng):
    h, w = frames[0].shape[:2]
    d = _depth(h, w)
    k = np.ones((7, 7), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vign = 1 - 0.45 * s * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    out = []
    for i, f in enumerate(frames):
        J = f.astype(np.float32) / 255
        gamma = 1 + 2.2 * s
        J = J ** gamma * (1 - 0.55 * s)
        J = _desat(J, 0.55 * s)
        J[..., 0] = J[..., 0] * (1 + 0.55 * s) + 0.015 * s  # blue shift (scotopic)
        J[..., 1] *= 1 + 0.12 * s
        J[..., 2] *= 1 - 0.25 * s
        J *= vign[..., None]
        J *= (1 - 0.35 * s * d)[..., None]  # far away is darker (no street lights)
        # headlight / taillight blooms: compact bright blobs (vehicles), not thin lane paint
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        m8 = cv2.morphologyEx((g > 195).astype(np.uint8), cv2.MORPH_OPEN, k)
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        keep = (st[:, cv2.CC_STAT_AREA] >= 25) & (st[:, cv2.CC_STAT_AREA] <= 1800) & \
               (st[:, cv2.CC_STAT_WIDTH] <= 70) & (st[:, cv2.CC_STAT_HEIGHT] <= 70)
        keep[0] = False
        m = keep[lab].astype(np.float32)
        core = cv2.GaussianBlur(m, (0, 0), 3)
        halo = cv2.GaussianBlur(m, (0, 0), 14)
        wide = cv2.GaussianBlur(m, (0, 0), 40)
        warm = np.array([0.75, 0.9, 1.0], np.float32)
        bloom = (0.9 * core + 1.0 * halo + 1.2 * wide)[..., None] * warm * s
        J = J + bloom * 0.45
        # sensor grain: luminance + chroma, stronger in darks
        n = rng.normal(0, 1, (h // 2, w // 2, 3)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
        J = J + n * (0.012 + 0.05 * s) * (1.2 - np.clip(J.mean(2, keepdims=True), 0, 1))
        out.append(_to_u8(J * 255))
    return out


# ---------------------------------------------------------------- glare
def glare(frames, s, rng):
    h, w = frames[0].shape[:2]
    sx, sy = w * rng.uniform(0.62, 0.82), h * rng.uniform(-0.05, 0.12)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - sx, yy - sy)
    sun = np.exp(-(r / (0.10 * w)) ** 2) * 1.6 + np.exp(-r / (0.35 * w)) * 0.9
    streak = np.exp(-((yy - sy) / 5) ** 2) * np.exp(-np.abs(xx - sx) / (0.45 * w))
    streak += 0.5 * np.exp(-((yy - sy) / 18) ** 2) * np.exp(-np.abs(xx - sx) / (0.3 * w))
    # ghosts on the line through image centre
    cx, cy = w / 2, h / 2
    ghosts = np.zeros((h, w), np.float32)
    gcol = np.zeros((h, w, 3), np.float32)
    for t, rad, a, col in [(0.4, 30, .25, (1, .8, .5)), (0.75, 60, .15, (.6, 1, .6)), (1.3, 22, .3, (1, .6, .9)),
                           (1.7, 95, .12, (.5, .8, 1)), (2.0, 40, .2, (1, .9, .6))]:
        gx, gy = sx + (cx - sx) * t, sy + (cy - sy) * t
        disc = np.clip(1 - np.hypot(xx - gx, yy - gy) / rad, 0, 1) ** 0.6 * a
        gcol += disc[..., None] * np.array(col, np.float32)
    warm = np.array([0.80, 0.93, 1.0], np.float32)
    L = (sun[..., None] * warm + streak[..., None] * np.array([1.0, 0.95, 0.9], np.float32) + gcol) * 255
    out = []
    for i, f in enumerate(frames):
        flick = 1 + 0.04 * np.sin(i * 0.7) + 0.02 * rng.standard_normal()
        J = f.astype(np.float32)
        J = (J - 128) * (1 - 0.35 * s) + 128 + 30 * s  # veiling glare
        J[..., 2] *= 1 + 0.06 * s
        J[..., 0] *= 1 - 0.08 * s
        hi = np.clip(J - 200, 0, None)
        J += cv2.GaussianBlur(hi, (0, 0), 12) * 1.5 * s  # bloom around specular cars
        J = J + L * (s * 1.1) * flick
        out.append(_to_u8(J))
    return out


FX = {"fog": fog, "rain": rain, "snow": snow, "night": night, "glare": glare}


def apply(frames_bgr, kind, severity, seed=0):
    if kind not in FX:
        raise ValueError(f"kind must be one of {KINDS}")
    s = float(np.clip(severity, 0, 1))
    if s <= 0:
        return [f.copy() for f in frames_bgr]
    return FX[kind](frames_bgr, s, np.random.default_rng(seed))


# ---------------------------------------------------------------- io
def read_frames(path, n=NFRAMES):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
        fr.append(f)
    cap.release()
    if not fr:
        raise RuntimeError(f"no frames in {path}")
    while len(fr) < n:  # pad by holding last frame
        fr.append(fr[-1].copy())
    return fr


def write_frames(frames, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-frames:v", str(NFRAMES),
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                          str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    tmp.replace(out)


def condition_for(kind, severity, base_cond=None, base="seed"):
    bc = dict(base_cond or {})
    weather = {"fog": "fog", "rain": "rain", "snow": "snow"}.get(kind, bc.get("weather", "clear"))
    tod = "night" if kind == "night" else bc.get("time", "day")
    cond = {"weather": weather, "time": tod, "intensity": "heavy" if severity >= 0.6 else "light",
            "kind": "hybrid" if base == "cosmos" else "physics", "severity": round(float(severity), 3),
            "base": base, "physics": kind}
    if kind == "glare":
        cond["glare"] = True
    return cond


def run(inp, kind, severity, out, seed=0, manifest=None, seed_id=None, base_cond=None, base=None):
    t0 = time.time()
    frames = read_frames(inp)
    res = apply(frames, kind, severity, seed)
    write_frames(res, out)
    secs = round(time.time() - t0, 2)
    entry = None
    if manifest:
        from degrade import rel, seed_id_for, upsert_manifest
        base = base or ("cosmos" if base_cond else "seed")
        sid = seed_id or seed_id_for(Path(inp))
        entry = {"clip_id": Path(out).stem, "seed_id": sid, "src": rel(Path(out)),
                 "condition": condition_for(kind, severity, base_cond, base),
                 "prompt": None, "generator": f"weather.py/{kind}@{severity:.2f}", "gpu": None,
                 "gen_seconds": secs, "input": rel(Path(inp))}
        upsert_manifest(Path(manifest), entry)
    return secs, entry


def sweep(kinds=None):
    od = ROOT / "data" / "sweep_weather"
    od.mkdir(parents=True, exist_ok=True)
    seeds = [ROOT / "data/seeds/i24_scene1_p1c2_04.mp4", ROOT / "data/seeds/i24_scene1_p1c1_04.mp4"]
    sevs = [0.4, 0.7, 1.0]
    ti = 3 * FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    for kind in kinds or KINDS:
        rows = []
        for sp in seeds:
            fr = read_frames(sp)
            tiles = [(fr[ti], "seed")]
            for sv in sevs:
                out = od / f"{sp.stem}__{kind}_s{int(sv * 100):03d}.mp4"
                t0 = time.time()
                res = apply(fr, kind, sv, seed=1)
                write_frames(res, out)
                print(f"{out.name}: {time.time() - t0:.1f}s", flush=True)
                tiles.append((res[ti], f"{kind} {sv:.1f}"))
            row = []
            for im, lab in tiles:
                t = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
                cv2.putText(t, lab, (10, 28), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(t, lab, (10, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                row.append(t)
            rows.append(np.hstack(row))
        cv2.imwrite(str(od / f"sheet_{kind}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path)
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--severity", type=float, default=0.7)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for particles/noise")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--seed-id")
    ap.add_argument("--base-condition", help="JSON condition of the input (e.g. a Cosmos night variant)")
    ap.add_argument("--base", choices=["seed", "cosmos"], help="default: cosmos if --base-condition given")
    ap.add_argument("--sweep", nargs="?", const="all", help="run the test sweep into data/sweep_weather/ (optionally kinds csv)")
    a = ap.parse_args()
    if a.sweep:
        return sweep(None if a.sweep == "all" else a.sweep.split(","))
    if not (a.inp and a.kind and a.out):
        ap.error("--in, --kind, --out required")
    bc = json.loads(a.base_condition) if a.base_condition else None
    secs, entry = run(a.inp, a.kind, a.severity, a.out, a.seed, a.manifest, a.seed_id, bc, a.base)
    print(json.dumps({"out": str(a.out), "seconds": secs, "clip_id": entry and entry["clip_id"]}))


if __name__ == "__main__":
    main()
