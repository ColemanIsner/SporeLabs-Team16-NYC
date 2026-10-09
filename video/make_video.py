"""Build the demo video: voiceover + burned-in captions from video/script.md over a recording of the Story UI.

Incremental: every sentence's audio is cached by its text, and every screen is recorded as its own clip,
cached by (step, length, UI source). Editing one line re-voices that sentence and re-records only its screen.

Needs the UI running (cd ui && npm run dev) plus:  pip install playwright && playwright install chromium
Voice: video/tts.py, run with the interpreter on its `PYTHON = ` docstring line (or $TTS_PYTHON).
Run:  python video/make_video.py [--fresh]   ->  video/spore_demo.mp4, spore_demo_nocaptions.mp4, spore_demo.srt
"""
import argparse, asyncio, base64, hashlib, json, os, re, shutil, subprocess
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent
BUILD = HERE / "build"
URL = "http://localhost:5173/"
# Record a smaller CSS viewport at 1.5x so every page fills the 1080p frame.
VW, VH, DPR = 1280, 720, 1.5
W, H, FPS = 1920, 1080, 30
LEAD, TAIL, GAP = 0.6, 1.0, 0.3  # seconds: screen before voice, after voice, between sentences
PARALLEL = 3  # screens recorded at once
# Video-only CSS: hide the nav buttons; their space at the bottom holds the captions.
# Also let the narrow content blocks use the full stage width.
VIDEO_CSS = (".st-nav{visibility:hidden;height:80px;padding:0!important}"
             ".st-bars,.st-live,.st-grid,.st-pick,.st-cands,.st-next-list,.st-fix{max-width:none!important}")


def sha(*parts) -> str:
    return hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()[:16]


def parse_script(path: Path) -> list[tuple[str, list[str]]]:
    steps, cur = [], None
    for line in path.read_text().splitlines():
        if m := re.match(r"^## (\w+)", line):
            cur = (m.group(1), [])
            steps.append(cur)
        elif cur is not None and line.strip():
            cur[1].append(line.strip())
    return [(k, [s for s in re.split(r"(?<=[.!?])\s+", " ".join(ls)) if s]) for k, ls in steps]


def duration(f: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def ffmpeg(*args, cwd=None):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True, cwd=cwd)


# ---- voice: one cached WAV per sentence ----

def tts_python() -> str:
    if os.environ.get("TTS_PYTHON"):
        return os.environ["TTS_PYTHON"]
    m = re.search(r"^PYTHON = (.+)$", (HERE / "tts.py").read_text(), re.M)
    return m.group(1).strip().strip("'\"")


def synth_all(steps) -> dict:
    """Returns {(step, i): (wav, seconds)}; only sentences whose text/voice/tts.py changed are synthesized."""
    vdir = BUILD / "voice"
    vdir.mkdir(parents=True, exist_ok=True)
    salt = sha((HERE / "tts.py").read_text(), os.environ.get("TTS_VOICE", ""), os.environ.get("TTS_SPEED", ""))
    want = {(k, i): vdir / f"{sha(salt, s)}.wav" for k, sents in steps for i, s in enumerate(sents)}
    text = {(k, i): s for k, sents in steps for i, s in enumerate(sents)}
    todo = [{"text": text[key], "out": str(f)} for key, f in want.items() if not f.exists()]
    if todo:
        (BUILD / "jobs.json").write_text(json.dumps(todo))
        subprocess.run([tts_python(), str(HERE / "tts.py"), "--batch", str(BUILD / "jobs.json")],
                       check=True, stdout=subprocess.DEVNULL)
    print(f"voice: {len(todo)} new, {len(want) - len(todo)} cached")
    return {key: (f, duration(f)) for key, f in want.items()}


# ---- screens: one cached silent clip per step ----

def ui_hash() -> str:
    files = sorted((ROOT / "ui" / "src").glob("*"))
    return sha(VIDEO_CSS, VW, VH, DPR, FPS, *(f.read_bytes() for f in files))


async def record_step(browser, steps, i, length, out: Path):
    """Record step i for `length` seconds, entering it with the UI's own transition from the step before."""
    key = steps[i][0]
    prev = steps[i - 1][0] if i else steps[-1][0]
    ctx = await browser.new_context(viewport={"width": VW, "height": VH}, device_scale_factor=DPR)
    await ctx.add_init_script(f"addEventListener('DOMContentLoaded',()=>{{const s=document.createElement('style');"
                              f"s.textContent={json.dumps(VIDEO_CSS)};document.head.appendChild(s)}})")
    pg = await ctx.new_page()
    await pg.goto(URL + "#" + prev)
    await pg.wait_for_timeout(1500)
    fdir = BUILD / "frames" / key
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    frames = []
    cdp = await ctx.new_cdp_session(pg)

    def on_frame(e):
        f = fdir / f"{len(frames):06d}.jpg"
        f.write_bytes(base64.b64decode(e["data"]))
        frames.append((e["metadata"]["timestamp"], f))
        asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": e["sessionId"]}))

    cdp.on("Page.screencastFrame", on_frame)
    await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H})
    await pg.wait_for_timeout(300)
    t0 = await pg.evaluate(f"(location.hash = {json.dumps('#' + key)}, Date.now() / 1000)")
    await pg.wait_for_timeout(length * 1000 + 300)
    await cdp.send("Page.stopScreencast")
    await ctx.close()
    # Frames before the switch show the previous step; start from the last of those (now the new step's first paint).
    keep = [(max(t - t0, 0), f) for t, f in frames if t >= t0 - 0.05] or frames[-1:]
    lst = fdir / "frames.txt"
    with lst.open("w") as fh:
        for k, (t, f) in enumerate(keep):
            nxt = keep[k + 1][0] if k + 1 < len(keep) else length
            if t >= length:
                break
            fh.write(f"file '{f.name}'\nduration {max(min(nxt, length) - t, 0.001):.4f}\n")
        fh.write(f"file '{keep[-1][1].name}'\n")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst.name, "-vf", f"fps={FPS},scale={W}:{H},format=yuv420p",
           "-t", f"{length:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-r", FPS, out.resolve(),
           cwd=fdir)
    shutil.rmtree(fdir)
    print(f"  recorded {key} ({length:.1f}s)")


async def record_all(steps, lengths, fresh) -> list[Path]:
    sdir = BUILD / "screens"
    sdir.mkdir(parents=True, exist_ok=True)
    uh = ui_hash()
    outs = [sdir / f"{k}_{sha(uh, k, steps[i - 1][0] if i else '', f'{lengths[i]:.2f}')}.mp4"
            for i, (k, _) in enumerate(steps)]
    todo = [i for i, f in enumerate(outs) if fresh or not f.exists()]
    print(f"screens: {len(todo)} to record, {len(outs) - len(todo)} cached")
    if todo:
        sem = asyncio.Semaphore(PARALLEL)
        async with async_playwright() as p:
            b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])

            async def one(i):
                async with sem:
                    await record_step(b, steps, i, lengths[i], outs[i])
            await asyncio.gather(*(one(i) for i in todo))
            await b.close()
    return outs


# ---- captions + final mux ----

def ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def caption_chunks(sentence: str, start: float, dur: float, max_words: int = 14):
    """Split long sentences into readable chunks (preferring a comma near the middle), timed by character share."""
    words = sentence.split()
    n = max(1, -(-len(words) // max_words))
    cuts, prev = [], 0
    for k in range(1, n):
        ideal = round(len(words) * k / n)
        near = [i for i in range(max(prev + 2, ideal - 4), min(len(words) - 1, ideal + 4))
                if words[i - 1][-1] in ",;:"]
        cut = min(near, key=lambda i: abs(i - ideal)) if near else ideal
        cuts.append(cut)
        prev = cut
    parts = [" ".join(words[a:b]) for a, b in zip([0, *cuts], [*cuts, len(words)])]
    total = sum(len(p) for p in parts)
    t = start
    for p in parts:
        d = dur * len(p) / total
        yield t, t + d, p
        t += d


def write_ass(cues, path: Path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Helvetica Neue,44,&H00FFFFFF,&H00FFFFFF,&H64000000,&H64000000,0,0,0,0,100,100,0,0,3,14,0,2,260,260,34,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(head + "".join(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{t}\n" for a, b, t in cues))


def assemble(steps, clips, lengths, screens, out: Path):
    # Screens back to back (same encoding, so no re-encode).
    (BUILD / "screens.txt").write_text("".join(f"file '{f.resolve()}'\n" for f in screens))
    silent = BUILD / "silent.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", BUILD / "screens.txt", "-c", "copy", silent)
    # Each sentence at its slot; captions use the same times.
    placed, cues, start = [], [], 0.0
    for i, (key, sents) in enumerate(steps):
        at = start + LEAD
        for j, s in enumerate(sents):
            f, d = clips[(key, j)]
            placed.append((at, f))
            cues += list(caption_chunks(s, at, d))
            at += d + GAP
        start += lengths[i]
    write_ass(cues, BUILD / "captions.ass")
    ins, filt = [], []
    for k, (at, f) in enumerate(placed):
        ins += ["-i", f.resolve()]
        ms = int(at * 1000)
        filt.append(f"[{k + 1}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
    filt.append("".join(f"[a{k}]" for k in range(len(placed))) +
                f"amix=inputs={len(placed)}:normalize=0,loudnorm=I=-16:TP=-1.5,apad[aout]")
    voice = BUILD / "voice.m4a"
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", *ins, "-filter_complex", ";".join(filt),
           "-map", "[aout]", "-t", f"{start:.3f}", "-c:a", "aac", "-b:a", "192k", voice.resolve(), cwd=BUILD)
    plain = out.with_name(out.stem + "_nocaptions" + out.suffix)
    ffmpeg("-i", silent, "-i", voice, "-map", "0:v", "-map", "1:a", "-c", "copy", "-shortest", plain)
    # Only the captioned cut re-encodes (ass runs in BUILD: the repo path has spaces).
    ffmpeg("-i", silent.resolve(), "-i", voice.resolve(), "-vf", "ass=captions.ass", "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "copy", "-shortest", out.resolve(), cwd=BUILD)
    out.with_suffix(".srt").write_text("\n".join(
        f"{k + 1}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n" for k, (a, b, t) in enumerate(cues)))
    (BUILD / "timeline.json").write_text(json.dumps([{"t": round(a, 2), "text": t} for a, _, t in cues], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "spore_demo.mp4"))
    ap.add_argument("--script", default=str(HERE / "script.md"))
    ap.add_argument("--fresh", action="store_true", help="re-record every screen (e.g. after results/ data changed)")
    a = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    steps = parse_script(Path(a.script))
    clips = synth_all(steps)
    lengths = [LEAD + sum(clips[(k, j)][1] for j in range(len(s))) + GAP * (len(s) - 1) + TAIL for k, s in steps]
    screens = asyncio.run(record_all(steps, lengths, a.fresh))
    assemble(steps, clips, lengths, screens, Path(a.out))
    print(f"wrote {a.out} ({duration(Path(a.out)):.1f}s) + _nocaptions.mp4 + .srt")


if __name__ == "__main__":
    main()
