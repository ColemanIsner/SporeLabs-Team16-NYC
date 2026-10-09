"""Narration TTS for the Spore demo video: Kokoro-82M (local, Apache-2.0), voice af_heart.

PYTHON = /private/tmp/claude-501/-Users-cisner-Documents-Coding-Work-hackathons-team16/27c0baac-74d4-4216-8b26-ac1a8306928f/scratchpad/tts312/bin/python

Setup (Python 3.12; Kokoro does not support 3.14 yet):
    /opt/homebrew/bin/python3.12 -m venv $SCRATCH/tts312
    $SCRATCH/tts312/bin/python -m pip install kokoro "misaki[en]" soundfile numpy
    # optional ASR check used during voice selection: pip install mlx-whisper
Weights (~330 MB) download once to the standard HF cache (hexgrad/Kokoro-82M).

We call misaki's G2P ourselves with no espeak fallback (the espeakng-loader
wheel crashes on this Mac) and run KModel directly; every word in script.md is
in misaki's lexicon. Brand names get explicit phonemes via PHONEMES below.

Usage:
    python video/tts.py "some text" out.wav        -> prints duration (s)
    python video/tts.py --batch jobs.json          -> jobs = [{"text":..., "out":...}, ...];
                                                      last stdout line = JSON list of durations
Env: TTS_VOICE (default af_heart; e.g. am_michael, am_fenrir, bm_george), TTS_SPEED (default 1.0).
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

VOICE = os.environ.get("TTS_VOICE", "af_heart")
SPEED = float(os.environ.get("TTS_SPEED", "1.0"))
VOICE_NAME = f"Kokoro-82M {VOICE}"
SR = 24000
PAD_S = 0.05  # silence kept at each end after trimming

# Spelling -> spoken form (applied before G2P; script keeps normal spellings).
TEXT_MAP = [
    (r"\bW&B\b", "W and B"),
    (r"\bSporeLabs\b", "Spore Labs"),
    (r"\bCoreWeave\b", "Core Weave"),
]
# Words forced to exact phonemes (misaki/Kokoro inline syntax [word](/phonemes/)).
PHONEMES = {
    "YOLO": "jˈOlO",           # yo-low
    "VSS": "vˌiˌɛsˈɛs",        # V-S-S
    "LLM": "ˌɛlˌɛlˈɛm",        # L-L-M
    "VAST": "vˈæst",           # rhymes with fast
    "VAST's": "vˈæsts",
    "NVIDIA": "ɛnvˈɪdiə",
    "Cosmos": "kˈɑzmOs",
}


def normalize(text: str) -> str:
    t = " ".join(text.split())
    for pat, rep in TEXT_MAP:
        t = re.sub(pat, rep, t)
    for word, ph in PHONEMES.items():
        t = re.sub(rf"(?<![\w\[]){re.escape(word)}(?![\w'])" if not word.endswith("'s")
                   else rf"(?<![\w\[]){re.escape(word)}(?!\w)", f"[{word}](/{ph}/)", t)
    return t


_model = _g2p = None
_voices: dict = {}


def _load():
    global _model, _g2p
    if _model is None:
        import torch
        from kokoro import KModel
        from misaki import en
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = KModel(repo_id="hexgrad/Kokoro-82M").to(dev).eval()
        _g2p = en.G2P(trf=False, british=VOICE.startswith("b"), fallback=None)
    return _model, _g2p


def _voice(name: str):
    if name not in _voices:
        import torch
        from huggingface_hub import hf_hub_download
        _voices[name] = torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{name}.pt"),
                                   weights_only=True)
    return _voices[name]


def phonemize(text: str) -> str:
    _, g2p = _load()
    ps, _ = g2p(normalize(text))
    return ps


def _trim(a: np.ndarray) -> np.ndarray:
    hop = int(0.01 * SR)
    n = len(a) // hop
    if n < 3:
        return a
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    loud = np.where(rms > rms.max() * 10 ** (-40 / 20))[0]
    if not len(loud):
        return a
    pad = int(PAD_S * SR)
    s = max(0, loud[0] * hop - pad)
    e = min(len(a), (loud[-1] + 1) * hop + pad)
    a = a[s:e].copy()
    f = int(0.005 * SR)  # 5 ms fades to avoid clicks
    a[:f] *= np.linspace(0, 1, f)
    a[-f:] *= np.linspace(1, 0, f)
    return a


def _render(ps: str) -> np.ndarray:
    import torch
    model, _ = _load()
    pack = _voice(VOICE)
    out = []
    # Kokoro context is 510 phonemes; split long input at sentence/clause punctuation.
    chunks, cur = [], ""
    for piece in re.split(r"(?<=[.!?;:,])\s+", ps):
        if cur and len(cur) + len(piece) + 1 > 400:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    for c in chunks:
        with torch.no_grad():
            out.append(model(c, pack[len(c) - 1], SPEED).cpu().numpy())
    return np.concatenate(out)


def synth(text: str, out_wav: str) -> float:
    """Synthesize text to a mono 24 kHz WAV; return duration in seconds."""
    a = _trim(_render(phonemize(text)))
    peak = np.abs(a).max()
    if peak > 0.95:
        a = a * (0.95 / peak)
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sf.write(out_wav, a.astype(np.float32), SR, subtype="PCM_16")
    return len(a) / SR


def main(argv):
    if len(argv) == 2 and argv[0] == "--batch":
        jobs = json.load(open(argv[1]))
        durs = [round(synth(j["text"], j["out"]), 3) for j in jobs]
        print(json.dumps(durs))
    elif len(argv) == 2:
        print(round(synth(argv[0], argv[1]), 3))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
