"""Weave tracing for the Spore loop. No-ops when W&B is not configured."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_enabled = False


def _identity(fn=None, **_kwargs):
    """Return the function unchanged. Supports @traced and @traced()."""
    if fn is None:
        return lambda f: f
    return fn


traced = _identity


def enabled() -> bool:
    return _enabled


def _api_key() -> str:
    return os.environ.get("WANDB_API_KEY", "").strip()


def init(project: str = "sporelabs-hackathon") -> bool:
    """Call weave.init when WANDB_API_KEY is set. Otherwise leave tracing off.

    Returns True only when Weave is active. Never raises if the key is missing,
    weave cannot be imported, or weave.init fails.
    """
    global _enabled, traced
    if not _api_key():
        _enabled = False
        traced = _identity
        return False
    try:
        import weave

        weave.init(project)
        traced = weave.op
    except Exception:
        _enabled = False
        traced = _identity
        return False
    _enabled = True
    return True
