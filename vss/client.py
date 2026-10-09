"""Minimal VSS (VAST Video Search & Summarization) client.

Credentials come from repo-root .env (copied from /config/team-16.config on the lab VM).
Uses curl under the hood: the public host sits behind Cloudflare, which rejects
Python's default user agent.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://team-16-vss.thecosmoslabs.com"
_TOKEN_CACHE = ROOT / ".vss_token"


def env() -> dict:
    out = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _curl(*args: str, timeout: int = 120) -> str:
    r = subprocess.run(["curl", "-s", "-m", str(timeout), *args], capture_output=True, text=True)
    return r.stdout


def token(refresh: bool = False) -> str:
    if not refresh and _TOKEN_CACHE.exists() and time.time() - _TOKEN_CACHE.stat().st_mtime < 3000:
        return _TOKEN_CACHE.read_text()
    e = env()
    res = _curl("-X", "POST", f"{BASE}/api/v1/auth/login", "-H", "Content-Type: application/json",
                "-d", json.dumps({"username": e["USERNAME"], "password": e["PASSWORD"]}))
    tok = json.loads(res)["access_token"]
    _TOKEN_CACHE.write_text(tok)
    _TOKEN_CACHE.chmod(0o600)
    return tok


def _get(path: str):
    return json.loads(_curl(f"{BASE}{path}", "-H", f"Authorization: Bearer {token()}"))


def _post(path: str, body: dict):
    return json.loads(_curl("-X", "POST", f"{BASE}{path}", "-H", f"Authorization: Bearer {token()}",
                            "-H", "Content-Type: application/json", "-d", json.dumps(body)))


def explore_all() -> list[dict]:
    out, off = [], 0
    while True:
        ch = _get(f"/api/v1/videos/explore?scope=all&limit=48&offset={off}").get("chunks", [])
        out += ch
        if len(ch) < 48:
            return out
        off += 48


def search(query: str, k: int = 10, metadata_filters: dict | None = None, **extra) -> dict:
    body = {"query": query, "top_k": k, "metadata_filters": metadata_filters or {}, **extra}
    return _post("/api/v1/search", body)


def download(source: str, out_path: str | Path) -> Path:
    """Download an s3:// chunk or segment through the backend's range-capable stream proxy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/api/v1/videos/stream?source={quote(source, safe='')}&token={token()}"
    subprocess.run(["curl", "-s", "-m", "300", "-o", str(out_path), url], check=True)
    return out_path


def upload(path: str | Path, fields: dict) -> dict:
    args = ["-X", "POST", f"{BASE}/api/v1/videos/upload", "-H", f"Authorization: Bearer {token()}",
            "-F", f"file=@{path}"]
    for k, v in fields.items():
        args += ["-F", f"{k}={v}"]
    return json.loads(_curl(*args, timeout=300))
