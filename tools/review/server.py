"""Footage review: browse every clip in data/, compare to its seed, rate + reply.

Run:  python3 tools/review/server.py   ->  http://localhost:8765
Replies are written to results/reviews.json (agents read this).
"""
import json
import mimetypes
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REVIEWS = ROOT / "results" / "reviews.json"
PAGE = Path(__file__).with_name("index.html")
LOCK = threading.Lock()


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def manifests():
    by_src = {}
    for m in DATA.rglob("*.json"):
        rows = load_json(m, [])
        if isinstance(rows, dict):
            rows = rows.get("clips", [])
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("src"):
                by_src[str(Path(r["src"]))] = r
    return by_src


def clips():
    meta = manifests()
    reviews = load_json(REVIEWS, {})
    out = []
    for p in sorted(DATA.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = str(p.relative_to(ROOT))
        m = meta.get(rel, {})
        clip_id = m.get("clip_id") or m.get("seed_id") or p.stem
        seed_id = m.get("seed_id") or (p.stem.split("__")[0] if "__" in p.stem else None)
        seed_src = None
        if seed_id and p.parent.name != "seeds":
            for cand in (p.parent.parent / "seeds" / f"{seed_id}.mp4", DATA / "seeds" / f"{seed_id}.mp4"):
                if cand.exists():
                    seed_src = str(cand.relative_to(ROOT))
                    break
        ev = load_json(ROOT / "results" / "evals" / f"{clip_id}.json", None)
        out.append({
            "id": rel, "src": rel, "group": p.parent.relative_to(DATA).as_posix(),
            "clip_id": clip_id, "seed_src": seed_src if seed_src != rel else None,
            "meta": m, "eval": ev, "review": reviews.get(rel),
            "mtime": p.stat().st_mtime,
        })
    return out


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            b = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)
        elif path == "/api/clips":
            self._json(clips())
        elif path.startswith("/file/"):
            f = (ROOT / path[len("/file/"):]).resolve()
            if ROOT not in f.parents or not f.is_file():
                return self._json({"error": "not found"}, 404)
            self._send_file(f)
        else:
            self._json({"error": "not found"}, 404)

    def _send_file(self, f):
        size = f.stat().st_size
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = int(m.group(2))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            REVIEWS.parent.mkdir(parents=True, exist_ok=True)
            reviews = load_json(REVIEWS, {})
            reviews[body["id"]] = {"rating": body.get("rating"), "note": body.get("note", ""),
                                   "clip_id": body.get("clip_id")}
            REVIEWS.write_text(json.dumps(reviews, indent=2))
        self._json({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Review UI: http://localhost:{port}  (replies -> {REVIEWS.relative_to(ROOT)})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
