"""Ask the archive: one live VSS search, judged by whether each hit's own VSS caption shows the query.

  python3 vss/ask.py "construction zone at night"      -> JSON on stdout

Used by the demo UI's live search box (ui/vite.config.ts /api/ask).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402
from real_check import kw_hits  # noqa: E402  (negation-aware keyword matcher)

STOP = {"with", "that", "this", "from", "into", "onto", "over", "under", "some", "there", "where", "when",
        "while", "near", "very", "show", "shows", "clip", "clips", "video", "footage", "scene", "the", "and"}


COND = {"night", "nighttime", "dark", "dusk", "dawn", "sunset", "twilight", "rain", "rainy", "raining", "wet", "fog",
        "foggy", "mist", "misty", "haze", "hazy", "snow", "snowy", "snowing", "ice", "icy", "glare", "sun", "storm"}


def words(q: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 2 and w not in STOP]


def terms(q: str) -> list[str]:
    words_ = words(q)
    # Prefix match so "foggy" hits "fog", "snowing" hits "snow", "construction" hits "constructions".
    return [pat(w) for w in words_]


def pat(w: str) -> str:
    return rf"\b{re.escape(w[:5] if len(w) > 5 else w)}\w*"


def ask(query: str, k: int = 10) -> dict:
    t0 = time.time()
    res = client.search(query, k=k).get("results", [])
    ws = words(query)
    pats = [pat(w) for w in ws]
    must = [pat(w) for w in ws if w in COND]  # weather/time words must all appear
    need = max(1, len(pats) - 1) if len(pats) >= 3 else len(pats)
    hits = []
    for r in res:
        cap = r.get("reasoning_content") or ""
        found = sorted(set(kw_hits(cap, pats)))
        ok_must = all(kw_hits(cap, [m]) for m in must)
        n_terms = sum(1 for p_ in pats if kw_hits(cap, [p_]))
        hits.append({
            "camera_id": r.get("camera_id"),
            "score": round(float(r.get("similarity_score") or r.get("score") or 0), 3),
            "highway": r.get("camera_id") == "i24_cam-1" or bool(re.search(r"\bhighway|interstate|freeway", cap, re.I)),
            "object_counts": r.get("object_counts"),
            "start": r.get("segment_start_sec"),
            "caption": cap[:280],
            "source": r.get("source"),
            "matched": found,
            "shows_it": ok_must and n_terms >= need,
        })
    n = sum(h["shows_it"] for h in hits)
    return {"query": query, "endpoint": "POST /api/v1/search", "seconds": round(time.time() - t0, 1), "top_k": k, "n_shows_it": n,
            "terms": [p.split("\\b")[-1].replace("\\w*", "") for p in pats], "hits": hits}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        sys.exit("usage: ask.py <query>")
    print(json.dumps(ask(q)))
