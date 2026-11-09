"""Vision-model judgments on candidate moments, from storyboard frames (no video download).

judge(vid, t, spec) -> {"match": bool, "confidence": float, "what": str}
Results are cached in data/judge_cache.json keyed by moment/video/time, so re-runs cost nothing.
"""
import json
import os
import subprocess
import threading

from . import store, verify, vlm

CACHE = os.path.join(store.DATA, "judge_cache.json")
SB_LAG_S = 7.0          # storyboard frame labelled t shows roughly t+7 s (measured on Hyrule Field truth)
_lock = threading.Lock()
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

PROMPT = """You are checking frames from a video. Context: {game}.
Target event: "{description}".
The image is a strip of {n} frames from the video, left to right, about 10 seconds apart; the MIDDLE frame is the candidate moment.
Is the middle frame part of the target event's scene, at or right around the moment it happens? Any shot of that scene
counts, including a close-up of a participant or an on-screen name card introducing them.
Answer false for: title screens, pause/quest/inventory menus, maps, loading screens, stream overlays such as
"starting soon", or moments where the event is only mentioned in dialogue or has not happened yet.
JSON keys: {{"match": true|false, "confidence": 0.0-1.0, "what": "what the middle frame actually shows, briefly"}}"""


def _strip(vid, t, out, offsets=(-10, 0, 10)):
    tiles = []
    for j, off in enumerate(offsets):
        p = f"{out}_{j}.png"
        if not verify._tile(vid, t + off - SB_LAG_S, p):
            return None
        tiles.append(p)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", p] for p in tiles], []) +
                   ["-filter_complex", f"hstack=inputs={len(tiles)}", out + ".png"])
    return out + ".png" if os.path.exists(out + ".png") else None


def judge(vid, t, spec):
    key = f"{spec.id}|{vid}|{int(round(t))}"
    with _lock:
        if key in _cache:
            return _cache[key]
    d = os.path.join(store.DATA, "judge")
    os.makedirs(d, exist_ok=True)
    img = _strip(vid, t, os.path.join(d, key.replace("|", "_")))
    if not img:
        return {"match": None, "confidence": 0.0, "what": "no storyboard frames"}
    try:
        r = vlm.ask_json(PROMPT.format(game=spec.game or "a video", description=spec.description, n=3), [img]) or {}
    except Exception as e:
        return {"match": None, "confidence": 0.0, "what": f"error: {e}"}
    res = {"match": bool(r.get("match")) if "match" in r else None,
           "confidence": float(r.get("confidence") or 0), "what": str(r.get("what", ""))[:200]}
    with _lock:
        _cache[key] = res
        json.dump(_cache, open(CACHE + ".tmp", "w"), indent=0)
        os.replace(CACHE + ".tmp", CACHE)
    return res

