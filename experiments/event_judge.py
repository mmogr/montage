"""EXPERIMENT (not in the main path; findings 2026-11-09): event-specific 8-frame grid judge for text-free cold
start. Dropped: it cut good seeds (HF 29 -> 13) and still accepted look-alike scenes (Deku, Gohma)."""
import json
import os
import subprocess

from montage import store, verify, vlm
from montage.judge import CACHE, SB_LAG_S, _cache, _lock


# ---- event-specific judge over a longer strip (for seeding without on-screen text) ----
# The scene-inclusive judge above accepts any shot of the right place, which is what seed filtering wants but not
# what cold start needs: a place/character often appears in several scenes (Deku Tree: first meeting, briefing, death).

EVENT_OFFSETS = (-40, -20, 0, 20, 40, 60, 80, 100)
EVENT_PROMPT = """You are checking frames from a video. Context: {game}.
Target event: "{description}".
The image is a grid of {n} frames, about 20 seconds apart, read left to right then top to bottom (frame 1 is top-left,
frame {n} is bottom-right). Together they cover about {span} seconds of the video.
Does this stretch of video show THE TARGET EVENT ITSELF? Be strict: the same place, character or creature appearing
in a different scene (an earlier conversation, a later revisit, a different phase, a menu, a map, a title screen)
does NOT count. Use what the frames show (what changes across them), not just who or what is visible.
If it does, give the number of the first frame where the event is happening.
JSON keys: {{"match": true|false, "start_frame": 1-{n} or null, "confidence": 0.0-1.0, "why": "brief"}}"""


def _grid(vid, t, out, offsets=EVENT_OFFSETS, cols=4):
    tiles = []
    for j, off in enumerate(offsets):
        p = f"{out}_{j}.png"
        if not verify._tile(vid, t + off - SB_LAG_S, p):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=240x135",
                            "-frames:v", "1", p])
        tiles.append(p)
    rows = len(tiles) // cols
    layout = "|".join(f"{c * 240}_{r * 135}" for r in range(rows) for c in range(cols))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", p] for p in tiles], []) +
                   ["-filter_complex", f"xstack=inputs={len(tiles)}:layout={layout}", out + ".png"])
    return out + ".png" if os.path.exists(out + ".png") else None


def judge_event(vid, t, spec):
    """-> {"match", "start_t", "confidence", "why"}; start_t is the video time of the first frame showing the event."""
    key = f"event|{spec.id}|{vid}|{int(round(t))}"
    with _lock:
        if key in _cache:
            return _cache[key]
    d = os.path.join(store.DATA, "judge")
    os.makedirs(d, exist_ok=True)
    img = _grid(vid, t, os.path.join(d, key.replace("|", "_")))
    if not img:
        return {"match": None, "start_t": None, "confidence": 0.0, "why": "no storyboard frames"}
    n = len(EVENT_OFFSETS)
    try:
        r = vlm.ask_json(EVENT_PROMPT.format(game=spec.game or "a video", description=spec.description, n=n,
                                             span=EVENT_OFFSETS[-1] - EVENT_OFFSETS[0]), [img]) or {}
    except Exception as e:
        return {"match": None, "start_t": None, "confidence": 0.0, "why": f"error: {e}"}
    sf = r.get("start_frame")
    start = t + EVENT_OFFSETS[int(sf) - 1] if isinstance(sf, (int, float)) and 1 <= int(sf) <= n else None
    res = {"match": bool(r.get("match")) if "match" in r else None, "start_t": start,
           "confidence": float(r.get("confidence") or 0), "why": str(r.get("why", ""))[:200]}
    with _lock:
        _cache[key] = res
        json.dump(_cache, open(CACHE + ".tmp", "w"), indent=0)
        os.replace(CACHE + ".tmp", CACHE)
    return res
