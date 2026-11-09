"""Local cache layout and loaders. Everything per-video is query-independent and reused across moments."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
VIDEOS = os.path.join(DATA, "videos")      # <id>.info.json (+comments, heatmap, chapters), <id>.<lang>.json3
INDEX = os.path.join(DATA, "index")        # <id>.screen.tsv  on-screen text from storyboards
SLICES = os.path.join(DATA, "slices")      # <id>_<start>-<end>.mp4  short 144p clips for exact refinement
BIN = os.path.join(ROOT, "bin")


def load_corpus(path):
    return [l.split("\t")[0].strip() for l in open(path) if l.strip() and not l.startswith("#")]


_info_cache = {}


def load_info(vid):
    if vid not in _info_cache:
        p = os.path.join(VIDEOS, f"{vid}.info.json")
        _info_cache[vid] = json.load(open(p)) if os.path.exists(p) else None
    return _info_cache[vid]


def load_transcript(vid):
    """[(seconds, lowercase text)] from YouTube captions (json3) or a local Whisper transcript (same shape)."""
    for lang in ("en", "en-orig", "en-US", "whisper"):
        p = os.path.join(VIDEOS, f"{vid}.{lang}.json3")
        if os.path.exists(p):
            out = []
            for e in json.load(open(p)).get("events", []):
                txt = "".join(s.get("utf8", "") for s in e.get("segs") or []).strip()
                if txt:
                    out.append((e["tStartMs"] / 1000.0, txt.lower()))
            return out
    return []


def load_screen(vid):
    """(meta, [(seconds, confidence, text)]) from the storyboard OCR index, or (None, []) if not indexed."""
    p = os.path.join(INDEX, f"{vid}.screen.tsv")
    if not os.path.exists(p):
        return None, []
    meta, rows = {}, []
    for line in open(p):
        if line.startswith("#"):
            meta = dict(kv.split("=", 1) for kv in line[1:].split())
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            rows.append((float(parts[0]), float(parts[1]), parts[2]))
    return meta, rows
