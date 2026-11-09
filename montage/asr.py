"""Local speech-to-text (mlx-whisper, large-v3-turbo) so statement mode never depends on YouTube's caption endpoint,
which rate-limits per IP. Audio downloads are not throttled (measured: 11 MB in 2.4 s while captions were refused)
and transcription runs ~65x real time on Apple Silicon (11.4 min of audio in 10.5 s, warm).

The transcript is written as data/videos/<id>.whisper.json3 in YouTube's json3 shape, so everything that reads
captions (store.load_transcript, statements.turns) uses it unchanged.
"""
import glob
import json
import os
import threading

from . import fetch, store

MODEL = os.environ.get("MONTAGE_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
AUDIO = os.path.join(store.DATA, "audio")
MAX_AUDIO_S = 45 * 60          # long livestreams: transcribe the first 45 min (noted in results by the caller)
_lock = threading.Lock()       # one transcription at a time on the GPU


def path(vid):
    return os.path.join(store.VIDEOS, f"{vid}.whisper.json3")


def has_transcript(vid):
    return any(os.path.exists(os.path.join(store.VIDEOS, f"{vid}.{l}.json3")) for l in ("en", "en-orig", "en-US", "whisper"))


def audio(vid, duration=None):
    """Whole audio track in the smallest format (speech needs little bitrate). Never --download-sections: that
    streams through ffmpeg at ~real time (measured: 7.8 MB in 4 min); a plain download is seconds."""
    os.makedirs(AUDIO, exist_ok=True)
    have = [p for p in glob.glob(os.path.join(AUDIO, f"{vid}.*")) if not p.endswith((".part", ".ytdl"))]
    if have:
        return have[0]
    ok, err = fetch.ytdlp(["-f", "worstaudio[acodec=opus]/worstaudio[ext=m4a]/bestaudio[ext=m4a]/bestaudio",
                           "-o", os.path.join(AUDIO, "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"],
                          timeout=600, tries=2)
    have = [p for p in glob.glob(os.path.join(AUDIO, f"{vid}.*")) if not p.endswith((".part", ".ytdl"))]
    return have[0] if have else None


def _trimmed(a):
    """First MAX_AUDIO_S seconds as 16 kHz mono wav (local ffmpeg, fast)."""
    import subprocess
    out = os.path.splitext(a)[0] + ".trim.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-t", str(MAX_AUDIO_S), "-ac", "1", "-ar", "16000", out],
                   timeout=600)
    return out if os.path.exists(out) else a


def transcribe(vid):
    """-> "cached" | "ok" | reason. Writes <id>.whisper.json3."""
    if os.path.exists(path(vid)):
        return "cached"
    info = store.load_info(vid) or {}
    a = audio(vid, info.get("duration"))
    if not a:
        return "audio download failed"
    if (info.get("duration") or 0) > MAX_AUDIO_S:
        a2 = _trimmed(a)
        if a2 != a:
            os.remove(a)
            a = a2
    import mlx_whisper
    with _lock:
        r = mlx_whisper.transcribe(a, path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False)
    events = [{"tStartMs": int(s["start"] * 1000), "dDurationMs": int((s["end"] - s["start"]) * 1000),
               "segs": [{"utf8": s["text"].strip()}]} for s in r.get("segments", []) if s.get("text", "").strip()]
    with open(path(vid) + ".tmp", "w") as f:
        json.dump({"source": "whisper", "model": MODEL, "events": events}, f)
    os.replace(path(vid) + ".tmp", path(vid))
    try:
        os.remove(a)              # keep transcripts, not audio
    except OSError:
        pass
    return "ok"
