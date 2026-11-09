"""All network access goes through here: one global rate limiter, retries with backoff,
429 detection, timeouts, cached outputs, and counters for the eval's cost columns."""
import os
import random
import subprocess
import threading
import time

from . import store

STATS = {"ytdlp_calls": 0, "http_429": 0, "timeouts": 0, "failures": 0, "slices": 0, "slice_bytes": 0}
_lock = threading.Lock()
_next_slot = [0.0]
MIN_GAP_S = 1.0          # minimum spacing between yt-dlp launches (all threads)
_slots = threading.BoundedSemaphore(6)   # max concurrent yt-dlp processes


# ---- per-job controls: a deadline / cancel signal, and a caption circuit breaker ----
# YouTube rate-limits captions per IP; retrying each video with its own backoff just sleeps against a shared limit
# (a 40-video statement search spent 8+ minutes on it, 1 caption obtained). After a few refusals, stop asking.
JOB = {"deadline": None, "cancel": False, "caption_429": 0, "captions_off": False, "blocked": 0}
CAPTION_BREAKER_AT = 2
BLOCKED_AT = 4         # any YouTube request refused this many times in a job (429 / bot check): stop fetching
BLOCK_SIGNS = ("429", "Too Many Requests", "Sign in to confirm", "not a bot")


def new_job(budget_s=None):
    JOB.update(deadline=(time.monotonic() + budget_s) if budget_s else None, cancel=False, caption_429=0,
               captions_off=False, blocked=0)


def stopped():
    """True once the job is cancelled or past its time budget: callers finish with what they have."""
    return (JOB["cancel"] or JOB["blocked"] >= BLOCKED_AT
            or (JOB["deadline"] is not None and time.monotonic() > JOB["deadline"]))


def youtube_blocked():
    return JOB["blocked"] >= BLOCKED_AT


def _wait_turn():
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot[0])
        _next_slot[0] = start + MIN_GAP_S + random.uniform(0, 0.5)
    time.sleep(max(0.0, start - time.monotonic()))


def ytdlp(args, timeout=420, tries=3):
    """Run yt-dlp with global pacing; back off on HTTP 429. Returns (ok, stderr)."""
    err = ""
    for attempt in range(tries):
        if stopped():
            return False, "stopped (cancelled or out of time)"
        _wait_turn()
        with _slots:
            STATS["ytdlp_calls"] += 1
            # Popen + poll so a cancelled / out-of-time job kills its download within a second.
            import tempfile
            errf = tempfile.TemporaryFile(mode="w+")
            pr = subprocess.Popen(["uvx", "yt-dlp", "--no-warnings", "-q"] + args, stdout=subprocess.DEVNULL,
                                  stderr=errf, text=True)
            t_end = time.monotonic() + timeout
            while pr.poll() is None and time.monotonic() < t_end and not stopped():
                time.sleep(0.5)
            if pr.poll() is None:
                pr.kill()
                pr.wait()
                if stopped():
                    return False, "stopped (cancelled or out of time)"
                STATS["timeouts"] += 1
                err = "timeout"
            else:
                errf.seek(0)
                err = errf.read()
                errf.close()
                if pr.returncode == 0:
                    return True, err
        if any(b in err for b in BLOCK_SIGNS):
            with _lock:
                JOB["blocked"] += 1
            if youtube_blocked():
                break
        if "429" in err:
            STATS["http_429"] += 1
            if attempt + 1 < tries:
                time.sleep(20 * (attempt + 1) + random.uniform(0, 5))
        else:
            time.sleep(2 * (attempt + 1))
    STATS["failures"] += 1
    return False, err


def slice_path(vid, a, b, height=144):
    suffix = "" if height == 144 else f"@{height}"
    return os.path.join(store.SLICES, f"{vid}_{int(a)}-{int(b)}{suffix}.mp4")


def fetch_slice(vid, a, b, height=144):
    """Video-only H.264 clip of [a, b] seconds at <= height (cached). Returns path or None."""
    os.makedirs(store.SLICES, exist_ok=True)
    out = slice_path(vid, a, b, height)
    if os.path.exists(out):
        return out
    ok, _ = ytdlp(["-f", f"bv*[vcodec^=avc1][height<={height}][protocol=https]/bv*[vcodec^=avc1][height<={max(height, 240)}]",
                   "--download-sections", f"*{int(a)}-{int(b)}", "-o", out,
                   f"https://www.youtube.com/watch?v={vid}"])
    for f in os.listdir(store.SLICES):   # clean partials left by timeouts
        if f.startswith(os.path.basename(out)) and f.endswith((".part", ".ytdl")):
            os.remove(os.path.join(store.SLICES, f))
    if ok and os.path.exists(out):
        STATS["slices"] += 1
        STATS["slice_bytes"] += os.path.getsize(out)
        return out
    return None


def ocr_video(path, pattern, step=1.0):
    """Seconds (relative to clip start) where on-screen text matches pattern."""
    r = subprocess.run([os.path.join(store.BIN, "ocr_find"), path, pattern, str(step)],
                       capture_output=True, text=True, timeout=600)
    return [float(l.split("\t")[0]) for l in r.stdout.splitlines() if l.strip()]


def fetch_meta(vid, comments=100, captions=True):
    """info.json (+comments, heatmap, chapters, storyboard URLs), then captions as a separate paced call,
    so a caption 429 never loses the metadata. Cached. Returns (has_info, has_captions)."""
    os.makedirs(store.VIDEOS, exist_ok=True)
    info = os.path.join(store.VIDEOS, f"{vid}.info.json")
    url = f"https://www.youtube.com/watch?v={vid}"
    if not os.path.exists(info):
        ytdlp(["--skip-download", "--write-info-json"] +
              (["--write-comments", "--extractor-args", f"youtube:max_comments={comments},{comments},0,0;comment_sort=top"]
               if comments else []) + ["-o", os.path.join(store.VIDEOS, "%(id)s.%(ext)s"), url])
    has_caps = any(os.path.exists(os.path.join(store.VIDEOS, f"{vid}.{l}.json3")) for l in ("en", "en-orig", "en-US"))
    if os.path.exists(info) and not has_caps and captions and not JOB["captions_off"]:
        ok, err = ytdlp(["--skip-download", "--write-auto-subs", "--write-subs", "--sub-langs", "en,en-orig",
                         "--sub-format", "json3", "-o", os.path.join(store.VIDEOS, "%(id)s.%(ext)s"), url], tries=1)
        if "429" in err:
            with _lock:
                JOB["caption_429"] += 1
                if JOB["caption_429"] >= CAPTION_BREAKER_AT:
                    JOB["captions_off"] = True
        has_caps = any(os.path.exists(os.path.join(store.VIDEOS, f"{vid}.{l}.json3")) for l in ("en", "en-orig", "en-US"))
    return os.path.exists(info), has_caps
