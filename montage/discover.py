"""Fresh corpus for a query: YouTube search -> metadata (+captions when the mode needs them) -> storyboard OCR
index -> CLIP embeddings. Everything is cached per video, so repeat queries over the same videos are cheap.
Records wall time per stage."""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from . import embed, fetch, sbindex


def search(queries, per_query=15, max_duration_s=4 * 3600, min_duration_s=20):
    seen, out = set(), []
    for q in queries:
        fetch._wait_turn()
        r = subprocess.run(["uvx", "yt-dlp", "--no-warnings", "-q", "--flat-playlist", "--dump-json",
                            f"ytsearch{per_query}:{q}"], capture_output=True, text=True, timeout=180)
        fetch.STATS["ytdlp_calls"] += 1
        for line in r.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid, dur = d.get("id"), d.get("duration") or 0
            if vid and vid not in seen and min_duration_s <= dur <= max_duration_s:
                seen.add(vid)
                out.append({"id": vid, "title": d.get("title"), "duration": dur, "query": q})
    return out


def prepare(vids, captions=False, embeddings=True, max_hours=4, log=print):
    """Fetch + index every video. Returns ({vid: status}, {stage: seconds})."""
    times, status = {}, {v: [] for v in vids}
    t = time.time()
    with ThreadPoolExecutor(4) as ex:
        for v, (info, caps) in zip(vids, ex.map(lambda v: fetch.fetch_meta(v, comments=0, captions=captions), vids)):
            status[v].append("info" if info else "NO-INFO")
            if captions:
                status[v].append("captions" if caps else "no-captions")
    times["metadata" + ("+captions" if captions else "")] = round(time.time() - t, 1)
    log(f"  metadata: {times[list(times)[-1]]}s")
    t = time.time()
    with ThreadPoolExecutor(3) as ex:
        for v, st in zip(vids, ex.map(lambda v: sbindex.index_video(v, max_hours), vids)):
            status[v].append(f"index:{st}")
    times["storyboard OCR index"] = round(time.time() - t, 1)
    log(f"  storyboard OCR index: {times['storyboard OCR index']}s")
    if embeddings:
        t = time.time()
        for v in vids:
            status[v].append(f"emb:{embed.video(v)}")
        times["CLIP embeddings"] = round(time.time() - t, 1)
        log(f"  CLIP embeddings: {times['CLIP embeddings']}s")
    return status, times
