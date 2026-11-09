"""On-screen-text index from YouTube storyboards (no video download): sheets from i.ytimg.com -> macOS Vision OCR
-> data/index/<id>.screen.tsv (seconds \t confidence \t text). Same format as scripts/sb_index.py."""
import json
import math
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import store

SB_DIR = os.path.join(store.DATA, "sb")
OCR = os.path.join(store.BIN, "ocr_storyboard")


def _get(url, path, tries=3):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception:
            time.sleep(1 + 2 * i)
    return False


def index_video(vid, max_hours=4):
    out = os.path.join(store.INDEX, f"{vid}.screen.tsv")
    if os.path.exists(out):
        return "cached"
    info_p = os.path.join(store.VIDEOS, f"{vid}.info.json")
    if not os.path.exists(info_p):
        return "no info"
    info = json.load(open(info_p))
    sbs = {f["format_id"]: f for f in info.get("formats", []) if f.get("format_note") == "storyboard"}
    f = sbs.get("sb0") or sbs.get("sb1")
    if not f or not f.get("fragments"):
        return "no storyboard"
    interval = 1 / f["fps"]
    per_sheet = f["columns"] * f["rows"]
    n = min(len(f["fragments"]), math.ceil(max_hours * 3600 / interval / per_sheet))
    d = os.path.join(SB_DIR, vid)
    os.makedirs(d, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        jobs = [ex.submit(lambda u, p: os.path.exists(p) or _get(u, p), fr["url"], os.path.join(d, f"{i:04d}.jpg"))
                for i, fr in enumerate(f["fragments"][:n])]
        ok = sum(1 for j in jobs if j.result())
    if ok < n:
        return f"sheets {ok}/{n}"
    r = subprocess.run([OCR, d, str(f["columns"]), str(f["rows"]), repr(interval)],
                       capture_output=True, text=True, timeout=1800)
    os.makedirs(store.INDEX, exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        fh.write(f"# interval={interval} cols={f['columns']} rows={f['rows']} fmt={f['format_id']} frames={n * per_sheet}\n")
        fh.write(r.stdout)
    os.replace(out + ".tmp", out)
    return f"ok {n} sheets"
