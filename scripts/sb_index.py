"""Build a whole-video on-screen-text index from YouTube storyboards (no video download).

For each video: fetch storyboard sheets (sb0 = 320x180, ~1 frame/10s) from i.ytimg.com,
OCR every frame, write data/index/<id>.screen.tsv  (seconds \t confidence \t text).
Usage: python3 -I sb_index.py VIDEOS_DIR ID_LIST_FILE [max_hours]
"""
import json, os, sys, subprocess, urllib.request, math, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB_DIR = os.path.join(ROOT, "data", "sb")
IDX_DIR = os.path.join(ROOT, "data", "index")
OCR = os.path.join(ROOT, "bin", "ocr_storyboard")


def fetch(url, path, tries=3):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception:
            time.sleep(1 + 2 * i)
    return False


def index_video(vid, vdir, max_hours):
    out = os.path.join(IDX_DIR, f"{vid}.screen.tsv")
    if os.path.exists(out):
        return vid, "cached"
    info = json.load(open(os.path.join(vdir, f"{vid}.info.json")))
    sbs = {f["format_id"]: f for f in info.get("formats", []) if f.get("format_note") == "storyboard"}
    f = sbs.get("sb0") or sbs.get("sb1")
    if not f or not f.get("fragments"):
        return vid, "no storyboard"
    interval = 1 / f["fps"]
    per_sheet = f["columns"] * f["rows"]
    n = min(len(f["fragments"]), math.ceil(max_hours * 3600 / interval / per_sheet))
    d = os.path.join(SB_DIR, vid)
    os.makedirs(d, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(8) as ex:
        jobs = []
        for i, fr in enumerate(f["fragments"][:n]):
            p = os.path.join(d, f"{i:04d}.jpg")
            jobs.append(ex.submit(lambda u, p: os.path.exists(p) or fetch(u, p), fr["url"], p))
        ok = sum(1 for j in jobs if j.result())
    if ok < n:
        return vid, f"sheets {ok}/{n}"
    r = subprocess.run([OCR, d, str(f["columns"]), str(f["rows"]), repr(interval)],
                       capture_output=True, text=True, timeout=1800)
    with open(out + ".tmp", "w") as fh:
        fh.write(f"# interval={interval} cols={f['columns']} rows={f['rows']} fmt={f['format_id']} frames={n*per_sheet}\n")
        fh.write(r.stdout)
    os.replace(out + ".tmp", out)
    return vid, f"ok {n} sheets, {r.stdout.count(chr(10))} lines"


def main():
    vdir, idfile = sys.argv[1], sys.argv[2]
    max_hours = float(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(IDX_DIR, exist_ok=True)
    ids = [l.split()[0] for l in open(idfile) if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(3) as ex:
        for vid, st in ex.map(lambda v: (lambda: index_video(v, vdir, max_hours))() if True else None, ids):
            print(vid, st, flush=True)


if __name__ == "__main__":
    main()
