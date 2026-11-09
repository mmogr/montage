"""Snap candidate windows to a deterministic on-screen game event (title card) via OCR.

For each candidate from analyze.py: download a short video-only slice around the
candidate time, OCR it, and if the title card is found, emit a precise clip.
Usage: python3 -I locate.py CANDIDATES_JSON OUT_JSON [min_score]
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICES = os.path.join(ROOT, "data", "slices")
OCR = os.path.join(ROOT, "bin", "ocr_find")
PATTERN = r"hyrule\s*field"
BEFORE, AFTER = 110, 50          # slice around candidate (seconds)
CLIP_LEAD, CLIP_TAIL = 20, 14    # final clip relative to the title card


def ocr_slice(vid, start, end):
    os.makedirs(SLICES, exist_ok=True)
    out = os.path.join(SLICES, f"{vid}_{start}.mp4")
    if not os.path.exists(out):
        for attempt in range(2):
            try:
                subprocess.run(["uvx", "yt-dlp", "-q", "--no-warnings",
                                "-f", "bv*[vcodec^=avc1][height<=144][protocol=https]/bv*[vcodec^=avc1][height<=240]",
                                "--download-sections", f"*{start}-{end}",
                                "-o", out, f"https://www.youtube.com/watch?v={vid}"],
                               capture_output=True, timeout=420)
                break
            except subprocess.TimeoutExpired:
                for f in os.listdir(SLICES):
                    if f.startswith(f"{vid}_{start}.") and f.endswith((".part", ".ytdl")):
                        os.remove(os.path.join(SLICES, f))
    if not os.path.exists(out):
        return None, "download failed"
    r = subprocess.run([OCR, out, PATTERN, "1"], capture_output=True, text=True, timeout=300)
    hits = [float(l.split("\t")[0]) for l in r.stdout.splitlines() if l.strip()]
    return hits, None


def locate(v):
    tries = [v["best"]] + [a for a in v.get("alternatives", []) if abs(a["t"] - v["best"]["t"]) > 300][:1]
    for cand in tries:
        t = int(cand["t"])
        start = max(0, t - BEFORE)
        hits, err = ocr_slice(v["id"], start, t + AFTER)
        if hits:
            card = start + min(hits)
            return {**v, "card_t": card, "clip": [max(0, card - CLIP_LEAD), card + CLIP_TAIL],
                    "ocr_frames": len(hits), "status": "card"}
        last_err = err or "no card in slice"
    return {**v, "status": last_err}


def main():
    src, out = sys.argv[1], sys.argv[2]
    min_score = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    cands = [v for v in json.load(open(src)) if v["best"] and v["best"]["score"] >= min_score]
    print(f"locating {len(cands)} candidates")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(locate, cands))
    json.dump(res, open(out, "w"), indent=1)
    for r in res:
        c = f"{int(r['card_t'])//3600}:{int(r['card_t'])%3600//60:02d}:{int(r['card_t'])%60:02d}" if r.get("card_t") else "-"
        drift = int(r["card_t"] - r["best"]["t"]) if r.get("card_t") else None
        print(f"{r['status']:18} card={c:>8} drift={str(drift):>5}  {r['channel'][:28]}")
    ok = sum(1 for r in res if r["status"] == "card")
    print(f"\n{ok}/{len(res)} snapped to title card")


if __name__ == "__main__":
    main()
