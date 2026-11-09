"""Contact sheets for human verification of proposed event times.

Each row = one (video, t): frames at t-10, t, t+5, t+40 (storyboard tiles; exact-slice frame at t when available).
Usage (from mt.py): build_sheets([(vid, t), ...], out_prefix) -> [png paths], row index printed.
"""
import glob
import math
import os
import subprocess

from . import store, fetch


def _tile(vid, t, out):
    meta, _ = store.load_screen(vid)
    if not meta:
        return False
    iv, cols, rows = float(meta["interval"]), int(meta["cols"]), int(meta["rows"])
    k = max(0, int(round(t / iv)))
    sheet, idx = divmod(k, cols * rows)
    p = os.path.join(store.DATA, "sb", vid, f"{sheet:04d}.jpg")
    if not os.path.exists(p):
        return False
    r, c = divmod(idx, cols)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vf",
                    f"crop=iw/{cols}:ih/{rows}:{c}*iw/{cols}:{r}*ih/{rows},scale=240:135", out])
    return os.path.exists(out)


def _slice_frame(vid, t, out):
    for p in glob.glob(os.path.join(store.SLICES, f"{vid}_*-*.mp4")):
        a, b = os.path.basename(p)[len(vid) + 1:-4].split("@")[0].split("-")
        if int(a) <= t <= int(b):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t - int(a) + 0.5), "-i", p,
                            "-frames:v", "1", "-vf", "scale=240:135", out])
            return os.path.exists(out)
    return False


def build_sheets(items, prefix, per_sheet=10, offsets=(-10, 0, 5, 40)):
    tmp = prefix + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for i, (vid, t) in enumerate(items):
        frames = []
        for j, off in enumerate(offsets):
            f = os.path.join(tmp, f"{i}_{j}.png")
            ok = (off == 0 and _slice_frame(vid, t, f)) or _tile(vid, t + off, f)
            if not ok:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                                "color=c=gray:s=240x135", "-frames:v", "1", f])
            frames.append(f)
        row = os.path.join(tmp, f"row{i}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", f] for f in frames], []) +
                       ["-filter_complex", f"hstack=inputs={len(frames)}", row])
        rows.append(row)
    sheets = []
    for s in range(math.ceil(len(rows) / per_sheet)):
        chunk = rows[s * per_sheet:(s + 1) * per_sheet]
        out = f"{prefix}_{s}.png"
        if len(chunk) == 1:
            subprocess.run(["cp", chunk[0], out])
        else:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", r] for r in chunk], []) +
                           ["-filter_complex", f"vstack=inputs={len(chunk)}", out])
        sheets.append(out)
    return sheets
