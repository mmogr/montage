"""Visual propagation over storyboard CLIP embeddings (no text, no downloads).

From confirmed anchors in seed videos, collect frame embeddings around each anchor with their offsets
(the "look" of the moment and its surroundings). In another video, every frame that closely matches a seed frame
votes for anchor_time = frame_time - offset; the earliest cluster backed by enough distinct seed frames wins.
"""
import os

import numpy as np

from . import store

EMB = os.path.join(store.DATA, "emb")
SB_LAG_S = 7.0


def load(vid):
    p = os.path.join(EMB, f"{vid}.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["emb"].astype(np.float32), d["t"] + SB_LAG_S     # times corrected to true video time


def prototypes(seed_times, window=(-20, 40)):
    embs, offs, src = [], [], []
    for vid, t in seed_times.items():
        e, ts = load(vid)
        if e is None:
            continue
        m = (ts >= t + window[0]) & (ts <= t + window[1])
        embs.append(e[m]); offs.append(ts[m] - t); src += [vid] * int(m.sum())
    if not embs:
        return None
    return np.concatenate(embs), np.concatenate(offs), np.array(src)


def guess(vid, protos, sim_thr=0.9, min_support=3, gap=15, exclude_self=True, occurrence="first"):
    """Return (anchor_time, support, mean_sim) or (None, 0, 0)."""
    e, ts = load(vid)
    if e is None or protos is None:
        return None, 0, 0.0
    P, offs, src = protos
    keep = src != vid if exclude_self else np.ones(len(src), bool)
    P, offs, src = P[keep], offs[keep], src[keep]
    if len(P) == 0:
        return None, 0, 0.0
    S = e @ P.T                                     # frames x prototypes
    fi, pj = np.nonzero(S >= sim_thr)
    if len(fi) == 0:
        return None, 0, 0.0
    votes = sorted(zip(ts[fi] - offs[pj], src[pj], S[fi, pj]))
    groups, cur = [], [votes[0]]
    for v in votes[1:]:
        if v[0] - cur[-1][0] > gap:
            groups.append(cur); cur = []
        cur.append(v)
    groups.append(cur)
    ok = [g for g in groups if len({s for _, s, _ in g}) >= min_support]   # distinct seed VIDEOS agreeing
    if not ok:
        return None, 0, 0.0
    g = ok[0] if occurrence == "first" else max(ok, key=lambda g: len({s for _, s, _ in g}))
    tt = sorted(x for x, _, _ in g)
    return float(tt[len(tt) // 2]), len({s for _, s, _ in g}), float(np.mean([x for _, _, x in g]))


def calibrate(seed_times, grid=(0.86, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95), tol=10, min_support=3):
    """Pick sim_thr without truth: leave-one-seed-out. For each threshold, hide each seed video, predict it from
    the others, and count how often the prediction lands within tol of the seed's own anchor. Score = correct
    minus wrong (a wrong guess costs as much as a correct one gains); ties go to the stricter threshold."""
    protos = prototypes(seed_times)
    best, table = None, []
    for thr in grid:
        ok = bad = 0
        for vid, t in seed_times.items():
            g, n, s = guess(vid, protos, sim_thr=thr, min_support=min_support, exclude_self=True)
            if g is None:
                continue
            if abs(g - t) <= tol:
                ok += 1
            else:
                bad += 1
        table.append((thr, ok, bad))
        if best is None or ok - bad >= best[1] - best[2]:
            best = (thr, ok, bad)
    # Seeds that cannot predict each other are not one consistent moment (e.g. the same place shown in several
    # scenes). Abstain (None) rather than propagate: the caller must report "seeds inconsistent".
    if best[1] < 3 or best[1] <= 2 * best[2]:
        return None, table
    return best[0], table


def onset(vid, t, bar=0.9, max_back_s=120):
    """Walk back from a confirmed frame while frames still look like it (cosine >= bar); return the run's start.
    A query's best-matching frame is usually mid-scene (the field is most 'field-like' once you are well into it);
    the event is when the scene begins."""
    e, ts = load(vid)
    if e is None:
        return t
    i = int(np.argmin(np.abs(ts - t)))
    ref, j = e[i], i
    while j > 0 and ts[i] - ts[j - 1] <= max_back_s and float(e[j - 1] @ ref) >= bar:
        j -= 1
    return float(ts[j])


def scenes(vid, q, k=3, gap_s=60):
    """Top-k separate peaks (>= gap_s apart) of similarity to a query embedding: [(sim, t)]."""
    e, ts = load(vid)
    if e is None:
        return []
    s = e @ q
    out = []
    for i in np.argsort(-s):
        if all(abs(ts[i] - t) > gap_s for _, t in out):
            out.append((float(s[i]), float(ts[i])))
            if len(out) == k:
                break
    return out
