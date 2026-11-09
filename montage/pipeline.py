"""Spec-driven moment localization over a corpus, with switchable signals for ablations.

Per video:
  1. screen   - search the storyboard on-screen-text index for spec.screen_text (whole video, ~10 s grain)
  2. cues     - otherwise score transcript / comment / chapter / heatmap evidence for a coarse guess
  3. refine   - download a short 144p slice around the coarse time, OCR at 1 fps, snap to the exact second
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import hashlib

from . import fetch, store

TS_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


@dataclass
class Config:
    name: str
    transcript: bool = True
    comments: bool = True
    chapters: bool = True
    heatmap: bool = True
    screen: bool = True
    context: bool = False            # learn on-screen context from pass-1 anchors, use it for the rest
    refine: bool = True
    require_anchor: bool = True      # drop cue-only guesses that refinement could not confirm
    require_context: bool = True     # with a context model, drop screen hits it does not support
    sb_offset_s: float = 0.0         # calibration: storyboard frame time -> true time
    min_ocr_conf: float = 0.3
    max_span_s: float = 300          # anchor text visible longer than this is persistent HUD, not an event
    vlm_seeds: bool = False          # vision model confirms pass-1 candidates before context is learned from them
    vlm_gate: bool = False           # vision model checks every final prediction; confident "no" drops it
    seed_target: int = 1000          # check every pass-1 seed (cached); the check removes bad seeds, it must not shrink the pool
    seed_max_checks: int = 60
    gate_min_conf: float = 0.6
    visual_fill: bool = False        # videos with no anchor: CLIP propagation from the run's own screen anchors (self-calibrated)


CONFIGS = {
    "cues":                Config("cues", screen=False, refine=False, require_anchor=False),
    "transcript":          Config("transcript", comments=False, chapters=False, heatmap=False, screen=False, refine=False, require_anchor=False),
    "cues+refine":         Config("cues+refine", screen=False),                       # = prototype
    "screen":              Config("screen", transcript=False, comments=False, chapters=False, heatmap=False, refine=False),
    "screen+refine":       Config("screen+refine", transcript=False, comments=False, chapters=False, heatmap=False),
    "screen+cues+refine":  Config("screen+cues+refine"),
    "screen+context+refine": Config("screen+context+refine", transcript=False, comments=False, chapters=False, heatmap=False, context=True),
    "all":                 Config("all", context=True),
    "screen+context+refine+vseed": Config("screen+context+refine+vseed", transcript=False, comments=False, chapters=False,
                                          heatmap=False, context=True, vlm_seeds=True),
    "screen+context+refine+vseed+vgate": Config("screen+context+refine+vseed+vgate", transcript=False, comments=False,
                                                chapters=False, heatmap=False, context=True, vlm_seeds=True, vlm_gate=True),
    "screen+context+refine+vseed+vgate+vfill": Config("screen+context+refine+vseed+vgate+vfill", transcript=False,
                                                      comments=False, chapters=False, heatmap=False, context=True,
                                                      vlm_seeds=True, vlm_gate=True, visual_fill=True),
}


# ---------- signals ----------

def _any(pats, text):
    return any(re.search(p, text, re.I) for p in pats)


def transcript_hits(vid, pats):
    return [(t, x) for t, x in store.load_transcript(vid) if pats and _any(pats, x)]


def comment_hits(info, kws):
    dur = info.get("duration") or 0
    out = []
    for c in info.get("comments") or []:
        text = c.get("text") or ""
        stamps = [(int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))) for m in TS_RE.finditer(text)]
        if 0 < len(stamps) <= 3 and kws and _any(kws, text):
            out += [{"t": t, "likes": c.get("like_count") or 0} for t in stamps if 0 < t < dur]
    return out


def _standalone(text, pattern, slack=12):
    m = re.search(pattern, text, re.I)
    return bool(m) and len(text.strip()) - (m.end() - m.start()) <= slack


def title_pattern(game):
    """The game's own title appears on title screens / stream overlays; frames showing it are never the event."""
    name = re.split(r"[(\[]", game or "")[0]
    sub = name.split(":")[-1].strip()
    edition = {"remake", "remaster", "remastered", "hd", "d", "definitive", "edition", "deluxe", "director", "directors",
               "cut", "anniversary", "complete", "goty", "switch", "port"}   # edition words never on the title screen
    words = [w for w in re.findall(r"[A-Za-z]+", sub) if w.lower() not in edition]
    return r"\s*".join(words) if len(words) >= 2 else ""


def screen_hits(vid, pattern, min_conf, style="any", exclude=()):
    """Storyboard frames whose on-screen text matches the anchor. Title/name cards are short standalone
    lines; the same words inside dialogue are mentions, not the event. style=standalone keeps only
    standalone lines; style=any prefers them when the video has any."""
    meta, rows = store.load_screen(vid)
    if not meta or not pattern:
        return None, []
    excl = [p for p in exclude if p]
    iv = float(meta["interval"])
    bad = sorted({t for t, c, x in rows if excl and any(re.search(p, x, re.I) for p in excl)})
    near_bad = lambda t: any(abs(t - b) <= 2.5 * iv for b in bad)       # title screens span several frames
    m = [(t, x) for t, c, x in rows if c >= min_conf and re.search(pattern, x, re.I) and not near_bad(t)]
    alone = [t for t, x in m if _standalone(x, pattern)]
    if style == "standalone" or alone:
        return float(meta["interval"]), sorted(alone)
    return float(meta["interval"]), sorted(t for t, _ in m)


def cluster(ts, gap):
    groups = []
    for t in ts:
        if groups and t - groups[-1][-1] <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def cue_guess(vid, spec, cfg):
    """Best coarse time from cheap text evidence (the prototype's scorer, spec-driven)."""
    info = store.load_info(vid) or {}
    strong = transcript_hits(vid, spec.transcript_strong) if cfg.transcript else []
    weak = transcript_hits(vid, spec.transcript_weak) if cfg.transcript else []
    pre = transcript_hits(vid, spec.transcript_precursor) if cfg.transcript else []
    com = comment_hits(info, spec.comment_keywords) if cfg.comments else []
    chap = [c["start_time"] for c in info.get("chapters") or []
            if cfg.chapters and spec.chapter_pattern and re.search(spec.chapter_pattern, c.get("title", ""), re.I)]
    heat = [((h["start_time"] + h["end_time"]) / 2, h["value"]) for h in info.get("heatmap") or []
            if cfg.heatmap and h["value"] >= 0.6]
    cands = {t for t, _ in strong} | {c["t"] for c in com} | set(chap)
    best = None
    for t in cands:
        s = 2 * min(4, sum(1 for x, _ in strong if abs(x - t) < 90))
        s += 0.5 * min(6, sum(1 for x, _ in weak if abs(x - t) < 60))
        s += 1.5 * min(4, sum(1 for x, _ in pre if 0 < t - x < spec.precursor_window_s))
        ck = [c for c in com if abs(c["t"] - t) < 90]
        s += (4 + sum(min(c["likes"], 500) for c in ck) / 100) if ck else 0
        hp = [v for x, v in heat if abs(x - t) < 120]
        s += 2 * max(hp) if hp else 0
        s += 6 if any(abs(c - t) < 60 for c in chap) else 0
        if best is None or s > best[1]:
            best = (t, s)
    if not best:
        return None, 0
    cluster_ts = [x for x in cands if abs(x - best[0]) < 90]
    return min(cluster_ts), best[1]


# ---------- learned context (no game knowledge) ----------

WORD_RE = re.compile(r"[a-z]{4,}")
STOP = set("that this with have from your there their what when will just they been were into than then them only very".split())


def _word_rows(vid):
    _, rows = store.load_screen(vid)
    return [(t, set(WORD_RE.findall(x.lower()))) for t, c, x in rows if c >= 0.5]


def _collect(seed_times, window, max_video_freq):
    votes = {}  # word -> [(seed, offset)]
    for vid, t in seed_times.items():
        wr = _word_rows(vid)
        freq = {}
        for _, ws in wr:
            for w in ws:
                freq[w] = freq.get(w, 0) + 1
        seen = {}
        for rt, ws in wr:
            off = rt - t
            if not window[0] <= off <= window[1]:
                continue
            for w in ws:
                if w not in STOP and freq.get(w, 0) <= max_video_freq:   # skip overlays / HUD text
                    seen.setdefault(w, off)                              # first appearance near the anchor
        for w, off in seen.items():
            votes.setdefault(w, []).append((vid, off))
    return votes


def _select(votes, n, min_share, min_count, max_spread, anchor_band):
    cues = []
    for w, vs in votes.items():
        offs = sorted(o for _, o in vs)
        if len(vs) < min_count or len(vs) / n < min_share:
            continue
        med = offs[len(offs) // 2]
        if abs(med) <= anchor_band:          # the anchor's own text: no independent evidence
            continue
        if sorted(abs(o - med) for o in offs)[len(offs) // 2] <= max_spread:
            cues.append((w, med))
    return cues


def support(vid, t, cues, gap=12, _cache={}):
    """Number of distinct context words seen on screen near their expected offset from t."""
    if vid not in _cache:
        _cache[vid] = _word_rows(vid)
    found = set()
    for w, off in cues:
        if any(w in ws and abs(rt - (t + off)) <= gap for rt, ws in _cache[vid]):
            found.add(w)
    return len(found)


def learn_context(seed_times, window=(-60, 120), max_video_freq=8, max_spread=12, anchor_band=5):
    """Learn on-screen words that reliably appear at a fixed offset from the anchor, from videos
    whose anchor time is known. Robust to wrong seeds (e.g. revisits): learn loosely, keep the seeds
    the context supports, re-learn strictly from those. Returns [(word, offset_s)]."""
    if len(seed_times) < 3:
        return []
    votes = _collect(seed_times, window, max_video_freq)
    cues = _select(votes, len(seed_times), 0.25, 3, max_spread, anchor_band)
    if len(cues) < 2:
        return cues
    good = {v: t for v, t in seed_times.items() if support(v, t, cues) >= 2}
    if len(good) >= 3:
        cues = _select(_collect(good, window, max_video_freq), len(good), 0.4, 2, max_spread, anchor_band) or cues
    return cues


def context_guess(vid, cues, min_words=2, gap=15):
    """Each cue-word hit votes for anchor time = hit_time - offset; earliest cluster backed by >= min_words distinct words."""
    if len(cues) < min_words:
        return None, 0
    cue = dict(cues)
    votes = sorted((t - cue[w], w) for t, ws in _word_rows(vid) for w in ws if w in cue)
    groups, cur = [], []
    for v in votes:
        if cur and v[0] - cur[-1][0] > gap:
            groups.append(cur); cur = []
        cur.append(v)
    if cur:
        groups.append(cur)
    for g in groups:
        if len({w for _, w in g}) >= min_words:
            ts = sorted(t for t, _ in g)
            return ts[len(ts) // 2], len({w for _, w in g})
    return None, 0


# ---------- per-video ----------

def locate(vid, spec, cfg, ctx_cues=None, screen_only=False):
    rec = {"id": vid, "t": None, "coarse": None, "method": None, "score": None, "error": None}
    try:
        cands = []
        if cfg.screen:
            interval, hits = screen_hits(vid, spec.screen_text, cfg.min_ocr_conf, spec.anchor_style,
                                         list(spec.screen_exclude) + [title_pattern(spec.game)])
            if hits and spec.after_screen_text:
                # Ordering constraint: a recurring scene (a character who talks several times) becomes unique once it
                # must follow a precursor event. No precursor in the video -> no anchor (never guess).
                _, pre = screen_hits(vid, spec.after_screen_text, cfg.min_ocr_conf, "standalone", [title_pattern(spec.game)])  # a card, not a mention
                t0 = min(pre) if pre else None
                hits = [] if t0 is None else [h for h in hits if t0 < h <= t0 + spec.after_window_s]
                if not hits:
                    rec["error"] = "no anchor after precursor" if pre else "precursor not found"
            if hits:
                groups = [g for g in cluster(hits, gap=3 * interval) if g[-1] - g[0] <= cfg.max_span_s]  # long spans = HUD/tracker
                cands = [g[0] + cfg.sb_offset_s for g in groups]
                if spec.occurrence == "last":
                    cands = cands[::-1]
        use_ctx = bool(ctx_cues) and len(ctx_cues) >= 2 and not screen_only
        if cands and not use_ctx:
            rec.update(coarse=cands[0], method="screen", score=len(cands))
        elif use_ctx:
            sup = [(c, support(vid, c, ctx_cues)) for c in cands]
            ok = [c for c, n in sup if n >= 2]
            if ok:
                rec.update(coarse=ok[0], method="screen+ctx", score=max(n for _, n in sup))
            else:
                t, n = context_guess(vid, ctx_cues)
                if t is not None:
                    rec.update(coarse=t, method="context", score=n)
                elif cands and not cfg.require_context:
                    rec.update(coarse=cands[0], method="screen", score=0)
                elif cands:
                    rec["error"] = "screen hit(s) without context support (likely revisit)"
        if screen_only and rec["coarse"] is None:
            return rec
        if rec["coarse"] is None and (cfg.transcript or cfg.comments or cfg.chapters or cfg.heatmap):
            t, s = cue_guess(vid, spec, cfg)
            if t is not None:
                rec.update(coarse=float(t), method="cues", score=round(s, 2))
        if rec["coarse"] is None:
            return rec
        rec["t"] = rec["coarse"]
        if cfg.refine and spec.screen_text:
            c = rec["coarse"]
            a, b = {"screen": (c - spec.refine_before_s, c + spec.refine_after_s),
                    "screen+ctx": (c - spec.refine_before_s, c + spec.refine_after_s),
                    "context": (c - 2 * spec.refine_before_s, c + 2 * spec.refine_after_s)}.get(
                        rec["method"], (c - spec.search_before_s, c + spec.search_after_s))
            a = max(0, int(a)); b = int(b)
            pat = (f"^.{{0,12}}(?:{spec.screen_text}).{{0,12}}$" if spec.anchor_style == "standalone"
                   else spec.screen_text)
            path = fetch.fetch_slice(vid, a, b)
            hits = fetch.ocr_video(path, pat) if path else []
            if path and not hits and b - a <= 60:      # small text (dialogue) needs more pixels; short windows only
                hi = fetch.fetch_slice(vid, a, b, height=360)
                hits = fetch.ocr_video(hi, pat) if hi else []
                path = hi or path
            if hits:
                rec.update(t=a + min(hits), method=rec["method"] + "+refine")
            elif rec["method"] in ("cues", "context") and cfg.require_anchor:
                rec.update(t=None, error="refine: no anchor in slice" if path else "refine: download failed")
    except Exception as e:  # one bad video never kills a run
        rec.update(t=None, error=f"{type(e).__name__}: {e}")
    return rec


def vlm_seed_filter(seeds, spec, cfg):
    """Keep only seeds the vision model confirms; check in a fixed pseudo-random order until enough."""
    from . import judge
    order = sorted(seeds, key=lambda v: hashlib.md5(f"{spec.id}{v}".encode()).hexdigest())
    good, log = {}, []
    for v in order[:cfg.seed_max_checks]:
        j = judge.judge(v, seeds[v], spec)
        log.append((v, seeds[v], j))
        if j["match"]:
            good[v] = seeds[v]
            if len(good) >= cfg.seed_target:
                break
    return good, log


def vlm_gate(recs, spec, cfg):
    from . import judge
    with ThreadPoolExecutor(2) as ex:
        js = list(ex.map(lambda r: judge.judge(r["id"], r["t"], spec) if r["t"] is not None else None, recs))
    for r, j in zip(recs, js):
        if j and j["match"] is False and j["confidence"] >= cfg.gate_min_conf:
            r.update(t=None, error=f"vlm rejected: {j['what'][:120]}")
        elif j:
            r["vlm"] = j
    return recs


def visual_fill(recs):
    """Fill videos that have no anchor by CLIP propagation from this run's own screen anchors (findings 2026-11-09:
    HF recall 0.861 -> 0.944 at P 1.0). The threshold is self-calibrated leave-one-seed-out; if the seeds do not
    predict each other, nothing is filled."""
    from . import visual
    seeds = {r["id"]: r["t"] for r in recs if r["t"] is not None and str(r["method"]).startswith("screen")}
    if len(seeds) < 3:
        return {"seeds": len(seeds), "threshold": None, "filled": 0, "note": "too few anchors to propagate"}
    thr, table = visual.calibrate(seeds)
    if thr is None:
        return {"seeds": len(seeds), "threshold": None, "filled": 0, "note": "abstained: seeds inconsistent", "table": table}
    protos = visual.prototypes(seeds)
    n = 0
    for r in recs:
        if r["t"] is None:
            t, sup, sim = visual.guess(r["id"], protos, sim_thr=thr, min_support=3)
            if t is not None:
                r.update(t=round(t, 1), coarse=t, method="visual", score=sup, error=None)
                n += 1
    return {"seeds": len(seeds), "threshold": thr, "filled": n}


def run(spec, corpus, cfg, workers=6):
    t0 = time.time()
    before = dict(fetch.STATS)
    cues = None
    with ThreadPoolExecutor(workers) as ex:
        if cfg.context and cfg.screen:
            # pass 1: direct screen anchors; their (refined) times seed the context model
            first = {r["id"]: r for r in ex.map(lambda v: locate(v, spec, cfg, screen_only=True), corpus)}
            seeds = {v: r["t"] for v, r in first.items() if r["t"] is not None}
            if cfg.vlm_seeds:
                seeds, seed_log = vlm_seed_filter(seeds, spec, cfg)
            cues = learn_context(seeds)
            # pass 2: every video, screen hits now verified by context (rejects revisits)
            recs = list(ex.map(lambda v: locate(v, spec, cfg, ctx_cues=cues), corpus))
        else:
            recs = list(ex.map(lambda v: locate(v, spec, cfg), corpus))
    if cfg.vlm_gate:
        recs = vlm_gate(recs, spec, cfg)
    fill_info = visual_fill(recs) if cfg.visual_fill else None
    cost = {k: fetch.STATS[k] - before[k] for k in fetch.STATS}
    if cues is not None:
        cost["context_cues"] = sorted(cues, key=lambda c: c[1])
    out = {"wall_s": round(time.time() - t0, 1), "slices": cost["slices"],
           "slice_mb": round(cost["slice_bytes"] / 1e6, 1), "http_429": cost["http_429"], "failures": cost["failures"]}
    if cues is not None:
        out["context_cues"] = cost["context_cues"]
    if fill_info is not None:
        out["visual_fill"] = fill_info
    from . import vlm as _vlm
    out["vlm_calls"] = _vlm.STATS["calls"]
    return {r["id"]: r for r in recs}, out
