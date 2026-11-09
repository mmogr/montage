"""Find a target moment across many YouTube playthroughs using cheap signals.

Signals per video (no video download, no API keys):
  - comment timestamps whose text matches the moment's keywords
  - transcript (auto-caption) keyword hits, plus an "anchor sequence" prior
  - Most Replayed heatmap peaks
  - creator chapters
Usage: python3 -I analyze.py DATA_DIR MOMENT_NAME OUT_JSON
"""
import json, re, sys, glob, os
from collections import defaultdict

MOMENTS = {
    "hyrule_field": {
        # strong: names the moment directly
        "strong": [r"hyrule field", r"\bthe field\b", r"kaepora", r"\bowl\b"],
        # weak: emotional / descriptive words people use there
        "weak": [r"open world", r"so big", r"huge", r"music", r"theme", r"beautiful", r"gorgeous",
                 r"breathtaking", r"goosebumps", r"chills", r"wow", r"whoa", r"holy"],
        # anchor: events that happen just before (Saria's farewell on the bridge)
        "anchor_before": [r"\bsaria\b", r"\bbridge\b", r"ocarina"],
        "comment_kw": [r"hyrule field", r"\bfield\b", r"\bowl\b", r"kaepora", r"saria", r"bridge"],
        "lead": 15, "tail": 45,
    },
}

TS_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def parse_ts(m):
    h, mi, s = m.group(1), m.group(2), m.group(3)
    return (int(h) if h else 0) * 3600 + int(mi) * 60 + int(s)


def load_transcript(vid, d):
    for lang in ("en", "en-orig", "en-US"):
        p = os.path.join(d, f"{vid}.{lang}.json3")
        if os.path.exists(p):
            ev = json.load(open(p)).get("events", [])
            out = []
            for e in ev:
                txt = "".join(s.get("utf8", "") for s in e.get("segs", []) or []).strip()
                if txt:
                    out.append((e["tStartMs"] / 1000.0, txt.lower()))
            return out
    return []


def comment_hits(info, kws, dur):
    hits = []
    for c in info.get("comments") or []:
        text = c.get("text") or ""
        stamps = [parse_ts(m) for m in TS_RE.finditer(text)]
        stamps = [t for t in stamps if 0 < t < dur]
        if not stamps:
            continue
        low = text.lower()
        kw = any(re.search(k, low) for k in kws)
        for t in stamps:
            hits.append({"t": t, "kw": kw, "likes": c.get("like_count") or 0, "text": text[:200]})
    return hits


def transcript_hits(tr, pats):
    return [(t, txt) for t, txt in tr if any(re.search(p, txt) for p in pats)]


def heat_peaks(info, min_val=0.6):
    hm = info.get("heatmap") or []
    return [((h["start_time"] + h["end_time"]) / 2, h["value"]) for h in hm if h["value"] >= min_val]


def score_video(vid, d, spec):
    info = json.load(open(os.path.join(d, f"{vid}.info.json")))
    dur = info.get("duration") or 0
    tr = load_transcript(vid, d)
    ch = comment_hits(info, spec["comment_kw"], dur)
    strong = transcript_hits(tr, spec["strong"])
    weak = transcript_hits(tr, spec["weak"])
    anchors = transcript_hits(tr, spec["anchor_before"])
    heat = heat_peaks(info)
    chapters = [c for c in info.get("chapters") or [] if re.search(r"hyrule field|field", c.get("title", ""), re.I)]

    # candidate times = every strong signal; score each by nearby evidence
    cands = set(t for t, _ in strong) | set(h["t"] for h in ch if h["kw"]) | set(c["start_time"] for c in chapters)
    scored = []
    for t in cands:
        s, why = 0.0, []
        n_strong = sum(1 for x, _ in strong if abs(x - t) < 90)
        if n_strong: s += 2 * min(n_strong, 4); why.append(f"{n_strong} transcript strong")
        n_weak = sum(1 for x, _ in weak if abs(x - t) < 60)
        if n_weak: s += 0.5 * min(n_weak, 6); why.append(f"{n_weak} transcript weak")
        n_anchor = sum(1 for x, _ in anchors if 0 < t - x < 600)
        if n_anchor: s += 1.5 * min(n_anchor, 4); why.append(f"{n_anchor} anchor before")
        ck = [h for h in ch if h["kw"] and abs(h["t"] - t) < 90]
        if ck: s += 4 + sum(min(h["likes"], 500) for h in ck) / 100; why.append(f"{len(ck)} comment kw")
        hp = [v for x, v in heat if abs(x - t) < 120]
        if hp: s += 2 * max(hp); why.append(f"heat {max(hp):.2f}")
        if any(abs(c["start_time"] - t) < 60 for c in chapters): s += 6; why.append("chapter")
        # prefer first visit: penalise late candidates mildly relative to first strong hit
        scored.append({"t": t, "score": round(s, 2), "why": why})
    scored.sort(key=lambda x: -x["score"])

    best = scored[0] if scored else None
    if best:
        # refine: snap to earliest strong signal inside the best cluster (first entry, not later revisits)
        cluster = [x for x in cands if abs(x - best["t"]) < 90]
        best["t"] = min(cluster)
    return {
        "id": vid, "title": info.get("title"), "channel": info.get("channel"),
        "upload_date": info.get("upload_date"), "duration": dur, "views": info.get("view_count"),
        "has_transcript": bool(tr), "has_heatmap": bool(info.get("heatmap")),
        "n_comments": len(info.get("comments") or []), "n_comment_ts": len(ch),
        "best": best, "alternatives": scored[1:4],
        "evidence": {
            "transcript_strong": [(round(t), x) for t, x in strong if best and abs(t - best["t"]) < 120][:8],
            "comments": [h for h in ch if h["kw"]][:5],
        },
    }


def main():
    d, moment, out = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = MOMENTS[moment]
    res = []
    for p in sorted(glob.glob(os.path.join(d, "*.info.json"))):
        vid = os.path.basename(p).split(".")[0]
        try:
            r = score_video(vid, d, spec)
        except Exception as e:
            print("ERR", vid, e); continue
        if (r["upload_date"] or "0") < "20261105":
            continue  # pre-launch: not the remake
        res.append(r)
    res.sort(key=lambda r: -(r["best"]["score"] if r["best"] else -1))
    json.dump(res, open(out, "w"), indent=1)
    for r in res:
        b = r["best"]
        hhmmss = f"{int(b['t'])//3600}:{int(b['t'])%3600//60:02d}:{int(b['t'])%60:02d}" if b else "-"
        print(f"{(b['score'] if b else 0):6.1f}  {hhmmss:>8}  tr={int(r['has_transcript'])} hm={int(r['has_heatmap'])} cts={r['n_comment_ts']:3d}  {r['channel'][:22]:22}  {', '.join(b['why']) if b else 'no candidates'}")


if __name__ == "__main__":
    main()
