"""Statement moments: "<who> saying <what>" across news-style videos.

Per video (no video download):
  1. turns     - caption events split into speaker turns at ">>" markers and long pauses
  2. onscreen  - storyboard OCR lines (chyrons: names, party, titles), time-aligned
  3. extract   - a text LLM reads metadata + turns + on-screen text and returns qualifying segments,
                 each with speaker, role, evidence (chyron text or transcript span) and a quote.
                 Evidence-required: no evidence -> speaker "Unknown" (per prior art, docs/research/).
  4. dedupe    - the same statement re-uploaded by several outlets is grouped by quote similarity.

Spec fields used (TOML, mode = "statements"): description, who, what, after, exclude_sources.
"""
import json
import re

from . import store, vlm

MAX_DOC_CHARS = 12000          # per LLM call; long videos are windowed around topic keywords (short docs keep brief soundbites visible)


def turns(vid, pause_s=2.0):
    """[(start, end, text)] split at '>>' speaker markers and long caption gaps."""
    p = None
    for lang in ("en", "en-orig", "en-US", "whisper"):
        import os
        cand = os.path.join(store.VIDEOS, f"{vid}.{lang}.json3")
        if os.path.exists(cand):
            p = cand
            break
    if not p:
        return []
    evs = []
    for e in json.load(open(p)).get("events", []):
        txt = "".join(s.get("utf8", "") for s in e.get("segs") or []).replace("\n", " ").strip()
        if txt:
            evs.append((e["tStartMs"] / 1000.0, (e.get("dDurationMs") or 0) / 1000.0, txt))
    out, cur, cur_start, last_end = [], [], None, None
    for t, d, txt in evs:
        parts = re.split(r"\s*>>+\s*", txt)
        for i, part in enumerate(parts):
            boundary = (i > 0) or (last_end is not None and t - last_end > pause_s and cur)
            if boundary and cur:
                out.append((cur_start, last_end or t, " ".join(cur)))
                cur, cur_start = [], None
            if part:
                if cur_start is None:
                    cur_start = t
                cur.append(part)
        last_end = t + d
    if cur:
        out.append((cur_start, last_end, " ".join(cur)))
    return out


def onscreen(vid, min_len=6):
    """Deduplicated on-screen text lines [(t, text)] from the storyboard index (chyrons etc.)."""
    meta, rows = store.load_screen(vid)
    seen, out = {}, []
    for t, c, x in rows:
        x = x.strip()
        if c < 0.5 or len(x) < min_len:
            continue
        k = re.sub(r"[^a-z]", "", x.lower())[:24]
        if k in seen and t - seen[k] < 60:
            continue
        seen[k] = t
        out.append((t, x))
    return out


def _fmt(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def build_doc(vid, spec, windows=None):
    info = store.load_info(vid) or {}
    T = turns(vid)
    S = onscreen(vid)
    if windows:
        T = [x for x in T if any(a <= x[0] <= b for a, b in windows)]
        S = [x for x in S if any(a - 30 <= x[0] <= b + 30 for a, b in windows)]
    head = (f"VIDEO {vid}\nTITLE: {info.get('title')}\nCHANNEL: {info.get('channel')}\n"
            f"UPLOADED: {info.get('upload_date')}\nDURATION: {_fmt(info.get('duration') or 0)}\n"
            f"DESCRIPTION: {(info.get('description') or '')[:600]}\n")
    scr = "\n".join(f"[{_fmt(t)}] {x}" for t, x in S)
    tr = "\n".join(f"[{_fmt(a)}-{_fmt(b)}] {x}" for a, b, x in T)
    doc = f"{head}\nON-SCREEN TEXT (OCR of frames; chyrons show names/party):\n{scr}\n\nTRANSCRIPT TURNS (captions; '>>' boundaries = speaker changes, unreliable):\n{tr}\n"
    return doc, T


PROMPT = """You find moments in videos for this request: "{description}".
A qualifying moment needs ALL of:
- WHO: {who}
- WHAT: {what}
- WHEN: the statement is made on or after {after} (check UPLOADED date and content).
Exclude: {exclude}

Identify who is speaking in each turn ONLY from evidence: an on-screen chyron naming them near that time, a host
introducing/addressing them right before ("Senator X, ...", "joining us now ..."), a self-introduction, or the
title/description clearly saying the video is that person's speech. A chyron alone is not enough if the turn
reads like a host or narrator talking ABOUT the person; the turn must read as the person speaking themselves.
Hosts, anchors, reporters, pundits, analysts, comedians and voters are not the WHO unless they meet it.
On-screen text that looks like a social-media post (an @handle, "X"/platform names, a quoted sentence shown as a graphic)
means the person's WRITTEN post is being shown or read aloud by someone else - that is not the person speaking.
Reported speech does not count: if the words refer to the person in the third person ("she said", "her brother",
"he told reporters", "the senator thanked voters"), someone else is describing them - not qualifying. The quote you
return must be the person's own first-person words.
Transcripts may come from speech recognition and misspell names; take the name from on-screen text when available
and never invent a person from a garbled name.
If you cannot establish the speaker from evidence, use speaker "Unknown" and do not mark it as qualifying.
If the TRANSCRIPT is empty: you may rely on the TITLE/DESCRIPTION only when they explicitly state that the video is that
person's own speech/remarks about the results (e.g. "X delivers concession speech", "X speaks after the election");
then return the whole video as one segment and start the evidence with "title-only:".

Return every qualifying segment (merge consecutive turns by the same speaker; segments can be long).
JSON: {{"segments": [{{"start": "H:MM:SS", "end": "H:MM:SS", "speaker": "name", "role": "e.g. Senator (R-SD)",
"evidence": "the chyron text or transcript words that identify the speaker", "quote": "one representative sentence",
"why": "why it meets WHO and WHAT"}}], "video_note": "one line: what this video is"}}
Return {{"segments": []}} if nothing qualifies.

{doc}"""


def _secs(s):
    try:
        parts = [int(p) for p in str(s).split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def topic_windows(T, pats, pad=180):
    hits = [a for a, b, x in T if any(re.search(p, x, re.I) for p in pats)]
    wins = []
    for h in sorted(hits):
        if wins and h - pad <= wins[-1][1]:
            wins[-1][1] = h + pad
        else:
            wins.append([h - pad, h + pad])
    return [(max(0, a), b) for a, b in wins]


def extract(vid, spec, model=None, think=True):
    info = store.load_info(vid) or {}
    if spec.after and (info.get("upload_date") or "99999999") < spec.after.replace("-", ""):
        return {"id": vid, "segments": [], "note": f"skipped: uploaded {info.get('upload_date')} before {spec.after}", "llm_calls": 0}
    doc, T = build_doc(vid, spec)
    calls = []
    if len(doc) <= MAX_DOC_CHARS:
        calls.append(doc)
    else:   # long video: window around topic keywords, chunk to size
        wins = topic_windows(T, spec.topic_keywords or [r"elect", r"midterm", r"result"])
        chunk = []
        for w in wins:
            chunk.append(w)
            d, _ = build_doc(vid, spec, chunk)
            if len(d) > MAX_DOC_CHARS and len(chunk) > 1:
                calls.append(build_doc(vid, spec, chunk[:-1])[0])
                chunk = [w]
        if chunk:
            calls.append(build_doc(vid, spec, chunk)[0][:MAX_DOC_CHARS])
    segs, notes = [], []
    for d in calls:
        r = vlm.ask_json(PROMPT.format(description=spec.description, who=spec.who, what=spec.what,
                                       after=spec.after, exclude=spec.exclude_sources, doc=d),
                         model=model, think=think, max_tokens=6000, timeout=1800) or {}
        notes.append(r.get("video_note", ""))
        for s in r.get("segments") or []:
            a, b = _secs(s.get("start")), _secs(s.get("end"))
            if a is None or b is None or str(s.get("speaker", "")).lower() == "unknown":
                continue
            segs.append({"start": a, "end": max(b, a + 1), "speaker": s.get("speaker"), "role": s.get("role"),
                         "evidence": s.get("evidence"), "quote": s.get("quote"), "why": s.get("why")})
    return {"id": vid, "segments": segs, "note": " | ".join(n for n in notes if n), "llm_calls": len(calls)}


def _shingles(text, n=4):
    w = re.findall(r"[a-z']+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def dedupe(results, threshold=0.25):
    """Group the same statement across uploads by transcript overlap; returns list of groups [(vid, seg)]."""
    items = []
    for r in results:
        T = turns(r["id"])
        for s in r["segments"]:
            text = " ".join(x for a, b, x in T if a < s["end"] and b > s["start"])
            items.append((r["id"], s, _shingles(text)))
    groups = []
    for it in items:
        for g in groups:
            sh = g[0][2]
            if sh and it[2] and len(sh & it[2]) / max(1, min(len(sh), len(it[2]))) >= threshold and it[0] != g[0][0]:
                g.append(it)
                break
        else:
            groups.append([it])
    return [[(v, s) for v, s, _ in g] for g in groups]


GATE_PROMPT = """These are frames (left to right) sampled across a video segment that a system attributed to: {speaker} ({role}).
Answer from what is visible only. Do NOT identify anyone from their face; use only visible text (chyrons, captions, graphics).
1. Is a real person shown speaking in these frames (at a podium, interview, rally, press gaggle, panel), as opposed to
   a social-media post/quote graphic, a results map or data graphic, a news anchor desk with the person absent, or b-roll?
2. If any on-screen text names someone, does it contradict the attribution to {speaker}?
JSON keys: {{"person_speaking": true|false, "graphic_or_post_only": true|false, "name_text_contradicts": true|false,
"confidence": 0.0-1.0, "what": "short description"}}"""


def gate(vid, seg, min_conf=0.6):
    """Vision check of a predicted segment from storyboard frames. Returns (keep, judgment)."""
    import os
    import subprocess
    from . import judge, verify
    a, b = seg["start"], seg["end"]
    ts = [a + (b - a) * f for f in (0.15, 0.5, 0.85)] if b - a > 12 else [(a + b) / 2]
    d = os.path.join(store.DATA, "judge")
    os.makedirs(d, exist_ok=True)
    tiles = []
    for j, t in enumerate(ts):
        p = os.path.join(d, f"stmt_{vid}_{int(a)}_{j}.png")
        if verify._tile(vid, t - judge.SB_LAG_S, p):
            tiles.append(p)
    if not tiles:
        return True, {"what": "no frames"}
    strip = os.path.join(d, f"stmt_{vid}_{int(a)}.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", p] for p in tiles], []) +
                   ["-filter_complex", f"hstack=inputs={len(tiles)}" if len(tiles) > 1 else "null", strip])
    r = vlm.ask_json(GATE_PROMPT.format(speaker=seg.get("speaker"), role=seg.get("role")), [strip]) or {}
    conf = float(r.get("confidence") or 0)
    bad = (r.get("graphic_or_post_only") is True or r.get("person_speaking") is False
           or r.get("name_text_contradicts") is True)
    return not (bad and conf >= min_conf), r
