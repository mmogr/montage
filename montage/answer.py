"""End to end: a sentence -> plan a spec -> search the corpus -> markdown a chat client can render.

This is the body of the future MCP tool. Output is plain markdown: video titles with timestamped links, plus how
the query was read and what was excluded, so a human can spot a misreading.

  uv run --with numpy python -I mt.py ask "QUERY" --corpus eval/corpus/oot_remake_v1.tsv
"""
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor

from . import pipeline, planner, statements, store
from .spec import Spec

# No slice downloads (YouTube throttles them): storyboard anchors + 7 s storyboard calibration + learned context +
# vision seed check + vision gate + visual fill. Measured in findings.md.
MOMENT_CFG = pipeline.Config("answer", transcript=False, comments=False, chapters=False, heatmap=False, refine=False,
                             context=True, vlm_seeds=True, vlm_gate=True, visual_fill=True, sb_offset_s=7.0)


def _ts(t):
    t = int(max(0, t))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _video(vid):
    info = store.load_info(vid) or {}
    return (info.get("title") or vid).replace("[", "(").replace("]", ")"), info.get("channel") or info.get("uploader") or ""


def _link(vid, t):
    return f"https://www.youtube.com/watch?v={vid}&t={int(max(0, t))}s"


def _spec(d):
    allowed = set(Spec.__dataclass_fields__)
    return Spec(**{k: v for k, v in d.items() if k in allowed})


class Cancelled(Exception):
    pass


SHORTLIST = 10      # statement mode: videos whose captions are fetched and read (captions are rate-limited)


def answer(query, corpus=None, log=print, per_query=12, res=None, budget_s=900):
    """corpus=None: discover videos for this query (search -> fetch -> index), timing each stage.
    res (optional dict) is filled with structured results for apps: description, interpretation, search, items."""
    res = {} if res is None else res
    from . import fetch
    fetch.new_job(budget_s)
    t0 = time.time()
    hist, times, notes = [], {}, []
    res["notes"] = notes
    log("planning...")
    t = time.time()
    d = planner.plan(query)
    times["plan draft"] = round(time.time() - t, 1)
    if corpus is None:
        from . import discover
        qs = d.get("queries") or [query]
        t = time.time()
        # Statement mode: search wide (metadata + storyboard text are cheap and unthrottled); only the SHORTLIST gets
        # captions and an LLM read, so the expensive part stays bounded.
        found = discover.search(qs, per_query=per_query)
        times["search"] = round(time.time() - t, 1)
        corpus = [x["id"] for x in found]
        log(f"search {qs}: {len(corpus)} videos in {times['search']}s; fetching + indexing...")
        # Captions are never fetched here: statement mode shortlists first (below), moment mode doesn't need them.
        st, tt = discover.prepare(corpus, captions=False, embeddings=d.get("mode") != "statements", log=log)
        times.update(tt)
        corpus = [v for v in corpus if any(x.startswith("index:ok") or x == "index:cached" for x in st[v])
                  or (d.get("mode") == "statements" and "info" in st[v])]
        if d.get("mode") == "statements":
            t = time.time()
            corpus = _shortlist(d, corpus, notes, log)
            times["shortlist + captions"] = round(time.time() - t, 1)
    if fetch.JOB["cancel"]:
        raise Cancelled()
    t = time.time()
    d = planner.refine(query, d, corpus, rounds=2, log=hist)
    times["plan probe+revise"] = round(time.time() - t, 1)
    sp = _spec(d)
    res.update(query=query, mode=sp.mode, description=sp.description or query, interpretation=sp.interpretation,
               n_videos=len(corpus), items=[])
    log(f"plan ready: {sp.description or sp.id}; looking through {len(corpus)} videos...")
    lines = [f"## {sp.description or query}", ""]
    if sp.interpretation:
        lines += [f"*How I read the query:* {sp.interpretation}", ""]
    t = time.time()
    if sp.mode == "statements":
        lines += _statements(sp, corpus, res, log)
    else:
        lines += _moments(sp, corpus, res, log)
    times["find moments"] = round(time.time() - t, 1)
    if fetch.JOB["cancel"]:
        raise Cancelled()
    if fetch.youtube_blocked():
        notes.append("YouTube started refusing requests from this machine (rate limit / bot check), so fetching stopped "
                     "early; results cover the videos fetched before that. Try again later; cached videos stay usable.")
    elif fetch.stopped():
        notes.append(f"Stopped at the {budget_s // 60}-minute time budget; results cover the videos checked so far.")
    if notes:
        lines[2:2] = ["> " + n for n in notes] + [""]
    total = time.time() - t0
    res.update(times=times, total_s=round(total, 1))
    lines += ["", f"<sub>{len(corpus)} videos searched in {total:.0f}s ("
                  + ", ".join(f"{k} {v:.0f}s" for k, v in times.items()) + ").</sub>"]
    return "\n".join(lines), sp, hist + [("times", times, None)]


def _shortlist(d, corpus, notes, log):
    """Statement mode: rank videos by cheap, unthrottled evidence (title/description keywords, party chyrons read
    from storyboards), then fetch captions only for the best SHORTLIST. A caption circuit breaker in fetch stops
    asking YouTube after repeated refusals; those videos are read from title/description/chyrons only."""
    import re
    from . import fetch
    kws = [k for k in (d.get("topic_keywords") or []) if k]
    party = re.compile(r"\((R|D|I)\)|\b[RDI]-[A-Z]{2}\b|SEN\.|REP\.|GOV\.|SPEAKER|PRESIDENT", re.I)

    def score(v):
        info = store.load_info(v) or {}
        text = f"{info.get('title', '')} {info.get('description', '')[:1500]}"
        sc = sum(1 for k in kws if _safe_search(k, text))
        _, rows = store.load_screen(v)
        sc += min(5, sum(1 for t, c, x in rows or [] if c >= 0.5 and party.search(x)))
        return sc
    ranked = sorted(corpus, key=score, reverse=True)[:SHORTLIST]
    # Transcripts come from local Whisper on downloaded audio, never from YouTube's caption endpoint (rate-limited
    # per IP; findings 2026-11-09). Captions already cached are used as-is.
    from . import asr
    need = [v for v in ranked if not asr.has_transcript(v)]
    log(f"shortlisted {len(ranked)} of {len(corpus)} videos; transcribing {len(need)} locally...")
    got, failed, long_ = len(ranked) - len(need), [], 0
    for i, v in enumerate(need, 1):
        if fetch.stopped():
            break
        log(f"transcribing {i}/{len(need)}")
        st = asr.transcribe(v)
        if st in ("ok", "cached"):
            got += 1
            long_ += ((store.load_info(v) or {}).get("duration") or 0) > asr.MAX_AUDIO_S
        else:
            failed.append(st)
    if failed:
        notes.append(f"{len(failed)} of {len(ranked)} shortlisted videos could not be transcribed ({failed[0]}); "
                     "they were read from titles, descriptions and on-screen names only.")
    if long_:
        notes.append(f"{long_} long video(s): only the first {asr.MAX_AUDIO_S // 60} minutes were transcribed.")
    log(f"transcripts for {got}/{len(ranked)} shortlisted videos")
    return ranked


def _safe_search(pat, text):
    import re
    try:
        return re.search(pat, text, re.I)
    except re.error:
        return pat.lower() in text.lower()


def _look_search(sp, vids, per_video=2, log=None):
    """CLIP text->image search for the planner's visual description; best scenes per video checked by the vision
    model (scene-inclusive judge, cached). Yields (vid, t, judgment) for the first confirmed scene per video."""
    from . import embed, judge, visual
    q = embed.text(sp.look)
    from . import fetch
    for i, v in enumerate(vids, 1):
        if fetch.stopped():
            return
        if log:
            log(f"checking video {i}/{len(vids)} by image search")
        for sim, t in visual.scenes(v, q, k=per_video):
            j = judge.judge(v, t, sp)
            if j["match"]:
                yield v, t, j
                break


def _moments(sp, corpus, res, log=print):
    if sp.screen_text:
        log("matching on-screen text")
        preds, cost = pipeline.run(sp, corpus, MOMENT_CFG)
    else:
        preds, cost = {v: {"id": v, "t": None, "method": None} for v in corpus}, {}
    # Videos the text path could not anchor: search for what the moment LOOKS like and let the vision model check.
    # Covers kinds of moments with no shared on-screen text (and planners that pick text anyway).
    rest = [v for v in corpus if preds[v]["t"] is None]
    if sp.look and rest:
        for v, t, j in _look_search(sp, rest, log=log):
            preds[v] = {"id": v, "t": t, "method": "look", "what": j.get("what", "")}
    found = sorted((r for r in preds.values() if r["t"] is not None), key=lambda r: _video(r["id"])[0].lower())
    how = [f"anchor `{sp.screen_text}`" if sp.screen_text else "no on-screen anchor"]
    if sp.look:
        how.append(f"image search for “{sp.look}”")
    if sp.after_screen_text:
        how.append(f"after `{sp.after_screen_text}`")
    if sp.screen_exclude:
        how.append("excluding " + ", ".join(f"`{x}`" for x in sp.screen_exclude[:12]))
    fill = cost.get("visual_fill") or {}
    out = [f"*Search:* {'; '.join(how)}.", ""]
    res["search"] = "; ".join(how)
    for r in found:
        title, ch = _video(r["id"])
        res["items"].append({"id": r["id"], "title": title, "channel": ch, "t": round(r["t"], 1),
                             "start": max(0, int(r["t"] - sp.clip_lead_s)), "link": _link(r["id"], r["t"] - sp.clip_lead_s),
                             "method": r["method"], "note": str(r.get("what") or "")[:160]})
    if not found:
        return out + ["No matching moments found."]
    out.append(f"**{len(found)} videos:**")
    out.append("")
    for i, r in enumerate(found, 1):
        title, ch = _video(r["id"])
        start = r["t"] - sp.clip_lead_s
        how = {"visual": "found by image similarity to the other results",
               "look": f"image search + vision check: {str(r.get('what', ''))[:90]}"}.get(r["method"], "")
        out.append(f"{i}. [{title}]({_link(r['id'], start)}) at **{_ts(r['t'])}**" + (f" · {ch}" if ch else "") +
                   (f" · {how}" if how else ""))
    if fill.get("filled"):
        out += ["", f"<sub>{fill['filled']} of these were found by visual similarity to the others "
                    f"(threshold {fill['threshold']}).</sub>"]
    return out


def _statements(sp, corpus, res, log=print):
    done = [0]

    from . import fetch

    def one(v):
        if fetch.stopped():
            return {"id": v, "segments": [], "note": "skipped: stopped", "llm_calls": 0}
        done[0] += 1
        log(f"reading video {done[0]}/{len(corpus)}")
        try:
            return statements.extract(v, sp, think=False)
        except Exception as e:
            return {"id": v, "segments": [], "note": f"error: {e}", "llm_calls": 0}
    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(one, corpus))
    # Hard filter: the planner and reader sometimes let commentators through. Unless the query itself asks for that
    # kind of person, drop segments whose role is a media role.
    import re
    media = re.compile(r"commentat|pundit|\bhost\b|anchor|analyst|youtuber|podcast|strategist|reporter|journalist|"
                       r"correspondent|influencer|comedian", re.I)
    if not media.search(res.get("query", "")):
        for r in results:
            r["segments"] = [x for x in r["segments"] if not media.search(str(x.get("role") or ""))]
    groups = statements.dedupe(results)
    out = [f"*Who counts:* {sp.who}", ""]
    res["search"] = f"who: {sp.who}"
    for g in groups:
        v, s = g[0]
        title, ch = _video(v)
        res["items"].append({"id": v, "title": title, "channel": ch, "t": s["start"], "start": int(s["start"]),
                             "end": int(s["end"]), "link": _link(v, s["start"]), "method": "statement",
                             "speaker": s.get("speaker"), "role": s.get("role"), "note": str(s.get("quote") or "")[:200],
                             "also": [{"id": v2, "title": _video(v2)[0], "link": _link(v2, s2["start"])} for v2, s2 in g[1:]]})
    if not groups:
        return out + ["No matching statements found."]
    by_speaker = {}
    for g in groups:
        v, s = g[0]
        by_speaker.setdefault(s.get("speaker") or "Unknown speaker", []).append(g)
    out.append(f"**{len(groups)} statements from {len(by_speaker)} people:**")
    out.append("")
    for spk in sorted(by_speaker):
        role = next((s.get("role") for g in by_speaker[spk] for _, s in g if s.get("role")), "")
        out.append(f"- **{spk}**" + (f" ({role})" if role else ""))
        for g in by_speaker[spk]:
            v, s = g[0]
            title, ch = _video(v)
            ev = str(s.get("quote") or "").strip().replace("\n", " ")
            out.append(f"  - [{title}]({_link(v, s['start'])}) {_ts(s['start'])}–{_ts(s['end'])}" +
                       (f" · {ch}" if ch else "") + (f" — “{ev[:140]}”" if ev else ""))
            if len(g) > 1:
                out.append("    also in: " + ", ".join(f"[{_video(v2)[0][:40]}]({_link(v2, s2['start'])})"
                                                       for v2, s2 in g[1:]))
    return out
