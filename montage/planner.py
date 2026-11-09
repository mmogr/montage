"""Planner: natural-language query -> spec, then probe the indexes and revise.

plan(query)                    one-shot draft from the query alone
probe(spec, corpus)            what the draft's anchors/cues actually match in the cached indexes
revise(query, spec, report)    corrected spec
plan_with_probe(query, corpus) draft -> probe -> revise (-> probe -> revise)

The prompts contain only the spec format and general rules: no game/domain knowledge, no hand-written specs.
"""
import json
import os
import re
import statistics

from . import store, statements, vlm
from .spec import Spec

SCHEMA = r"""
A spec tells a video-moment search engine what to look for. Two modes:

mode = "moment": the same kind of moment recurs across many videos of shared content (e.g. many playthroughs of a
  game, reaction videos to one trailer, broadcasts of one match). Located mainly via ON-SCREEN TEXT that the engine
  OCRs from frames sampled every ~10 s (title cards, area names, boss/character name cards, dialogue boxes,
  subtitles, scoreboards), then verified and refined.
  Fields:
    id (snake_case), description (one sentence: the event), game (what the shared content is, e.g. "Title (year)")
    screen_text: Python regex (case-insensitive) matched against single OCR'd text lines. Must match the on-screen
      text that appears AT the event. OCR is noisy: allow small variations (e.g. "hyrule\s*fiel").
    anchor_style: "standalone" if the text is a card/name shown on its own line (title card, boss name card, speaker
      label), so the same words inside a sentence are only mentions; "any" for dialogue/subtitle lines.
    occurrence: "first" (the first time it happens in a video) or "last".
    after_screen_text: optional regex for on-screen text of an event that always happens BEFORE the target (e.g. a
      boss name card before the boss's death scene). Use it when the anchor text (a character's name label, a place)
      appears in several scenes and only the one after that precursor is the target. The anchor must then come
      within after_window_s (default 2400) seconds after the precursor's first appearance.
    screen_exclude: list of regexes; a frame containing any of these is never the event (e.g. title screens, menus,
      stream overlays).
    transcript_strong / transcript_weak / transcript_precursor: regex lists for what streamers SAY at / around /
      shortly before the event (captions are a weak fallback signal).
    comment_keywords: regex list for timestamped viewer comments about the moment.
    clip_lead_s / clip_tail_s: seconds of clip before/after the anchor.
    For a KIND of moment in unrelated videos (e.g. "dogs seeing snow for the first time") there is usually no shared
      on-screen text: leave screen_text empty and describe the moment in look.
    look: one short phrase describing what a single frame AT the event looks like (scene, objects, colors), no
      on-screen text; used for image search when no text marks the event (e.g. "a giant one-eyed spider in a dark cave").

mode = "statements": people saying something, found in news/speech/interview videos via captions + on-screen
  chyrons. Fields:
    id, description, who (precise: which people qualify and which do not; people must themselves be what the query
      names; hosts, pundits, commentators and YouTubers talking ABOUT them do not qualify unless the query asks for
      them), what (what they must be talking about),
    after (YYYY-MM-DD; statements must be on/after), exclude_sources (what kinds of videos/segments do not count),
    topic_keywords: regex list used to window long transcripts (use word stems, e.g. "conced" not "concede").

Both modes also take "queries": 2-4 YouTube search strings likely to return videos that contain the moment or
statements (e.g. playthrough/first-time/reaction videos for a game moment; news clips/press conferences for
statements, aimed at the people speaking rather than at analysis shows, e.g. "<role or name> press conference",
"<name> speech", "<name> reacts"; compilations for a kind of everyday moment). Keep them short, as a person would
type them.
Both modes also take "interpretation": how you read the query and what you deliberately left out.
Interpret the query inclusively: anything a reasonable person asking it would want counts (e.g. "fans reacting to
a trailer" includes delighted, disappointed and mixed reactions). Describe who/what qualifies by kind, never by
naming specific people or videos to exclude. Put your reading of the query, and anything you chose to leave out, in
"interpretation" (one or two sentences) so the user can correct it.
Return a JSON object with "mode" and the fields for that mode only.
"""

PLAN_PROMPT = """You write search specs for a video-moment engine.
{schema}
User query: "{query}"
Today's date: {today}.
Think about what the event looks like on screen and in speech, what text would be visible at that instant, and what
look-alike text could cause false matches (menus, title screens, quest logs, maps, mentions in dialogue, replays,
earlier/later occurrences).
Copy the name of the content (title, edition, year, remake/remaster) into "game" exactly as the query gives it. Do not
substitute a release you know better: releases newer than your knowledge exist.
Write the spec."""

REVISE_PROMPT = """You write search specs for a video-moment engine.
{schema}
User query: "{query}"
Today's date: {today}.

Your draft spec:
{spec}

PROBE REPORT - what the draft actually matches in the cached index of {n} candidate videos:
{report}

Use the report to fix the spec. Typical problems: the anchor regex matches menu/quest-log/HUD text, a title screen,
map labels, or the words inside dialogue (mentions) instead of the event itself; it matches too early in videos
(intros/title screens) or persists for minutes (HUD); it matches nothing because the real on-screen wording differs
(look at the explored text lines for the actual wording). Prefer anchors that appear once, briefly, at the event.
Return the full corrected spec as JSON."""


def _today():
    import datetime
    return datetime.date.today().isoformat()


def _clean(d):
    allowed = set(Spec.__dataclass_fields__)
    out = {k: v for k, v in (d or {}).items() if k in allowed and v is not None}   # null must not become "None"
    out.setdefault("id", "planned")
    out.setdefault("description", "")
    out.setdefault("game", "")
    return out


# Thinking off: with it on, the local 27B model spent its whole token budget reasoning and returned empty content
# for every query (planner eval run 1). Off, a spec comes back in ~45 s.
THINK = os.environ.get("MONTAGE_PLANNER_THINK") == "1"


def plan(query, model=None):
    d = vlm.ask_json(PLAN_PROMPT.format(schema=SCHEMA, query=query, today=_today()), model=model, think=THINK,
                     timeout=1800, max_tokens=16000 if THINK else 4000)
    if not d:
        raise RuntimeError("planner returned no parseable spec")
    return _clean(d)


# ---------------- probe ----------------

def _words(q):
    stop = set("the a an of to in on for and or first time people reacting reaction reactions when with their from "
               "seeing they them see video videos moment moments".split())
    return [w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in stop]


def probe_moment(spec, corpus, query):
    lines = []          # (vid, t, conf, text, rel_pos)
    explore = {}
    qwords = _words(query + " " + spec.get("description", ""))
    for v in corpus:
        meta, rows = store.load_screen(v)
        if not meta:
            continue
        dur = (store.load_info(v) or {}).get("duration") or 1
        by_t = {}
        for t, c, x in rows:
            by_t.setdefault(t, []).append(x)
        pat = spec.get("screen_text") or "$^"
        for t, c, x in rows:
            if c < 0.5:
                continue
            try:
                if re.search(pat, x, re.I):
                    lines.append((v, t, x, t / dur, by_t.get(t, [])))
            except re.error as e:
                return f"screen_text is not a valid regex: {e}"
            xl = x.lower()
            for w in qwords:
                if w in xl and len(x) <= 60:
                    k = re.sub(r"\s+", " ", x.strip().upper())[:40]
                    explore.setdefault(k, set()).add(v)
    vids = {l[0] for l in lines}
    rep = [f"screen_text {spec.get('screen_text')!r}: {len(lines)} matching OCR lines in {len(vids)}/{len(corpus)} videos."]
    from .pipeline import title_pattern
    if spec.get("after_screen_text"):
        try:
            ap = re.compile(spec["after_screen_text"], re.I)
        except re.error as e:
            return f"after_screen_text is not a valid regex: {e}"
        win = spec.get("after_window_s") or 2400
        pre_v = after_v = 0
        for v in corpus:
            _, rows = store.load_screen(v)
            pre = [t for t, c, x in rows or [] if c >= 0.5 and ap.search(x)]
            if pre:
                pre_v += 1
                t0 = min(pre)
                if any(t0 < l[1] <= t0 + win for l in lines if l[0] == v):
                    after_v += 1
        rep.append(f"after_screen_text {spec['after_screen_text']!r}: precursor seen in {pre_v}/{len(corpus)} videos; "
                   f"the anchor matches within {win}s after it in {after_v} videos (only these can be found).")
    tp = title_pattern(spec.get("game", ""))
    tvids = sum(1 for v in corpus if tp and any(re.search(tp, x, re.I) for _, c, x in (store.load_screen(v)[1] or [])))
    rep.append(f"game {spec.get('game')!r} gives title pattern {tp!r} (frames showing it are excluded as title screens/"
               f"overlays); it matches on-screen text in {tvids}/{len(corpus)} videos."
               + (" If that is 0, the game name is probably wrong; check the explored lines below." if not tvids else ""))
    if lines:
        forms = {}
        for l in lines:
            k = re.sub(r"\s+", " ", l[2].strip())[:60]
            forms[k] = forms.get(k, 0) + 1
        rep.append("Most common matched lines: " + "; ".join(f'"{k}" x{n}' for k, n in sorted(forms.items(), key=lambda kv: -kv[1])[:10]))
        pos = [l[3] for l in lines]
        rep.append(f"Position in video (fraction of duration): median {statistics.median(pos):.2f}; "
                   f"{sum(p < 0.03 for p in pos)} of {len(pos)} matches are in the first 3% of a video.")
        # persistence: per video, span of consecutive matches
        spans = []
        for v in vids:
            ts = sorted(l[1] for l in lines if l[0] == v)
            run_start = prev = ts[0]
            for t in ts[1:]:
                if t - prev > 30:
                    spans.append(prev - run_start); run_start = t
                prev = t
            spans.append(prev - run_start)
        rep.append(f"Visible duration per occurrence: median {statistics.median(spans):.0f}s, max {max(spans):.0f}s "
                   f"(long = persistent HUD/menu). Occurrences per video: median "
                   f"{statistics.median([sum(1 for l in lines if l[0]==v) for v in vids]):.0f} frames.")
        standalone = sum(1 for l in lines if len(l[2].strip()) <= 30)
        rep.append(f"{standalone}/{len(lines)} matches are short standalone lines (<=30 chars); the rest are inside longer text.")
        co = {}
        for l in lines:
            for other in l[4]:
                if other != l[2] and 4 <= len(other) <= 40:
                    k = re.sub(r"\s+", " ", other.strip())[:40]
                    co[k] = co.get(k, 0) + 1
        rep.append("Text that appears in the SAME frame as matches (often part of the same card, e.g. a subtitle under a "
                   "name; putting it in screen_exclude would remove those matches): " +
                   "; ".join(f'"{k}" x{n}' for k, n in sorted(co.items(), key=lambda kv: -kv[1])[:12]))
        # The anchor often matches a name that appears in several scenes. Show the text that follows matches so a
        # scene-specific line can be chosen instead (e.g. a line only spoken at the event).
        follow = {}
        hits_by_v = {}
        for l in lines:
            hits_by_v.setdefault(l[0], []).append(l[1])
        for v, ts in hits_by_v.items():
            _, rows = store.load_screen(v)
            grams = set()
            for t, c, x in rows or []:
                if c >= 0.5 and len(x) >= 15 and any(0 <= t - h <= 90 for h in ts) and not re.search(pat, x, re.I):
                    w = re.findall(r"[a-z]{2,}", x.lower())
                    grams |= {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}
            for g in grams:
                follow.setdefault(g, set()).add(v)
        top = [(k, len(vs)) for k, vs in sorted(follow.items(), key=lambda kv: -len(kv[1])) if len(vs) >= 3][:30]
        if top:
            rep.append("Word phrases on screen within 90 s after matches (OCR is noisy, so phrases not whole lines), with "
                       "how many videos show each. If the anchor is a name seen in several scenes, a phrase said only at "
                       "the event is a better anchor: " + "; ".join(f'"{k}" in {n}' for k, n in top))
        excl = [p for p in (spec.get("screen_exclude") or []) if p]
        removed, by = [], {}
        for l in lines:
            for p in excl:
                try:
                    hit = next((o for o in l[4] if re.search(p, o, re.I)), None)
                except re.error as e:
                    return f"screen_exclude {p!r} is not a valid regex: {e}"
                if hit is not None:
                    removed.append(l)
                    by.setdefault(p, []).append(hit.strip()[:40])
                    break
        kept = [l for l in lines if l not in removed]
        rep.append(f"AFTER screen_exclude: {len(kept)} matches survive in {len({l[0] for l in kept})}/{len(corpus)} videos; "
                   f"exclusions removed {len(removed)} matches in {len({l[0] for l in removed})} videos" +
                   ("" if not by else ": " + "; ".join(f"{p!r} removed {len(h)} (e.g. \"{h[0]}\")" for p, h in by.items())) +
                   ". Removing real event frames loses videos; only exclude text that marks NON-event frames.")
    ex = sorted(explore.items(), key=lambda kv: -len(kv[1]))[:20]
    rep.append("EXPLORE - short on-screen lines containing query words, with how many videos show them: " +
               "; ".join(f'"{k}" in {len(vs)} videos' for k, vs in ex))
    return "\n".join(rep)


def probe_statements(spec, corpus, query):
    after = (spec.get("after") or "").replace("-", "")
    infos = [store.load_info(v) or {} for v in corpus]
    before = sum(1 for d in infos if after and (d.get("upload_date") or "0") < after)
    kws = spec.get("topic_keywords") or []
    hits = 0
    for v in corpus:
        T = statements.turns(v)
        if any(re.search(k, x, re.I) for _, _, x in T for k in kws):
            hits += 1
    caps = sum(1 for v in corpus if statements.turns(v))
    chy = {}
    for v in corpus:
        for t, x in statements.onscreen(v):
            if re.search(r"\((R|D|I)\)|\b[RDI]-[A-Z]{2}\b|SEN\.|REP\.|GOV\.|SPEAKER|PRESIDENT", x):
                k = x.strip()[:50]
                chy[k] = chy.get(k, 0) + 1
    return "\n".join([
        f"{len(corpus)} candidate videos; {caps} have captions; {before} were uploaded before 'after'={spec.get('after')!r}.",
        f"topic_keywords {kws}: match captions in {hits}/{caps} captioned videos.",
        "Chyron-like on-screen lines seen: " + "; ".join(f'"{k}" x{n}' for k, n in sorted(chy.items(), key=lambda kv: -kv[1])[:15]),
        "Titles: " + " | ".join((d.get("title") or "")[:70] for d in infos[:25]),
    ])


def probe(spec, corpus, query):
    return (probe_statements if spec.get("mode") == "statements" else probe_moment)(spec, corpus, query)


def revise(query, spec, report, n, model=None):
    d = vlm.ask_json(REVISE_PROMPT.format(schema=SCHEMA, query=query, today=_today(), spec=json.dumps(spec, indent=1),
                                          report=report, n=n), model=model, think=THINK,
                     max_tokens=16000 if THINK else 4000, timeout=1800)
    if not d:
        print("planner: revise returned no parseable spec; keeping the previous one", flush=True)
        return spec
    return _clean(d)


def anchor_coverage(spec, corpus):
    """Videos with at least one confident on-screen line matching screen_text and surviving screen_exclude."""
    pat, excl, n = spec.get("screen_text"), [p for p in (spec.get("screen_exclude") or []) if p], 0
    if not pat:
        return 0
    try:
        re.compile(pat); [re.compile(p) for p in excl]
    except re.error:
        return 0
    for v in corpus:
        meta, rows = store.load_screen(v)
        by_t = {}
        for t, c, x in rows or []:
            by_t.setdefault(t, []).append(x)
        if any(c >= 0.5 and re.search(pat, x, re.I) and not any(re.search(p, o, re.I) for p in excl for o in by_t[t])
               for t, c, x in rows or []):
            n += 1
    return n


def plan_with_probe(query, corpus, rounds=2, model=None, log=None):
    return refine(query, plan(query, model), corpus, rounds, model, log)


def refine(query, spec, corpus, rounds=2, model=None, log=None):
    """Probe a drafted spec against the (cached or freshly indexed) corpus and revise it."""
    hist = [("draft", spec, None)]
    if spec.get("mode") == "statements":
        rounds = 0      # measured (findings run 4): statements probe+revise was worse than one-shot (R 0.75 vs 0.875)
    for r in range(rounds):
        rep = probe(spec, corpus, query)
        new = revise(query, spec, rep, len(corpus), model)
        # Guard: a revision whose anchor reaches far fewer videos is usually an over-fit or broken regex (seen: a
        # cleaned-up copy of one OCR'd sentence matching 0 videos). Keep the previous spec and tell the log.
        if new.get("mode", "moment") == "moment" and spec.get("screen_text"):
            before, after = anchor_coverage(spec, corpus), anchor_coverage(new, corpus)
            if after < max(3, before // 3):
                hist.append((f"revise{r+1}-rejected", new, f"anchor coverage {before} -> {after} videos"))
                continue
        spec = new
        hist.append((f"revise{r+1}", spec, rep))
    if log is not None:
        log.extend(hist)
    return spec


def to_toml(spec):
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return json.dumps(str(v))     # JSON string escaping is valid TOML basic-string escaping
    return "[spec]\n" + "\n".join(f"{k} = {val(v)}" for k, v in spec.items()) + "\n"
