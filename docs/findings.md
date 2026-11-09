# Findings log

Dated, measured results that shaped the design. Numbers are from `eval/` on the frozen corpora; see
`eval/README.md` for methodology.

## 2026-11-09 — visual (CLIP) propagation and text-free cold start

Storyboard frames (≈10 s apart) are embedded with open_clip ViT-B-32 (`scripts/embed_storyboards.py`).
`montage/visual.py` propagates confirmed anchors from seed videos to others (frames vote for
anchor = frame time − offset; support = distinct seed videos agreeing) and self-calibrates the similarity
threshold by leave-one-seed-out on the seeds (no truth used). Calibration **abstains** when seeds cannot predict
each other (best correct < 3 or correct ≤ 2 × wrong).

**Seeded from the text pipeline's own anchors** (`scripts/visual_eval.py`): self-calibrated thresholds HF 0.90,
Gohma 0.90, Zelda 0.94, Deku 0.93 (all pass the abstain rule). Visual-only P 0.97–1.0, R 0.75–0.86. Text + visual
fill keeps P 1.0 and raises Hyrule Field recall 0.861 → 0.944. Gohma's two text misses are not recovered.

**Cold start, no on-screen text at all** (description → CLIP text→image search → vision check → seeds →
calibrated propagation):

| variant | Hyrule Field | Deku Tree death | Gohma |
|---|---|---|---|
| v1: top-1 frame/video, scene-inclusive 3-frame judge (`coldstart_eval.py`) | 29 seeds; ±30 s P 1.0 R 0.81 (0.86 with onset); ±10 s P 0.74 | 0 predictions (calibration picked an empty threshold; now an explicit abstain) | same |
| v2: top-3 scenes/video, strict 8-frame "event itself" judge (`coldstart_v2.py`) | 13 seeds; ±30 s P 1.0 R 0.36 | **abstain** | **abstain** |

- Distinctive-looking moments (an open field) cold-start well: right videos, P 1.0 at ±30 s. Timing is limited by
  the 10 s storyboard grid (seed error spread ≈ ±10 s; onset alignment moves the median +5 → +1.5 s). Tight
  timing needs the slice refine step, not better seeding.
- **Recurring look-alike scenes defeat single-pass visual judging.** The Deku Tree appears at the game start, before
  the dungeon and at its death; Gohma-like spiders appear long before the boss. The vision model, reading
  240×135 thumbnails, confidently rationalised wrong scenes ("frame 3 shows his farewell words…") even with a
  strict prompt; the strict prompt mainly cost recall on the easy case. Calibration caught both failures and
  abstained rather than emitting wrong clips.
- **Ordering disambiguates.** CLIP-only test: "the tree scene that comes after the strongest giant-spider scene"
  lands at a consistent ≈ −300 s from the truth marker in 15/22 videos (the truth marker is one line deep in a
  ~5-minute farewell speech, so the scene start is that far before it). A `visual_precursor` / "after X" spec
  field is the text-free analogue of `transcript_precursor`. Not yet built into the pipeline.
- Kept: v1 judge for cold start of distinctive scenes; the abstain rule; onset alignment. Dropped: the strict
  grid judge as a seed filter.

## 2026-11-09 — planner (query → spec)

Local Qwen3.8-27B via gglib. Variants: one-shot draft; draft + probe/revise (the probe runs the draft against
the cached on-screen-text index and reports what it actually matches). Same pipeline config as the hand-written
specs (`scripts/plan_eval.py`: storyboard anchors + learned context + vision seed check + vision gate, no slice
refine), ±10 s.

Failures found and fixed on the way:
- Thinking on: the model spent the whole token budget reasoning and returned empty content for every query
  (run 1 measured nothing). Planner now runs with thinking off (~45–60 s per call) and raises on an empty spec.
- Regex backslashes in JSON (`"\s"`) made replies unparseable; `vlm.parse_json` repairs them and restores
  `\b` that JSON reads as backspace; one retry at temperature 0.4.
- The model substituted a game it knows ("Tears of the Kingdom", "Ocarina of Time (2011)") for the 2026 remake.
  `game` drives automatic title-screen exclusion, so the prompt now says to copy the name verbatim and the probe
  reports how many videos the game's title pattern actually appears in.
- Gohma probe-revise made things worse (R 0.56 → 0.19): the report listed the boss card's own subtitle
  ("PARASITIC ARMORED ARACHNID") as co-occurring text and the model excluded it. The probe now reports what
  `screen_exclude` removes (count + examples) and frames co-occurring text as likely part of the same card; two
  revise rounds so the model sees the effect of its own edits.

Results, run 3 (`eval/results/planner_eval_run3.json`; ±10 s; same pipeline config for all three):

| query | one-shot | probe + 2 revisions | hand-written |
|---|---|---|---|
| Hyrule Field (36 present) | P 0.79 R 0.42 | **P 0.85 R 0.61** (run 2: R 0.69) | P 0.79 R 0.42 |
| Gohma (16) | P 1.0 R 0.56 | P 1.0 R 0.56 | P 1.0 R 0.56 |
| Zelda meeting (5) | R 0 (no matches) | P 0.8 R 0.8 | P 0.8 R 0.8 |
| Deku Tree death (23) | P 0 (25 wrong) | R 0 | P 0.96 R 0.96 |
| GOP midterm reactions (16 segments) | P 1.0 R 0.69 | P 0.91 R 0.63 | P 1.0 R 0.94 |

- The probe is what makes the planner work for on-screen-text moments: it never lost to one-shot on moments,
  rescued Zelda (0 → 0.8) and beat the hand spec on Hyrule Field (it accepted OCR variants like "HYRULE FIELD D").
  Run-to-run variance is visible (HF probe R 0.69 vs 0.61 between runs).
- **Deku Tree death fails** for the same reason the cold start does: the planner anchors on the "GREAT DEKU TREE"
  speaker label, which appears in every Deku Tree conversation, and the vision gate cannot tell those scenes
  apart. The hand spec works only because it anchors on a line from the death speech ("…coming to an end").
  Events that are a specific point inside a recurring scene need a scene-specific line, an ordering constraint
  ("after Gohma"), or speech.
- Politics: the planner read the query more narrowly than the truth labels: one-shot required "party
  representatives" (dropping candidates); the probe version saw candidate titles and excluded concession speeches
  by naming the candidates. Interpretation, not mechanics. The schema now asks for inclusive interpretation, bans
  name-based exclusions, and returns an `interpretation` field for the user to correct.
- After run 3: revisions whose anchor coverage collapses (< max(3, ⅓ of before)) are rejected; the title
  pattern drops edition words (remake/3D/HD/…); the probe reports recurring word phrases that follow anchor matches.

Run 4 (`planner_eval_run4.json`; Deku + politics after the inclusive-interpretation rule, coverage guard,
edition-stripped title pattern and follow-on phrase probe):

| query | one-shot | probe + 2 revisions | hand-written |
|---|---|---|---|
| Deku Tree death | R 0 (planner chose no anchor: "no distinct text at the death") | P 0 (33 wrong; name-label anchor) | P 0.96 R 0.96 |
| GOP midterm reactions | **P 1.0 R 0.875** | P 0.92 R 0.75 (1 Democrat false hit) | P 1.0 R 0.875 (run 3: 0.94) |

- Politics: with inclusive interpretation the one-shot planner spec **matches the hand spec** (same 14/16; only
  misses are a 2.4 h livestream segment and one caption-less concession). The probe version is worse: its
  revision broadened `who` to "strategists/analysts" and added chyron names as keywords. The statements probe adds
  noise rather than signal → **use one-shot for statements mode, probe+revise for moment mode** until the
  statements probe is redesigned.
- Deku stays unsolved by the planner (expected; see run 3 note). Stopped prompt-tuning against it to avoid fitting
  the test; the fix is structural (ordering constraint / scene-specific line / speech).

## 2026-11-09 — ordering constraint (`after_screen_text`)

Built: the anchor must come within `after_window_s` after the first standalone on-screen match of a precursor
(e.g. a boss card); no precursor → no anchor. The probe reports precursor coverage. Verified it filters correctly.
It does **not** rescue Deku Tree death: with the precursor "GOHMA" card (readable in only 17/62 storyboards), the
first "GREAT DEKU TREE" label after it lands +975…+1648 s after the death on the 3 present videos that get one, so the
speaker label is simply not storyboard-readable during the farewell speech. The planner's revisions also excluded
the death-scene lines themselves ("is dead", "last words"). Left as a known limitation: events inside a recurring
scene need a scene-specific line (script knowledge) or a visual precursor at scene precision plus refine. Planner
`screen_exclude` and `interpretation` are surfaced in the output so a human can spot self-sabotaging exclusions.

## 2026-11-09 — visual fill in the pipeline; first end-to-end run

`Config.visual_fill` (config `screen+context+refine+vseed+vgate+vfill`): after the gate, videos with no anchor are
filled by CLIP propagation from the run's own screen anchors, threshold self-calibrated, abstain if inconsistent.

| moment | before (vseed+vgate) | + visual fill | fills on unlabelled videos |
|---|---|---|---|
| Hyrule Field | P 1.0 R 0.861 | **P 1.0 R 0.944** | — |
| Gohma | P 1.0 R 0.75 | **P 1.0 R 0.875** | — |
| Zelda meeting | P 1.0 R 0.8 | **P 1.0 R 1.0** | 19, all 19 confirmed on contact sheets (scene + name card within ~10 s) |
| Deku Tree death | P 1.0 R 0.913 | **P 1.0 R 1.0** | 11 (not yet reviewed) |

No slice downloads in the fill step; it runs in seconds on cached embeddings.

**End to end** (`mt.py ask "QUERY" --corpus …`, `montage/answer.py`): sentence → planner (probe+revise) →
pipeline (no slice downloads) → markdown with titles, timestamped links, the planner's interpretation and its
exclusions. First untuned query, "streamers getting the Kokiri Sword": 219 s, 2 videos returned.
- Both returned videos are right (the sword in the equipment menu just after getting it), but the anchor is a
  proxy: the storyboards never OCR "You got the Kokiri Sword"; "Kokiri Sword" is only readable on the equipment
  screen, in 7/62 videos.
- **The vision gate was inconsistent on near-identical inputs**: of 5 frames all showing the same equipment screen
  shortly after the pickup, it accepted 2 ("inventory screen showing the Kokiri Sword") and rejected 3 ("menu, not
  a scene"). With only 2 survivors, visual fill had too few seeds. The gate is a reliability weak point on
  borderline frames; a planner-supplied note on whether a menu/card frame IS the event would help.
- The planner's description contained a game-knowledge error ("receives the Kokiri Sword from Saria").

## 2026-11-09 — from scratch: sentence → YouTube search → index → answer

`mt.py ask "QUERY"` with no corpus: planner drafts the spec + search queries → `discover.search` (yt-dlp
ytsearch) → `discover.prepare` (metadata, storyboard OCR, CLIP; captions only for statement mode) → probe/revise
on the fresh videos → find moments → markdown. Videos the text path cannot anchor get CLIP image search for the
planner's `look` description + vision check (`answer._look_search`), so kinds of moments without shared on-screen
text work even when the planner picks a text anchor anyway.

"dogs seeing snow for the first time" (domain never seen before):
- Cold run (26 new videos): 260 s total: plan 26 s, search 7 s, metadata 32 s, storyboard OCR 116 s, CLIP 16 s,
  probe+revise 56 s. No throttling. Only 1 result: the planner chose a title-overlay text anchor.
- With the look fallback (24 videos, mostly cached): 394 s, **21/24 videos returned**. Contact-sheet review: ~18
  clearly dogs in snow at the timestamp; 1 wrong video (Houston Zoo: a capybara/sea lion the vision model called
  "a brown dog"); 1–2 at an off moment (dog indoors). Timestamps are "a good dog-in-snow moment", not necessarily
  the first contact; for compilations any moment is valid.
- Cost is dominated by the local model (planning ~1.5–2.5 min, vision checks ~3 min), not by fetching/indexing
  (~3 min cold for 25 videos, seconds when cached).

## 2026-11-09 — local app (`./montage.sh serve`, http://127.0.0.1:8766)

Job queue (one at a time) + progress log + saved jobs (`data/jobs/`) + results page (thumbnails that play at the
timestamp, copy-markdown). First query through the app, "cats knocking things off tables" (32 fresh videos):
520 s total (plan 29, search 7, metadata 44, OCR 23, CLIP 10, probe+revise 74, find moments 333). 11 results; by
contact sheet ~9 clearly a cat pushing things off a counter/shelf within ~10 s, 2 borderline (a cat stealing food in a
news story; an explainer illustrated with AI images). Recall is the weak side: 20/31 videos returned nothing, many of
them compilations that certainly contain the moment — the look search checks only the top-2 CLIP scenes per video.

## 2026-11-09 — statement mode through the app: caption throttling

First politics query from the user through the app stalled: 40 videos, captions retried per video with backoff
against a per-IP limit → 8+ min, 1 caption. Fixed in the fetch layer (not the engine): caption circuit breaker (stop
asking after 2 refusals; 1 try per video), job time budget (15 min) with a shared stop signal, cancel, statement-mode
shortlist (search wide; metadata + storyboard text are unthrottled; captions + LLM read only for the best 10),
degraded-mode note in results.
- Rerun 1 (16 videos, 10 shortlisted): 394 s, captions 6/10, 2 statements (Tillis, Trump), both right.
- Rerun 2 (35 videos, speaker-aimed queries, 10 shortlisted): 323 s, breaker tripped after 7.5 s, captions 4/10,
  3 statements: Trump (right), DeSantis (quoted inside a commentary video), and a YouTuber commentator (wrong: fails
  `who`).
- Audio is NOT throttled: the m4a for a caption-refused video downloaded in 2.4 s (11 MB). Local transcription of
  the shortlist (≤10 clips, cached) would remove the dependency that is now the main limit on statement recall.

## 2026-11-09 — statement mode with local Whisper (no YouTube caption requests)

Shortlisted videos without cached captions: audio download (smallest format, whole track; never
`--download-sections`, which streams at ~real time) → mlx-whisper large-v3-turbo (≈65× real time) → json3. Rerun of
the user's query: 611 s total, **transcripts 10/10** (5 local, 137 s incl. downloads), **6 statements** (vs 2 and 3
without). Right: Trump, Andy Barr (election-night victory remarks), MTG (on-record quote). Wrong/doubtful: Ben
Shapiro (commentator: planner again put "commentators" in `who`), "Susan Collins" (an anchor paraphrasing her,
third person), "Darlene Graham (R-SC)" (garbled name, likely an anchor's narration). Fixes after this run: explicit
third-person / reported-speech rule and ASR-name caution in the reader prompt; hard filter dropping media roles
unless the query asks for them. Rate limiting is no longer the limit for statement mode; attribution precision is.
- Rerun after the attribution fixes (different search results: planner queries vary run to run): 496 s, 3
  statements, all from real politicians (2× DeSantis inside a commentary video that plays his clips; Trump). No
  commentators or reported speech got through (planner still wrote "pundits" into the description; the role filter
  removed them). Precision up, recall down/variable: which videos the searches surface differs a lot between runs.
