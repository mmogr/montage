# Evaluation methodology

**Corpus.** `corpus/oot_remake_v1.tsv` is a frozen list of 62 launch-window (uploaded ≥ 2026-11-05)
Ocarina of Time Remake playthroughs. Their metadata, comments and captions are cached in `data/videos/` and
are never re-fetched, because YouTube search results and auto-captions change over time and a moving
corpus makes runs incomparable.

**Truth is the game event, not the reaction.** Each `truth/<moment>.json` gives, per video, the time
(seconds) of a deterministic in-game event, such as the first frame an area title card is visible. How long
the clip runs before and after that event is a system parameter, not part of the truth.
Status is `present`, `absent` (the video clearly never reaches the moment, e.g. a Day 2 VOD that starts
later in the game) or `unknown` (not established). `unknown` videos are excluded from both precision and
recall and are reported as a count.

**Verification.** A `present` label requires a human (or reviewer) to have looked at frames around the
time. OCR and the system may *propose* times, but nothing becomes truth from OCR alone, so the truth does not
share the system's failure modes. Exact seconds come from 1 fps OCR on a short slice, confirmed against a
contact sheet.

**Pooling.** Truth is the union of verified finds from every method tried (prototype run, storyboard
index, manual inspection), as in TREC-style pooled evaluation. Recall is therefore relative to the pool and is
optimistic until the pool is exhausted. The `unknown` count shows how much of the corpus is unlabelled.

**Scoring** (`montage/evaluate.py`): a hit is a prediction within ±`tolerance_s` (default 5 s) of the truth
time on a present video. Precision = hits / predictions on labelled videos; recall = hits / present.
Timing error is reported as median and p90 absolute seconds. Cost columns: slices downloaded, MB, wall
time, HTTP 429s.

**Ablations.** `python3 -I mt.py ablate SPEC --corpus … --truth …` runs every signal configuration in
`montage/pipeline.py:CONFIGS` and writes one row per configuration to `results/<moment>__ablation.json`.

**Verified negatives.** A truth entry may list `negatives: [t, …]`: times a reviewer confirmed are
*not* the event (title screens, quest menus, map labels, in-dialogue mentions). A prediction that lands on
one counts as a precision error even when the video's own status is `unknown`.

## Statement mode ("<who> saying <what>")

`truth/gop_midterm_reactions.json` labels *segments* (speaker, start, end) per video, not single event times.
Statuses: `labelled` (fully read from captions), `coarse` (no captions; whole-video speech labelled from
title/description/chyrons), `partial` (long livestream searched, not fully read: only overlaps are scored),
`ambiguous` (WHO genuinely unclear, e.g. a former official turned commentator: skipped), `unknown` (skipped).
Labels were written from captions/chyrons/metadata *before* any system output was seen; two later corrections
are marked `LABEL FIX` with their reason. `duplicates` lists known re-uploads of the same speech.

Scoring (`evaluate.score_statements`): a prediction is a hit when it is in the same video, overlaps a truth
segment by ≥ min(5 s, 30% of the shorter), and names the same speaker (token match). One prediction may cover
several truth segments. Also reported: boundary IoU (median), and how many labelled-negative videos received
any false hit.

**Dev vs held-out.** `corpus/politics_midterms_v1.tsv` has been used for error analysis and fixes, so it is a
*development* set. `corpus/politics_midterms_heldout_ids.txt` (15 videos) is untouched and is the test set.
Most of it currently lacks captions because of YouTube caption rate-limiting (HTTP 429); re-fetch before using.
