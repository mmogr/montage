# Prior art: speaker attribution & statement retrieval in news video (no face recognition)

Research pass 2026-11-09 (web research by a subagent; figures are as reported by the cited sources,
some self-reported — see Caveats). Purpose: inform the "who said what about X" path of montage.

## Key numbers
- **Written (chyron/overlay) names vs spoken names** — Poignant et al., Interspeech 2013
  (https://www.isca-archive.org/interspeech_2013/poignant13_interspeech.pdf):
  written-name extraction P 95.7% / R 97.7% (OCR CER 2.6%); spoken names P 73.5% / R 50% (ASR WER 16.9%).
  A written name co-occurs with that person *speaking* 95.8% of the time; a spoken name only 31.5%.
  Written names can name at most ~73% of speakers (oracle) — ~1/4 of speakers are never named on screen.
  Main error source: anchors/journalists (~45% of speech time) are almost never named in overlays.
- **MediaEval Multimodal Person Discovery 2015/16, REPERE** (https://ceur-ws.org/Vol-1436/Paper3.pdf,
  https://hal.archives-ouvertes.fr/hal-01690581): no prior name list; names must come from OCR or ASR and
  every returned name needs an *evidence* shot. Baseline 78.35% EwMAP; participants ~86–90% (self-reported).
- **Transcript-only speaker naming**: Liu & Liu NAACL 2007 (https://aclanthology.org/N07-2026.pdf) 68.9%
  (oracle 79.1%); MediaSum-based benchmark best P 80.3% (https://arxiv.org/abs/2407.12094); LLM mapping of
  anonymous diarization IDs to names 74% exact match, main error = name on the adjacent segment
  (https://arxiv.org/html/2604.21202). Evidence-required prompting with "Unknown" fallback:
  https://arxiv.org/pdf/2606.24066. DiarizationLM: https://arxiv.org/abs/2401.03506.

## Systems
- **Internet Archive TV News Archive + Third Eye** — caption search, 60 s clips; chyron OCR for BBC/CNN/Fox/MSNBC
  with links back to clips (https://archive.org/services/third-eye.php).
- **GDELT** — caption search, OCR/ASR "AI Explorer", 1 frame / 4 s Visual Explorer; found OCR search beat visual
  search for TV news because chyrons repeat the name all segment long
  (https://blog.gdeltproject.org/lessons-from-the-trenches-how-ocr-text-search-beat-out-visual-search-for-television-news/);
  speaker labels by diff-aligning captions with CNN rush transcripts (demonstrated, not evaluated).
- **Stanford Cable TV News Analyzer** — face-based identity, no chyrons/speaker ID; explicitly warns
  on-screen ≠ speaking (https://arxiv.org/abs/2008.06007).
- **Twelve Labs** — joint video/audio/text embeddings; useful for topical search, not identity.

## Statements, stance, segmentation
- Quote attribution in written news (Quotebank/Quobert, 86.9%):
  https://dlab.epfl.ch/people/west/pub/Vaucher-Spitz-Catasta-West_WSDM-21.pdf
- Claim detection from debate transcripts is hard (CLEF CheckThat!, https://checkthat.gitlab.io/).
- Zero-shot LLM stance is competitive but target-dependent and shows political bias
  (https://arxiv.org/pdf/2212.14548, https://arxiv.org/pdf/2404.12171).
- No recent broadcast-specific topic segmentation; TreeSeg (https://arxiv.org/abs/2407.12028) is closest.
  Cheap boundaries: caption `>>>` = story change, `>>` = speaker change, chyron change.
- "Who said what about topic X" over transcripts: no direct prior work found — novel component.

## Near-duplicates
- Audio fingerprinting with offset recovery: Panako (https://archives.ismir.net/ismir2014/paper/000122.pdf), Chromaprint.
- Text (full-story duplicates) + keyframes (partial overlaps) combine best (TRECVID survey).

## `>>` reliability
- `>>`/`>>>` come from human broadcast captioning conventions; YouTube ASR emitting `>>` is undocumented.
  Captioners apply speaker marks inconsistently (Liu & Kubala 1999,
  https://www.isca-archive.org/eurospeech_1999/liu99_eurospeech.pdf); >80% of true speaker changes coincide
  with pauses. → measure `>>` reliability on our own labelled set; combine with pauses + chyron changes.

## Adopt (ranked)
1. Chyron OCR as primary identity, time-aligned to caption turns; parse party/state ("(R) NEW YORK");
   fuzzy-correct names against a canonical list.
2. Evidence-required naming (chyron frame or transcript span like "Senator X, welcome"); else "Unknown".
3. Identify hosts from channel/show metadata and exclude their turns.
4. LLM second pass linking names to turns, quoting evidence; watch for adjacent-segment errors.
5. Cheap eval data: MediaSum CNN transcripts have speaker labels.

## Pitfalls
- Chyron ≠ speaking (cutaways, anchor intros, paraphrase) → require first-person plausibility of the turn.
- ~25% of speakers unnameable from chyrons.
- Storyboard tiles are small; measure OCR error on real tiles.
- Some chyrons are headlines, not names; re-uploads may carry chyrons from other moments.

## Caveats
MediaEval participant scores, Quobert's 86.9%, and an AWS case study about C-SPAN face matching are
self/vendor-reported. Figures were gathered by a research subagent and not independently re-verified.
