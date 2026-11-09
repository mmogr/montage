# montage

Describe a moment, get the timestamps.

```
./montage.sh ask "dogs seeing snow for the first time"
```

montage searches YouTube, looks through the videos it finds, and returns the exact points where that moment
happens, as links that open at the right second:

> **21 videos:**
>
> 1. [Cutest Puppy First Time Seeing Snow! (Hilarious Reaction)](https://www.youtube.com/watch?v=9BRyB_XyxeA&t=76s) at **1:21** · Herc Bond
> 2. [Dogs Getting The Zoomies For Their First Snow Day Of The Year | The Dodo](https://www.youtube.com/watch?v=cAAw4QAgaMo&t=2s) at **0:07** · The Dodo
> 3. [Dogs Reacting to First Snow   Funniest Dog Reaction](https://www.youtube.com/watch?v=NiTVEpugGNE&t=214s) at **3:39** · Funny Pet's Life
> …

It works for different kinds of moments:

- **A shared moment across many videos:** "streamers stepping into Hyrule Field for the first time", "first time
  fighting Gohma". The same event shows up in dozens of playthroughs at different times.
- **A kind of moment:** "dogs seeing snow for the first time", "cats knocking things off tables".
- **Who said what:** "republicans reacting to the midterm results". It finds the segments where a qualifying
  person is actually speaking, not a host talking about them.

Everything runs locally. No cloud APIs.

## How it works

1. **Plan.** A local model turns your sentence into a search plan: YouTube queries, what text appears on screen at
   the moment, what it looks like, who counts as a speaker. It tries the plan against the videos it found and
   revises it.
2. **Gather.** For each video it pulls metadata and YouTube's storyboard thumbnails (about one frame per 10 s),
   and reads the on-screen text with macOS Vision. No video is downloaded. Statement queries also transcribe
   audio locally with Whisper.
3. **Find.** It locates the moment from on-screen text, from what the frames look like (CLIP), or from the
   transcript. A vision model then checks each candidate frame. Confirmed moments in some videos help find the
   same moment in the others.
4. **Answer.** You get a list of videos with timestamped links, plus a note on how it read your query, so you can
   tell when it misunderstood.

A search takes about 4–10 minutes, mostly spent on local model calls. Everything is cached per video, so
repeat searches over the same videos are much faster.

## Requirements

- macOS on Apple Silicon (Vision OCR, mlx-whisper)
- [uv](https://docs.astral.sh/uv/) and `ffmpeg`
- gglib running with a vision-capable model (default `Qwen3.8-27B`)

Build the two OCR helpers once:

```
mkdir -p bin
swiftc -O scripts/ocr_storyboard.swift -o bin/ocr_storyboard
swiftc -O scripts/ocr_find.swift -o bin/ocr_find
```

Python dependencies (yt-dlp, open_clip, mlx-whisper, …) are pulled in by `uv` on first run. The first search also
downloads the CLIP and Whisper models.

## Usage

**The app.** A local page with a search box, live progress, and results you can play in place:

```
./montage.sh serve          # then open http://127.0.0.1:8766
```

Searches run one at a time and are saved, so you can close the page and come back later.

**The command line:**

```
./montage.sh ask "people meeting Princess Zelda for the first time in the Ocarina of Time remake" --out zelda.md
```

**Configuration** (environment variables):

| Variable | Default |
|---|---|
| `MONTAGE_LLM_BASE` | `http://127.0.0.1:8080/v1` (gglib's OpenAI-compatible endpoint) |
| `MONTAGE_VLM_MODEL` | `Qwen3.8-27B` |
| `MONTAGE_LLM_KEY` | read from `gglib config settings reveal proxy-api-key` |
| `MONTAGE_ASR_MODEL` | `mlx-community/whisper-large-v3-turbo` |

## How well it works

Results are measured, not eyeballed. `eval/` has hand-verified truth for several moments across 62 playthroughs
and a set of political news videos. `docs/findings.md` logs every result that shaped the design, including the
things that didn't work.

Roughly:

- Moments marked by on-screen text (area names, boss cards, character introductions): precision 1.0, recall
  0.88–1.0 on the labelled set.
- Kinds of moments on fresh queries: about 85–90% of returned clips are right.
- Who-said-what: attributions are mostly right, but how many statements it finds varies between runs.

Known gaps: a moment buried inside a scene that repeats (the same character talking at several points) is hard
to pin down without text; long compilations are under-searched; timestamps are good to about ±10 s.

## Layout

```
montage/      engine: planner, discovery, pipeline, vision checks, transcription, app
scripts/      evaluation and indexing scripts, OCR helpers (Swift)
eval/         corpora, truth labels, results
docs/         findings log and background research
specs/        hand-written and planner-generated search specs
experiments/  measured ideas that were not adopted
```
