"""Moment spec: everything moment-specific lives here, never in code.
Plain strings/lists/numbers only, so a model can write one later."""
import tomllib
from dataclasses import dataclass, field


@dataclass
class Spec:
    id: str
    description: str
    game: str
    # discovery (unused by eval runs on a frozen corpus)
    queries: list = field(default_factory=list)
    since: str = ""
    min_duration_s: int = 0
    title_exclude: list = field(default_factory=list)
    # cheap text cues
    transcript_strong: list = field(default_factory=list)
    transcript_weak: list = field(default_factory=list)
    transcript_precursor: list = field(default_factory=list)
    precursor_window_s: int = 600
    comment_keywords: list = field(default_factory=list)
    chapter_pattern: str = ""
    # deterministic on-screen anchor
    screen_text: str = ""          # regex matched against OCR'd on-screen text
    occurrence: str = "first"      # first | last | all
    screen_exclude: list = field(default_factory=list)  # a frame that also shows any of these is not the event
    after_screen_text: str = ""    # ordering: the event comes after the first on-screen match of this (e.g. a boss card)
    after_window_s: int = 2400     # ... and within this many seconds of it
    anchor_style: str = "any"      # standalone = only short lines that are essentially the anchor (title/name cards)
    refine_before_s: int = 15      # exact-refinement slice around a screen-index hit
    refine_after_s: int = 15
    search_before_s: int = 110     # slice around a cue-based guess when no screen hit
    search_after_s: int = 50
    # statement mode ("<who> saying <what>")
    mode: str = "moment"            # moment | statements
    who: str = ""
    what: str = ""
    after: str = ""
    exclude_sources: str = ""
    topic_keywords: list = field(default_factory=list)
    # output
    clip_lead_s: int = 20
    clip_tail_s: int = 14
    interpretation: str = ""   # planner's reading of the query, shown to the user so they can correct it
    look: str = ""            # what the event looks like, for CLIP text->image search when there is no on-screen text


def load(path):
    d = tomllib.load(open(path, "rb"))
    flat = {}
    for section in d.values() if all(isinstance(v, dict) for v in d.values()) else [d]:
        flat.update(section)
    return Spec(**flat)
