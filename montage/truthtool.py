"""Build/extend truth files from reviewed proposals.

Workflow:
  1. proposals = {vid: t} from any methods (pooled)
  2. sheets(moment, proposals) -> contact sheets + row index for human review
  3. record(moment, event, decisions) where decisions = {vid: ("present", t) | ("absent", None) | ("unknown", None)}
     merges into eval/truth/<moment>.json (existing present labels are kept unless overridden).
"""
import json
import os

from . import store, verify

TRUTH_DIR = os.path.join(store.ROOT, "eval", "truth")


def sheets(moment, proposals, offsets=(-10, 0, 5, 40)):
    items = sorted(proposals.items(), key=lambda kv: kv[0])
    paths = verify.build_sheets(items, os.path.join(store.DATA, "check", f"truth_{moment}"), offsets=offsets)
    index = [(i, v, round(t, 1)) for i, (v, t) in enumerate(items)]
    return paths, index


def record(moment, event, corpus, decisions, note, tolerance_s=5):
    path = os.path.join(TRUTH_DIR, f"{moment}.json")
    truth = json.load(open(path)) if os.path.exists(path) else {
        "moment": moment, "event": event, "tolerance_s": tolerance_s,
        "videos": {v: {"status": "unknown", "t": None, "verified": None, "note": ""} for v in corpus}}
    for v in corpus:
        truth["videos"].setdefault(v, {"status": "unknown", "t": None, "verified": None, "note": ""})
    for v, (status, t) in decisions.items():
        truth["videos"][v] = {"status": status, "t": None if t is None else round(t, 1), "verified": note, "note": ""}
    json.dump(truth, open(path, "w"), indent=1)
    c = {}
    for x in truth["videos"].values():
        c[x["status"]] = c.get(x["status"], 0) + 1
    return c
