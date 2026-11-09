"""Score pipeline predictions against frame-verified ground truth.

Truth file (eval/truth/<moment>.json):
  {"moment": str, "event": str, "tolerance_s": 5,
   "videos": {"<id>": {"status": "present"|"absent"|"unknown", "t": float|null,
                       "verified": "frames"|..., "note": str}}}
Predictions: {"<id>": {"t": float|null, ...}}  (missing id or t=None means "no clip").

Hit      = prediction with |t - truth_t| <= tolerance on a present video.
Precision = hits / predictions made on labelled (present|absent) videos, plus predictions on
            unlabelled videos that land on a verified negative ("negatives": [t, ...]).
Recall    = hits / present videos.
Predictions on unknown videos are counted separately (cannot be scored).
"""
import json
import re
import statistics


def load_truth(path):
    return json.load(open(path))


def score(truth, preds):
    tol = truth.get("tolerance_s", 5)
    vids = truth["videos"]
    present = {v: x for v, x in vids.items() if x["status"] == "present"}
    labelled = {v for v, x in vids.items() if x["status"] in ("present", "absent")}
    made = {v: p for v, p in preds.items() if p and p.get("t") is not None}
    # verified negatives: times confirmed NOT to be the event (title screens, menus, mentions) make an
    # otherwise-unlabelled prediction scoreable as wrong
    neg = {v for v, p in made.items() if v not in labelled and
           any(abs(p["t"] - n) <= tol + 10 for n in vids.get(v, {}).get("negatives", []))}
    scored = {v: p for v, p in made.items() if v in labelled or v in neg}
    hits, errs, wrong = [], [], []
    for v, p in scored.items():
        if v in present:
            e = p["t"] - present[v]["t"]
            errs.append(abs(e))
            (hits if abs(e) <= tol else wrong).append((v, round(e, 1)))
        elif v in neg:
            wrong.append((v, "verified-negative"))
        else:
            wrong.append((v, "absent"))
    missed = [v for v in present if v not in made]
    abs_errs = sorted(errs)
    q = lambda p: abs_errs[min(len(abs_errs) - 1, int(p * len(abs_errs)))] if abs_errs else None
    return {
        "present": len(present), "absent": sum(1 for x in vids.values() if x["status"] == "absent"),
        "unknown": sum(1 for x in vids.values() if x["status"] == "unknown"),
        "predictions": len(made), "pred_on_unknown": len(made) - len(scored),
        "hits": len(hits),
        "precision": round(len(hits) / len(scored), 3) if scored else None,
        "recall": round(len(hits) / len(present), 3) if present else None,
        "err_median_s": round(statistics.median(abs_errs), 1) if abs_errs else None,
        "err_p90_s": round(q(0.9), 1) if abs_errs else None,
        "wrong": wrong, "missed": missed,
    }


def table(rows, cols=("config", "precision", "recall", "hits", "predictions", "present",
                      "err_median_s", "err_p90_s", "slices", "slice_mb", "wall_s", "http_429")):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


# ---------- statement mode ----------

def _same_speaker(a, b):
    """Surname/token match, tolerant of titles and OCR ("Sen. John Thune" vs "John Thune")."""
    tok = lambda s: {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2} - {"sen", "rep", "gov", "the", "speaker", "president", "vice", "senator", "governor"}
    ta, tb = tok(a), tok(b)
    return bool(ta & tb)


def score_statements(truth, preds, min_overlap_s=5):
    """preds: {vid: {"segments": [{"start","end","speaker"}]}}.
    Hit: same video, temporal overlap >= min(min_overlap_s, 30% of the shorter), same speaker.
    Precision counts predictions on labelled/coarse videos, and on partial videos only where they overlap a
    truth segment of a different speaker or nothing is known (they are skipped). Ambiguous/unknown skipped."""
    vids = truth["videos"]
    tp, fp, ious, wrong = 0, 0, [], []
    matched = set()
    for v, p in preds.items():
        st = vids.get(v, {}).get("status", "unknown")
        if st in ("unknown", "ambiguous"):
            continue
        for s in p.get("segments", []):
            hits = []
            for i, g in enumerate(vids[v]["segments"]):
                ov = min(s["end"], g["end"]) - max(s["start"], g["start"])
                need = min(min_overlap_s, 0.3 * min(s["end"] - s["start"], g["end"] - g["start"]))
                if ov >= need and _same_speaker(s.get("speaker"), g["speaker"]):
                    hits.append(i)
                    if st != "coarse":
                        ious.append(ov / (max(s["end"], g["end"]) - min(s["start"], g["start"])))
            hit = hits[0] if hits else None
            if hits:
                tp += 1
                matched.update((v, i) for i in hits)
            elif st == "partial":
                continue          # unlabelled stretch of a partially labelled video
            else:
                fp += 1
                wrong.append((v, s.get("speaker"), int(s["start"]), int(s["end"])))
    total = sum(len(x["segments"]) for x in vids.values() if x["status"] in ("labelled", "coarse", "partial"))
    missed = [(v, g["speaker"], g["start"]) for v, x in vids.items() if x["status"] in ("labelled", "coarse", "partial")
              for i, g in enumerate(x["segments"]) if (v, i) not in matched]
    neg_videos = [v for v, x in vids.items() if x["status"] == "labelled" and not x["segments"]]
    fp_videos = sorted({w[0] for w in wrong if w[0] in neg_videos})
    return {"predictions_scored": tp + fp, "hits": tp, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "segments_found": len(matched), "segments_total": total,
            "recall": round(len(matched) / total, 3) if total else None,
            "iou_median": round(statistics.median(ious), 2) if ious else None,
            "negative_videos_with_false_hits": f"{len(fp_videos)}/{len(neg_videos)}",
            "wrong": wrong, "missed": missed}
