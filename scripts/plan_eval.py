"""Evaluate the planner: query -> spec (one-shot, and with probe+revise) -> pipeline -> score vs truth.
Hand-written specs are run under the same config as the baseline.

python3 -I scripts/plan_eval.py [moment ...]
"""
import copy
import dataclasses
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from montage import evaluate, pipeline, planner, statements, store  # noqa: E402
from montage import spec as specmod  # noqa: E402

QUERIES = {
    "hyrule_field": ("moment", "reactions of people stepping into Hyrule Field for the first time in the Ocarina of Time remake"),
    "gohma": ("moment", "people fighting Gohma for the first time in the Ocarina of Time remake"),
    "zelda_meeting": ("moment", "people's reactions when they meet Princess Zelda for the first time in the Ocarina of Time remake"),
    "deku_tree_death": ("moment", "reactions to the Great Deku Tree dying in the Ocarina of Time remake"),
    "gop_midterm_reactions": ("statements", "republican politicians reacting to the midterm results"),
}
CORPUS = {"moment": "eval/corpus/oot_remake_v1.tsv", "statements": "eval/corpus/politics_midterms_v1.tsv"}
# No slice downloads: storyboard anchors (+7 s calibration) + learned context + vision seed check + vision gate.
MOMENT_CFG = pipeline.Config("plan_eval", transcript=False, comments=False, chapters=False, heatmap=False,
                             refine=False, context=True, vlm_seeds=True, vlm_gate=True, sb_offset_s=7.0)
TOL = 10


def run_moment(sp, corpus, truth):
    preds, cost = pipeline.run(sp, corpus, MOMENT_CFG)
    t = copy.deepcopy(truth)
    t["tolerance_s"] = TOL
    s = evaluate.score(t, preds)
    return {k: s[k] for k in ("precision", "recall", "hits", "predictions", "present", "err_median_s", "wrong", "missed")}, preds


def run_statements(sp, corpus, truth):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda v: statements.extract(v, sp, think=False), corpus))
    preds = {r["id"]: r for r in res}
    s = evaluate.score_statements(truth, preds)
    return {k: s[k] for k in ("precision", "recall", "hits", "segments_found", "segments_total",
                              "negative_videos_with_false_hits", "wrong", "missed")}, preds


def main():
    names = sys.argv[1:] or list(QUERIES)
    os.makedirs(os.path.join(ROOT, "specs", "planned"), exist_ok=True)
    out_path = os.path.join(ROOT, "eval", "results", "planner_eval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for m in names:
        mode, q = QUERIES[m]
        corpus = store.load_corpus(os.path.join(ROOT, CORPUS[mode]))
        truth = evaluate.load_truth(os.path.join(ROOT, "eval", "truth", f"{m}.json"))
        runner = run_statements if mode == "statements" else run_moment
        row = {"query": q}
        # baseline: hand-written spec
        hand = specmod.load(os.path.join(ROOT, "specs", f"{m}.toml"))
        row["hand"], _ = runner(hand, corpus, truth)
        # planner variants
        for variant, rounds in (("oneshot", 0), ("probe", 2)):
            t0 = time.time()
            log = []
            d = planner.plan_with_probe(q, corpus, rounds=rounds, log=log)
            d["id"] = f"planned_{m}_{variant}"
            open(os.path.join(ROOT, "specs", "planned", f"{m}__{variant}.toml"), "w").write(planner.to_toml(d))
            json.dump([(n, s, r) for n, s, r in log], open(os.path.join(ROOT, "specs", "planned", f"{m}__{variant}.log.json"), "w"), indent=1)
            try:
                sp = specmod.Spec(**d)
                score, _ = runner(sp, corpus, truth)
            except Exception as e:
                score = {"error": f"{type(e).__name__}: {e}"}
            score["plan_s"] = round(time.time() - t0)
            row[variant] = score
            print(m, variant, {k: v for k, v in score.items() if k not in ("wrong", "missed")}, flush=True)
        print(m, "hand", {k: v for k, v in row["hand"].items() if k not in ("wrong", "missed")}, flush=True)
        results[m] = row
        json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
