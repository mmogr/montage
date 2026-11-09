"""Does visual propagation (CLIP on storyboards) find moments the text pipeline missed?

Seeds: the system's own verified screen-text anchors (best config results), never truth.
Reports, per moment and threshold: visual-only precision/recall, recovered text-misses, and text+visual-fill.
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from montage import evaluate, store, visual  # noqa: E402

BEST = "screen+context+refine+vseed+vgate"
TOL = 10


def main():
    corpus = store.load_corpus(os.path.join(ROOT, "eval/corpus/oot_remake_v1.tsv"))
    rows = []
    for m in ["hyrule_field", "gohma", "zelda_meeting", "deku_tree_death"]:
        truth = copy.deepcopy(evaluate.load_truth(os.path.join(ROOT, f"eval/truth/{m}.json")))
        truth["tolerance_s"] = TOL
        text = json.load(open(os.path.join(ROOT, f"eval/results/{m}__{BEST}.json")))["predictions"]
        seeds = {v: r["t"] for v, r in text.items() if r["t"] is not None and str(r["method"]).startswith("screen")}
        protos = visual.prototypes(seeds)
        text_missed = [v for v, x in truth["videos"].items() if x["status"] == "present" and (text.get(v) or {}).get("t") is None]
        for thr in (0.85, 0.9, 0.93):
            for sup in (2, 3):
                vis = {}
                for v in corpus:
                    t, n, s = visual.guess(v, protos, sim_thr=thr, min_support=sup)
                    if t is not None:
                        vis[v] = {"t": t}
                sv = evaluate.score(truth, vis)
                rec = [v for v in text_missed if v in vis and abs(vis[v]["t"] - truth["videos"][v]["t"]) <= TOL]
                fill = {v: (text[v] if (text.get(v) or {}).get("t") is not None else vis.get(v)) for v in corpus}
                fill = {v: p for v, p in fill.items() if p}
                sf = evaluate.score(truth, fill)
                rows.append((m, thr, sup, len(seeds), sv["precision"], sv["recall"], sv["err_median_s"],
                             f"{len(rec)}/{len(text_missed)}", sf["precision"], sf["recall"]))
    print("| moment | sim_thr | min_support | seeds | visual P | visual R | visual err_med | text-misses recovered | text+visual P | text+visual R |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    base = []
    for m in ["hyrule_field", "gohma", "zelda_meeting", "deku_tree_death"]:
        truth = copy.deepcopy(evaluate.load_truth(os.path.join(ROOT, f"eval/truth/{m}.json"))); truth["tolerance_s"] = TOL
        s = evaluate.score(truth, json.load(open(os.path.join(ROOT, f"eval/results/{m}__{BEST}.json")))["predictions"])
        base.append(f"{m}: text-only P={s['precision']} R={s['recall']}")
    print("\n" + "; ".join(base))


if __name__ == "__main__":
    main()
