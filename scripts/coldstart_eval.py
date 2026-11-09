"""Cold start with NO on-screen text: words -> CLIP text-to-image search over storyboards -> vision-model check
-> seeds -> self-calibrated visual propagation. Scored against the same truth.

uv run --with torch --with open_clip_torch --with numpy --with pillow python -I scripts/coldstart_eval.py
"""
import copy
import os
import sys

import numpy as np
import open_clip
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from montage import evaluate, judge, store, visual  # noqa: E402
from montage.spec import Spec  # noqa: E402

GAME = "The Legend of Zelda: Ocarina of Time (2026 remake), gameplay videos with streamer overlays"
CASES = {   # visual descriptions only: what the frame looks like, no on-screen text
    "hyrule_field": ("The player steps out of Kokiri Forest into Hyrule Field for the first time",
                     "a vast open green grassy field with rolling hills under a bright cloudy sky, video game"),
    "deku_tree_death": ("The Great Deku Tree's farewell speech and death after Gohma is defeated",
                        "a giant ancient tree with a wise old face, a small boy in green standing before it, video game"),
    "gohma": ("First encounter with Queen Gohma, the boss of Inside the Deku Tree",
              "a giant one-eyed spider insect monster in a dark cave, video game boss"),
}
TOP_PER_VIDEO, CANDIDATES = 1, 30
ONSET = os.environ.get("ONSET") == "1"
ONLY = [a for a in sys.argv[1:]]


def main():
    corpus = store.load_corpus(os.path.join(ROOT, "eval/corpus/oot_remake_v1.tsv"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    for m, (desc, look) in CASES.items():
        if ONLY and m not in ONLY:
            continue
        with torch.no_grad():
            q = torch.nn.functional.normalize(model.encode_text(tok([look]).to(dev)).float(), dim=-1).cpu().numpy()[0]
        cands = []
        for v in corpus:
            e, ts = visual.load(v)
            if e is None:
                continue
            s = e @ q
            for i in np.argsort(-s)[:TOP_PER_VIDEO]:
                cands.append((float(s[i]), v, float(ts[i])))
        cands.sort(reverse=True)
        sp = Spec(id=f"cold_{m}", description=desc, game=GAME)
        seeds = {}
        for s, v, t in cands[:CANDIDATES]:
            j = judge.judge(v, t, sp)
            if j["match"]:
                seeds[v] = visual.onset(v, t) if ONSET else t
        thr, table = visual.calibrate(seeds) if len(seeds) >= 3 else (None, [])
        protos = visual.prototypes(seeds)
        preds = {}
        if thr is None:
            print(f"{m}: ABSTAIN - {len(seeds)} seeds do not predict each other (leave-one-out table {table})", flush=True)
            continue
        for v in corpus:
            g, n, sc = visual.guess(v, protos, sim_thr=thr, min_support=3)
            if g is not None:
                preds[v] = {"t": g}
        truth = copy.deepcopy(evaluate.load_truth(os.path.join(ROOT, f"eval/truth/{m}.json")))
        out = []
        for tol in (10, 30):
            truth["tolerance_s"] = tol
            r = evaluate.score(truth, preds)
            out.append(f"tol±{tol}s: P={r['precision']} R={r['recall']} err_med={r['err_median_s']}")
        errs = sorted(round(preds[v]["t"] - x["t"], 1) for v, x in truth["videos"].items() if x["status"] == "present" and v in preds)
        print(f"{m}: {len(seeds)}/{CANDIDATES} candidates confirmed as seeds; thr={thr}; {len(preds)} predictions\n   "
              + " | ".join(out) + f"\n   signed errors vs truth event: {errs}", flush=True)


if __name__ == "__main__":
    main()
