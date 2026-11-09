"""EXPERIMENT (dropped, see docs/findings.md). Cold start v2 (no on-screen text): description -> CLIP text query -> top-3 separate scenes per video ->
event-specific grid judge (8 frames over 140 s, strict about 'same place, different scene') -> seeds at the judged
start -> leave-one-out calibration (abstains if seeds disagree) -> visual propagation. Scored at several tolerances.

uv run --with torch --with open_clip_torch --with numpy --with pillow python -I scripts/coldstart_v2.py [moment ...]
"""
import copy
import os
import sys

import numpy as np
import open_clip
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from montage import evaluate, store, visual  # noqa: E402
import event_judge as judge  # noqa: E402
from montage.spec import Spec  # noqa: E402
from coldstart_eval import CASES, GAME  # noqa: E402

SCENES_PER_VIDEO, CANDIDATES = 3, 45


def main():
    corpus = store.load_corpus(os.path.join(ROOT, "eval/corpus/oot_remake_v1.tsv"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
    tok = open_clip.get_tokenizer("ViT-B-32")
    for m in (sys.argv[1:] or list(CASES)):
        desc, look = CASES[m]
        with torch.no_grad():
            q = torch.nn.functional.normalize(model.encode_text(tok([look])).float(), dim=-1).numpy()[0]
        cands = sorted(((s, v, t) for v in corpus for s, t in visual.scenes(v, q, k=SCENES_PER_VIDEO)), reverse=True)
        sp = Spec(id=f"cold_{m}", description=desc, game=GAME)
        truth = copy.deepcopy(evaluate.load_truth(os.path.join(ROOT, f"eval/truth/{m}.json")))
        seeds, judged = {}, []
        for s, v, t in cands[:CANDIDATES]:
            j = judge.judge_event(v, t, sp)
            tv = truth["videos"].get(v, {})
            judged.append((v, t, j["match"], j["start_t"], tv.get("t")))
            if j["match"] and j["start_t"] is not None and (v not in seeds or j["start_t"] < seeds[v]):
                seeds[v] = j["start_t"]
        seed_err = sorted(round(t - truth["videos"][v]["t"]) for v, t in seeds.items()
                          if truth["videos"].get(v, {}).get("status") == "present")
        thr, table = visual.calibrate(seeds, tol=20) if len(seeds) >= 3 else (None, [])
        print(f"{m}: {len(seeds)} seed videos from {CANDIDATES} scene candidates; seed errors vs truth {seed_err}")
        if thr is None:
            print(f"   ABSTAIN: seeds do not predict each other; leave-one-out {table}", flush=True)
            continue
        protos = visual.prototypes(seeds)
        preds = {}
        for v in corpus:
            g, n, sc = visual.guess(v, protos, sim_thr=thr, min_support=3)
            if g is not None:
                preds[v] = {"t": g}
        out = []
        for tol in (10, 30, 120):
            truth["tolerance_s"] = tol
            r = evaluate.score(truth, preds)
            out.append(f"±{tol}s P={r['precision']} R={r['recall']}")
        errs = sorted(round(preds[v]["t"] - x["t"]) for v, x in truth["videos"].items()
                      if x["status"] == "present" and v in preds)
        print(f"   thr={thr} {len(preds)} predictions | " + " | ".join(out) + f"\n   signed errors {errs}", flush=True)


if __name__ == "__main__":
    main()
