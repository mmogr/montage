"""montage CLI.
  python3 -I mt.py ablate SPEC --corpus CORPUS --truth TRUTH [--configs a,b,...] [--sb-offset S]
  python3 -I mt.py run    SPEC --corpus CORPUS --config NAME
  uv run --with numpy --with torch --with open_clip_torch --with pillow python -I mt.py serve [--port 8766]
  uv run --with numpy --with torch --with open_clip_torch --with pillow python -I mt.py ask "QUERY" [--corpus CORPUS] [--out FILE.md]
"""
import argparse
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from montage import evaluate, pipeline, spec as specmod, store  # noqa: E402


def score_all():
    """Re-score every saved run against the current truth files; write eval/results/REPORT.md."""
    import glob
    res = os.path.join(ROOT, "eval", "results")
    order = ["transcript", "cues", "cues+refine", "screen", "screen+refine", "screen+context+refine", "all"]
    out = ["# Ablation report", "", "Hit = within ±5 s of the frame-verified event time. Recall is relative to the pooled truth "
           "(see eval/README.md). Cost columns are from the run that produced the predictions (cached slices cost 0).", ""]
    for tpath in sorted(glob.glob(os.path.join(ROOT, "eval", "truth", "*.json"))):
        truth = evaluate.load_truth(tpath)
        m = truth["moment"]
        rows = []
        for n in order:
            p = os.path.join(res, f"{m}__{n}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append({"config": n, **{k: d["cost"].get(k) for k in ("slices", "slice_mb", "wall_s", "http_429")},
                         **evaluate.score(truth, d["predictions"])})
        c = evaluate.score(truth, {})
        out += [f"## {m}", "", f"{truth.get('event') or ''}", "",
                f"truth: {c['present']} present, {c['absent']} absent, {c['unknown']} unknown", "", evaluate.table(rows), ""]
    open(os.path.join(res, "REPORT.md"), "w").write("\n".join(out))
    print("\n".join(out))


def run_statements(argv):
    """python3 -I mt.py statements SPEC --corpus C --truth T [--think] [--name N] [--workers 2]"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from montage import statements, vlm
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--corpus", required=True); ap.add_argument("--truth")
    ap.add_argument("--think", action="store_true"); ap.add_argument("--name", default="default")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    sp = specmod.load(a.spec)
    corpus = store.load_corpus(a.corpus)
    t0 = time.time()

    def one(v):
        try:
            return statements.extract(v, sp, think=a.think)
        except Exception as e:
            return {"id": v, "segments": [], "note": f"error: {e}", "llm_calls": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, corpus))
    groups = statements.dedupe(results)
    preds = {r["id"]: r for r in results}
    out = {"spec": sp.id, "config": {"name": a.name, "think": a.think, "model": vlm.MODEL},
           "cost": {"wall_s": round(time.time() - t0, 1), "llm_calls": sum(r["llm_calls"] for r in results)},
           "predictions": preds,
           "duplicate_groups": [[(v, s["start"]) for v, s in g] for g in groups if len(g) > 1]}
    path = os.path.join(ROOT, "eval", "results", f"{sp.id}__{a.name}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out["cost"]))
    for r in results:
        for s in r["segments"]:
            print(f"  {r['id']} {s['start']:>5}-{s['end']:<5} {s['speaker']} | {str(s.get('evidence'))[:60]}")
    print("duplicate groups:", out["duplicate_groups"])
    if a.truth:
        sc = evaluate.score_statements(evaluate.load_truth(a.truth), preds)
        print(json.dumps({k: v for k, v in sc.items() if k not in ("wrong", "missed")}))
        print("wrong:", sc["wrong"]); print("missed:", sc["missed"])


def ask(argv):
    from montage import answer
    ap = argparse.ArgumentParser()
    ap.add_argument("query"); ap.add_argument("--corpus"); ap.add_argument("--out")
    ap.add_argument("--per-query", type=int, default=12)
    a = ap.parse_args(argv)
    md, sp, hist = answer.answer(a.query, store.load_corpus(a.corpus) if a.corpus else None,
                                 log=lambda m: print(m, file=sys.stderr), per_query=a.per_query)
    if a.out:
        open(a.out, "w").write(md)
        json.dump({"query": a.query, "spec": dataclasses.asdict(sp), "planner_log": hist},
                  open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(md)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from montage import service
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return service.main()
    if len(sys.argv) > 1 and sys.argv[1] == "ask":
        return ask(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        return score_all()
    if len(sys.argv) > 1 and sys.argv[1] == "statements":
        return run_statements(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ablate"])
    ap.add_argument("spec")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth")
    ap.add_argument("--config", default="screen+cues+refine")
    ap.add_argument("--configs", default=",".join(pipeline.CONFIGS))
    ap.add_argument("--sb-offset", type=float, default=0.0)
    a = ap.parse_args()

    sp = specmod.load(a.spec)
    corpus = store.load_corpus(a.corpus)
    names = [a.config] if a.cmd == "run" else a.configs.split(",")
    truth = evaluate.load_truth(a.truth) if a.truth else None
    rows = []
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    for n in names:
        cfg = dataclasses.replace(pipeline.CONFIGS[n], sb_offset_s=a.sb_offset)
        preds, cost = pipeline.run(sp, corpus, cfg)
        json.dump({"spec": sp.id, "config": dataclasses.asdict(cfg), "cost": cost, "predictions": preds},
                  open(os.path.join(ROOT, "eval", "results", f"{sp.id}__{n}.json"), "w"), indent=1)
        row = {"config": n, **cost}
        if truth:
            row.update(evaluate.score(truth, preds))
        rows.append(row)
        print(f"done {n}: {cost}", file=sys.stderr)
    if truth:
        json.dump(rows, open(os.path.join(ROOT, "eval", "results", f"{sp.id}__ablation.json"), "w"), indent=1)
        print(evaluate.table(rows))
    else:
        for r in preds.values():
            print(r["id"], r["t"], r["method"], r["error"] or "")


if __name__ == "__main__":
    main()
