"""Image embeddings for every storyboard frame (CLIP), for visual search and visual propagation.

Writes data/emb/<id>.npz with: emb (N x D float16, L2-normalised), t (N seconds, storyboard label time).
Usage: uv run --with torch --with open_clip_torch --with pillow --with numpy python -I scripts/embed_storyboards.py CORPUS
"""
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def meta_of(vid):
    p = os.path.join(ROOT, "data", "index", f"{vid}.screen.tsv")
    if not os.path.exists(p):
        return None
    line = open(p).readline()
    return dict(kv.split("=", 1) for kv in line[1:].split()) if line.startswith("#") else None


def main():
    corpus = [l.split("\t")[0].strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=dev)
    model.eval()
    out_dir = os.path.join(ROOT, "data", "emb")
    os.makedirs(out_dir, exist_ok=True)
    for vid in corpus:
        out = os.path.join(out_dir, f"{vid}.npz")
        m = meta_of(vid)
        if os.path.exists(out) or not m:
            continue
        iv, cols, rows = float(m["interval"]), int(m["cols"]), int(m["rows"])
        sb = os.path.join(ROOT, "data", "sb", vid)
        sheets = sorted(f for f in os.listdir(sb) if f.endswith(".jpg"))
        embs, ts, batch, bt = [], [], [], []
        for si, f in enumerate(sheets):
            img = Image.open(os.path.join(sb, f)).convert("RGB")
            tw, th = img.width // cols, img.height // rows
            for r in range(rows):
                for c in range(cols):
                    k = si * cols * rows + r * cols + c
                    batch.append(preprocess(img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))))
                    bt.append(k * iv)
                    if len(batch) == 256:
                        with torch.no_grad():
                            e = model.encode_image(torch.stack(batch).to(dev)).float()
                        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16))
                        ts += bt; batch, bt = [], []
        if batch:
            with torch.no_grad():
                e = model.encode_image(torch.stack(batch).to(dev)).float()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16))
            ts += bt
        np.savez(out, emb=np.concatenate(embs), t=np.array(ts, dtype=np.float32))
        print(vid, len(ts), flush=True)


if __name__ == "__main__":
    main()
