"""CLIP embeddings for storyboard frames and text queries (open_clip ViT-B-32, laion2b). Needs torch, open_clip,
pillow, numpy; imported lazily so the rest of montage runs without them.
Writes data/emb/<id>.npz: emb (N x D float16, L2-normalised), t (storyboard label seconds)."""
import os
import threading

from . import store

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
EMB = os.path.join(store.DATA, "emb")
_m = {}
_lock = threading.Lock()


def _model():
    with _lock:
        if not _m:
            import open_clip
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=dev)
            model.eval()
            _m.update(model=model, preprocess=preprocess, tok=open_clip.get_tokenizer(MODEL), dev=dev, torch=torch)
    return _m


def text(s):
    m = _model()
    torch = m["torch"]
    with torch.no_grad():
        e = m["model"].encode_text(m["tok"]([s]).to(m["dev"])).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0]


def video(vid):
    import numpy as np
    from PIL import Image
    out = os.path.join(EMB, f"{vid}.npz")
    meta, _ = store.load_screen(vid)
    if os.path.exists(out):
        return "cached"
    if not meta:
        return "no storyboard index"
    m = _model()
    torch = m["torch"]
    iv, cols, rows = float(meta["interval"]), int(meta["cols"]), int(meta["rows"])
    sb = os.path.join(store.DATA, "sb", vid)
    sheets = sorted(f for f in os.listdir(sb) if f.endswith(".jpg"))
    embs, ts, batch, bt = [], [], [], []

    def flush():
        with _lock, torch.no_grad():
            e = m["model"].encode_image(torch.stack(batch).to(m["dev"])).float()
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16))
        ts.extend(bt)
        batch.clear(); bt.clear()
    for si, f in enumerate(sheets):
        img = Image.open(os.path.join(sb, f)).convert("RGB")
        tw, th = img.width // cols, img.height // rows
        for r in range(rows):
            for c in range(cols):
                batch.append(m["preprocess"](img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))))
                bt.append((si * cols * rows + r * cols + c) * iv)
                if len(batch) == 256:
                    flush()
    if batch:
        flush()
    if not embs:
        return "no frames"
    os.makedirs(EMB, exist_ok=True)
    np.savez(out, emb=np.concatenate(embs), t=np.array(ts, dtype=np.float32))
    return f"ok {len(ts)} frames"
