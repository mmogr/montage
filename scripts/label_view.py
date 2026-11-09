"""Print a compact labelling view of a video: metadata, then 20 s caption chunks interleaved with on-screen text."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montage import store, statements

def view(vid, chunk=20, width=170, t0=0, t1=1e9):
    d = store.load_info(vid)
    print(f"##### {vid} | {d['upload_date']} | {d['duration']}s | {d['channel']} | {d['title']}")
    p = next((os.path.join(store.VIDEOS, f"{vid}.{l}.json3") for l in ("en", "en-orig", "en-US")
              if os.path.exists(os.path.join(store.VIDEOS, f"{vid}.{l}.json3"))), None)
    evs = []
    if p:
        for e in json.load(open(p)).get("events", []):
            txt = "".join(s.get("utf8", "") for s in e.get("segs") or []).replace("\n", " ").strip()
            if txt: evs.append((e["tStartMs"] / 1000, txt))
    scr = statements.onscreen(vid, min_len=8)
    buckets = {}
    for t, x in evs:
        if t0 <= t <= t1: buckets.setdefault(int(t // chunk), []).append(x)
    for t, x in scr:
        if t0 <= t <= t1: buckets.setdefault(int(t // chunk), []).append(f"«SCREEN: {x}»")
    for k in sorted(buckets):
        s = " ".join(buckets[k])
        print(f"  {k*chunk:5d}s {s[:width]}")

for v in sys.argv[1:]:
    view(v)
