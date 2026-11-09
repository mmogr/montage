"""Build a self-contained montage player page from located clips.
Usage: python3 -I build_player.py LOCATED_JSON VIDEOS_DIR OUT_HTML "Title"
"""
import json, os, sys, html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_transcript  # noqa: E402


def quote_for(vid, vdir, a, b):
    lines = [x for t, x in load_transcript(vid, vdir) if a - 1 <= t <= b]
    txt = " ".join(l.replace(">>", "").strip() for l in lines)
    txt = " ".join(txt.split())
    return txt[:220] + ("…" if len(txt) > 220 else "")


def main():
    src, vdir, out, title = sys.argv[1:5]
    clips = []
    for r in json.load(open(src)):
        if r.get("status") != "card":
            continue
        a, b = r["clip"]
        clips.append({"id": r["id"], "start": int(a), "end": int(b), "channel": r["channel"],
                      "title": r["title"], "views": r.get("views") or 0,
                      "quote": quote_for(r["id"], vdir, a, b)})
    clips.sort(key=lambda c: -c["views"])
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_template.html")).read()
    page = tpl.replace("__TITLE__", html.escape(title)).replace("__CLIPS__", json.dumps(clips))
    open(out, "w").write(page)
    print(f"{len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
