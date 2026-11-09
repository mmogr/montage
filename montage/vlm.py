"""Vision-language model client over any OpenAI-compatible endpoint (default: local gglib proxy).

Env:
  MONTAGE_LLM_BASE   default http://127.0.0.1:8080/v1
  MONTAGE_VLM_MODEL  default Qwen3.8-27B
  MONTAGE_LLM_KEY    if unset, read from `gglib config settings reveal proxy-api-key`
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

BASE = os.environ.get("MONTAGE_LLM_BASE", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MONTAGE_VLM_MODEL", "Qwen3.8-27B")
STATS = {"calls": 0, "seconds": 0.0, "failures": 0}
_key = [os.environ.get("MONTAGE_LLM_KEY")]


def _api_key():
    if not _key[0]:
        try:
            _key[0] = subprocess.run(["gglib", "config", "settings", "reveal", "proxy-api-key"],
                                     capture_output=True, text=True, timeout=20).stdout.strip() or "none"
        except Exception:
            _key[0] = "none"
    return _key[0]


def _img_part(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def ask(prompt, images=(), model=None, max_tokens=1500, timeout=600, temperature=0.0, think=False):
    """Send text + images; return the reply text (reasoning stripped)."""
    content = [{"type": "text", "text": prompt}] + [_img_part(p) for p in images]
    body = {"model": model or MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": temperature}
    if not think:   # visual yes/no checks gain nothing from reasoning and run ~3.5x faster without it
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {_api_key()}"})
    t0 = time.time()
    STATS["calls"] += 1
    try:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    msg = json.load(r)["choices"][0]["message"]
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))        # server busy / swapping models
            except urllib.error.URLError:
                if attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))
    except Exception:
        STATS["failures"] += 1
        raise
    finally:
        STATS["seconds"] += time.time() - t0
    text = msg.get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def ask_json(prompt, images=(), **kw):
    """ask() and parse the last JSON object in the reply; None if none parses."""
    for attempt in range(2):     # temperature 0 would repeat the same broken reply; the retry samples a little
        text = ask(prompt + "\n\nReply with only a JSON object.", images, **(kw if attempt == 0 else {**kw, "temperature": 0.4}))
        d = parse_json(text)
        if d is not None:
            return d
        LAST_BAD.append(text[-4000:])
    return None


LAST_BAD = []


def parse_json(text):
    """Last JSON object in text. Repairs the commonest model mistake: regex backslashes left unescaped (\\s, \\b)."""
    for m in reversed(list(re.finditer(r"\{.*\}", text, re.S))):
        raw = m.group(0)
        for cand in (raw, re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", raw)):
            try:
                return _unctl(json.loads(cand))
            except json.JSONDecodeError:
                continue
    return None


def _unctl(x):
    """A single-backslash \\b or \\f in a regex parses as a backspace/form-feed control char; restore the regex."""
    if isinstance(x, str):
        return x.replace("\x08", "\\b").replace("\x0c", "\\f")
    if isinstance(x, list):
        return [_unctl(v) for v in x]
    if isinstance(x, dict):
        return {k: _unctl(v) for k, v in x.items()}
    return x
