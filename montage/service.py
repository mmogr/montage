"""Local montage app: a job queue in front of answer(), with a small web page.

One job runs at a time (gglib serves one model). Each job keeps a live progress log and is saved to
data/jobs/<id>.json, so results survive restarts and pages can be revisited.

  uv run --with numpy --with torch --with open_clip_torch --with pillow python -I mt.py serve [--port 8766]

API:
  POST /api/jobs {"query": "..."}      -> job
  POST /api/jobs/<id>/cancel           -> stop a queued or running job (it finishes its current step)
  GET  /api/jobs                       -> recent jobs (summary)
  GET  /api/jobs/<id>                  -> job with log, structured result, markdown
  GET  /api/jobs/<id>/markdown         -> the markdown answer (text/markdown)
"""
import argparse
import json
import os
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import store

JOBS = os.path.join(store.DATA, "jobs")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_jobs, _lock, _q = {}, threading.Lock(), queue.Queue()


def _save(job):
    os.makedirs(JOBS, exist_ok=True)
    with open(os.path.join(JOBS, job["id"] + ".json.tmp"), "w") as f:
        json.dump(job, f, indent=1)
    os.replace(os.path.join(JOBS, job["id"] + ".json.tmp"), os.path.join(JOBS, job["id"] + ".json"))


def _load_all():
    if not os.path.isdir(JOBS):
        return
    for f in os.listdir(JOBS):
        if f.endswith(".json"):
            try:
                j = json.load(open(os.path.join(JOBS, f)))
            except Exception:
                continue
            if j.get("status") in ("queued", "running"):        # interrupted by a restart
                j["status"], j["error"] = "error", "interrupted (service restarted)"
            _jobs[j["id"]] = j


def submit(query):
    job = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "query": query.strip(), "status": "queued",
           "created": time.time(), "started": None, "finished": None, "log": [], "result": None, "markdown": None,
           "error": None}
    with _lock:
        _jobs[job["id"]] = job
        _save(job)
    _q.put(job["id"])
    return job


def _worker():
    from . import answer
    while True:
        jid = _q.get()
        job = _jobs[jid]
        job.update(status="running", started=time.time())
        _save(job)

        def log(msg, job=job):
            with _lock:
                job["log"].append([round(time.time() - job["started"], 1), str(msg)])
                _save(job)
        try:
            if job.get("cancel"):
                raise answer.Cancelled()
            res = {}
            md, sp, hist = answer.answer(job["query"], log=log, res=res)
            with _lock:
                job.update(status="done", result=res, markdown=md, finished=time.time())
        except answer.Cancelled:
            with _lock:
                job.update(status="cancelled", finished=time.time())
        except Exception as e:
            with _lock:
                job.update(status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
                job["log"].append([round(time.time() - job["started"], 1), traceback.format_exc()[-1500:]])
        with _lock:
            _save(job)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/") or "/"
        if p == "/" or p.startswith("/job/"):
            return self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), "text/html")
        if p == "/api/jobs":
            with _lock:
                lst = sorted(_jobs.values(), key=lambda j: -j["created"])[:50]
                return self._send(200, [{"id": j["id"], "query": j["query"], "status": j["status"],
                                         "created": j["created"],
                                         "n": len((j.get("result") or {}).get("items") or [])} for j in lst])
        if p.startswith("/api/jobs/"):
            parts = p.split("/")
            with _lock:
                job = _jobs.get(parts[3])
                if not job:
                    return self._send(404, {"error": "no such job"})
                if len(parts) > 4 and parts[4] == "markdown":
                    return self._send(200, job.get("markdown") or "", "text/markdown")
                return self._send(200, {**job, "queue_position": list(_q.queue).index(job["id"]) + 1
                                        if job["status"] == "queued" and job["id"] in _q.queue else 0})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = self.path.rstrip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "jobs"] and parts[4] == "cancel":
            from . import fetch
            with _lock:
                job = _jobs.get(parts[3])
                if not job:
                    return self._send(404, {"error": "no such job"})
                if job["status"] in ("queued", "running"):
                    job["cancel"] = True
                    if job["status"] == "running":
                        fetch.JOB["cancel"] = True      # long loops and fetches stop at their next check
                    _save(job)
            return self._send(200, {"ok": True})
        if self.path.rstrip("/") != "/api/jobs":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        q = str(body.get("query") or "").strip()
        if not q or len(q) > 500:
            return self._send(400, {"error": "query must be 1-500 characters"})
        self._send(201, submit(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    _load_all()
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"montage on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
