from __future__ import annotations

import os
import webbrowser
from threading import Timer

import uvicorn
from dotenv import load_dotenv


load_dotenv()


def server_config() -> tuple[str, int, bool]:
    """Return local defaults while honoring Render and OuiPanel ports."""
    render = os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"}
    hosted = render or "PORT" in os.environ or "SERVER_PORT" in os.environ
    port = int(os.getenv("SERVER_PORT") or os.getenv("PORT") or "8000")
    host = os.getenv("HOST", "0.0.0.0" if hosted else "127.0.0.1")
    open_browser = not hosted and host in {"127.0.0.1", "localhost"}
    return host, port, open_browser


def worker_count() -> int:
    """How many server processes to run (audit 2026-10-14, #3).

    One process answered one slow request at a time: 8 parallel requests took
    11.3 s on the live site. Hosted, it runs WEB_CONCURRENCY processes (4 by
    default) - uvicorn's own multi-process mode, the equivalent of gunicorn
    -w 4 -k uvicorn.workers.UvicornWorker without another dependency. SQLite
    is opened in WAL mode with a busy_timeout (core/db.py), so the processes
    share it safely.

    Only when analysis runs on a separate worker: in-process mode analyses in
    the web process, and four of them would each start analyses of their own.
    Locally it stays one process unless WEB_CONCURRENCY says otherwise.
    """
    hosted = any(name in os.environ for name in ("RENDER", "PORT", "SERVER_PORT"))
    requested = os.getenv("WEB_CONCURRENCY", "").strip()
    count = int(requested) if requested.isdigit() else (4 if hosted else 1)
    if os.getenv("WARRIORIQ_WORKER_MODE", "inprocess").strip().lower() == "inprocess":
        return 1
    return max(1, count)


def main():
    host, port, open_browser = server_config()
    local_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{local_host}:{port}"
    print("\nWarriorIQ")
    print("Local website:", url)
    print("Keep this terminal open while WarriorIQ is running.\n")
    if open_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    workers = worker_count()
    print(f"Server processes: {workers}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, proxy_headers=True, workers=workers)


if __name__ == "__main__":
    main()
