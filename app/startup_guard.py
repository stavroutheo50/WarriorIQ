"""Keep a failed start from taking the whole site down (walkthrough, 2026-11-16).

Around 16:12 UTC on 16 Nov 2026 every route answered with Passenger's own
"We're sorry, but something went wrong" page for about a minute, then
recovered. It came ten minutes after a merge, which is when the owner deploys:
.cpanel.yml copies the tree into the live app root while the old process is
still serving, and a process Passenger starts in that window imports a
half-copied tree. The import raises, Passenger shows its error page, and the
site stays down until a later spawn finds a complete tree.

This wraps the import. If it fails, the traceback goes to stderr (Passenger's
log, one clear line saying what to look for) and visitors get a WarriorIQ page
saying it will be back shortly, with Retry-After, instead of the host's error.
The import is tried again on a later request, at most every RETRY_SECONDS, so
the site comes back by itself the moment the tree is whole again - without
waiting for a restart.

Deliberately imports nothing from the application and nothing heavy: it has
to work precisely when the application cannot be imported.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

RETRY_SECONDS = 10.0

# Self-contained: no stylesheet, font or image request that could fail too.
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>{title} · WarriorIQ</title>
<style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#07101f;color:#f7f7f3;
font-family:Inter,"Segoe UI",Arial,sans-serif}}main{{max-width:560px;padding:40px 28px;margin:24px;
border:1px solid #243653;border-radius:22px;background:linear-gradient(145deg,#111b2e,#0b1324)}}
b{{font-size:22px;letter-spacing:-1px}}b span{{color:#2f6fed}}h1{{font-size:34px;line-height:1.05;margin:18px 0 12px}}
p{{color:#a7b4c8;line-height:1.6}}a{{color:#7fb1ff}}</style></head><body><main>
<b>WARRIOR<span>IQ</span></b><h1>{title}</h1><p>{message}</p></main></body></html>"""


def branded_page(title: str, message_html: str) -> bytes:
    """A standalone WarriorIQ error page. message_html is trusted markup written in this codebase."""
    return _PAGE.format(title=title, message=message_html).encode("utf-8")


OUTAGE_PAGE = branded_page(
    "Back in a moment.",
    "WarriorIQ is restarting after an update. Your fights and reports are safe. This page will work again "
    'within a minute: <a href="">try again</a>.')


def _log(message: str) -> None:
    # stderr is what Passenger writes to its log; print is enough and cannot fail on a missing logger.
    print(f"warrioriq startup: {message}", file=sys.stderr, flush=True)


def guarded(loader, retry_seconds: float = RETRY_SECONDS):
    """A WSGI application that is loader()'s result, or the outage page until loader() succeeds."""
    state = {"app": None, "tried_at": 0.0}
    lock = threading.Lock()

    def attempt() -> None:
        started = time.monotonic()
        try:
            state["app"] = loader()
            _log(f"application loaded in {time.monotonic() - started:.1f}s")
        except Exception:                                            # noqa: BLE001 - anything at all
            _log("APPLICATION FAILED TO START (serving the outage page, retrying in "
                 f"{retry_seconds:.0f}s):\n{traceback.format_exc()}")
        state["tried_at"] = time.monotonic()

    attempt()

    def application(environ, start_response):
        app = state["app"]
        if app is None:
            with lock:
                if state["app"] is None and time.monotonic() - state["tried_at"] >= retry_seconds:
                    attempt()
                app = state["app"]
        if app is not None:
            return app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "text/html; charset=utf-8"), ("Retry-After", str(int(retry_seconds))),
            ("Cache-Control", "no-store"), ("Content-Length", str(len(OUTAGE_PAGE))),
        ])
        return [b"" if environ.get("REQUEST_METHOD") == "HEAD" else OUTAGE_PAGE]

    return application
