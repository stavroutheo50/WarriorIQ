"""Serve a label pack so the answers land on disk as they are given.

Opening label.html straight off the filesystem looked like the simple option
and cost somebody a finished pack. Two things go wrong on a `file://` page and
both are silent:

  * Chrome does not reliably keep `localStorage` for a file:// origin, and the
    page's save was wrapped in a bare try/except - so the counter climbed to
    "60 of 60" while nothing was written anywhere.
  * `URL.revokeObjectURL` ran on the line after `a.click()`. The download is
    asynchronous, so the blob was destroyed before the browser read it: a blank
    tab and no file.

Both are fixed in the page. This removes the category. Served over http there
is a real origin, a real fetch, and every answer is written to disk the moment
it is given - so there is no button to remember and nothing to lose by closing
the tab.

    tools/serve_label_pack.py                 # every pack under labelpack/
    tools/serve_label_pack.py --job fam3      # just this one
    tools/serve_label_pack.py --port 8765

Bound to 127.0.0.1 by default. Nothing there is authenticated, because nothing
there is reachable from off this machine.

`--host 0.0.0.0` exists so a pack can be labelled on a phone, and it changes
that bargain: the clips are crops of identifiable people, and a LAN is not a
trusted room. So binding anywhere but loopback mints a one-off key and refuses
every request that does not carry it. The key rides in the URL once, then in a
cookie, because the page POSTs each answer to an absolute /<job>/save and would
otherwise lose the key on the first save. It dies with the process.
"""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
import socket
import sys
import webbrowser
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKS = PROJECT_ROOT / "labelpack"
# The finished answers go next to the pack, so a pack and its labels travel
# together and ingest_labels.py can be pointed straight at them.
MAX_BODY_BYTES = 8 * 1024 * 1024


# Set only when binding off loopback. None means "loopback, no key needed".
ACCESS_KEY: str | None = None
KEY_COOKIE = "labelpack_key"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):        # noqa: A002 - quieter than the default
        return

    def _offered_key(self) -> str | None:
        """The key from this request, whether it came in the URL or a cookie."""
        query = parse_qs(urlsplit(self.path).query)
        if query.get("k"):
            return query["k"][0]
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookies.get(KEY_COOKIE)
        return morsel.value if morsel else None

    def _authorized(self) -> bool:
        if ACCESS_KEY is None:
            return True
        # compare_digest so a wrong key cannot be found one character at a time.
        offered = self._offered_key()
        return bool(offered) and hmac.compare_digest(offered, ACCESS_KEY)

    def _reject(self) -> None:
        self._send(403, b"open the link with its key", "text/plain; charset=utf-8")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if ACCESS_KEY is not None and code < 400:
            self.send_header(
                "Set-Cookie",
                f"{KEY_COOKIE}={ACCESS_KEY}; Path=/; SameSite=Strict; Max-Age=86400",
            )
        self.end_headers()
        self.wfile.write(body)

    def _pack_dir(self, job: str) -> Path | None:
        # Resolve and confirm the result is still inside labelpack/, so a job
        # name cannot walk out of it.
        candidate = (PACKS / job).resolve()
        if candidate.parent != PACKS.resolve() or not candidate.is_dir():
            return None
        return candidate

    def do_GET(self) -> None:                 # noqa: N802 - stdlib's spelling
        if not self._authorized():
            return self._reject()
        path = self.path.split("?", 1)[0].strip("/")
        if not path:
            return self._send(200, self._index().encode("utf-8"), "text/html; charset=utf-8")
        parts = path.split("/")
        if len(parts) == 2 and parts[1] == "":
            parts = parts[:1]
        job = parts[0]
        pack = self._pack_dir(job)
        if pack is None:
            return self._send(404, b"no such pack", "text/plain; charset=utf-8")
        page = pack / "label.html"
        if not page.exists():
            return self._send(404, b"pack has no label.html", "text/plain; charset=utf-8")
        return self._send(200, page.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:                # noqa: N802 - stdlib's spelling
        if not self._authorized():
            return self._reject()
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) != 2 or parts[1] != "save":
            return self._send(404, b"not here", "text/plain; charset=utf-8")
        pack = self._pack_dir(parts[0])
        if pack is None:
            return self._send(404, b"no such pack", "text/plain; charset=utf-8")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, b"bad length", "text/plain; charset=utf-8")
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._send(400, b"bad length", "text/plain; charset=utf-8")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._send(400, b"bad json", "text/plain; charset=utf-8")
        if not isinstance(payload, dict) or payload.get("job") != parts[0]:
            return self._send(400, b"payload is for another pack", "text/plain; charset=utf-8")

        destination = pack / f"{parts[0]}-labels.json"
        # Written beside the pack, then moved into place, so an interrupted
        # write cannot truncate a good file. This is somebody's only copy.
        temporary = destination.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temporary.replace(destination)
        done = len(payload.get("labels") or []) + len(payload.get("unsure") or [])
        print(f"  {parts[0]}: {done} answered -> {destination.relative_to(PROJECT_ROOT)}")
        return self._send(200, b'{"ok":true}', "application/json")

    def _index(self) -> str:
        rows = []
        for pack in sorted(p for p in PACKS.iterdir() if (p / "label.html").exists()):
            saved = pack / f"{pack.name}-labels.json"
            state = "not started"
            if saved.exists():
                try:
                    data = json.loads(saved.read_text(encoding="utf-8"))
                    state = f"{len(data.get('labels') or []) + len(data.get('unsure') or [])} answered"
                except ValueError:
                    state = "saved"
            rows.append(
                f'<li><a href="/{pack.name}/">{pack.name}</a> <span>{state}</span></li>')
        return (
            "<!doctype html><meta charset=utf-8><title>WarriorIQ label packs</title>"
            "<style>body{background:#0f1115;color:#eef1f5;font:16px/1.6 system-ui;"
            "margin:0;padding:40px}h1{font-size:19px}li{margin:10px 0}"
            "a{color:#ffb020}span{color:#98a2b3;font-size:13px;margin-left:10px}</style>"
            "<h1>WarriorIQ label packs</h1><p style='color:#98a2b3;font-size:14px'>"
            "Answers save to disk as you give them. Closing the tab loses nothing.</p><ul>"
            + "".join(rows) + "</ul>")


def _lan_address() -> str:
    """This machine's address on the network the phone is also on.

    Asking the routing table which interface would reach the internet beats
    resolving the hostname, which on a machine with a VPN or WSL returns an
    address the phone cannot route to. Nothing is sent; connect() on UDP only
    picks a route.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))      # TEST-NET-1, never actually routed
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve label packs so answers save themselves.")
    parser.add_argument("--job", help="serve one pack and open it (default: list them all)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach this from a phone on the same "
                             "network. Off loopback the server mints a key and "
                             "every request must carry it.")
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    args = parser.parse_args()

    if not PACKS.is_dir():
        print("no labelpack/ directory - build a pack first with tools/build_label_pack.py")
        return 1

    global ACCESS_KEY
    loopback = args.host in {"127.0.0.1", "::1", "localhost"}
    if not loopback:
        ACCESS_KEY = secrets.token_urlsafe(24)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    suffix = f"{args.job}/" if args.job else ""
    host = args.host if loopback else _lan_address()
    url = f"http://{host}:{args.port}/{suffix}"
    if ACCESS_KEY:
        url += f"?k={ACCESS_KEY}"
    print(f"Label packs at {url}")
    if ACCESS_KEY:
        print("That key is needed once; after that it rides in a cookie.")
        print("It is new every run and dies with this process. Anyone on this")
        print("network who has the link can read the clips, so do not paste it")
        print("anywhere but the phone you are labelling on.")
        if args.open_browser:
            # A phone cannot use a browser opened here, and printing the link
            # for a human to retype is the point.
            args.open_browser = False
    print("Answers are written to labelpack/<job>/<job>-labels.json as you go.")
    print("Press Ctrl+C when you are finished.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
