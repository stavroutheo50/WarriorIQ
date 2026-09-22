"""Find out what the web host actually accepts in one request body.

Why this is not already known
-----------------------------
`SETTINGS.max_upload_bytes` is 130 MiB and its comment attributes that to the
host: "the body is refused at exactly 130 MiB, three times running, with a 500
rather than a 413". Every part of that is consistent with a second explanation
nobody checked - that WarriorIQ refused its own upload:

  * 130 MiB is exactly this application's default ceiling, not a round number
    any web host would pick;
  * `UploadBodyLimitMiddleware` returns a clean 413 for an over-size body that
    carries a Content-Length, and lets a `MultiPartException` escape as a
    **500** when it does not - which is the recorded symptom exactly;
  * asked directly, the live host answers `100 Continue` to a declared 200 MiB
    on `/upload`, so Apache is not refusing on size before the body arrives;
  * `a2wsgi` passes Content-Length through correctly and streams the body, so
    it is not losing the header on the way in.

**Since settled, and the answer is that the host has no such limit.** A
136 MiB body was pushed at the live /upload: 134 MiB of it crossed the wire
and the application answered 401 on its own terms. Nothing in front of
WarriorIQ refused it. The 130 MiB ceiling is WarriorIQ's own
`max_upload_bytes` and can be raised with WARRIORIQ_MAX_UPLOAD_BYTES.

What this tool is still for is the other two walls, which that test did not
touch: whether a body arrives streamed or buffered, and how long one request
is allowed to take.

What this measures
------------------
Three walls, which are separate and get confused with each other:

  size      the largest body that reaches the application at all
  delivery  whether it arrives streamed or buffered upstream, which is what
            a chunk size has to be chosen against
  time      how long one request is allowed to take, which bites a slow
            uplink long before any byte ceiling does

Running it
----------
On the host, set WARRIORIQ_UPLOAD_PROBE=1 in /home/dchoodxm/warrioriq/.env,
restart the app, then from anywhere:

    python tools/probe_upload_limits.py --base https://warrioriq.eu \\
        --cookie "warrioriq_session=...; warrioriq_csrf=..." --sizes 8,64,120,136,200

Copy the whole cookie string from the browser's dev tools: both the session
and the CSRF token are needed, and the CSRF header is derived from it here.

Turn the flag back off afterwards. The endpoint is admin-only and writes
nothing, but an endpoint that exists to absorb large bodies should not be left
switched on.

Sizes are MiB and are sent in ascending order; the first failure stops the
run, because everything above it would fail the same way and cost real
bandwidth to prove it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SIZES = (8, 64, 120, 136, 200)


def probe(base: str, cookie: str, mib: int, timeout: float) -> dict:
    """Send `mib` mebibytes and report what came back."""
    url = base.rstrip("/") + "/api/upload/probe"
    payload = b"\0" * (mib * 1048576)
    request = urllib.request.Request(url, data=payload, method="PUT")
    request.add_header("Content-Type", "application/octet-stream")
    request.add_header("Content-Length", str(len(payload)))
    if cookie:
        request.add_header("Cookie", cookie)
        # Every state-changing route requires the visitor's CSRF token echoed
        # in a header, and this one is no exception - an endpoint that absorbs
        # large bodies is the last place to make one. The token is already in
        # the cookie string being pasted in, so it is lifted from there rather
        # than asked for separately.
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "warrioriq_csrf" and value:
                request.add_header("X-CSRF-Token", value)
                break
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            elapsed = time.perf_counter() - started
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = {"raw": body[:200]}
            return {"sent_mib": mib, "status": response.status,
                    "seconds": round(elapsed, 1), "server": parsed}
    except urllib.error.HTTPError as error:
        elapsed = time.perf_counter() - started
        detail = error.read().decode("utf-8", "replace")[:200]
        return {"sent_mib": mib, "status": error.code,
                "seconds": round(elapsed, 1), "error": detail}
    except Exception as error:  # noqa: BLE001 - a dropped connection is a result
        elapsed = time.perf_counter() - started
        return {"sent_mib": mib, "status": None, "seconds": round(elapsed, 1),
                "error": f"{type(error).__name__}: {error}"}


def verdict(results: list[dict]) -> list[str]:
    lines = []
    accepted = [r for r in results if r.get("status") == 200]
    refused = [r for r in results if r.get("status") != 200]
    if accepted:
        largest = max(r["sent_mib"] for r in accepted)
        lines.append(f"Largest body that reached the application: {largest} MiB.")
        fastest = min(accepted, key=lambda r: r["sent_mib"])
        rate = fastest["sent_mib"] * 1048576 / max(0.1, fastest["seconds"]) / 1024
        lines.append(f"Upload rate on this connection: about {rate:.0f} KiB/s.")
        streamed = [r for r in accepted if isinstance(r.get("server"), dict)
                    and r["server"].get("streamed")]
        if streamed:
            biggest = max(r["server"].get("largest_piece_bytes", 0) for r in streamed)
            lines.append(f"The body arrives streamed, largest piece {biggest} bytes - "
                         "so a chunk size can be chosen for the wire rather than "
                         "for an upstream buffer.")
        else:
            lines.append("The body arrives in one piece, so something upstream "
                         "buffers it whole. Choose a chunk size that is "
                         "comfortable to hold in memory, not just on the wire.")
    if refused:
        first = min(refused, key=lambda r: r["sent_mib"])
        lines.append(f"First refusal at {first['sent_mib']} MiB "
                     f"(status {first['status']}).")
        if accepted and max(r["sent_mib"] for r in accepted) >= 130:
            lines.append("A body over 130 MiB reached the application, so the "
                         "130 MiB ceiling is WarriorIQ's own and can be raised "
                         "with WARRIORIQ_MAX_UPLOAD_BYTES.")
    elif accepted and max(r["sent_mib"] for r in accepted) > 130:
        lines.append("Nothing was refused, and a body over 130 MiB got through. "
                     "The ceiling is WarriorIQ's own: raise "
                     "WARRIORIQ_MAX_UPLOAD_BYTES rather than designing around "
                     "a host limit that is not there.")
    lines.append("")
    lines.append("One rate is not a rate. The same path to the live host "
                 "measured 183 KiB/s and 3.5 MiB/s within an hour - the same "
                 "260 MB file is twenty-three minutes or seventy seconds "
                 "depending which one you sampled. Read the number above as "
                 "one draw from that spread, not as this connection's speed, "
                 "and do not size a timeout from it.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="e.g. https://warrioriq.eu")
    parser.add_argument("--cookie", default="", help="an admin session cookie")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--timeout", type=float, default=1800.0)
    arguments = parser.parse_args()

    sizes = sorted(int(s) for s in arguments.sizes.split(",") if s.strip())
    print(f"Probing {arguments.base} with {sizes} MiB\n")
    results = []
    for mib in sizes:
        print(f"  sending {mib:>4} MiB ... ", end="", flush=True)
        result = probe(arguments.base, arguments.cookie, mib, arguments.timeout)
        results.append(result)
        if result.get("status") == 200:
            server = result.get("server", {})
            print(f"accepted in {result['seconds']}s "
                  f"(server saw {server.get('received_mib')} MiB in "
                  f"{server.get('pieces')} piece(s))")
        else:
            print(f"REFUSED status={result['status']} after {result['seconds']}s")
            print(f"    {str(result.get('error'))[:160]}")
            # Everything larger fails the same way and costs bandwidth to prove.
            break

    print()
    for line in verdict(results):
        print(line)
    print()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
