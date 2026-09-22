"""Where the upload ceiling actually is, and what it does when you hit it.

The 130 MiB ceiling is attributed to the web host. These pin down what can be
established without the host, which turned out to be most of it.
"""

from __future__ import annotations

import unittest

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from browser_client import BrowserClient
from app.main import app
from core.config import SETTINGS
from core.upload_security import UploadBodyLimitMiddleware

MIB = 1048576


def _sink_app():
    """Just the body-limit middleware, with nothing in front of it.

    The real /upload sits behind admission, which answers 401 before the body
    is ever weighed - so a test against it measures authentication, not the
    limit.
    """
    async def sink(request):
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
        return PlainTextResponse(str(total))

    bare = Starlette(routes=[Route("/upload", sink, methods=["POST"])])
    bare.add_middleware(UploadBodyLimitMiddleware)
    return bare


class BodyLimitBehaviourTests(unittest.TestCase):
    """What WarriorIQ itself does at the ceiling."""

    def setUp(self):
        self.client = TestClient(_sink_app(), raise_server_exceptions=False)
        self.limit = min(SETTINGS.max_fight_bytes, SETTINGS.max_upload_bytes) + MIB

    def tearDown(self):
        self.client.close()

    def test_a_body_under_the_ceiling_is_carried(self):
        size = self.limit - 2 * MIB
        response = self.client.post("/upload", content=b"x" * size,
                                    headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(response.text), size)

    def test_an_over_size_body_with_a_length_is_refused_cleanly(self):
        response = self.client.post("/upload", content=b"x" * (self.limit + MIB),
                                    headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status_code, 413)
        self.assertIn("maximum allowed size", response.text)

    def test_an_over_size_body_without_a_length_returns_500_not_413(self):
        """The bug, pinned.

        max_upload_bytes' comment records the live symptom as "refused at
        exactly 130 MiB, three times running, with a 500 rather than a 413"
        and attributes it to the web host. This reproduces that symptom with
        no host involved: when there is no Content-Length to pre-check, the
        MultiPartException raised inside the streaming guard escapes as a 500.

        130 MiB is also exactly this application's own default, which is not a
        number a web host would pick.

        None of that proves the host has no limit of its own - only that the
        evidence for one is also explained by this. tools/probe_upload_limits.py
        settles it against the real host.

        This test asserts the *current* behaviour so the fix is a deliberate
        change rather than a silent one.
        """
        oversize = self.limit + MIB

        def streamed():
            sent = 0
            while sent < oversize:
                piece = min(MIB, oversize - sent)
                sent += piece
                yield b"x" * piece

        response = self.client.post("/upload", content=streamed(),
                                    headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(
            response.status_code, 500,
            "if this is now 413 the escape was fixed - update max_upload_bytes' "
            "comment, which blames the host for it")


class AdmissionRunsBeforeTheBodyIsWeighedTests(unittest.TestCase):
    def test_an_anonymous_upload_is_refused_without_reading_the_video(self):
        """Why the limit cannot be tested through the real route."""
        client = BrowserClient(app)
        try:
            response = client.post("/upload", content=b"x" * (4 * MIB),
                                   headers={"Content-Type": "application/octet-stream"})
            self.assertEqual(response.status_code, 401)
        finally:
            client.close()


class ProbeEndpointTests(unittest.TestCase):
    """The probe must be invisible unless deliberately switched on."""

    def setUp(self):
        self.client = BrowserClient(app)
        self.previous = SETTINGS.upload_probe_enabled

    def tearDown(self):
        object.__setattr__(SETTINGS, "upload_probe_enabled", self.previous)
        self.client.close()

    def test_it_is_absent_by_default(self):
        object.__setattr__(SETTINGS, "upload_probe_enabled", False)
        self.assertEqual(self.client.put("/api/upload/probe", content=b"x").status_code, 404)

    def test_it_is_still_absent_to_a_stranger_when_switched_on(self):
        """404 rather than 403: an endpoint that absorbs large bodies should
        not advertise itself to someone who cannot use it."""
        object.__setattr__(SETTINGS, "upload_probe_enabled", True)
        self.assertEqual(self.client.put("/api/upload/probe", content=b"x").status_code, 404)

    def test_it_writes_nothing(self):
        """A probe that left files behind would fill a shared host's disk."""
        import pathlib

        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "app" / "main.py").read_text(encoding="utf-8")
        start = source.index("async def upload_probe")
        body = source[start:source.index('@app.get("/health"', start)]
        # The docstring explains that it writes nothing, which is prose about
        # the code rather than code. Scan what runs.
        opening = body.index('"""')
        code = body[:opening] + body[body.index('"""', opening + 3) + 3:]
        for forbidden in ("open(", ".write", "UPLOADS", "shutil", "Path("):
            self.assertNotIn(forbidden, code, f"the probe must not use {forbidden}")


if __name__ == "__main__":
    unittest.main()
