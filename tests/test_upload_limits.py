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

    def test_an_over_size_body_without_a_length_is_refused_cleanly_too(self):
        """This returned 500, and that 500 is why the ceiling is 130 MiB.

        The middleware rewrote the status on the way out, which only works
        when something downstream turns the exception into a response - and
        only the multipart parser does. A raw body had nobody to catch it, so
        the MultiPartException escaped as a 500.

        max_upload_bytes' comment records the live symptom as "refused at
        exactly 130 MiB, three times running, with a 500 rather than a 413",
        attributes it to the web host, and sets the ceiling to match. The
        symptom reproduces with no host involved, and 130 MiB is exactly this
        application's own default.
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
        self.assertEqual(response.status_code, 413)
        self.assertIn("maximum allowed size", response.text)

    def test_a_refusal_the_application_already_answered_is_left_alone(self):
        """The one case where answering here would be wrong: if a response is
        already on the wire, its status cannot be taken back."""
        import inspect

        from core.upload_security import UploadBodyLimitMiddleware as middleware

        source = inspect.getsource(middleware.__call__)
        self.assertIn("if responded or not (exceeded or timed_out):", source)
        self.assertIn("raise", source)


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

    def test_it_is_absent_to_a_caller_with_no_csrf_token_either(self):
        """How an outsider actually meets it, and the case this test missed.

        The suite's BrowserClient adds the CSRF header automatically, so this
        class was reaching the handler's flag check and seeing its 404. A
        caller without a token met require_csrf first and got a 403 - which
        says the route exists. Live, that is exactly what the disabled probe
        answered. The gate is a dependency ahead of require_csrf now, so a
        switched-off endpoint looks like one that was never built.
        """
        from fastapi.testclient import TestClient as RawClient

        object.__setattr__(SETTINGS, "upload_probe_enabled", False)
        raw = RawClient(app)
        try:
            self.assertEqual(raw.put("/api/upload/probe", content=b"x").status_code, 404)
        finally:
            raw.close()

    def test_the_chunked_endpoints_hide_the_same_way(self):
        from fastapi.testclient import TestClient as RawClient

        previous = SETTINGS.chunked_upload_enabled
        object.__setattr__(SETTINGS, "chunked_upload_enabled", False)
        raw = RawClient(app)
        job = "abcdef123456"
        try:
            for method, path, kwargs in (
                    ("post", "/api/upload/begin", {"json": {}}),
                    ("put", "/api/upload/%s/chunk?offset=0" % job, {"content": b"x"}),
                    ("get", "/api/upload/%s/status" % job, {}),
                    ("post", "/api/upload/%s/abort" % job, {}),
                    ("post", "/api/upload/%s/finish" % job, {})):
                with self.subTest(path=path):
                    self.assertEqual(
                        getattr(raw, method)(path, **kwargs).status_code, 404,
                        "%s reveals itself while switched off" % path)
        finally:
            raw.close()
            object.__setattr__(SETTINGS, "chunked_upload_enabled", previous)

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
