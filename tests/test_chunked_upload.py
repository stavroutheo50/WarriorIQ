"""Uploading a fight in pieces, so a phone connection can lose one.

A phone films 1080p at 8-17 Mbps, so two minutes of fight is 140-260 MB. Sent
as one request body that is two walls at once: a body ceiling, and a single
request that has to survive the whole transfer. Measured against the live host
the transfer runs at about 187 KiB/s, so 260 MB is roughly twenty-three
minutes against an upload_timeout_seconds of 900 - and one dropped connection
at minute twenty costs the whole fight, because a single POST has nothing to
resume from.
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from browser_client import BrowserClient as TestClient

import app.main as webapp
from app.main import app
from core import chunked_upload
from core.chunked_upload import ChunkedUploadError, UploadSession
from core.config import SETTINGS, UPLOADS

MIB = 1048576


def _session(**overrides) -> UploadSession:
    base = dict(job_id="a1b2c3d4e5f6", account_id=1, suffix=".mp4",
                declared_bytes=1024, received_bytes=0,
                original_name="fight.mp4", form={})
    base.update(overrides)
    return UploadSession(**base)


class ResumeCursorTests(unittest.TestCase):
    """The part file's size is the cursor. There is no second ledger."""

    def setUp(self):
        self.job = "abcdef123456"
        chunked_upload.discard(self.job)
        self.session = chunked_upload.begin(
            self.job, 1, filename="fight.mp4", declared_bytes=6, form={})

    def tearDown(self):
        chunked_upload.discard(self.job)

    def test_pieces_in_order_assemble_the_file(self):
        self.assertEqual(chunked_upload.append(self.session, 0, b"abc"), 3)
        self.assertEqual(chunked_upload.append(self.session, 3, b"def"), 6)
        path = chunked_upload.finalise(self.session)
        self.addCleanup(path.unlink, True)
        self.assertEqual(path.read_bytes(), b"abcdef")

    def test_a_piece_that_does_not_continue_is_told_where_to_resume(self):
        chunked_upload.append(self.session, 0, b"abc")
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.append(self.session, 99, b"def")
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(caught.exception.offset, 3,
                         "the client must be told the real end of the file")

    def test_a_replayed_piece_is_refused_rather_than_duplicated(self):
        """A retry after a response was lost must not append twice."""
        chunked_upload.append(self.session, 0, b"abc")
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.append(self.session, 0, b"abc")
        self.assertEqual(caught.exception.offset, 3)
        self.assertEqual(chunked_upload.part_path(self.job).stat().st_size, 3)

    def test_an_upload_larger_than_it_declared_is_refused(self):
        chunked_upload.append(self.session, 0, b"abcdef")
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.append(self.session, 6, b"more")
        self.assertEqual(caught.exception.status, 413)

    def test_an_incomplete_upload_will_not_finalise(self):
        chunked_upload.append(self.session, 0, b"abc")
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.finalise(self.session)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.offset, 3)

    def test_a_piece_bigger_than_the_agreed_chunk_is_refused(self):
        big = _session(declared_bytes=SETTINGS.upload_chunk_bytes * 4)
        chunked_upload.discard(big.job_id)
        chunked_upload.begin(big.job_id, 1, filename="f.mp4",
                             declared_bytes=big.declared_bytes, form={})
        self.addCleanup(chunked_upload.discard, big.job_id)
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.append(big, 0, b"x" * (SETTINGS.upload_chunk_bytes + 1))
        self.assertEqual(caught.exception.status, 413)


class SessionSafetyTests(unittest.TestCase):
    def tearDown(self):
        chunked_upload.discard("abcdef123456")

    def test_a_job_id_that_is_not_one_is_refused(self):
        """It is a path component, so it is validated rather than trusted."""
        for bad in ("../../etc/passwd", "abc", "ABCDEF123456", "", "a" * 40,
                    "abcdef12345/x"):
            with self.subTest(job_id=bad):
                with self.assertRaises(ChunkedUploadError) as caught:
                    chunked_upload.load(bad, 1)
                self.assertEqual(caught.exception.status, 404)

    def test_another_account_cannot_continue_someone_elses_upload(self):
        chunked_upload.begin("abcdef123456", 1, filename="f.mp4",
                             declared_bytes=4, form={})
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.load("abcdef123456", 2)
        self.assertEqual(caught.exception.status, 404,
                         "404 rather than 403: existence is not confirmed")

    def test_a_format_the_server_cannot_read_is_refused_before_any_bytes(self):
        for name in ("clip.wmv", "clip.flv", "clip.3gp", "clip.mpg"):
            with self.subTest(name=name):
                with self.assertRaises(ChunkedUploadError):
                    chunked_upload.suffix_for(name)
        self.assertEqual(chunked_upload.suffix_for("fight.MOV"), ".mov")

    def test_a_video_over_the_ceiling_is_refused_before_any_bytes(self):
        with self.assertRaises(ChunkedUploadError) as caught:
            chunked_upload.begin("abcdef123456", 1, filename="f.mp4",
                                 declared_bytes=SETTINGS.max_chunked_upload_bytes + 1,
                                 form={})
        self.assertEqual(caught.exception.status, 413)
        self.assertIn("MB", caught.exception.detail)

    def test_beginning_truncates_anything_left_at_that_path(self):
        """A job id is fresh; anything already there is debris and must not
        become the first bytes of somebody's fight."""
        chunked_upload.part_path("abcdef123456").write_bytes(b"stale")
        chunked_upload.begin("abcdef123456", 1, filename="f.mp4",
                             declared_bytes=4, form={})
        self.assertEqual(chunked_upload.part_path("abcdef123456").stat().st_size, 0)


class CeilingTests(unittest.TestCase):
    def test_the_ceiling_is_512_mb_and_well_under_max_fight_bytes(self):
        self.assertEqual(SETTINGS.max_chunked_upload_bytes, 512 * MIB)
        self.assertLess(SETTINGS.max_chunked_upload_bytes, SETTINGS.max_fight_bytes)

    def test_a_chunk_is_far_below_any_plausible_body_ceiling(self):
        """The point of chunking: no single request goes near the wall."""
        self.assertLessEqual(SETTINGS.upload_chunk_bytes, 32 * MIB)
        self.assertLess(SETTINGS.upload_chunk_bytes, SETTINGS.max_upload_bytes / 4)

    def test_a_full_round_fits_in_a_sane_number_of_requests(self):
        """260 MB is the top of the range a phone produces for two minutes."""
        requests = (260 * 1000 * 1000) / SETTINGS.upload_chunk_bytes
        self.assertLess(requests, 60, "too many round trips")
        self.assertGreater(requests, 4, "chunks so large the ceiling matters again")


class RoutesTests(unittest.TestCase):
    """The endpoints refuse a stranger and disappear when switched off."""

    def setUp(self):
        self.client = TestClient(app)
        self.previous = SETTINGS.chunked_upload_enabled

    def tearDown(self):
        object.__setattr__(SETTINGS, "chunked_upload_enabled", self.previous)
        self.client.close()

    def test_every_endpoint_refuses_an_anonymous_caller(self):
        object.__setattr__(SETTINGS, "chunked_upload_enabled", True)
        job = "abcdef123456"
        for method, path in (("post", "/api/upload/begin"),
                             ("put", f"/api/upload/{job}/chunk"),
                             ("get", f"/api/upload/{job}/status"),
                             ("post", f"/api/upload/{job}/abort"),
                             ("post", f"/api/upload/{job}/finish")):
            with self.subTest(path=path):
                kwargs = {} if method == "get" else {"content": b"{}"}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertIn(response.status_code, {401, 403},
                              f"{path} answered {response.status_code}")

    def test_the_whole_path_disappears_when_switched_off(self):
        """The rollback: turning the flag off leaves /upload as it was."""
        object.__setattr__(SETTINGS, "chunked_upload_enabled", False)
        response = self.client.post("/api/upload/begin", content=b"{}")
        self.assertEqual(response.status_code, 404)

    def test_the_single_request_upload_is_still_there(self):
        """/upload stays the fallback; it is not replaced."""
        paths = {route.path for route in app.routes}
        self.assertIn("/upload", paths)


class ReuseTests(unittest.TestCase):
    """finish calls the existing pipeline rather than restating it."""

    def setUp(self):
        self.source = (Path(__file__).resolve().parents[1]
                       / "app" / "main.py").read_text(encoding="utf-8")

    def _finish(self) -> str:
        start = self.source.index("async def chunked_upload_finish")
        return self.source[start:self.source.index("@app.put(\"/api/upload/probe\"", start)]

    def test_finish_calls_the_single_request_handler(self):
        body = self._finish()
        self.assertIn("await upload(", body)
        # None of the pipeline is restated here; a second copy is a second
        # copy to keep in step.
        for stage in ("looks_like_video", "scan_upload", "normalize_container",
                      "get_video_info"):
            self.assertNotIn(stage, body, f"{stage} is duplicated in finish")

    def test_the_assembled_file_is_moved_rather_than_copied(self):
        """Copying would write a second half-gigabyte file on a shared host."""
        source = (Path(__file__).resolve().parents[1]
                  / "core" / "chunked_upload.py").read_text(encoding="utf-8")
        self.assertIn("part.replace(destination)", source)
        self.assertNotIn("shutil.copy", source)

    def test_a_failed_pipeline_gives_the_reservation_back(self):
        """max_pending_uploads is 2, so a leaked lease locks the account out."""
        body = self._finish()
        self.assertIn("_release_chunked", body)
        self.assertIn("except HTTPException", body)

    def test_deferred_work_is_attached_to_the_response(self):
        """Calling the handler directly means FastAPI does not wire these up."""
        self.assertIn("response.background = background", self._finish())


class LeaseTests(unittest.TestCase):
    def test_a_chunk_pushes_the_expiry_out(self):
        """Without this the sweep reclaims an upload that is still arriving -
        exactly the slow transfer this path exists to carry."""
        source = (Path(__file__).resolve().parents[1]
                  / "app" / "main.py").read_text(encoding="utf-8")
        start = source.index("async def chunked_upload_chunk")
        body = source[start:source.index("@app.get(\"/api/upload/{job_id}/status\")", start)]
        self.assertIn("extend_lease", body)

    def test_the_part_file_ages_out_with_its_job(self):
        """core/retention.py groups by owning_stem, so the sidecar and the
        part file are swept together with everything else for that job."""
        from core.retention import owning_stem

        self.assertEqual(owning_stem(UPLOADS / "abcdef123456.part"), "abcdef123456")


class EndToEndTests(unittest.TestCase):
    """A real signed-in account, over real HTTP, through every endpoint.

    The unit tests above exercise the store. This drives the routes, which is
    where CSRF, admission, ownership and the handover to the existing pipeline
    actually meet.
    """

    def setUp(self):
        from core.auth import create_account, issue_session

        self.client = TestClient(app)
        # Unique per test: the suite shares one database for the whole run,
        # so a fixed address collides with the previous test's account.
        self.email = "chunked-%s@example.com" % uuid.uuid4().hex[:10]
        self.account = create_account(self.email, "a-long-enough-password")
        self.client.cookies.set("warrioriq_session",
                                issue_session(int(self.account["id"])))
        # Any page issues the CSRF cookie; the header has to echo it.
        self.client.get("/analyze/kickboxing")
        self.token = self.client.cookies.get("warrioriq_csrf")
        self.headers = {"X-CSRF-Token": self.token}
        self.previous = SETTINGS.chunked_upload_enabled
        object.__setattr__(SETTINGS, "chunked_upload_enabled", True)
        # A test clip is far smaller than the 8 MiB production chunk, so at
        # the real size everything here would arrive in one piece and the
        # assembly this exists to test would never run.
        self.previous_chunk = SETTINGS.upload_chunk_bytes
        object.__setattr__(SETTINGS, "upload_chunk_bytes", 64 * 1024)
        # begin shares the fight-upload rate budget with /upload, which is
        # correct - a chunked upload is a fight upload - but the limiter keys
        # on client IP and every test client is the same one. Opening a dozen
        # sessions here spent the budget of whatever ran next, which failed
        # with a 429 that had nothing to do with it.
        webapp._rate_windows.clear()

    def tearDown(self):
        webapp._rate_windows.clear()

    def tearDown(self):
        webapp._rate_windows.clear()
        object.__setattr__(SETTINGS, "chunked_upload_enabled", self.previous)
        object.__setattr__(SETTINGS, "upload_chunk_bytes", self.previous_chunk)
        self.client.close()

    def _forget(self, job):
        """Release everything begin reserved.

        A session holds a storage lease and an analysis reservation until
        finish or abort. Tests that opened one and walked away left both
        behind in the shared runtime, which is not a test-only problem - it is
        the same leak max_pending_uploads punishes in production, so the
        cleanup is the real release rather than deleting the file.
        """
        from core.db import release_analysis
        from core.upload_security import release_upload_storage

        chunked_upload.discard(job)
        try:
            release_analysis(int(self.account["id"]), job)
        except Exception:  # noqa: BLE001 - already released is fine
            pass
        release_upload_storage(job)

    def _begin(self, size, **extra):
        payload = {
            "filename": "fight.mp4", "size": size,
            "rights_confirmed": True, "people_permissions_confirmed": True,
            "minor_permission_status": "no_minors",
            "fight_type": "competition", "ruleset": "K1", "analysis_target": "BOTH",
        }
        payload.update(extra)
        return self.client.post("/api/upload/begin", json=payload, headers=self.headers)

    def test_a_session_opens_and_reports_the_chunk_size_to_use(self):
        response = self._begin(64)
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.addCleanup(self._forget, body["job_id"])
        self.assertEqual(body["chunk_size"], SETTINGS.upload_chunk_bytes)
        self.assertEqual(body["offset"], 0)
        self.assertRegex(body["job_id"], r"^[0-9a-f]{12}$")

    def test_consent_is_required_before_a_single_byte_is_accepted(self):
        """Stricter than the single-request path, not looser: the answers
        arrive before the footage rather than alongside it."""
        for missing in ("rights_confirmed", "people_permissions_confirmed"):
            with self.subTest(missing=missing):
                response = self._begin(64, **{missing: False})
                self.assertEqual(response.status_code, 400)
        response = self._begin(64, minor_permission_status="")
        self.assertEqual(response.status_code, 400)

    def test_pieces_arrive_and_the_offset_advances(self):
        opened = self._begin(9).json()
        job = opened["job_id"]
        self.addCleanup(self._forget, job)
        first = self.client.put("/api/upload/%s/chunk?offset=0" % job,
                                content=b"abcd", headers=self.headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["offset"], 4)
        self.assertFalse(first.json()["complete"])
        second = self.client.put("/api/upload/%s/chunk?offset=4" % job,
                                 content=b"efghi", headers=self.headers)
        self.assertEqual(second.json()["offset"], 9)
        self.assertTrue(second.json()["complete"])

    def test_a_lost_response_is_recovered_from_the_offset_it_returns(self):
        """The resume path, which is the whole point on a phone connection."""
        job = self._begin(9).json()["job_id"]
        self.addCleanup(self._forget, job)
        self.client.put("/api/upload/%s/chunk?offset=0" % job,
                        content=b"abcd", headers=self.headers)
        # The client believes it is still at zero and retries.
        stale = self.client.put("/api/upload/%s/chunk?offset=0" % job,
                                content=b"abcd", headers=self.headers)
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["offset"], 4, "told where to resume")
        # And a page reload can ask outright.
        status = self.client.get("/api/upload/%s/status" % job)
        self.assertEqual(status.json()["offset"], 4)
        resumed = self.client.put("/api/upload/%s/chunk?offset=4" % job,
                                  content=b"efghi", headers=self.headers)
        self.assertTrue(resumed.json()["complete"])

    def test_another_account_cannot_touch_the_upload(self):
        from core.auth import create_account, issue_session

        job = self._begin(9).json()["job_id"]
        self.addCleanup(self._forget, job)
        other = create_account("stranger-%s@example.com" % uuid.uuid4().hex[:10],
                               "a-long-enough-password")
        thief = TestClient(app)
        try:
            thief.cookies.set("warrioriq_session", issue_session(int(other["id"])))
            thief.get("/analyze/kickboxing")
            headers = {"X-CSRF-Token": thief.cookies.get("warrioriq_csrf")}
            response = thief.put("/api/upload/%s/chunk?offset=0" % job,
                                 content=b"xxxx", headers=headers)
            self.assertEqual(response.status_code, 404)
        finally:
            thief.close()
        self.assertEqual(chunked_upload.part_path(job).stat().st_size, 0,
                         "a stranger's bytes must not reach the file")

    def test_aborting_gives_the_capacity_straight_back(self):
        job = self._begin(9).json()["job_id"]
        self.assertTrue(chunked_upload.part_path(job).exists())
        response = self.client.post("/api/upload/%s/abort" % job, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(chunked_upload.part_path(job).exists())
        self.assertFalse(chunked_upload.meta_path(job).exists())

    def test_finishing_an_incomplete_upload_is_refused(self):
        job = self._begin(9).json()["job_id"]
        self.addCleanup(self._forget, job)
        self.client.put("/api/upload/%s/chunk?offset=0" % job,
                        content=b"abcd", headers=self.headers)
        response = self.client.post("/api/upload/%s/finish" % job, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["offset"], 4)

    def test_a_complete_upload_that_is_not_a_video_is_refused_by_the_pipeline(self):
        """The handover works: finish reaches the container check the
        single-request path uses, and that check does its job."""
        payload = b"not a video at all"
        job = self._begin(len(payload)).json()["job_id"]
        self.addCleanup(self._forget, job)
        self.client.put("/api/upload/%s/chunk?offset=0" % job,
                        content=payload, headers=self.headers)
        response = self.client.post("/api/upload/%s/finish" % job, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a video", response.text.lower())
        # And nothing was left behind pretending to be a fight.
        self.assertFalse((UPLOADS / ("%s.mp4" % job)).exists())

    def test_every_write_endpoint_requires_the_csrf_token(self):
        """Driven with a raw client, because the suite's BrowserClient adds
        the header automatically - which is right for every other test and
        exactly wrong for this one."""
        from fastapi.testclient import TestClient as RawClient

        job = self._begin(9).json()["job_id"]
        self.addCleanup(self._forget, job)
        raw = RawClient(app)
        raw.cookies.set("warrioriq_session", self.client.cookies.get("warrioriq_session"))
        raw.cookies.set("warrioriq_csrf", self.token)
        try:
            for method, path, kwargs in (
                    ("post", "/api/upload/begin", {"json": {}}),
                    ("put", "/api/upload/%s/chunk?offset=0" % job, {"content": b"ab"}),
                    ("post", "/api/upload/%s/abort" % job, {}),
                    ("post", "/api/upload/%s/finish" % job, {})):
                with self.subTest(path=path):
                    response = getattr(raw, method)(path, **kwargs)
                    self.assertEqual(response.status_code, 403,
                                     "%s accepted a request with no token" % path)
        finally:
            raw.close()

    def _real_clip(self) -> bytes:
        """A short H.264 clip, because the pipeline decodes what it is given.

        mp4v is what cv2 writes by default and no browser decodes it, so it is
        transcoded - the same reason the upload route normalises containers.
        """
        import math
        import subprocess
        import tempfile

        try:
            import cv2
            import imageio_ffmpeg
            import numpy as np
        except ImportError:  # pragma: no cover - only on a trimmed install
            self.skipTest("cv2 / ffmpeg are not available")
        folder = Path(tempfile.mkdtemp(prefix="wiq-clip-"))
        raw, clip = folder / "raw.mp4", folder / "fight.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
        rng = np.random.default_rng(1)
        background = np.full((180, 320, 3), (74, 48, 32), np.uint8)
        for _ in range(200):
            x, y = int(rng.integers(0, 316)), int(rng.integers(0, 176))
            background[y:y + 3, x:x + 3] = rng.integers(40, 220, 3)
        for index in range(30 * 11):
            seconds = index / 30
            frame = background.copy()
            for side, centre in enumerate((0.38, 0.58)):
                left = int(320 * centre + math.sin(seconds * 3 + side) * 16)
                cv2.rectangle(frame, (left, 90), (left + 20, 170),
                              (237, 111, 47) if side else (77, 72, 224), -1)
            writer.write(frame)
        writer.release()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-g", "30", str(clip)], check=True)
        payload = clip.read_bytes()
        self.addCleanup(shutil.rmtree, folder, True)
        return payload

    def _upload_in_pieces(self, payload: bytes, **extra):
        opened = self._begin(len(payload), round_count=1, **extra).json()
        job, size = opened["job_id"], opened["chunk_size"]
        self.addCleanup(self._forget, job)
        sent = pieces = 0
        while sent < len(payload):
            response = self.client.put(
                "/api/upload/%s/chunk?offset=%d" % (job, sent),
                content=payload[sent:sent + size], headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            sent = response.json()["offset"]
            pieces += 1
        return job, pieces, self.client.post("/api/upload/%s/finish" % job,
                                             headers=self.headers)

    def test_a_real_video_uploads_in_pieces_and_becomes_a_job(self):
        """The whole path, end to end, on a file the decoder accepts."""
        from app.state import get_job

        payload = self._real_clip()
        job, pieces, finished = self._upload_in_pieces(payload)
        self.assertGreater(pieces, 1, "the clip should have needed more than one piece")
        self.assertEqual(finished.status_code, 201, finished.text[:200])
        self.assertEqual(finished.json()["job_id"], job)
        self.assertEqual(finished.json()["next_url"], "/frame/%s" % job)
        self.assertTrue(get_job(job), "the fight did not reach the job store")
        self.assertTrue((UPLOADS / ("%s.mp4" % job)).exists())
        # The working files are gone, not left to be swept later.
        self.assertFalse(chunked_upload.part_path(job).exists())
        self.assertFalse(chunked_upload.meta_path(job).exists())

    def test_finish_answers_json_whatever_the_caller_asked_for(self):
        """The handler renders the frame page unless the caller asks for JSON.

        finish is always called by a script, and the first run of this path
        got a 200 HTML page - which the success check then read as a failure
        and handed the reservation back underneath a fight that had actually
        uploaded. The contract is finish's own now, not a consequence of what
        the client happened to send.
        """
        payload = self._real_clip()
        job, _, finished = self._upload_in_pieces(payload)
        self.assertEqual(finished.status_code, 201)
        self.assertTrue(finished.headers["content-type"].startswith("application/json"))
        # Even when the caller explicitly asks for HTML.
        self.assertIn("job_id", finished.json())

    def test_a_refusal_releases_the_reservation_however_it_is_spelled(self):
        """The success check listed status codes and treated everything else
        as a failure, which is the wrong way round: any 4xx or 5xx is the
        refusal, and anything else is a fight that uploaded."""
        source = (Path(__file__).resolve().parents[1]
                  / "app" / "main.py").read_text(encoding="utf-8")
        start = source.index("async def chunked_upload_finish")
        body = source[start:source.index('@app.put("/api/upload/probe"', start)]
        self.assertIn("response.status_code >= 400", body)
        self.assertNotIn("not in {201, 303}", body)


class ClientTests(unittest.TestCase):
    """The page sends in pieces, and knows how to stop."""

    def setUp(self):
        self.page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                     / "analyze.html").read_text(encoding="utf-8")

    def test_it_sends_pieces_sequentially(self):
        """Not in parallel: the server is a small number of Passenger
        workers, and parallel chunks would occupy all of them for the length
        of one upload."""
        self.assertIn("while (offset < file.size)", self.page)
        self.assertIn("file.slice(offset", self.page)
        self.assertNotIn("Promise.all(", self.page)

    def test_it_resumes_from_the_offset_the_server_reports(self):
        self.assertIn("response.status === 409", self.page)
        self.assertIn("offset = told.offset", self.page)

    def test_it_falls_back_to_the_single_request_upload(self):
        """The flag is a real rollback: a 404 from begin means an older
        deploy or the feature switched off, and the old path still works."""
        self.assertIn("if (opened.status === 404) throw FALLBACK", self.page)
        self.assertIn("legacySend()", self.page)
        self.assertIn("const request=new XMLHttpRequest();", self.page)

    def test_it_gives_the_reservation_back_when_cancelled(self):
        self.assertIn("/abort", self.page)

    def test_it_retries_a_dropped_chunk_before_giving_up(self):
        self.assertIn("if (++attempts > 3) throw", self.page)

    def test_every_call_carries_the_csrf_header(self):
        for call in ("/api/upload/begin", "/chunk?offset=", "/finish"):
            with self.subTest(call=call):
                start = self.page.index(call)
                window = self.page[start:start + 320]
                self.assertIn("wiqCsrfHeaders", window,
                              f"{call} is sent without a CSRF token")


if __name__ == "__main__":
    unittest.main()
