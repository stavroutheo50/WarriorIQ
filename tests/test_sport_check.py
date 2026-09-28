"""The optional "does this video show the sport you chose?" check."""
from __future__ import annotations

import base64
import os
import unittest
from unittest import mock

from core import sport_check

JPEG = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0" + b"0" * 64).decode()


class ConfigurationTests(unittest.TestCase):
    def test_off_unless_a_provider_and_its_key_are_both_set(self):
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "", "ANTHROPIC_API_KEY": "k"}):
            self.assertIsNone(sport_check.provider())
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": ""}):
            self.assertIsNone(sport_check.provider())
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}):
            self.assertEqual(sport_check.provider(), "anthropic")
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "openai", "OPENAI_API_KEY": "k"}):
            self.assertEqual(sport_check.provider(), "openai")
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "somebody-else", "OPENAI_API_KEY": "k"}):
            self.assertIsNone(sport_check.provider())

    def test_nothing_is_sent_when_off(self):
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": ""}), \
             mock.patch.object(sport_check, "_ask_anthropic") as ask:
            self.assertIsNone(sport_check.detect_sport([b"x"]))
            ask.assert_not_called()


class FrameTests(unittest.TestCase):
    def test_accepts_up_to_three_jpegs(self):
        self.assertEqual(len(sport_check.decode_frames([JPEG, JPEG, JPEG])), 3)

    def test_refuses_anything_else(self):
        for frames in ([], [JPEG] * 4, ["data:image/png;base64,AAAA"], ["not a data url"],
                       ["data:image/jpeg;base64," + base64.b64encode(b"GIF89a").decode()],
                       ["data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"0" * 500_000).decode()]):
            with self.subTest(n=len(frames)), self.assertRaises(ValueError):
                sport_check.decode_frames(frames)


class AnswerTests(unittest.TestCase):
    def _detect(self, answer=None, error=None):
        with mock.patch.dict(os.environ, {"WARRIORIQ_SPORT_CHECK_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}), \
             mock.patch.object(sport_check, "_ask_anthropic", side_effect=error, return_value=answer):
            return sport_check.detect_sport([b"x"])

    def test_a_clean_answer_passes_through(self):
        got = self._detect({"sport": "taekwondo", "confidence": 0.93, "reason": "doboks, no gloves"})
        self.assertEqual(got, {"sport": "taekwondo", "confidence": 0.93, "reason": "doboks, no gloves"})

    def test_any_failure_is_silence_not_an_error(self):
        self.assertIsNone(self._detect(error=RuntimeError("network down")))
        self.assertIsNone(self._detect({"sport": "curling", "confidence": 0.9, "reason": ""}))
        self.assertIsNone(self._detect({"sport": "boxing", "confidence": "high", "reason": ""}))
        self.assertIsNone(self._detect(None))

    def test_only_a_confident_different_sport_is_a_mismatch(self):
        v = sport_check.verdict
        self.assertFalse(v("kickboxing", None)["mismatch"])
        self.assertFalse(v("kickboxing", {"sport": "kickboxing", "confidence": 0.99, "reason": ""})["mismatch"])
        self.assertFalse(v("kickboxing", {"sport": "unclear", "confidence": 0.99, "reason": ""})["mismatch"])
        self.assertFalse(v("kickboxing", {"sport": "taekwondo", "confidence": 0.6, "reason": ""})["mismatch"])
        hit = v("kickboxing", {"sport": "taekwondo", "confidence": 0.9, "reason": "doboks"})
        self.assertTrue(hit["mismatch"])
        self.assertEqual(hit["detected"], "taekwondo")


class EndpointTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as webapp
        self.webapp = webapp
        self.client = TestClient(webapp.app)

    def test_off_means_nothing_happens(self):
        with mock.patch.object(sport_check, "provider", return_value=None):
            got = self.client.post("/api/sport-check", json={"sport": "kickboxing", "frames": [JPEG]},
                                   headers={"X-CSRF-Token": "x"})
        self.assertIn(got.status_code, (200, 403))
        if got.status_code == 200:
            self.assertEqual(got.json(), {"available": False})

    def test_the_upload_page_only_carries_the_check_when_it_is_on(self):
        page = self.webapp.templates.env.get_template("analyze.html")
        source = page.environment.loader.get_source(page.environment, "analyze.html")[0]
        self.assertIn("{% if sport_check_enabled %}", source)
        self.assertIn("runSportCheck(file)", source)


if __name__ == "__main__":
    unittest.main()
