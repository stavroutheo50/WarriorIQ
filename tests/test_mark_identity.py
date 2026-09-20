"""The identity marker: the cheap instrument, and the ways it silently lied before."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mark_identity", ROOT / "tools" / "mark_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MarkIdentityTests(unittest.TestCase):
    def test_the_save_target_does_not_depend_on_a_trailing_slash(self):
        """The label pack lost 32 answers to exactly this, so it is pinned here.

        A relative fetch target resolves against the *directory* of the current
        page. Served at /<job> with no trailing slash that directory is "/", so
        the POST goes somewhere the server does not route - and the page cannot
        tell, because a 404 is a perfectly good response.
        """
        page = _module().PAGE
        self.assertIn('fetch("/" + encodeURIComponent(JOB) + "/marks"', page)
        self.assertEqual(page.count("fetch("), 1, "one call site, so one thing to get right")

    def test_it_does_not_claim_to_be_saving_before_it_has_saved(self):
        """The pill read 'saving to disk' in green over 32 answers that 404'd."""
        page = _module().PAGE
        self.assertIn('saveState = "unproven"', page)
        self.assertIn("not saved yet", page)

    def test_a_held_key_cannot_run_away_when_the_window_loses_focus(self):
        """keyup never arrives if the key is released elsewhere.

        Without this the mark stays open and swallows the rest of the fight,
        which would read as "wrong for 94 seconds" and be believed.
        """
        page = _module().PAGE
        self.assertIn('addEventListener("blur"', page)
        self.assertIn('v.addEventListener("pause"', page)

    def test_tracking_is_reduced_to_the_boxes_and_sorted_by_time(self):
        from core.config import OUTPUTS

        module = _module()
        job = OUTPUTS / "__marktest__"
        job.mkdir(parents=True, exist_ok=True)
        try:
            (job / "tracking.jsonl").write_text("\n".join([
                json.dumps({"time_seconds": 2.0, "fighter_A": {"observation": {"box": [1, 2, 3, 4]}}}),
                "not json at all",
                json.dumps({"time_seconds": 1.0, "fighter_B": {"observation": {"box": [5, 6, 7, 8]}}}),
                json.dumps({"time_seconds": 3.0, "fighter_A": {"observation": {}}}),
            ]), encoding="utf-8")
            frames = module.load_tracking("__marktest__")
        finally:
            (job / "tracking.jsonl").unlink(missing_ok=True)
            (job / ".state.lock").unlink(missing_ok=True)
            job.rmdir()
        self.assertEqual([f["t"] for f in frames], [1.0, 2.0, 3.0], "unparseable lines are skipped, order is time")
        self.assertEqual(frames[0]["B"], [5.0, 6.0, 7.0, 8.0])
        self.assertNotIn("A", frames[2], "a frame with no box must not claim one")


if __name__ == "__main__":
    unittest.main()
