"""The pre-upload check must agree with the probe that runs after the upload.

The landing page's step 01 - "a quick quality check catches hard-to-see footage
before the analysis starts" - was not true. The browser read the file's size
and its duration; videoWidth was never touched, so the one thing that decides
whether an analysis can work went unexamined until the worker had the file.

These cover the half that can be checked in Python: that the thresholds the
browser is handed are the worker's own, and that the arithmetic converting a
subject's height into detector pixels matches `core.preflight`.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from core import preflight
from core.preflight_client import (
    DETECTOR_FLOOR_PX,
    DETECTOR_RELIABLE_PX,
    MIN_TRUSTWORTHY_MOTION_SHARE,
    client_thresholds,
    subject_px_in_network,
)


class ThresholdsComeFromTheProbeTests(unittest.TestCase):
    def test_the_browser_is_given_the_workers_own_numbers(self):
        limits = client_thresholds()
        self.assertEqual(limits["target_subject_px"], preflight.TARGET_SUBJECT_PX)
        self.assertEqual(limits["min_inference_size"], preflight.MIN_INFERENCE_SIZE)
        self.assertEqual(limits["max_inference_size"], preflight.MAX_INFERENCE_SIZE)
        self.assertEqual(limits["min_usable_long_edge"], preflight.MIN_USABLE_LONG_EDGE)
        self.assertEqual(limits["min_usable_seconds"], preflight.MIN_USABLE_SECONDS)
        self.assertEqual(limits["well_framed_share"], preflight.WELL_FRAMED_SHARE)

    def test_every_threshold_survives_json(self):
        """It is rendered into the page as JSON, so a value that will not
        serialise is a page that will not parse."""
        json.loads(json.dumps(client_thresholds()))

    def test_the_detector_band_is_ordered(self):
        self.assertLess(DETECTOR_FLOOR_PX, DETECTOR_RELIABLE_PX)
        self.assertLessEqual(DETECTOR_RELIABLE_PX, preflight.TARGET_SUBJECT_PX)


class NetworkPixelArithmeticTests(unittest.TestCase):
    """The number shown to the filmer is the number the detector will see."""

    def test_it_matches_the_probes_own_inference_size(self):
        for long_edge in (640, 854, 1280, 1920, 2560, 3840):
            for subject in (40, 60, 90, 150, 200, 400):
                with self.subTest(long_edge=long_edge, subject=subject):
                    size = preflight.inference_size_for_subject(long_edge, subject)
                    self.assertAlmostEqual(
                        subject_px_in_network(long_edge, subject),
                        subject * size / long_edge, places=6)

    def test_the_same_subject_gets_worse_as_the_source_gets_bigger(self):
        """The counter-intuitive result the whole check turns on.

        A fighter 90 px tall is fine in a 640-wide clip and hopeless in a 4K
        one, because the downscale to a clamped inference size shrinks them
        further. Measuring source pixels alone would therefore give the wrong
        answer on exactly the phone footage this is for - which is why the
        browser measures a share of frame height rather than a pixel count.
        """
        at = {edge: subject_px_in_network(edge, 90) for edge in (640, 1280, 1920, 3840)}
        self.assertGreater(at[640], DETECTOR_RELIABLE_PX)
        self.assertLess(at[1920], DETECTOR_FLOOR_PX)
        self.assertLess(at[3840], at[1920])
        self.assertLess(at[1920], at[1280])
        self.assertLess(at[1280], at[640])

    def test_nothing_is_claimed_about_a_video_with_no_measurements(self):
        self.assertEqual(subject_px_in_network(0, 90), 0.0)
        self.assertEqual(subject_px_in_network(1920, 0), 0.0)

    def test_the_motion_floor_admits_a_distant_pair_of_fighters(self):
        """Set by measurement, not by what sounded like a lot of movement.

        At 0.004 this floor rejected the far-away case the check exists for:
        two fighters 8% of the height of a 1080p frame moved 0.28%-0.52% of
        the pixels, median 0.39%, and were reported as a video in which
        nothing moves.
        """
        self.assertLess(MIN_TRUSTWORTHY_MOTION_SHARE, 0.0028)


class PageWiringTests(unittest.TestCase):
    """The check has to actually be on the page, reading these numbers."""

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.page = (root / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.script = (root / "app" / "static" / "preflight.js").read_text(encoding="utf-8")

    def test_the_page_loads_the_check_and_hands_it_the_limits(self):
        self.assertIn("/static/preflight.js", self.page)
        self.assertIn("const PREFLIGHT_LIMITS={{preflight_limits|safe}}", self.page)
        self.assertIn("runPreflight(file)", self.page)

    def test_the_check_reads_the_frame_size_it_was_accused_of_ignoring(self):
        self.assertIn("videoWidth", self.script)
        self.assertIn("videoHeight", self.script)

    def test_it_warns_and_never_blocks(self):
        """A check this new must not be able to refuse somebody's fight.

        Nothing here is validated well enough for that, and a check that is
        wrong AND final is worse than one that is wrong and advisory.
        """
        for forbidden in ("uploadSubmit.disabled=true", "rejectFile("):
            self.assertNotIn(forbidden, self._preflight_block())

    def _preflight_block(self) -> str:
        start = self.page.index("const runPreflight")
        return self.page[start:self.page.index("const restore=", start)]

    def test_the_three_measurements_the_audit_asked_for_are_reported(self):
        block = self._preflight_block()
        self.assertIn("m.width", block)          # source resolution
        self.assertIn("m.fps", block)            # frame rate
        self.assertIn("m.subjectShare", block)   # subject size in frame

    def test_it_says_when_it_could_not_measure_rather_than_guessing(self):
        block = self._preflight_block()
        self.assertIn("camera moving", block)
        self.assertIn("nothing moving", block)
        # And the reasons the panel words are reasons the script can produce.
        for reason in ("camera moving", "nothing moving", "no duration"):
            self.assertIn("'" + reason + "'", self.script)

    def test_no_threshold_is_hard_coded_in_the_page(self):
        """Every number compared against comes from the server's dict, so a
        threshold cannot drift between the warning and the analysis."""
        block = self._preflight_block()
        for name in ("detector_floor_px", "detector_reliable_px", "min_usable_fps",
                     "min_usable_long_edge"):
            self.assertIn(f"PREFLIGHT_LIMITS.{name}", block)
        # No bare three-digit pixel thresholds left inline.
        self.assertNotIn("< 120", block.replace(" ", ""))
        self.assertNotIn("<190", block.replace(" ", ""))

    def test_the_claim_on_the_landing_page_is_now_backed_by_a_measurement(self):
        """Step 01 promises this check. It is the reason the module exists, so
        if the promise is reworded this test should be read again rather than
        quietly passing on a claim nothing backs."""
        root = Path(__file__).resolve().parents[1]
        landing = (root / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIsNotNone(
            re.search(r"quality check catches hard-to-see footage", landing),
            "the landing page promise moved; this check should follow it")
        main = (root / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('templates.env.globals["preflight_limits"]', main)


if __name__ == "__main__":
    unittest.main()
