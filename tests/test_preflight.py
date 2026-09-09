"""The pre-flight probe: what it measures, and what it refuses to promise."""

import unittest

from core.preflight import (
    MIN_USABLE_LONG_EDGE,
    MIN_USABLE_SECONDS,
    MAX_INFERENCE_SIZE,
    MIN_INFERENCE_SIZE,
    Preflight,
    REFERENCE_SUBJECT_PX,
    TARGET_SUBJECT_PX,
    _judge,
    inference_size_for_subject,
)


class InferenceSizeFromSubjectTests(unittest.TestCase):
    """Choose the size from how tall the subject is, not from the resolution.

    Measured on fight 3 by sweeping the inference size alone (568x320, fighter
    76 px): the subject's height *in the network input* is what governs
    detection, from 0.3 people found at 69 px to 15.0 at 214 px.
    """

    def _subject_in_network(self, long_edge, subject_px):
        return subject_px * inference_size_for_subject(long_edge, subject_px) / long_edge

    def test_small_and_large_sources_both_land_near_the_target(self):
        for long_edge, subject in ((568, 76), (480, 60), (1920, 400), (3840, 900)):
            self.assertAlmostEqual(
                self._subject_in_network(long_edge, subject), TARGET_SUBJECT_PX,
                delta=25, msg=f"{long_edge} wide, subject {subject}px")

    def test_a_phone_video_is_not_downscaled_into_the_dead_zone(self):
        """The rule this replaces gave every source over 960 the default 640.

        A 1080p clip of a fighter 200 px tall put them at 200 * 640/1920 = 67 px
        in the network, where the sweep measured 0.3 people found per frame.
        """
        from core.pose_tracker import inference_size

        old = inference_size(1920, 1080)
        self.assertEqual(old, 640, "the fallback rule is unchanged")
        self.assertLess(200 * old / 1920, 70, "which is the dead zone")

        new = inference_size_for_subject(1920, 200)
        self.assertGreater(200 * new / 1920, 170, "the measured rule is not")

    def test_a_close_subject_would_ask_for_less_work(self):
        """The function alone would shrink for an already-large subject.

        Note that QualityController does *not* take this saving: it floors the
        measured size at the old rule's choice, because lowering 1600 to 1440
        on fight 3 cost coverage. This asserts the arithmetic, not the policy.
        """
        wide = inference_size_for_subject(1920, 200)
        close = inference_size_for_subject(1920, 700)
        self.assertLess(close, wide)

    def test_sizes_are_clamped_and_stride_aligned(self):
        for long_edge, subject in ((1920, 5), (1920, 1900), (640, 1), (4000, 3999)):
            size = inference_size_for_subject(long_edge, subject)
            self.assertGreaterEqual(size, MIN_INFERENCE_SIZE)
            self.assertLessEqual(size, MAX_INFERENCE_SIZE)
            self.assertEqual(size % 32, 0, f"{long_edge}/{subject} is not stride aligned")

    def test_unknown_input_does_not_silently_change_the_analysis(self):
        from core.config import SETTINGS

        self.assertEqual(inference_size_for_subject(0, 0), SETTINGS.default_imgsz)
        self.assertEqual(inference_size_for_subject(1920, 0), SETTINGS.default_imgsz)


class JudgementTests(unittest.TestCase):
    """The wording is the product here, so the wording is tested."""

    def _report(self, **kwargs):
        report = Preflight(width=1920, height=1080, fps=30.0, frame_count=900,
                           measured=True, **kwargs)
        _judge(report)
        return report

    def test_footage_no_better_than_the_reference_is_always_flagged(self):
        """Fight 1's fighters are 59 px - one under the smallest reference.

        A range test let the worst footage this project owns pass with nothing
        said. Anything no bigger than the reference is at least as hard.
        """
        for subject in (40, 59, 60, max(REFERENCE_SUBJECT_PX)):
            report = self._report(subject_height_px=subject,
                                  subject_share_of_height=subject / 1080,
                                  subject_px_in_network=200, people_in_frame=4)
            self.assertTrue(
                any("no bigger than the tournament footage" in w for w in report.warnings),
                f"{subject}px was not flagged")

    def test_a_subject_too_small_to_detect_blocks_rather_than_warns(self):
        report = self._report(subject_height_px=30, subject_share_of_height=0.03,
                              subject_px_in_network=90, people_in_frame=2)
        self.assertTrue(report.blocking)
        self.assertFalse(report.can_analyse)

    def test_a_clean_recording_is_never_told_it_will_be_accurate(self):
        """Nothing here has been validated on footage better than the three
        wide tournament fights, so a green light would be invented."""
        report = self._report(subject_height_px=400, subject_share_of_height=0.37,
                              subject_px_in_network=210, people_in_frame=3)
        self.assertFalse(report.blocking)
        self.assertFalse(report.warnings)
        blob = " ".join(report.advice).lower()
        self.assertIn("not a promise", blob)
        # Claim words are fine inside the disclaimer that negates them - the
        # sentence is "not a promise the analysis will be accurate". What must
        # never appear is a claim that survives without one.
        for sentence in blob.split("."):
            if "not a promise" in sentence or "only that" in sentence:
                continue
            for forbidden in ("will work", "will be accurate", "good to go",
                              "guaranteed", "should be fine"):
                self.assertNotIn(forbidden, sentence)

    def test_a_busy_hall_is_reported_with_the_measured_rate(self):
        report = self._report(subject_height_px=400, subject_share_of_height=0.37,
                              subject_px_in_network=210, people_in_frame=16)
        self.assertTrue(any("one clip in eight" in w for w in report.warnings))

    def test_an_unmeasured_video_never_claims_it_can_be_analysed(self):
        self.assertFalse(Preflight().can_analyse)



class VerificationTagTests(unittest.TestCase):
    """Search Console ownership must be real or absent, never invented."""

    def test_no_token_renders_no_tag(self):
        from core.config import SETTINGS

        self.assertEqual(SETTINGS.site_verification_token, "",
                         "the default must be empty - a wrong token fails "
                         "verification silently and looks like never having tried")


class InferenceSizeIsAFloorTests(unittest.TestCase):
    """The measured size may raise the old rule's choice, never lower it.

    Measured on fight 3: letting it lower 1600 to 1440 cost coverage, A 0.4451
    -> 0.3973 and B 0.1987 -> 0.1598, on a deterministic pipeline.
    """

    def test_a_smaller_measured_size_does_not_shrink_a_small_source(self):
        from core.pose_tracker import QualityController, inference_size

        rule = inference_size(568, 320)
        controller = QualityController(30.0, 568, 320, measured_imgsz=1440)
        self.assertEqual(controller.base_imgsz, rule)
        self.assertGreaterEqual(controller.base_imgsz, 1440)

    def test_a_larger_measured_size_is_taken(self):
        from core.pose_tracker import QualityController, inference_size

        self.assertEqual(inference_size(1920, 1080), 640)
        controller = QualityController(30.0, 1920, 1080, measured_imgsz=960)
        self.assertEqual(controller.base_imgsz, 960,
                         "the high-resolution case this exists to fix")

    def test_no_measurement_falls_back_to_the_rule(self):
        from core.pose_tracker import QualityController, inference_size

        for w, h in ((568, 320), (1920, 1080), (0, 0)):
            self.assertEqual(QualityController(30.0, w, h).base_imgsz,
                             inference_size(w, h))


class UnusableFileTests(unittest.TestCase):
    """Say the real reason, not the first reason that happens to be true.

    A one-frame file and a 64x48 file both used to fall through to "no people
    could be found... the camera is too far away", which sends the filmer off
    to fix a thing that is not broken.
    """

    def test_a_clip_too_short_to_hold_an_exchange_says_so(self):
        report = Preflight(width=1920, height=1080, fps=30.0, frame_count=1)
        self.assertLess(report.frame_count / report.fps, MIN_USABLE_SECONDS)

    def test_the_thresholds_are_defensible(self):
        # The action windows in core/ are 0.6-1.5s, so under 2s cannot hold one.
        self.assertGreaterEqual(MIN_USABLE_SECONDS, 1.5)
        # And a frame smaller than this cannot show a limb at any framing.
        self.assertLessEqual(MIN_USABLE_LONG_EDGE, 480)


if __name__ == "__main__":
    unittest.main()
