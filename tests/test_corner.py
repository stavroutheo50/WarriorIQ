"""Corner colour has to be read from evidence, and refused when there is none."""
from __future__ import annotations

import unittest

import numpy as np

from core.corner import (
    CLEAR_COLOUR, DECIDE_AFTER_FRAMES, DECIDED_FRACTION, CornerReader,
    CornerReading, assign_corners, colour_scores, read_corners, score_detection,
)

RED_BGR = (40, 40, 210)
BLUE_BGR = (205, 70, 40)
# The competition mat in this library is itself red and blue, which is the
# whole reason sampling happens inside the fighter rather than across the box.
# This one is blue-hued on purpose: a sampler that strays onto the floor should
# be caught by a test, not in production.
MAT_BGR = (150, 80, 60)
# A fighter wearing no corner colour still wears something. Near-grey kit is
# unsaturated, so it casts no vote either way - which is the correct answer for
# a competitor whose corner cannot be read.
NEUTRAL_BGR = (118, 120, 122)


class Person:
    """The two attributes core/corner.py reads off a detection."""

    def __init__(self, box, keypoints=None, referee_prob=0.0):
        self.box = np.asarray(box, dtype=np.float32)
        self.keypoints = keypoints
        self.referee_prob = referee_prob


def skeleton(cx, cy, height):
    """COCO-ordered keypoints for an upright figure centred on (cx, cy).

    The hands are held out from the body on purpose. An earlier version of this
    helper put the wrists at the same coordinates as the hips, which dropped
    the glove discs inside the torso quad and made a gear-painted fighter read
    red on the torso - the fixture inventing the very confusion the module is
    meant to resolve.
    """
    half, quarter = height / 2.0, height / 4.0
    points = [[float(cx), float(cy - half)]] * 5                             # nose, eyes, ears
    points += [[cx - quarter, cy - quarter], [cx + quarter, cy - quarter]]   # shoulders
    points += [[cx - quarter * 1.4, cy - quarter * 0.2],
               [cx + quarter * 1.4, cy - quarter * 0.2]]                     # elbows
    points += [[cx - quarter * 1.7, cy + quarter * 0.2],
               [cx + quarter * 1.7, cy + quarter * 0.2]]                     # wrists
    points += [[cx - quarter * 0.6, cy + quarter],
               [cx + quarter * 0.6, cy + quarter]]                           # hips
    points += [[cx - quarter * 0.6, cy + half * 0.75],
               [cx + quarter * 0.6, cy + half * 0.75]]                       # knees
    points += [[cx - quarter * 0.6, cy + half], [cx + quarter * 0.6, cy + half]]  # ankles
    return [[float(x), float(y)] for x, y in points]


def frame_with(fighters, region, size=(240, 420)):
    """A mat-coloured frame with each fighter painted in one region only.

    Painting a single region is the point: footage that carries the corner on
    the uniform and footage that carries it on the gloves both exist, and the
    reader is supposed to work out which without being told.
    """
    height, width = size
    frame = np.full((height, width, 3), MAT_BGR, dtype=np.uint8)
    people = []
    for cx, cy, tall, colour in fighters:
        keypoints = skeleton(cx, cy, tall)
        box = [cx - tall / 4.0, cy - tall / 2.0, cx + tall / 4.0, cy + tall / 2.0]
        # Every fighter is painted: one in a corner colour, one in neutral kit.
        # Leaving a fighter unpainted would sample bare mat and measure the
        # floor's colour as if it were theirs. The fill is wider than the box
        # so the gear discs at head and ankles land on a body rather than off
        # its edge onto the mat, which is what a real fighter looks like.
        pad = int(tall * 0.14)
        x1, x2 = int(cx - tall / 2) - pad, int(cx + tall / 2) + pad
        y1, y2 = int(cy - tall / 2) - pad, int(cy + tall / 2) + pad
        frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = NEUTRAL_BGR
        if colour is not None:
            if region == "torso":
                x1, x2 = int(cx - tall / 4), int(cx + tall / 4)
                y1, y2 = int(cy - tall / 4), int(cy)
                frame[y1:y2, x1:x2] = colour
            else:                                   # gear: head, hands, feet
                for index in (0, 9, 10, 15, 16):
                    px, py = keypoints[index]
                    r = max(3, int(tall * 0.07))
                    frame[int(py) - r:int(py) + r, int(px) - r:int(px) + r] = colour
        people.append(Person(box, keypoints))
    return frame, people


class ColourScoreTests(unittest.TestCase):
    def test_red_and_blue_read_as_themselves(self):
        red = np.array([RED_BGR] * 40, dtype=np.uint8)
        blue = np.array([BLUE_BGR] * 40, dtype=np.uint8)
        self.assertGreater(colour_scores(red)[0], 0.9)
        self.assertGreater(colour_scores(blue)[1], 0.9)

    def test_unsaturated_pixels_do_not_vote(self):
        """Shadow, blur and washed-out floor are not evidence of a corner."""
        grey = np.array([[128, 128, 128]] * 60, dtype=np.uint8)
        self.assertEqual(colour_scores(grey), (0.0, 0.0))

    def test_too_few_pixels_is_not_a_reading(self):
        self.assertEqual(colour_scores(np.array([RED_BGR], dtype=np.uint8)), (0.0, 0.0))
        self.assertEqual(colour_scores(None), (0.0, 0.0))


class RegionTests(unittest.TestCase):
    def test_a_torso_colour_is_read_from_the_torso(self):
        frame, people = frame_with([(120, 120, 120, RED_BGR)], "torso")
        red, _ = score_detection(frame, people[0].box, people[0].keypoints, "torso")
        self.assertGreater(red, CLEAR_COLOUR)

    def test_gear_colour_is_missed_by_torso_sampling(self):
        """The reason the region cannot be hardcoded.

        Point-fighting puts the corner on gloves, headguard and footpads. A
        torso sample of that fighter measures their uniform, which carries no
        corner at all - which is exactly what happened on two of the four
        fights this was built against.
        """
        frame, people = frame_with([(120, 120, 120, RED_BGR)], "gear")
        torso_red, _ = score_detection(frame, people[0].box, people[0].keypoints, "torso")
        gear_red, _ = score_detection(frame, people[0].box, people[0].keypoints, "gear")
        self.assertLess(torso_red, CLEAR_COLOUR)
        self.assertGreater(gear_red, CLEAR_COLOUR)


class ReadCornersTests(unittest.TestCase):
    def samples(self, region, colours, count=20):
        out = []
        for _ in range(count):
            out.append(frame_with([(110, 120, 120, colours[0]),
                                   (300, 120, 120, colours[1])], region))
        return out

    def test_red_against_blue_on_the_torso_is_decided(self):
        reading = read_corners(self.samples("torso", (RED_BGR, BLUE_BGR)))
        self.assertTrue(reading.decided)
        self.assertEqual(reading.region, "torso")
        self.assertGreaterEqual(reading.separation, DECIDED_FRACTION)

    def test_red_against_blue_on_the_gear_is_decided_and_names_gear(self):
        reading = read_corners(self.samples("gear", (RED_BGR, BLUE_BGR)))
        self.assertTrue(reading.decided)
        self.assertEqual(reading.region, "gear")
        # The region that carries nothing must not be the one that wins.
        self.assertGreater(reading.per_region["gear"], reading.per_region["torso"])

    def test_two_fighters_in_the_same_colour_is_refused(self):
        """Measured on real footage: both competitors in blue, and no method
        can separate them. Saying so beats asserting a corner."""
        reading = read_corners(self.samples("gear", (BLUE_BGR, BLUE_BGR)))
        self.assertFalse(reading.decided)
        self.assertIsNone(reading.region)

    def test_one_coloured_fighter_is_not_a_separation(self):
        """A frame only separates when one reads red AND the other reads blue."""
        reading = read_corners(self.samples("torso", (RED_BGR, None)))
        self.assertFalse(reading.decided)

    def test_nothing_to_compare_is_reported_not_guessed(self):
        frame, people = frame_with([(110, 120, 120, RED_BGR)], "torso")
        reading = read_corners([(frame, people)])
        self.assertFalse(reading.decided)
        self.assertEqual(reading.frames_scored, 0)
        self.assertIn("two fighters", reading.reason)

    def test_empty_input_is_safe(self):
        reading = read_corners([])
        self.assertFalse(reading.decided)
        self.assertIsInstance(reading, CornerReading)

    def test_reading_serialises_for_the_report(self):
        reading = read_corners(self.samples("torso", (RED_BGR, BLUE_BGR)))
        payload = reading.to_dict()
        self.assertEqual(payload["region"], "torso")
        self.assertTrue(payload["decided"])
        self.assertIn("torso", payload["per_region"])


class AssignCornersTests(unittest.TestCase):
    """Which of these two is the red corner, in a region already decided.

    Split from the region decision on purpose. The region is a question about
    the whole fight and is answered from many frames; which fighter wears which
    colour is a question about two particular boxes and can only be answered
    from the frame they are in.
    """

    def pair(self, region, colours, size=(240, 420)):
        frame, people = frame_with(
            [(120, 120, 150, colours[0]), (300, 120, 150, colours[1])],
            region, size=size)
        return frame, people[0], people[1]

    def test_red_on_the_left_is_named_red(self):
        frame, a, b = self.pair("torso", (RED_BGR, BLUE_BGR))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "torso"),
            ("red", "blue"))

    def test_the_answer_follows_the_fighters_and_not_their_order(self):
        frame, a, b = self.pair("torso", (BLUE_BGR, RED_BGR))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "torso"),
            ("blue", "red"))

    def test_gear_colours_are_found_in_the_gear_region(self):
        frame, a, b = self.pair("gear", (RED_BGR, BLUE_BGR))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "gear"),
            ("red", "blue"))

    def test_two_fighters_in_one_colour_name_neither(self):
        frame, a, b = self.pair("torso", (BLUE_BGR, BLUE_BGR))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "torso"),
            (None, None))

    def test_one_uncoloured_fighter_names_neither(self):
        """Half an answer is not an answer: both have to read clearly."""
        frame, a, b = self.pair("torso", (RED_BGR, None))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "torso"),
            (None, None))

    def test_looking_in_the_wrong_region_names_neither(self):
        """Gear-coloured fighters, asked about the torso."""
        frame, a, b = self.pair("gear", (RED_BGR, BLUE_BGR))
        self.assertEqual(
            assign_corners(frame, a.box, a.keypoints, b.box, b.keypoints, "torso"),
            (None, None))


class CornerReaderTests(unittest.TestCase):
    """The accumulating form, which is what an analysis actually uses.

    read_corners() takes a finished list of samples. An analysis does not have
    one: it has frames arriving one at a time, and holding hundreds of them to
    ask a colour question afterwards is not affordable.
    """

    def frame(self, colours, region="torso"):
        return frame_with([(120, 120, 150, colours[0]), (300, 120, 150, colours[1])],
                          region)

    def test_it_agrees_with_read_corners_on_the_same_frames(self):
        samples = [self.frame((RED_BGR, BLUE_BGR)) for _ in range(5)]
        reader = CornerReader()
        for frame, people in samples:
            reader.observe(frame, people)
        self.assertEqual(reader.decide().region,
                         read_corners(samples).region)

    def test_frames_without_two_fighters_are_not_counted(self):
        frame, people = self.frame((RED_BGR, BLUE_BGR))
        reader = CornerReader()
        reader.observe(frame, people[:1])
        reader.observe(frame, [])
        reader.observe(None, people)
        self.assertEqual(reader.frames_scored, 0)
        reader.observe(frame, people)
        self.assertEqual(reader.frames_scored, 1)

    def test_nothing_observed_decides_nothing(self):
        reading = CornerReader().decide()
        self.assertFalse(reading.decided)
        self.assertIsNone(reading.region)

    def test_the_round_is_watched_for_long_enough_to_outvote_one_frame(self):
        """The whole point of DECIDE_AFTER_FRAMES.

        Measured on this project's footage, a single frame picks torso or gear
        depending only on which fighter is considered first, so the threshold
        has to be well above one - and below what a short clip can supply.
        """
        self.assertGreater(DECIDE_AFTER_FRAMES, 10)
        self.assertLess(DECIDE_AFTER_FRAMES, 200)


if __name__ == "__main__":
    unittest.main()
