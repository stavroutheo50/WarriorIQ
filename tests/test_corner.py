"""Corner colour has to be read from evidence, and refused when there is none."""
from __future__ import annotations

import unittest

import numpy as np

from core.corner import (
    CLEAR_COLOUR, DECIDED_FRACTION, CornerReading, colour_scores,
    read_corners, score_detection,
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


if __name__ == "__main__":
    unittest.main()
