"""The identity benchmark's scoring (tools/identity_benchmark.py)."""

import json
import unittest
from pathlib import Path

from tools.identity_benchmark import score

ROOT = Path(__file__).resolve().parents[1]


def record(frame, a=None, b=None):
    return {"source_frame": frame,
            "fighter_A": {"observation": None if a is None else {"box": a}},
            "fighter_B": {"observation": None if b is None else {"box": b}}}


A_BOX, B_BOX, REFEREE = [0, 0, 10, 20], [30, 0, 40, 20], [60, 0, 70, 20]
TRUTH = {"frames": [
    {"phase": "standing", "source_frame": 1, "A": A_BOX, "B": B_BOX},
    {"phase": "standing", "source_frame": 2, "A": A_BOX, "B": B_BOX},
    {"phase": "standing", "source_frame": 3, "A": None, "B": B_BOX},
    {"phase": "ground", "source_frame": 4, "pair": [0, 0, 40, 20]},
]}


class IdentityBenchmarkTests(unittest.TestCase):
    def test_right_swapped_other_and_missing_are_told_apart(self):
        result = score([record(1, A_BOX, B_BOX), record(2, B_BOX, REFEREE), record(3, None, None),
                        record(4, [0, 0, 38, 20], None)], TRUTH)
        self.assertEqual(result["A"], {"right": 1, "swapped": 1, "partial": 0, "other": 0, "missing": 0, "frames": 2})
        self.assertEqual(result["B"], {"right": 1, "swapped": 0, "partial": 0, "other": 1, "missing": 1, "frames": 3})
        self.assertEqual(result["ground"], {"on_the_pair": 1, "elsewhere": 0, "missing": 1})

    def test_a_smaller_box_on_the_right_fighter_is_partial_not_another_person(self):
        from tools.identity_benchmark import classify

        marked_a, marked_b = [201.0, 39.9, 269.8, 175.5], [109.1, 45.2, 158.4, 190.1]
        # ma604 at 32 s on the analysis PC: A's upper body only, IoU 0.497.
        self.assertEqual(classify([202, 39, 247, 143], marked_a, marked_b), "partial")
        # Half on the fighter, half on someone beside him: still another person.
        self.assertEqual(classify([217, 32, 308, 191], [206.6, 30.7, 258.4, 210.2], None), "other")
        # A small box on the other fighter is not credited to this one.
        self.assertEqual(classify([115, 60, 150, 150], [100, 40, 160, 200], [110, 50, 160, 190]), "other")
        # "right" is untouched.
        self.assertEqual(classify(marked_a, marked_a, marked_b), "right")

    def test_the_marked_bouts_are_well_formed(self):
        for path in sorted((ROOT / "dataset" / "regression" / "identity_pankration").glob("*.json")):
            truth = json.loads(path.read_text(encoding="utf-8"))
            standing = [f for f in truth["frames"] if f["phase"] == "standing"]
            self.assertGreaterEqual(len(standing), 15, path.name)
            for frame in standing:
                self.assertTrue(frame.get("A") or frame.get("B"), path.name)
            self.assertEqual(len(truth["fighter_a_box"]), 4)


if __name__ == "__main__":
    unittest.main()
