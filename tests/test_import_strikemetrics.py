"""tools/import_strikemetrics.py imports only fights whose two files line up."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import import_strikemetrics as importer  # noqa: E402

from core.strike_exam import not_an_answer_key  # noqa: E402
from core.temporal_model import ACTION_CLASSES  # noqa: E402

W, H = 1920, 1080


def _person(cx, kick_offset=0.0):
    """17 COCO joints around ``cx`` (pixels); the right ankle moves by kick_offset."""
    joints = np.zeros((17, 3), np.float32)
    ys = [200, 190, 190, 195, 195, 300, 300, 400, 400, 480, 480, 550, 550, 700, 700, 850, 850]
    for i, y in enumerate(ys):
        joints[i] = (cx + (-30 if i % 2 else 30), y, 0.9)
    joints[16, 0] += kick_offset
    joints[16, 1] -= kick_offset
    return joints


def _write_fight(root: Path, name_xml: str, name_csv: str, *, shift_people=0, strike_label="Leg Kick"):
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    (root / "Keypoint Files").mkdir(parents=True, exist_ok=True)
    tracks = []
    for t, frame in enumerate(range(0, 120, 10)):
        tracks.append(f'<track id="{t}" label="F1"><box frame="{frame}" outside="0" occluded="0" '
                      f'xtl="400" ytl="150" xbr="700" ybr="900" z_order="0"></box></track>')
    tracks.append(f'<track id="99" label="{strike_label}"><box frame="60" outside="0" occluded="0" '
                  f'xtl="450" ytl="150" xbr="750" ybr="900" z_order="0"></box></track>')
    (root / "Annotations" / name_xml).write_text(
        f"<annotations><meta><original_size><width>{W}</width><height>{H}</height></original_size></meta>"
        + "".join(tracks) + "</annotations>", encoding="utf-8")
    header = ["frame_id", "person_id"] + [f"keypoint_{i}_{k}" for i in range(17) for k in ("x", "y", "confidence")]
    rows = [",".join(header)]
    for frame in range(150):
        for pid, (cx, offset) in enumerate(((550 + shift_people, (frame % 12) * 8.0), (1300, 0.0))):
            joints = _person(cx, offset)
            values = [f"{v:.5f}" for j in joints for v in (j[0] / W, j[1] / H, j[2])]
            rows.append(",".join([str(frame), str(pid)] + values))
    (root / "Keypoint Files" / name_csv).write_text("\n".join(rows), encoding="utf-8")


def _args(out):
    return argparse.Namespace(fps=30.0, sample_fps=12.0, window=12, after=3, max_jump=0.15,
                              min_match=0.9, min_boxes=10, out=out)


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.out = self.root / "out"
        self.out.mkdir()

    def test_aligned_fight_becomes_a_round_kick_by_the_striker(self):
        _write_fight(self.root, "a.xml", "a.csv")
        made, refused = importer.import_fight(self.root / "Annotations" / "a.xml",
                                              self.root / "Keypoint Files" / "a.csv", self.out, _args(self.out))
        self.assertIsNone(refused)
        files = sorted(self.out.glob("*.npz"))
        self.assertEqual(len(files), 1)
        with np.load(files[0]) as data:
            self.assertEqual(data["x"].shape, (12, 102))
            self.assertTrue(np.isfinite(data["x"]).all())
            self.assertEqual(ACTION_CLASSES[int(data["y"])], "right_round_kick")
            fight = str(data["fight_id"])
        self.assertTrue(fight.startswith("strikemetrics_"))
        self.assertIsNotNone(not_an_answer_key(fight))   # training material, never an answer

    def test_misaligned_fight_is_skipped_with_the_reason(self):
        _write_fight(self.root, "b.xml", "b.csv", shift_people=700)   # skeletons outside every fighter box
        made, refused = importer.import_fight(self.root / "Annotations" / "b.xml",
                                              self.root / "Keypoint Files" / "b.csv", self.out, _args(self.out))
        self.assertIn("do not line up", refused)
        self.assertEqual(list(self.out.glob("*.npz")), [])

    def test_too_few_boxes_to_check_is_skipped(self):
        _write_fight(self.root, "c.xml", "c.csv")
        args = _args(self.out)
        args.min_boxes = 50
        _, refused = importer.import_fight(self.root / "Annotations" / "c.xml",
                                           self.root / "Keypoint Files" / "c.csv", self.out, args)
        self.assertIn("fighter boxes", refused)

    def test_jab_keeps_its_class(self):
        _write_fight(self.root, "d.xml", "d.csv", strike_label="Jab")
        importer.import_fight(self.root / "Annotations" / "d.xml", self.root / "Keypoint Files" / "d.csv",
                              self.out, _args(self.out))
        with np.load(next(self.out.glob("*.npz"))) as data:
            self.assertEqual(ACTION_CLASSES[int(data["y"])], "jab")


if __name__ == "__main__":
    unittest.main()
