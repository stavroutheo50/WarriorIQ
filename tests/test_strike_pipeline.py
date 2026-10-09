"""tools/strike_pipeline.py never trains on the exam clips, and stops honestly."""

from __future__ import annotations

import sys
import zlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strike_pipeline as pipeline  # noqa: E402


def _seq(directory: Path, fight: str, n: int, *, value=None, y=1):
    """n distinct windows, or n copies of one when ``value`` is given."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        fill = value if value is not None else zlib.crc32(f"{fight}{i}".encode()) % 100000
        np.savez(directory / f"{fight}__{i}.npz", x=np.full((12, 102), fill, np.float32), y=np.int64(y),
                 fight_id=np.array(fight))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        for name, value in (("TRAIN", self.root / "train"), ("EXAM", self.root / "exam")):
            patcher = patch.object(pipeline, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_exam_fights_are_held_out_of_training(self):
        source = self.root / "sequences_tkd_kick3"
        _seq(source, "tkd_train", 5)
        _seq(source, "tkd_test", 3)
        counts = pipeline.gather([source], {"tkd_test": "tkd_kick3_test"})
        self.assertEqual(counts, {"train": 5, "exam": 3, "duplicates": 0, "gap": 0})
        trained = {pipeline.fight_of(p) for p in (self.root / "train").glob("*.npz")}
        self.assertEqual(trained, {"tkd_train"})
        self.assertEqual(len(list((self.root / "exam" / "tkd_kick3_test").glob("*.npz"))), 3)

    def test_regathering_starts_clean(self):
        source = self.root / "s"
        _seq(source, "a", 2)
        pipeline.gather([source], {})
        (source / "a__0.npz").unlink()
        pipeline.gather([source], {})
        self.assertEqual(len(list((self.root / "train").glob("*.npz"))), 1)

    def test_repeated_windows_are_left_out_and_never_leak_the_exam(self):
        # The auto-labeller and the negative miner cut windows from the same tracking.
        own, auto = self.root / "sequences_own_negatives", self.root / "sequences_auto"
        _seq(own, "own_b883", 2, value=7.0, y=0)
        _seq(auto, "auto_negmine_b883", 1, value=7.0, y=0)
        _seq(auto, "auto_leak", 1, value=9.0)
        _seq(self.root / "exam_src", "tkd_test", 1, value=9.0)
        counts = pipeline.gather([own, auto, self.root / "exam_src"], {"tkd_test": "tkd_kick3_test"})
        self.assertEqual(counts, {"train": 1, "exam": 1, "duplicates": 3, "gap": 0})
        trained = {pipeline.fight_of(p) for p in (self.root / "train").glob("*.npz")}
        self.assertEqual(trained, {"own_b883"})

    def test_quiet_footage_is_split_in_time_between_training_and_exam(self):
        # 2026-10-09: V6 held out whole, the model called all 952 of its quiet
        # windows punches. Early part trains, late part is examined, gap between.
        source = self.root / "sequences_boxingvi"
        _seq(source, "boxingvi_V6", 10, y=0)
        counts = pipeline.gather([source], {}, {"boxingvi_V6": ("boxingvi_test", 0.3, 0.4)})
        self.assertEqual(counts, {"train": 3, "exam": 6, "duplicates": 0, "gap": 1})
        trained = sorted(p.name for p in (self.root / "train").glob("*.npz"))
        examined = sorted(p.name for p in (self.root / "exam" / "boxingvi_test").glob("*.npz"))
        self.assertEqual([n.rsplit("__", 1)[1] for n in trained], ["0.npz", "1.npz", "2.npz"])
        self.assertEqual([n.rsplit("__", 1)[1] for n in examined], [f"{i}.npz" for i in range(4, 10)])
        self.assertEqual({pipeline.fight_of(p) for p in (self.root / "train").glob("*.npz")}, {"boxingvi_V6a"})
        self.assertEqual({pipeline.fight_of(p) for p in (self.root / "exam" / "boxingvi_test").glob("*.npz")},
                         {"boxingvi_V6b"})

    def test_refusal_names_the_gap(self):
        _seq(self.root / "train", "a", 25)
        _seq(self.root / "train", "b", 3, y=0)
        reason = pipeline.refusal_reason(self.root / "train")
        self.assertIn("3 \"none\" windows", reason)
        self.assertIn("own footage", reason)

    def test_no_data_stops_before_training(self):
        calls = []
        with patch.object(pipeline, "DATASET", self.root / "empty"), \
                patch.object(pipeline, "step", lambda title, command: calls.append(title) or False):
            self.assertEqual(pipeline.main(["--skip-fetch"]), 1)
        self.assertNotIn("Train (round 1)", calls)


if __name__ == "__main__":
    unittest.main()
