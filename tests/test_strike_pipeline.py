"""tools/strike_pipeline.py never trains on the exam clips, and stops honestly."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strike_pipeline as pipeline  # noqa: E402


def _seq(directory: Path, fight: str, n: int):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.savez(directory / f"{fight}__{i}.npz", x=np.zeros((12, 102), np.float32), y=np.int64(1),
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
        self.assertEqual(counts, {"train": 5, "exam": 3})
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

    def test_no_data_stops_before_training(self):
        calls = []
        with patch.object(pipeline, "DATASET", self.root / "empty"), \
                patch.object(pipeline, "step", lambda title, command: calls.append(title) or False):
            self.assertEqual(pipeline.main(["--skip-fetch"]), 1)
        self.assertNotIn("Train (round 1)", calls)


if __name__ == "__main__":
    unittest.main()
