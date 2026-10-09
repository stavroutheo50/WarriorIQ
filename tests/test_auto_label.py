"""tools/auto_label.py keeps only what the rules and the model agree on, and marks it auto_."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import auto_label  # noqa: E402

from core.strike_exam import not_an_answer_key  # noqa: E402
from core.temporal_model import ACTION_CLASSES  # noqa: E402


class _Says(torch.nn.Module):
    """Answers one class with a fixed probability, whatever the window."""

    def __init__(self, name, probability):
        super().__init__()
        self.index, self.p = ACTION_CLASSES.index(name), probability

    def forward(self, x):
        rest = (1.0 - self.p) / (len(ACTION_CLASSES) - 1)
        probs = torch.full((x.shape[0], len(ACTION_CLASSES)), rest)
        probs[:, self.index] = self.p
        return probs.log()


def _job(root: Path, *, trusted=True, technique="left_hook", frames=120):
    job = root / "job1"
    job.mkdir()
    rows = []
    for i in range(frames):
        kp = (np.tile([[300.0, 300.0]], (17, 1)) + np.arange(17)[:, None] * 5 + i).tolist()
        observation = {"keypoints": kp, "box": [250, 250, 450, 700], "keypoint_conf": [0.9] * 17}
        rows.append({"source_frame": i * 5, "time_seconds": i / 6.0, "round_number": 1,
                     "fighter_A": {"observation": observation}, "fighter_B": {"observation": observation}})
    (job / "tracking.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    events = [{"fighter": "A", "technique": technique, "family": "punch", "peak_time": 5.0}]
    (job / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (job / "report.json").write_text(json.dumps({"integrity": {"identity_evidence_trusted": trusted}}),
                                     encoding="utf-8")
    return job


def _args():
    return argparse.Namespace(min_probability=0.9, min_none_probability=0.95, quiet_seconds=1.0,
                              max_none=3, window=12)


class AutoLabelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.out = self.root / "out"
        self.out.mkdir()

    def _labels(self):
        found = []
        for path in sorted(self.out.glob("*.npz")):
            with np.load(path) as data:
                found.append((ACTION_CLASSES[int(data["y"])], str(data["fight_id"])))
        return found

    def test_agreement_becomes_a_label_marked_auto(self):
        job = _job(self.root)
        auto_label.label_job(job, _Says("left_hook", 0.97), self.out, _args(), np.random.default_rng(0))
        labels = self._labels()
        self.assertIn(("left_hook", "auto_job1"), labels)
        self.assertTrue(all(not_an_answer_key(fight) for _, fight in labels))

    def test_disagreement_and_doubt_are_dropped(self):
        job = _job(self.root)
        tally = auto_label.label_job(job, _Says("cross", 0.97), self.out, _args(), np.random.default_rng(0))
        self.assertEqual(tally["dropped: rules and model disagree"], 1)
        self.assertNotIn("left_hook", [name for name, _ in self._labels()])
        out2 = self.root / "out2"
        out2.mkdir()
        tally = auto_label.label_job(job, _Says("left_hook", 0.6), out2, _args(), np.random.default_rng(0))
        self.assertEqual(tally["dropped: model not sure enough"], 1)

    def test_family_agreement_labels_with_the_model_technique(self):
        # 2026-10-09: rules and model never named the same technique on our fights.
        job = _job(self.root)
        args = _args()
        args.agree_on = "family"
        tally = auto_label.label_job(job, _Says("cross", 0.97), self.out, args, np.random.default_rng(0))
        self.assertEqual(tally["cross"], 1)
        self.assertIn(("cross", "auto_job1"), self._labels())
        out2 = self.root / "out2"
        out2.mkdir()
        tally = auto_label.label_job(job, _Says("left_round_kick", 0.97), out2, args, np.random.default_rng(0))
        self.assertEqual(tally["dropped: rules and model disagree"], 1)

    def test_quiet_moments_need_a_sure_none(self):
        job = _job(self.root)
        tally = auto_label.label_job(job, _Says("none", 0.99), self.out, _args(), np.random.default_rng(0))
        self.assertEqual(tally["none"], 6)            # 3 per fighter
        out2 = self.root / "out2"
        out2.mkdir()
        tally = auto_label.label_job(job, _Says("none", 0.8), out2, _args(), np.random.default_rng(0))
        self.assertEqual(tally["none"], 0)

    def test_failed_identity_is_not_read(self):
        job = _job(self.root, trusted=False)
        tally = auto_label.label_job(job, _Says("left_hook", 0.99), self.out, _args(), np.random.default_rng(0))
        self.assertEqual(tally["skipped: identity check did not pass"], 1)
        self.assertEqual(self._labels(), [])


if __name__ == "__main__":
    unittest.main()
