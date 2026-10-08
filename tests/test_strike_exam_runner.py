"""tools/strike_exam.py refuses evidence that could flatter the model, and writes a verdict."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strike_exam as runner  # noqa: E402

from core.temporal_model import ACTION_CLASSES, build_temporal_network, checkpoint_sha256  # noqa: E402


class _Constant(torch.nn.Module):
    """Always predicts one class: a model whose precision is known in advance."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        out = torch.zeros(x.shape[0], len(ACTION_CLASSES))
        out[:, self.index] = 1.0
        return out


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ckpt = self.tmp / "model.pt"
        net = build_temporal_network("gru_v1", 102, len(ACTION_CLASSES))
        torch.save({"state_dict": net.state_dict(), "architecture": "gru_v1", "input_dim": 102,
                    "classes": ACTION_CLASSES, "training_fights": ["train_fight"],
                    "held_out_fights": ["val_fight"]}, self.ckpt)
        self.sha = checkpoint_sha256(self.ckpt)

    def _windows(self, name, fight, label, n):
        directory = self.tmp / name
        directory.mkdir(exist_ok=True)
        for i in range(n):
            np.savez(directory / f"{fight}__{i:04d}.npz", x=np.zeros((12, 102), np.float32),
                     y=np.int64(ACTION_CLASSES.index(label)), fight_id=np.array(fight))
        return directory

    def test_refuses_training_validation_and_auto_label_fights(self):
        _, seen, _ = runner.load_model(self.ckpt)
        for fight in ("train_fight", "val_fight", "auto_job42", "strikemetrics_haggertyvnaito"):
            directory = self._windows(f"d_{fight}", fight, "jab", 2)
            with self.assertRaises(runner.ExamRefused):
                runner.load_windows([directory], seen)

    def test_precision_from_a_known_model(self):
        rows = runner.load_windows([self._windows("w", "exam1", "jab", 60),
                                    self._windows("n", "exam2", "none", 20)], set())
        guesses = runner.predict(_Constant(ACTION_CLASSES.index("cross")), rows)
        from core.strike_exam import window_checks

        checks = window_checks([(ACTION_CLASSES[r[1]], ACTION_CLASSES[g]) for r, g in zip(rows, guesses)])
        self.assertEqual(checks["punch"].predicted, 80)
        self.assertAlmostEqual(checks["punch"].precision, 0.75)   # 20 non-strikes called punches
        self.assertFalse(checks["punch"].passed())

    def _bout(self, made_by):
        result = self.tmp / f"result_{made_by[:4]}"
        result.mkdir(exist_ok=True)
        (result / "report.json").write_text(json.dumps({"classifier": {"temporal_checkpoint_sha256": made_by},
                                                          "rounds": [{"number": 1}]}))
        events = [{"fighter": "A", "family": "punch", "round_number": 1, "outcome": "missed"}] * 10
        (result / "events.json").write_text(json.dumps(events))
        official = self.tmp / "ufc.csv"
        with official.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.",
                                                        "TOTAL STR.", "HEAD", "BODY", "LEG", "TD"])
            writer.writeheader()
            for fighter in ("Ann", "Bea"):
                writer.writerow({"EVENT": "E1", "BOUT": "Ann vs. Bea", "ROUND": "Round 1", "FIGHTER": fighter,
                                 "KD": "0", "SIG.STR.": "5 of 9", "TOTAL STR.": "6 of 11", "HEAD": "0 of 0",
                                 "BODY": "0 of 0", "LEG": "0 of 0", "TD": "0 of 0"})
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps([{"result": str(result), "bout": "Ann vs. Bea",
                                         "fighter_a": "Ann", "fighter_b": "Bea"}]))
        return manifest, official

    def test_count_check_uses_only_results_from_this_model(self):
        manifest, official = self._bout(self.sha)
        check = runner.count_check(manifest, official, self.sha)
        # Bea had no strikes counted in an analysed round: that is 0 of 11, not left out.
        self.assertEqual([(ours, truth) for _, ours, truth in check.rounds], [(10, 11), (0, 11)])
        other, official = self._bout("f" * 64)
        with self.assertRaises(runner.ExamRefused):
            runner.count_check(other, official, self.sha)

    def test_cli_prints_and_writes_only_with_write(self):
        windows = self._windows("cli", "exam1", "left_round_kick", 3)
        out = self.tmp / "verdict.json"
        self.assertEqual(runner.main(["--checkpoint", str(self.ckpt), "--windows", str(windows),
                                      "--out", str(out)]), 0)
        self.assertFalse(out.exists())
        self.assertEqual(runner.main(["--checkpoint", str(self.ckpt), "--windows", str(windows),
                                      "--out", str(out), "--write"]), 0)
        verdict = json.loads(out.read_text())
        self.assertEqual(verdict["checkpoint_sha256"], self.sha)
        # No count check was given, so no sport can pass.
        self.assertEqual(verdict["sports"]["mma"]["passed_families"], [])

    def test_refusal_exit_code(self):
        windows = self._windows("bad", "auto_x", "jab", 1)
        self.assertEqual(runner.main(["--checkpoint", str(self.ckpt), "--windows", str(windows)]), 2)


if __name__ == "__main__":
    unittest.main()
