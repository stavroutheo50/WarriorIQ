"""Training variations keep the label true; the family loss rewards the right family (2026-10-09)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.temporal_augment import JOINTS, augment, mirror, speed
from core.temporal_model import ACTION_CLASSES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_temporal_model as trainer  # noqa: E402


def _window(seed=0, hidden=(3,)):
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.uniform(0.1, 0.9, (12, JOINTS, 2)), np.full((12, JOINTS, 1), 0.9)], axis=2)
    vel = rng.normal(0, 1, (12, JOINTS, 3))
    vel[..., 2] = np.hypot(vel[..., 0], vel[..., 1])
    pos[:, list(hidden)] = 0.0
    vel[:, list(hidden)] = 0.0
    return np.concatenate([pos.reshape(12, -1), vel.reshape(12, -1)], axis=1).astype(np.float32)


class AugmentTests(unittest.TestCase):
    def test_mirror_swaps_sides_and_labels_and_undoes_itself(self):
        x = _window()
        y = ACTION_CLASSES.index("left_hook")
        mx, my = mirror(x, y)
        self.assertEqual(ACTION_CLASSES[my], "right_hook")
        # Left wrist (9) becomes the right wrist (10), x reflected in the box.
        self.assertAlmostEqual(mx[0, 10 * 3], 1.0 - x[0, 9 * 3], places=5)
        self.assertAlmostEqual(mx[0, JOINTS * 3 + 10 * 3], -x[0, JOINTS * 3 + 9 * 3], places=5)
        back, by = mirror(mx, my)
        np.testing.assert_allclose(back, x, atol=1e-6)
        self.assertEqual(by, y)
        for name in ("jab", "cross", "none", "backfist"):
            self.assertEqual(ACTION_CLASSES[mirror(x, ACTION_CLASSES.index(name))[1]], name)

    def test_a_hidden_joint_stays_hidden(self):
        x = _window(hidden=(3,))
        mx, _ = mirror(x, 0)                      # joint 3 (left ear) becomes 4 (right ear)
        self.assertTrue(np.all(mx[:, 4 * 3:4 * 3 + 3] == 0))
        sx = speed(x, 1.2)
        self.assertTrue(np.all(sx[:, 3 * 3:3 * 3 + 3] == 0))

    def test_speed_one_is_the_same_window_and_faster_moves_faster(self):
        x = _window()
        np.testing.assert_allclose(speed(x, 1.0), x, atol=1e-6)
        fast = speed(x, 1.25)
        np.testing.assert_allclose(fast[-1, JOINTS * 3:], x[-1, JOINTS * 3:] * 1.25, rtol=1e-5)

    def test_augment_keeps_shape_finite_and_a_sensible_label(self):
        rng = np.random.default_rng(1)
        x = _window()
        for _ in range(50):
            ax, ay = augment(x, ACTION_CLASSES.index("left_round_kick"), rng)
            self.assertEqual(ax.shape, x.shape)
            self.assertTrue(np.isfinite(ax).all())
            self.assertIn(ACTION_CLASSES[ay], ("left_round_kick", "right_round_kick"))


class FamilyLossTests(unittest.TestCase):
    def test_the_right_family_with_the_wrong_technique_costs_little(self):
        families, class_family = trainer.family_targets()
        logits = torch.full((1, len(ACTION_CLASSES)), -10.0)
        logits[0, ACTION_CLASSES.index("cross")] = 10.0
        jab = torch.tensor([ACTION_CLASSES.index("jab")])
        kick = torch.tensor([ACTION_CLASSES.index("left_round_kick")])
        self.assertLess(float(trainer.family_loss(logits, jab, class_family, len(families))), 0.01)
        self.assertGreater(float(trainer.family_loss(logits, kick, class_family, len(families))), 5.0)


class TrainerRunTests(unittest.TestCase):
    def test_a_short_run_with_both_options_trains_and_records_them(self):
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        data.mkdir()
        for i in range(60):
            fight = "f1" if i % 2 else "f2"
            y = ACTION_CLASSES.index("jab") if i < 30 else 0
            np.savez(data / f"{fight}__{i}.npz", x=_window(seed=i), y=np.int64(y), fight_id=np.array(fight))
        out = root / "model.pt"
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "train_temporal_model.py"), "--data", str(data),
                                 "--epochs", "1", "--batch", "16", "--out", str(out), "--augment",
                                 "--family-weight", "1.0", "--architecture", "gru_v1"],
                                cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        payload = torch.load(out, map_location="cpu", weights_only=True)
        self.assertTrue(payload["augment"])
        self.assertEqual(payload["family_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
