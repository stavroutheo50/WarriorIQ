"""The owner's fights as identity test cases (dataset/regression/identity_owner)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_identity_benchmark as rib

FOLDER = Path(__file__).resolve().parents[1] / "dataset" / "regression" / "identity_owner"


def test_every_owner_fight_is_a_well_formed_truth_file():
    files = sorted(FOLDER.glob("*.json"))
    assert {p.stem for p in files} == {"f1", "f5736", "fb883"}
    assert all(p in rib.TRUTH for p in files)
    for path in files:
        truth = json.loads(path.read_text(encoding="utf-8"))
        first = truth["frames"][0]
        assert truth["start_seconds"] == first["time_seconds"] < truth["end_seconds"]
        assert truth["fighter_a_box"] == first["A"] and truth["fighter_b_box"] == first["B"]
        assert truth["fighter_a_box"] and truth["fighter_b_box"]
        for frame in truth["frames"]:
            for side in ("A", "B"):
                box = frame[side]
                assert box is None or (len(box) == 4 and box[0] < box[2] and box[1] < box[3])
        marked = sum(f[s] is not None for f in truth["frames"] for s in "AB")
        assert marked >= 15


def test_a_missing_video_is_reported_not_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        rib._download({"video": ["no/such/file.mp4"]})
