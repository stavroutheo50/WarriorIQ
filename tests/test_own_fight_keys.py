"""tools/build_own_fight_keys.py: marks from the labelling page's own-video mode (2026-11-23)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_own_fight_keys as keys  # noqa: E402


def _doc(**over):
    doc = {"kind": "own_video", "clip": "own:abc:1", "labeller": "x", "finished": True, "watched": 300,
           "video": {"name": "Team fights 2026.mp4", "size": 123, "duration": 1900.0},
           "bout": {"number": 1, "start": 100.0, "end": 280.0, "blue": [0.3, 0.6], "red": [0.7, 0.6], "ruleset": "K1"},
           "marks": [{"t": 120.5, "fighter": "A", "family": "punch", "sure": True},
                     {"t": 90.0, "fighter": "B", "family": "kick", "sure": True},        # before the bout: dropped
                     {"t": 150.25, "fighter": "B", "family": "kick", "sure": False},
                     {"t": 160.0, "fighter": "C", "family": "punch", "sure": True}],      # not a fighter: dropped
           "unreadable": [[90.0, 105.0], [200.0, 210.0], [300.0, 310.0]]}
    doc.update(over)
    return doc


def test_only_own_video_documents_are_read(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"data": _doc()}, {"clip": "S01", "marks": []}]), encoding="utf-8")
    assert [d["clip"] for d in keys.read_documents(path)] == ["own:abc:1"]


def test_the_answer_key_keeps_the_bout_and_absolute_times():
    key = keys.answer_key(_doc())
    assert key["covered"] == [100.0, 280.0]
    assert [(s["t"], s["fighter"], s["family"], s["sure"]) for s in key["strikes"]] == [
        (120.5, "A", "punch", True), (150.25, "B", "kick", False)]
    assert key["unreadable"] == [[100.0, 105.0], [200.0, 210.0]]          # clipped to the bout


def test_the_labeller_who_watched_most_is_used_per_bout():
    short, long = _doc(watched=50, labeller="a"), _doc(watched=170, labeller="b")
    assert keys.pick([short, long])[keys.bout_name(long)]["labeller"] == "b"
    assert keys.bout_name(long) == "own_Team_fights_2026_b01"


def test_the_tapped_point_picks_its_person():
    boxes = [[0, 0, 100, 200], [150, 0, 260, 200], [40, 20, 90, 120]]
    assert keys._containing(boxes, (60, 50)) == 2            # inside two boxes: the smaller
    assert keys._containing(boxes, (200, 100)) == 1
    assert keys._containing(boxes, (500, 100)) == 1          # outside all: the nearest
    assert keys._containing([], (1, 1)) is None
