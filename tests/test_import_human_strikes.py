"""A person's labels from the labelling page become answer keys (2026-11-10)."""

from __future__ import annotations

import json

from tools import benchmark_dense_strikes as dense
from tools import import_human_strikes as imp


def _fight(tmp_path, name="S04"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "track.jsonl.gz").write_bytes(b"")
    (folder / "strikes.json").write_text(json.dumps({
        "covered": [14.0, 57.0], "unreadable": [],
        "strikes": [{"t": 15.0, "fighter": "A", "family": "punch", "sure": True},
                    {"t": 20.0, "fighter": "B", "family": "punch", "sure": True}]}))
    return folder


def test_marks_become_a_key_in_video_time_and_claude_is_compared(tmp_path, monkeypatch, capsys):
    s04, s04k = _fight(tmp_path, "S04"), _fight(tmp_path, "S04K")
    monkeypatch.setattr(dense, "dense_fights", lambda: [s04, s04k])
    export = tmp_path / "labels.json"
    export.write_text("\n".join(json.dumps({"id": i, "data": d}) for i, d in enumerate([
        {"clip": "S04", "labeller": "p1", "finished": False, "watched": 10.0, "updated": "x",
         "unreadable": [[2.0, 3.0]],
         "marks": [{"t": 1.1, "fighter": "A", "family": "punch", "sure": True},
                   {"t": 4.0, "fighter": "B", "family": "kick", "sure": False},
                   {"t": 12.0, "fighter": "A", "family": "punch", "sure": True},    # past what was watched
                   {"t": 5.0, "fighter": "C", "family": "punch", "sure": True}]},   # not a fighter
        {"clip": "S04", "labeller": "p2", "finished": False, "watched": 3.0, "marks": []},
    ])))
    assert imp.main([str(export), "--compare"]) == 0
    key = json.loads((s04 / dense.HUMAN_LABELS).read_text())
    assert key["covered"] == [14.0, 24.0] and key["unreadable"] == [[16.0, 17.0]]
    assert [(s["t"], s["fighter"], s["family"], s["sure"]) for s in key["strikes"]] == [
        (15.1, "A", "punch", True), (18.0, "B", "kick", False)]
    assert "p2" in key["notes"]
    assert (s04k / dense.HUMAN_LABELS).read_text() == (s04 / dense.HUMAN_LABELS).read_text()
    out = capsys.readouterr().out
    assert "1 of the person's 1 found, 1 extra" in out and "not finished" in out
    # The benchmark scores against the person's key from now on.
    assert dense.load_truth(s04)["strikes"] == key["strikes"]
    assert dense.labelled_by_person(s04)


def test_without_a_person_the_benchmark_keeps_claudes_checked_labels(tmp_path):
    folder = _fight(tmp_path)
    assert not dense.labelled_by_person(folder)
    assert [s["t"] for s in dense.load_truth(folder)["strikes"]] == [15.0, 20.0]
