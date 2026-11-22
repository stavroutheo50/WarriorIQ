"""Fight Camp took 13 seconds (owner, 2026-11-22): every restart re-read each fight's tracking.jsonl.

The plausibility verdict is now saved beside the tracking file and reused
while the file and the checking code are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import core.fight_presence as fp


def _tracking(path: Path, frames: int = 300, far: bool = False) -> Path:
    def obs(x):
        return {"track_id": 1, "box": [x, 100, x + 120, 400], "confidence": 0.9}
    with path.open("w", encoding="utf-8") as out:
        for i in range(frames):
            out.write(json.dumps({"time_seconds": i / 30, "fighter_A": {"observation": obs(100)},
                                  "fighter_B": {"observation": obs(900 if far else 260)}}) + "\n")
    return path


def test_the_verdict_is_saved_and_reused_after_a_restart(tmp_path, monkeypatch):
    track = _tracking(tmp_path / "tracking.jsonl")
    first = fp.plausibility_from_tracking(track)
    saved = tmp_path / "tracking.jsonl.plausibility.json"
    assert saved.exists() and json.loads(saved.read_text())["verdict"] == first
    fp._TRACKING_CACHE.clear()                                   # a restart empties memory
    monkeypatch.setattr(fp, "FightPresence", None)               # recomputing would now fail
    assert fp.plausibility_from_tracking(track) == first


def test_a_changed_tracking_file_or_changed_code_is_checked_again(tmp_path, monkeypatch):
    track = _tracking(tmp_path / "tracking.jsonl")
    fp.plausibility_from_tracking(track)
    _tracking(track, frames=600, far=True)                       # rewritten: new size and time
    fp._TRACKING_CACHE.clear()
    calls = []
    real = fp.FightPresence
    monkeypatch.setattr(fp, "FightPresence", lambda *a: calls.append(1) or real(*a))
    fp.plausibility_from_tracking(track)
    assert calls, "a changed tracking file must be checked again"
    fp._TRACKING_CACHE.clear()
    calls.clear()
    monkeypatch.setattr(fp, "_CODE_VERSION", "a-newer-check")
    fp.plausibility_from_tracking(track)
    assert calls, "a changed check must not reuse an old verdict"


def test_an_unwritable_folder_still_answers(tmp_path, monkeypatch):
    track = _tracking(tmp_path / "tracking.jsonl")
    monkeypatch.setattr(fp, "_verdict_path", lambda path: tmp_path / "missing-dir" / "x.json")
    fp.plausibility_from_tracking(track)                         # answers, never raises


def test_the_background_sweep_saves_verdicts_once_per_process(tmp_path, monkeypatch):
    import app.main as web

    report = tmp_path / "job1" / "report.json"
    report.parent.mkdir()
    report.write_text("{}")
    _tracking(report.parent / "tracking.jsonl")
    monkeypatch.setattr("core.db.list_fight_report_paths", lambda: [str(report)])
    assert web._warm_plausibility_verdicts() == 1
    assert (report.parent / "tracking.jsonl.plausibility.json").exists()
    assert web._warm_plausibility_verdicts() == 0               # already saved: nothing re-read
