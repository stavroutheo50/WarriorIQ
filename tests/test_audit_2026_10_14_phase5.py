"""Audit of 14 Oct 2026, phase 5: upload and selection.

#14 the Start panel no longer covers the Opponent (B) card
#15 "More options" opens at full width
#16 "Which fighter?" starts on the profile's main fighter
#17 the browser rejects tiny or non-video files; the server reads the real type
#18 no toast covers the fighter select when a video is chosen
#19 kicks or knees in a boxing upload suggest another sport (never a block)
#20 trim: analyse only part of the video
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"
CSS = (ROOT / "app" / "static" / "fighter-selection.css").read_text(encoding="utf-8")


# --------------------------------------------------------------- #14 #15
def test_start_follows_the_fighter_cards_and_is_not_sticky():
    page = (TEMPLATES / "select.html").read_text(encoding="utf-8")
    assert "keys.after(choice)" in page and "guide.prepend(choice)" not in page
    assert ".fighter-lock-guide .analysis-choice{position:static" in CSS


def test_more_options_spans_the_full_width():
    assert ".stage-extras .select-more{grid-column:1/-1}" in CSS


# --------------------------------------------------------------------- #16
def test_the_main_fighter_is_preselected(monkeypatch):
    import app.main as web

    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert "{% if main_fighter_id and f.id == main_fighter_id %}selected{% endif %}" in page
    request = mock.Mock()
    monkeypatch.setattr(web, "_account", lambda request: {"profile_id": 7})
    monkeypatch.setattr(web, "list_fighters", lambda profile_id: [{"id": 1, "name": "Maria"},
                                                                   {"id": 2, "name": "Theodoulos Stavrou"}])
    monkeypatch.setattr(web, "get_profile", lambda profile_id: {"display_name": "theodoulos  stavrou"})
    assert web._main_fighter_id(request) == 2
    # No name match: the fighter most fights are filed under.
    monkeypatch.setattr(web, "get_profile", lambda profile_id: {"display_name": "Coach"})
    monkeypatch.setattr(web, "list_fights", lambda profile_id: [{"fighter_id": 1}, {"fighter_id": 1}, {"fighter_id": 2}])
    assert web._main_fighter_id(request) == 1
    # Nothing to go on with several fighters: nobody is pre-selected.
    monkeypatch.setattr(web, "list_fights", lambda profile_id: [])
    assert web._main_fighter_id(request) is None


# --------------------------------------------------------------------- #17
def test_the_browser_rejects_tiny_and_non_video_files():
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert "const MIN_VIDEO_BYTES=50*1024;" in page
    assert "if(f.size<MIN_VIDEO_BYTES){rejectFile(" in page
    assert "if(!(await looksLikeVideo(f)))" in page
    # The same container signatures the server checks.
    assert "at(4,'ftyp')||at(0,[0x1a,0x45,0xdf,0xa3])||(at(0,'RIFF')&&at(8,'AVI '))||at(0,'OggS')" in page


@pytest.mark.skipif(not (shutil.which("ffprobe") or shutil.which("ffmpeg")), reason="needs ffprobe or ffmpeg")
def test_the_server_reads_the_real_file_type(tmp_path):
    from core.video import has_video_stream

    forged = tmp_path / "forged.mp4"
    forged.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 100)
    real = tmp_path / "real.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=1:size=160x120:rate=10", "-c:v", "libx264", str(real)], check=True)
    assert has_video_stream(forged) is False
    assert has_video_stream(real) is True


def test_the_upload_refuses_a_file_with_no_video_stream():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "if SETTINGS.upload_stream_probe and await run_in_threadpool(has_video_stream, video_path) is False:" in source
    # On by default; only the test configuration switches it off (conftest.py).
    config = (ROOT / "core" / "config.py").read_text(encoding="utf-8")
    assert 'upload_stream_probe: bool = env_bool("WARRIORIQ_UPLOAD_STREAM_PROBE", True)' in config


# --------------------------------------------------------------------- #18
def test_no_toast_covers_the_fighter_select():
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert "Fight video selected. Press upload to send it." not in page


# --------------------------------------------------------------------- #19
def _boxing(events):
    return {"scorecard": {"sport": "boxing"}, "events": events}


def test_kicks_in_a_boxing_upload_suggest_another_sport():
    from core.sport_check import kicks_in_boxing

    kicks = [{"family": "kick", "confidence": 0.8}] * 3 + [{"family": "knee", "confidence": 0.7}]
    suggestion = kicks_in_boxing(_boxing(kicks))
    assert suggestion == {"kicks": 3, "knees": 1, "total": 4, "suggested": ["kickboxing", "muay_thai"]}
    # Too few, or too unsure, or not boxing: no suggestion.
    assert kicks_in_boxing(_boxing(kicks[:3])) is None
    assert kicks_in_boxing(_boxing([{"family": "kick", "confidence": 0.3}] * 10)) is None
    assert kicks_in_boxing({"scorecard": {"sport": "kickboxing"}, "events": kicks}) is None


def test_the_suggestion_never_blocks():
    page = (TEMPLATES / "result.html").read_text(encoding="utf-8")
    assert "{% if kicks_in_boxing %}" in page and "If it was boxing, ignore this." in page


# --------------------------------------------------------------------- #20
def test_trim_is_stored_for_the_analysis(monkeypatch):
    import app.main as web

    stored = {}
    monkeypatch.setattr(web, "update_job", lambda job_id, patch: stored.update(patch))
    job = {"video_duration": 300.0, "selection_seconds": 40.0}
    updated = web._apply_trim("j", job, 30.0, 150.0)
    assert stored == {"requested_start_seconds": 30.0, "end_seconds": 150.0}
    assert updated["requested_start_seconds"] == 30.0 and updated["end_seconds"] == 150.0
    # Nothing asked for: nothing changes.
    stored.clear()
    assert web._apply_trim("j", job, None, None) is job and stored == {}


def test_trim_must_keep_the_frame_the_fighters_were_tapped_on(monkeypatch):
    import app.main as web

    monkeypatch.setattr(web, "update_job", lambda job_id, patch: None)
    job = {"video_duration": 300.0, "selection_seconds": 40.0}
    with pytest.raises(HTTPException) as refused:
        web._apply_trim("j", job, 60.0, 120.0)
    assert refused.value.status_code == 400 and "outside the part you kept" in refused.value.detail
    with pytest.raises(HTTPException):
        web._apply_trim("j", job, 39.5, 40.5)          # under the 2 s minimum


def test_the_analysis_request_reads_the_trim():
    import app.main as web

    job = {"video_path": "v.mp4", "fight_type": "sparring", "ruleset": "K1", "round_count": 1,
           "round_duration_seconds": 0, "break_duration_seconds": 0,
           "requested_start_seconds": 30.0, "end_seconds": 150.0, "selection_seconds": 40.0}
    req = web._analysis_request("j", job, [0, 0, 10, 10], [20, 0, 30, 10], "A")
    assert (req.start_seconds, req.end_seconds, req.selection_seconds) == (30.0, 150.0, 40.0)


def test_the_selection_page_offers_trimming():
    page = (TEMPLATES / "select.html").read_text(encoding="utf-8")
    assert 'id="trimStart"' in page and 'id="trimEnd"' in page
    assert "trim_start_seconds:trim.start,trim_end_seconds:trim.end" in page
