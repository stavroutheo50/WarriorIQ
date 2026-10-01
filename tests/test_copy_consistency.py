"""Copy and UI bugs from QA 2026-09 (item 18): each page says one thing."""

from __future__ import annotations

import os
from pathlib import Path

from app import state
from core.report import identity_failure

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


def _page(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def test_a_claimed_job_is_not_still_staged_as_queued(tmp_path):
    job_id = "claimstage01"
    state.create_job(job_id, {"owner_key": "account:1", "status": "selecting"})
    try:
        state.prepare_job_run(job_id, {})
        # The oldest queued session is claimed first; make it this one.
        os.utime(state._session_path(job_id), (1, 1))
        claimed = state.claim_next_job("worker-test")
        assert claimed is not None and claimed[0] == job_id
        job = state.get_job(job_id)
        assert job["status"] == "running"
        assert job["stage"] != "queued"
        assert "GPU" not in job["message"]
    finally:
        state.delete_job(job_id)


def test_the_frame_picker_uses_the_four_step_bar_only():
    page = _page("frame.html")
    assert "of 03" not in page
    assert page.count('class="workflow-steps"') == 1


def test_the_progress_percent_says_what_it_measures():
    page = _page("progress.html")
    assert "Real pipeline progress" not in page
    assert "Of the whole job" in page
    assert "Video analysed through" in page
    assert "SAM2 follows" not in page


def test_the_preflight_advice_does_not_say_both_about_one_thing():
    assert "fix both that and the measurement" not in _page("analyze.html")


def test_a_camera_failure_rate_of_one_reads_as_english():
    tracking = {"fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9,
                "fighter_A_seed_source": "pose_detector", "fighter_B_seed_source": "pose_detector",
                "initial_iou_A": 0.9, "initial_iou_B": 0.9,
                "fighter_A_handoffs_per_minute": 1.0, "fighter_B_handoffs_per_minute": 1.0,
                "fighter_A_suspicious_handoffs_per_minute": 25.0}
    cause = identity_failure(tracking)
    assert cause["cause"] == "camera"
    assert "25 times a minute" in cause["headline"]
    assert " 1 times" not in cause["headline"]


def test_replay_and_report_do_not_give_two_counts_the_same_name():
    assert "analyzed frames`" not in _page("replay.html")
    assert "tracked frames" in _page("replay.html")


# --- previous audit items (QA 2026-09 "also verify") ---------------------------

STATIC = TEMPLATES.parent / "static"


def test_no_stylesheet_sets_text_below_twelve_pixels():
    import re

    for sheet in STATIC.glob("*.css"):
        text = sheet.read_text(encoding="utf-8")
        sizes = [float(size) for size in re.findall(r"font(?:-size)?:[^;}]*?\b(\d+(?:\.\d+)?)px", text)]
        assert all(size >= 12 for size in sizes), (sheet.name, sorted(set(s for s in sizes if s < 12)))


def test_footer_links_are_44px_both_ways_on_a_phone():
    a11y = (STATIC / "a11y.css").read_text(encoding="utf-8")
    assert "min-height: 44px" in a11y and "min-width: 44px" in a11y


def test_the_wordmark_is_one_word():
    base = _page("base.html")
    assert '<b class="brand-word">WARRIOR<span>IQ</span></b>' in base


def test_revealed_content_cannot_stay_hidden_if_the_observer_fails():
    motion = (STATIC / "motion.js").read_text(encoding="utf-8")
    assert "catch(error){revealAll()}" in motion
    # Hidden only under a class this same script adds.
    assert "body.motion-enabled .wiq-reveal{opacity:0" in (STATIC / "motion.css").read_text(encoding="utf-8")
