"""Strike counts switched off: the production default since 2026-10-04.

QA found a waist-up boxing clip counted as 33 kicks and 14 knees, and the
Accuracy Lab shows 10 real of 26 counted strikes. Until classification meets
the release targets on /validation, WARRIORIQ_PUBLISH_STRIKE_COUNTS stays
unset and no punch, kick or knee count reaches the report summary, the live
progress page, the replay chapter list or the story card. The rest of the
suite runs with the flag on (tests/conftest.py); this file switches it off
where it is read.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from core import report as report_module


@pytest.fixture
def counts_off(monkeypatch):
    import app.main as web

    monkeypatch.setattr(report_module, "STRIKE_COUNTS_PUBLISHED", False)
    monkeypatch.setattr(web, "STRIKE_COUNTS_PUBLISHED", False)
    return web


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


STRIKE_CLAIMS = (r"\d+\s*(punches|kicks|knees|strikes)\b", r"(kicks|strikes|punches) thrown",
                 r"Kick attempts", r"Strikes we counted", r"Watch every counted strike")


def test_the_default_is_off():
    env = {key: value for key, value in os.environ.items() if key != "WARRIORIQ_PUBLISH_STRIKE_COUNTS"}
    out = subprocess.run(
        [sys.executable, "-c", "from core.report import STRIKE_COUNTS_PUBLISHED as p; print(p)"],
        env=env, capture_output=True, text=True, check=True, cwd=os.getcwd())
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("sport", ["kickboxing", "boxing", "muay_thai", "taekwondo", "mma"])
def test_no_family_is_published_for_any_sport(counts_off, sport):
    from core.sport_policy import counting_policy

    assert report_module.published_families(sport) == ()
    policy = counting_policy(sport)
    assert policy.counted == ()
    assert "no strike counts" in policy.live_note


def test_report_summary_shows_no_strike_count(counts_off):
    from test_web import NoPunchClaimLeaksTests

    for trusted in (True, False):
        report = NoPunchClaimLeaksTests._report(trusted=trusted)
        text = _text(NoPunchClaimLeaksTests._render(
            report, strike_counts_published=False, families_shown=()))
        for pattern in STRIKE_CLAIMS:
            found = re.search(pattern, text, re.I)
            assert found is None, (trusted, pattern, text[max(0, found.start() - 60):found.end() + 40])


def test_story_card_carries_movement_and_no_counts(counts_off):
    from test_web import NoPunchClaimLeaksTests

    report = NoPunchClaimLeaksTests._report(trusted=True)
    report["metrics"]["A"].update({"guard_index": 0.64, "balance_index": 0.8,
                                   "ring_center_control": 0.5, "pressure_index": 0.1})
    card = report_module.share_card(report)
    me = card["fighters"]["A"]
    assert me["strikes"] is None and me["total"] is None and card["score"] is None
    assert {"label": "Guard up", "value": 64, "unit": "%"} in me["movement"]
    assert {"label": "Pressure", "value": 55, "unit": " of 100"} in me["movement"]

    from core.share_image import preview_png

    assert preview_png(card, "A", None)[:4] == b"\x89PNG"


def test_story_page_shows_movement_not_strikes(counts_off):
    from test_web import _real_template_env

    card = {"sport": "Boxing", "note": report_module.SHARE_CARD_NOTE, "score": None, "fighters": {
        side: {"strikes": None, "total": None, "strength": None, "working_on": None,
               "movement": [{"label": "Guard up", "value": 60 if side == "A" else 40, "unit": "%"}]}
        for side in ("A", "B")}}

    class Stub:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, k): return Stub()
        def __str__(self): return ""
        def __bool__(self): return False

    html = _real_template_env().get_template("story.html").render(
        request=Stub(state=Stub(csrf_token="t" * 43, account=None), url=Stub(path="/s")),
        card=card, side="A", other="B", corner=None, name=None, page_url="u", image_url="i",
        asset_version="t", fought_at=None)
    text = _text(html)
    assert "Guard up" in text and "60%" in text and "40%" in text
    assert not re.search(r"strikes thrown", text, re.I)


def test_live_status_sends_no_counts(counts_off):
    job = {"status": "running", "live_events": [{"family": "kick", "fighter": "A"}],
           "provisional_stats": {"attempt_counts_available": True, "fighters": {
               "A": {"observation_coverage": 0.7, "kick_attempts": 33, "knee_attempts": 14}}}}
    payload = counts_off._public_job_status("job", job)
    assert payload["live_events"] == []
    assert payload["provisional_stats"]["attempt_counts_available"] is False
    assert payload["provisional_stats"]["fighters"] == {"A": {"observation_coverage": 0.7}}


def test_live_page_hides_the_feed_and_strike_rows(counts_off):
    from test_web import _real_template_env

    class Stub:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, k): return Stub()
        def __str__(self): return ""
        def __bool__(self): return False

    html = _real_template_env().get_template("progress.html").render(
        request=Stub(state=Stub(csrf_token="t" * 43, csp_nonce="n", account=None), url=Stub(path="/p")),
        job_id="j", initial_status={}, live_counting_note="", strike_counts_published=False,
        asset_version="t")
    assert re.search(r'id="liveFeedCard"[^>]*\bhidden\b', html)
    assert re.search(r'<div hidden><span>Kick attempts', html)
    assert re.search(r'<div hidden><span>Total attempts', html)
    assert re.search(r'<div><span>Identity lock', html)


def test_a_stored_estimated_score_is_withheld_on_read(counts_off):
    report = {"scorecard": {"available": True, "status": "preliminary_unvalidated",
                            "totals": {"A": 30, "B": 27}, "sport": "kickboxing"}}
    counts_off._withhold_score_while_counts_are_off(report)
    assert report["scorecard"]["available"] is False
    assert report["scorecard"]["totals"] == {"A": None, "B": None}
    withheld = counts_off._score_withheld(report)
    assert "does not count strikes" in withheld["reason"]


def test_no_kick_minimum_table_without_counts(counts_off):
    from test_web import NoPunchClaimLeaksTests

    assert report_module.kick_minimum_check(NoPunchClaimLeaksTests._report(trusted=True)) is None
