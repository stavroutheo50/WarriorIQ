"""One truthful counting policy per sport, followed by every surface.

QA 2026-09: the upload page, live view, report and replay each described the
same counts differently, and the boxing copy warned about kicks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import analyzer
from core.sport_policy import SPORT_LABELS, counting_policy

SPORTS = tuple(SPORT_LABELS)


@pytest.mark.parametrize("sport", SPORTS)
def test_every_sentence_names_the_same_families(sport):
    policy = counting_policy(sport, published=True, validated=False)
    for family in policy.counted:
        assert family in policy.setup_line
        assert family in policy.live_note
        assert family in policy.report_frame
    assert policy.estimates
    assert "estimate" in policy.estimate_note


def test_boxing_counts_punches_and_never_talks_about_kicks():
    policy = counting_policy("boxing", published=True, validated=False)
    assert policy.counted == ("punches",)
    for text in (policy.setup_line, policy.estimate_note, policy.live_note,
                 policy.report_frame, policy.badge, policy.not_analysed_line):
        assert "kick" not in text.lower(), text
    assert "mixed up punches and kicks" not in policy.estimate_note


def test_only_kickboxing_claims_the_hand_checked_detail():
    """The one measurement is a kickboxing bout; other sports say they are unchecked."""
    for sport in ("boxing", "muay_thai", "taekwondo", "mma"):
        assert "not been checked by hand" in counting_policy(sport, published=True).estimate_note


def test_mma_discloses_what_it_does_not_analyse():
    policy = counting_policy("mma", published=True)
    assert "takedowns" in policy.not_analysed_line and "ground work" in policy.not_analysed_line
    assert "standing exchanges only" in policy.not_analysed_line


def test_unpublished_counts_say_so_everywhere():
    # Switched off, nothing is counted - kicks included (QA, 2026-10-04).
    policy = counting_policy("kickboxing", published=False, validated=False)
    assert policy.counted == ()
    assert "no strike counts yet" in policy.setup_line
    assert "punches" not in policy.live_note and "kicks" not in policy.live_note
    assert counting_policy("boxing", published=False, validated=False).badge == "No punch counts yet"
    assert policy.badge == "No strike counts yet"


def _event(family, ruleset_ok=True):
    return SimpleNamespace(attempted=True, family=family, peak_time=1.0, confidence=0.9,
                           outcome="clean", evidence={}, metadata={})


@pytest.mark.parametrize("ruleset,family,dropped", [
    ("WT_TAEKWONDO", "knee", True),      # taekwondo does not count knees
    ("WT_TAEKWONDO", "kick", False),
    ("BOXING", "kick", True),
    ("BOXING", "punch", False),
    ("K1", "knee", False),
    ("MUAY_THAI", "knee", False),
])
def test_live_feed_lists_only_what_the_report_counts(ruleset, family, dropped):
    reason = analyzer._attempt_drop_reason(_event(family), ruleset)
    assert (reason == "not_in_this_sport") is dropped, reason


def test_progress_page_carries_the_sports_live_note():
    import app.main as web

    template = (web.ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert "liveCountingNote" in template
    assert "leg strikes only" not in template.lower()


def test_replay_offers_counted_strikes_as_estimates(tmp_path, monkeypatch):
    import json

    import app.main as web
    from browser_client import BrowserClient

    report = {
        "video": {"focus_fighter": "A"}, "setup": {"start_seconds": 0.0},
        "scorecard": {"sport": "taekwondo"},
        "integrity": {"identity_evidence_trusted": True},
        "key_moments": [],
        "event_feed": [
            {"fighter": "A", "family": "kick", "time_seconds": 3.0, "verification": "observed"},
            {"fighter": "B", "family": "punch", "time_seconds": 7.5, "verification": "observed"},
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(web, "_authorized_job", lambda request, job_id: {"ruleset": "WT_TAEKWONDO"})
    monkeypatch.setattr(web, "_require_completed_artifact", lambda job_id, name: path)
    monkeypatch.setattr(web, "refresh_identity_integrity", lambda report: report)
    with BrowserClient(web.app) as client:
        page = client.get("/replay/abcdef123456").text
    assert "No action label passed the verification gate" not in page
    assert "Counted automatically (estimate)" in page
    assert "0:03.0 · Fighter A · Kick" in page
    assert counting_policy("taekwondo").replay_note in page


def test_the_legacy_constant_is_the_kickboxing_sentence():
    from core.report import ESTIMATE_NOTE

    assert counting_policy("kickboxing", published=True, validated=False).estimate_note == ESTIMATE_NOTE


def test_upload_forms_are_honest_about_mma_and_default_taekwondo_to_wt():
    """QA 2026-09: the MMA page said "Most rounds turn on the ground" without
    saying ground work is not analysed, and taekwondo defaulted to ITF."""
    import re

    import app.main as web
    from browser_client import BrowserClient

    with BrowserClient(web.app) as client:
        mma = client.get("/analyze/mma").text
        taekwondo = client.get("/analyze/taekwondo").text
    assert "Most rounds turn on the ground" not in mma
    assert "takedowns, ground work and submissions are not read" in mma
    options = re.findall(r'<option value="(\w+_TAEKWONDO)"', taekwondo)
    assert options[0] == "WT_TAEKWONDO"
