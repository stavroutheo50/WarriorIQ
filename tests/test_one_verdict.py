"""One verdict per report.

QA, 2026-10-04: the same report said "Not scored, we lost sight of a fighter"
and "Good observation evidence / Identity stability: Stable", and offered a
training plan and a success target "from the fighter's measured report" while
identity had not been confirmed. Every section now reads
core.report.identity_verdict.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"


def _report(cov_a=0.95, cov_b=0.95, **tracking):
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["tracking"].update({"fighter_A_coverage": cov_a, "fighter_B_coverage": cov_b, **tracking})
    report["metrics"]["A"]["pose_coverage"] = 0.9
    return report


def _page(report):
    from test_web import NoPunchClaimLeaksTests

    from app.main import _score_and_identity_as_shown, _score_withheld

    report = deepcopy(report)
    _score_and_identity_as_shown(report)
    html = NoPunchClaimLeaksTests._render(report, strike_counts_published=False, families_shown=(),
                                          score_withheld=_score_withheld(report))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def test_lost_sight_is_never_called_good_evidence():
    from app.main import _analysis_quality_summary, _score_withheld
    from core.report import identity_verdict

    report = _report(cov_a=0.70, cov_b=0.72)
    verdict = identity_verdict(report)
    assert verdict["trusted"] is True and verdict["followed_enough_to_score"] is False
    quality = _analysis_quality_summary(report)
    assert quality["label"] == "Partial observation evidence"
    report["scorecard"]["status"] = "insufficient_observation_coverage"
    report["scorecard"]["available"] = False
    assert "lost sight" in _score_withheld(report)["reason"]
    text = _page(report)
    assert "Good observation evidence" not in text and "Strong observation evidence" not in text


def test_an_untrusted_report_offers_no_plan_target_or_coaching_claim():
    from core.report import identity_verdict, refresh_identity_integrity

    # Fighter B lost: identity fails as a whole, even though A passed alone.
    report = _report(cov_a=0.95, cov_b=0.20)
    assert identity_verdict(report)["trusted"] is False
    refreshed = refresh_identity_integrity(deepcopy(report))
    assert refreshed["training_plan"]["A"] == [] and refreshed["coaching"]["A"]["improvements"] == []
    text = _page(report)
    for claim in ("Success target", "Keep doing", "Fix next", "not a generic template",
                  "Next training priority", "Identity stability Stable"):
        assert claim not in text, claim


def test_every_section_agrees_with_the_verdict():
    from app.main import _analysis_quality_summary, _numbers_state
    from core.report import identity_verdict, refresh_identity_integrity

    for cov_b in (0.20, 0.95):
        report = refresh_identity_integrity(_report(cov_b=cov_b))
        trusted = identity_verdict(report)["trusted"]
        assert report["integrity"]["identity_evidence_trusted"] is trusted
        assert _analysis_quality_summary(report)["identity_stable"] is trusted
        assert (_numbers_state(report)["state"] == "unverified") is (not trusted)


def test_the_measured_target_sentence_needs_a_measured_target():
    page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    sentence = ('This target comes from <span class="fighter-name-fit" title="{{ summary_name }}">'
                "{{ summary_name }}</span>’s measured report, not a generic template.")
    before = page[:page.index(sentence)]
    assert before.rstrip().endswith("{% if plan %}<p>")
