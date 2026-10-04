"""Copy the QA pass on 2026-10-04 found to be false."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _pricing(current_plan_key, payments_enabled=False):
    from test_web import PlanBadgeTests

    from core.payments import PLANS, plan_for_key
    from test_web import _real_template_env

    class Stub:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, key): return Stub()
        def __getitem__(self, key): return Stub()
        def __str__(self): return ""
        def __bool__(self): return False

    return _text(_real_template_env().get_template("pricing.html").render(
        request=Stub(url=Stub(path="/pricing"), state=Stub(account=None), cookies={}),
        plans=PLANS, roster_held=0, payments_enabled=payments_enabled,
        account={"plan": "free", "plan_override": None, "email": "a@b.c"},
        allowance={"plan": plan_for_key(current_plan_key), "remaining": None},
        current_plan_key=current_plan_key))


def test_a_gym_account_is_not_told_everyone_is_on_starter():
    page = _pricing("gym")
    assert "Everyone is on Starter" not in page
    assert "Your account is on Gym without charge" in page
    assert "Your current plan Gym" in page


def test_a_starter_account_is_told_it_is_on_starter():
    page = _pricing("free")
    assert "New accounts are on Starter, free" in page


def test_no_checkout_or_webhook_badge_while_billing_is_closed():
    closed = _pricing("free", payments_enabled=False)
    assert "Secure hosted checkout" not in closed and "Signed webhook verification" not in closed
    opened = _pricing("free", payments_enabled=True)
    assert "Secure hosted checkout" in opened


def test_a_current_report_whose_kit_could_not_be_compared_is_not_called_an_earlier_check():
    from core.report import identity_failure, lookalike_blocks_identity

    current = {"fighters_separable": False, "pair_similarity_method": "torso_hs_histogram",
               "kit_check_attempted": True, "kit_similarity": None, "identity_confusions_per_minute": 9.0,
               "identity_confusions": 5}
    assert lookalike_blocks_identity(current)
    cause = identity_failure(current)
    assert "earlier kit check" not in cause["headline"]
    assert "analyse" not in cause["advice"].lower()
    # A report analysed before the check existed still says so.
    legacy = {key: value for key, value in current.items() if key != "kit_check_attempted"}
    assert "earlier kit check" in identity_failure(legacy)["headline"]


def test_a_new_report_never_carries_the_older_reports_banner():
    from app.main import _analysed_span_summary

    new = {"video": {"analysed_span": {"start_seconds": 0.0, "end_seconds": 30.0,
                                       "video_duration_seconds": 30.0}}}
    summary = _analysed_span_summary(new)
    assert summary["note"] is None and summary["whole"] is True


def test_the_analysis_writes_the_kit_check_marker():
    source = (ROOT / "core" / "analyzer.py").read_text(encoding="utf-8")
    assert '"kit_check_attempted": True' in source


def test_the_boxing_live_page_never_lists_kicks():
    from test_web import _real_template_env

    class Stub:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, k): return Stub()
        def __str__(self): return ""
        def __bool__(self): return False

    html = _real_template_env().get_template("progress.html").render(
        request=Stub(state=Stub(csrf_token="t" * 43, csp_nonce="n", account=None), url=Stub(path="/p")),
        job_id="j", initial_status={}, live_counting_note="", strike_counts_published=True,
        live_families=("punch",), asset_version="t")
    for label in ("Kick attempts", "Kicks landed", "Kick accuracy"):
        assert re.search(rf"<div hidden><span>{label}", html), label
