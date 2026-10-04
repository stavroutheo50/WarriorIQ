"""Copy fixes from the QA pass on 2026-10-04."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")


def test_counts_agree_with_their_nouns():
    from core.coaching import count_of

    assert count_of(1, "attempt") == "1 attempt"
    assert count_of(0, "attempt") == "0 attempts"
    assert count_of(3, "combination") == "3 combinations"
    assert count_of(None, "fight") == "0 fights"


def test_coaching_never_says_one_attempts():
    from core import coaching

    source = Path(coaching.__file__).read_text(encoding="utf-8")
    assert "} attempts" not in source.replace("count_of", "")
    assert "count_of(attacks.get('attempts', 0), 'attempt')" in source


def test_the_summary_does_not_repeat_keep_doing_and_fix_next():
    summary = RESULT.split('id="report-summary"', 1)[1].split("</section>", 1)[0]
    assert "keep[0]" not in summary and "fix[0]" not in summary
    assert "{% set worked = keep[1] if keep|length > 1 else None %}" in summary


def test_no_empty_guard_timeline():
    from core.report_visuals import build

    report = {"metrics": {"A": {"moments": {"guard_index": {"low": [400.0, 512.3]}}}, "B": {}},
              "rounds": [{"end_seconds": 120.0}], "scorecard": {"sport": "kickboxing"}}
    visuals = build(report, "A", outcomes_counted=False)
    assert visuals["guard_low"] == []
    assert "{% if visuals.guard_low %}<span><i class=\"key-dot guard\"></i>" in RESULT


def test_no_jargon_fallbacks():
    for phrase in ("Keep the measured baseline", "Build the next clean sample"):
        assert phrase not in RESULT
    from core import coaching

    assert "measured baseline" not in Path(coaching.__file__).read_text(encoding="utf-8")
