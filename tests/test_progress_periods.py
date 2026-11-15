"""Fight Camp: the last 30, 60 and 90 days (owner's brief, 2026-11-14)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.progress_periods import period_summary

NOW = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)


def _point(days_ago: float, guard: float | None) -> dict:
    return {"created_at": (NOW - timedelta(days=days_ago)).isoformat(), "guard": guard, "job_id": f"j{days_ago}"}


def _session(days_ago: float, verdict: str = "counted") -> dict:
    return {"created_at": (NOW - timedelta(days=days_ago)).isoformat(), "verdict": verdict}


def test_each_period_counts_only_its_own_fights_and_sessions():
    points = [_point(80, 0.40), _point(50, 0.45), _point(20, 0.50), _point(5, 0.60)]
    sessions = [_session(2), _session(3), _session(10), _session(40), _session(85), _session(1, "too_short")]
    by_days = {period["days"]: period for period in period_summary(points, sessions, NOW)}
    assert [by_days[d]["fights"] for d in (30, 60, 90)] == [2, 3, 4]
    assert [by_days[d]["sessions"] for d in (30, 60, 90)] == [3, 4, 5]     # the rejected session never counts
    assert by_days[30]["active_weeks"] <= by_days[30]["weeks"] == 4


def test_a_change_needs_two_measured_fights_in_the_period():
    points = [_point(80, 0.40), _point(50, 0.45), _point(5, 0.60)]
    by_days = {period["days"]: period for period in period_summary(points, [], NOW)}
    assert by_days[30]["cards"] == []                         # one fight: no change is shown
    guard_60 = next(card for card in by_days[60]["cards"] if card["key"] == "guard")
    guard_90 = next(card for card in by_days[90]["cards"] if card["key"] == "guard")
    assert (guard_60["first"], guard_60["latest"]) == ("45%", "60%")
    assert (guard_90["first"], guard_90["latest"]) == ("40%", "60%")


def test_unmeasured_values_and_bad_dates_are_left_out():
    points = [_point(10, None), _point(5, None), {"created_at": "not a date", "guard": 0.9}]
    for period in period_summary(points, [_session(1), {"created_at": None, "verdict": "counted"}], NOW):
        assert period["cards"] == []
        assert period["sessions"] == 1


def test_the_section_shows_each_period_and_says_when_there_is_too_little():
    import app.main as web

    points = [_point(80, 0.40), _point(50, 0.45), _point(5, 0.60)]
    html = web.templates.env.get_template("_camp_periods.html").render(
        periods=period_summary(points, [_session(2)], NOW), progress={"card_width": 120, "card_height": 36})
    for days in (30, 60, 90):
        assert f'id="period{days}"' in html and f'data-period="{days}"' in html
    assert "Not enough fights in the last 30 days" in html
    assert "Guard up" in html and "45%" in html and "40%" in html
