"""Audit of 14 Oct 2026, phase 10: homepage and pricing.

#51 the allowance period is configurable per plan (day, month or video
    minutes); the current daily limits stay the default
#53 the original A/B interface preview stays (owner's decision, 2026-10-19)
#54 a public sample report, made-up data, labelled as a sample
#55 homepage sections: measured per sport, accuracy, FAQ, pricing summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from browser_client import BrowserClient

import app.main as web
from core import db as database
from core import payments
from core.auth import hash_password


# --------------------------------------------------------------------- #51
def test_the_current_limits_are_still_per_day():
    for key, plan in payments.PLANS.items():
        rule = payments.allowance(plan)
        if plan.get("unlimited"):
            assert rule["period"] == "unlimited"
        else:
            assert rule == {"period": "day", "amount": plan["daily_limit"]}, key
    assert payments.PLANS["athlete"]["limit_label"] == "3 analyses every day"


def test_a_plan_can_count_per_month_or_in_video_minutes():
    month = {"allowance_period": "month", "monthly_limit": 30, "daily_limit": 3}
    minutes = {"allowance_period": "video_minutes", "monthly_video_minutes": 120, "daily_limit": 3}
    assert payments.allowance(month) == {"period": "month", "amount": 30}
    assert payments.allowance_text(minutes) == "120 video minutes per month"
    # Asked for a period without its number: keeps counting per day.
    assert payments.allowance({"allowance_period": "month", "daily_limit": 3}) == {"period": "day", "amount": 3}


def _account_on(monkeypatch, plan: dict):
    account = database.create_account(f"p10-{uuid.uuid4().hex[:8]}@example.com", hash_password("Strong-Local-Password"))
    monkeypatch.setattr(database, "plan_for_key", lambda key: plan)
    return int(account["id"])


def test_a_minutes_plan_counts_minutes(monkeypatch):
    account_id = _account_on(monkeypatch, {"allowance_period": "video_minutes", "monthly_video_minutes": 10,
                                           "daily_limit": 1})
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    assert database.reserve_analysis(account_id, "p10a" + uuid.uuid4().hex[:8], now, video_seconds=360)
    assert database.reserve_analysis(account_id, "p10b" + uuid.uuid4().hex[:8], now, video_seconds=180)
    # 9 minutes used: a 2-minute video would pass the 10-minute limit.
    assert not database.reserve_analysis(account_id, "p10c" + uuid.uuid4().hex[:8], now, video_seconds=120)
    left = database.analysis_allowance(account_id, now)
    assert (left["unit"], left["used"], left["remaining"]) == ("minutes", 9.0, 1)


def test_a_monthly_plan_counts_per_month(monkeypatch):
    account_id = _account_on(monkeypatch, {"allowance_period": "month", "monthly_limit": 2, "daily_limit": 1})
    day1, day2 = datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 20, tzinfo=timezone.utc)
    assert database.reserve_analysis(account_id, "p10d" + uuid.uuid4().hex[:8], day1)
    assert database.reserve_analysis(account_id, "p10e" + uuid.uuid4().hex[:8], day1)   # past the daily 1
    assert not database.reserve_analysis(account_id, "p10f" + uuid.uuid4().hex[:8], day2)


def test_the_usage_column_is_a_backward_compatible_addition():
    database.init_db()
    with database.connection() as con:
        columns = {row[1]: row for row in con.execute("PRAGMA table_info(analysis_usage)")}
    assert columns["video_seconds"][3] == 0           # nullable


def test_pricing_copy_follows_the_period():
    page = BrowserClient(web.app).get("/pricing").text
    assert '<abbr title="Analyses per day">Per day</abbr>' in page
    assert "more daily analyses" not in page


# --------------------------------------------------------------------- #53
def test_the_hero_shows_a_real_tracking_example():
    # The empty interface preview became a real example (owner, 2026-10-28):
    # a stock clip with WarriorIQ's own tracking drawn on it, never a placeholder.
    # Since 2026-11-22 it is a still picture, not a video: the owner found the
    # moving clip behind the headline confusing. Still no sample report.
    # Since 2026-11-25 (home redesign "Corner Light") it is one frame with both
    # fighters in full view and the pose model's skeleton drawn on each.
    page = BrowserClient(web.app).get("/").text
    assert 'src="/static/home/hero-corners.jpg' in page
    assert "Real WarriorIQ tracking" in page
    hero = page[page.index('class="cg-hero"'):page.index("</section>", page.index('class="cg-hero"'))]
    assert "<video" not in hero and "autoplay" not in hero
    assert "<video" not in page                        # no looping video anywhere on the home page
    # The clip shows tracking only (owner, 2026-11-01): its strike names were
    # wrong too often, so neither the picture nor its description may claim them.
    assert "strike named" not in page and "each strike" not in hero
    # Removed entirely, not replaced (owner, 2026-11-09).
    assert "From a real WarriorIQ report" not in page and "home-proof" not in page
    assert "Demo clip coming soon" not in page and "goes here" not in page


# --------------------------------------------------------------------- #54
def test_there_is_no_sample_report():
    # Owner, 2026-11-02: "no sample reports should exist". Old links go home.
    client = BrowserClient(web.app)
    response = client.get("/sample-report", follow_redirects=False)
    assert response.status_code == 301 and response.headers["location"] == "/"
    home = client.get("/").text
    assert "/sample-report" not in home and "Sample report" not in home
    assert not (web.ROOT / "app" / "templates" / "sample_report.html").exists()


# --------------------------------------------------------------------- #55
def test_the_homepage_has_the_new_sections():
    page = BrowserClient(web.app).get("/").text
    for anchor in ('id="what-is-measured"', 'class="product-section faq', 'href="/accuracy"'):
        assert anchor in page, anchor
    for sport in ("Kickboxing", "Boxing", "Muay Thai", "Taekwondo", "MMA"):
        assert f"<h3>{sport}</h3>" in page
    assert "about a third" not in page                     # moved to /accuracy (2026-10-19)
    accuracy = BrowserClient(web.app).get("/accuracy").text
    assert "about a third" in accuracy and "11 of 16" in accuracy
    # The plans block left the home page in the 2026-11-25 redesign (owner
    # approved design G); prices live on /pricing, linked from the menu.
    assert 'id="plans"' not in page


@pytest.mark.parametrize("published", [True, False])
def test_the_counting_line_follows_the_switch(monkeypatch, published):
    monkeypatch.setattr(web, "STRIKE_COUNTS_PUBLISHED", published)
    measured = web._home_sections()["measured"]
    if published:
        assert measured["boxing"]["counted"] == "punches"
    else:
        assert all(not item["counted"] for item in measured.values())
