"""Owner's review after the glass redesign went live (2026-10-28).

The bar never shows the email and asks for a username instead; the email can
be shown on the public page only when the athlete ticks it; Home replaces
"Analyse a video"; Compare fights lives inside the Fight library.
"""

from __future__ import annotations

import uuid

from browser_client import BrowserClient

import app.main as web
from core import db as database


def _signed_in():
    web._rate_windows.clear()
    client = BrowserClient(web.app)
    email = f"owner28-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_group": "adult"}, follow_redirects=False)
    return client, email


def _bar(page: str) -> str:
    return page[page.index('<nav class="nav"'):page.index("</nav>", page.index('<nav class="nav"'))]


def test_the_bar_never_shows_the_email_and_asks_for_a_username():
    client, email = _signed_in()
    page = client.get("/camp").text
    assert email not in _bar(page)
    assert "Choose a username" in _bar(page)
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    database.set_public_profile(profile_id, f"user{uuid.uuid4().hex[:6]}", "private", "", "")
    handle = database.get_profile(profile_id)["handle"]
    bar = _bar(client.get("/camp").text)
    assert f"@{handle}" in bar and email not in bar and "Choose a username" not in bar


def test_the_email_is_on_the_public_page_only_when_ticked():
    client, email = _signed_in()
    handle = f"pub{uuid.uuid4().hex[:6]}"
    form = {"handle": handle, "visibility": "public", "bio": "", "gym": ""}
    client.post("/profile/public", data=form, follow_redirects=False)
    assert email not in BrowserClient(web.app).get(f"/athlete/{handle}").text
    client.post("/profile/public", data={**form, "show_email": "true"}, follow_redirects=False)
    assert email in BrowserClient(web.app).get(f"/athlete/{handle}").text
    client.post("/profile/public", data=form, follow_redirects=False)        # unticked again
    assert email not in BrowserClient(web.app).get(f"/athlete/{handle}").text


def test_home_replaces_analyse_and_compare_moves_into_the_library():
    client, _ = _signed_in()
    bar = _bar(client.get("/camp").text)
    assert 'data-nav="home" href="/" title="Your homepage">Home</a>' in bar
    assert "Analyse a video" not in bar and "Compare fights" not in bar
    template = (web.ROOT / "app" / "templates" / "history.html").read_text(encoding="utf-8")
    assert 'href="/compare">Compare fights</a>' in template
    assert 'href="/compare?a={{f.job_id}}">Compare</a>' in template


def test_the_live_board_puts_each_fighter_on_the_side_they_were_picked_on(monkeypatch):
    from app import state

    monkeypatch.setattr(web, "STRIKE_COUNTS_PUBLISHED", True)
    client = BrowserClient(web.app)
    client.get("/")
    owner = f"guest:{client.cookies.get(web.GUEST_COOKIE)}"
    job_id = "board" + uuid.uuid4().hex[:8]
    # A picked on the right of the frame, B on the left.
    state.create_job(job_id, {"owner_key": owner, "status": "running", "stage": "analysis",
                              "fighter_a_box": [900, 100, 1100, 700], "fighter_b_box": [100, 100, 300, 700]})
    try:
        page = client.get(f"/progress/{job_id}").text
        board = page[page.index('id="liveBoard"'):page.index('id="videoWrap"')]
        assert board.index('id="boardB"') < board.index('id="boardA"')
        assert "live · provisional" in board
    finally:
        state.delete_job(job_id)
    assert web._board_sides({"fighter_a_box": [0, 0, 10, 10], "fighter_b_box": [50, 0, 60, 10]}) == ["A", "B"]
    assert web._board_sides({}) == ["A", "B"]


def _landed(fighter, t, round_number=1, technique="jab"):
    from core.types import StrikeEvent

    return StrikeEvent(fighter=fighter, opponent="B" if fighter == "A" else "A", round_number=round_number,
                       start_frame=int(t * 30) - 3, peak_frame=int(t * 30), end_frame=int(t * 30) + 3,
                       start_time=t - .1, peak_time=t, end_time=t + .1, technique=technique, family="punch",
                       limb="lead_hand", outcome="likely_landed", landed=True, confidence=.8, contact_confidence=.7)


def test_the_live_score_comes_from_the_reports_scorer(monkeypatch):
    """Owner, 2026-11-01: a live score per fighter, not a live strike count."""
    from core import analyzer
    from core.scoring import score_fight

    monkeypatch.setattr(analyzer, "STRIKE_COUNTS_PUBLISHED", True)
    monkeypatch.setattr(analyzer, "STRIKE_COUNTS_PRECISION_VALIDATED", False)
    events = [_landed("A", 2.0), _landed("A", 5.0), _landed("B", 8.0)]
    live = analyzer._live_score(events, "BOXING", "BOTH", 1)
    card = score_fight(events, "BOXING", [1], [], reliable=True, estimated=True)
    assert live["totals"] == card["totals"] and live["status"] == "provisional"
    assert live["current_round"]["round"] == 1
    # No opponent's points in a one-fighter analysis, and nothing before a round is known.
    assert analyzer._live_score(events, "BOXING", "A", 1) is None
    assert analyzer._live_score([], "BOXING", "BOTH", None) is None


def test_the_live_page_shows_points_and_no_statistics_section():
    template = (web.ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert 'aria-label="Live score, provisional"' in template and '<b id="boardLabel">Points</b>' in template
    assert "total_strikes" not in template[template.index("function renderBoard"):]
    assert 'class="live-stats"' not in template and "Statistics update as footage is processed" not in template
    assert 'id="liveFeedCard"' in template
