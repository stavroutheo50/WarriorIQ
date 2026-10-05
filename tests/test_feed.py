"""The feed (core/feed.py, app.main.feed_page): only what really happened, only to whom it may."""

from core import feed


def _entry(points, day, entry_id):
    return {"id": entry_id, "points": points, "created_at": f"2026-09-{day:02d}T10:00:00+00:00"}


def test_a_level_is_dated_by_the_entry_that_reached_it():
    items = feed.level_ups([_entry(60, 1, 1), _entry(50, 3, 2), _entry(-200, 4, 3), _entry(100, 5, 4)])
    assert [(i["level"], i["at"][:10]) for i in items] == [(2, "2026-09-03"), (3, "2026-09-05")]
    # Reaching level 3 is a new rank (Prospect); level 2 is not.
    assert [i["new_rank"] for i in items] == [False, True]
    assert items[1]["rank"] == "Prospect"


def test_spending_points_announces_nothing():
    assert feed.level_ups([_entry(90, 1, 1), _entry(-200, 2, 2)]) == []


def test_one_big_entry_can_reach_two_levels():
    assert [i["level"] for i in feed.level_ups([_entry(250, 1, 1)])] == [2, 3]


def _session(day, verdict="counted"):
    return {"verdict": verdict, "created_at": f"2026-{day}T18:00:00+00:00"}


def test_a_streak_milestone_needs_consecutive_weeks():
    # Weeks of 7 Sep, 14 Sep, (gap), 28 Sep, 5 Oct.
    sessions = [_session("09-08"), _session("09-09"), _session("09-15"), _session("09-29"), _session("10-06")]
    items = feed.streak_milestones(sessions)
    assert [(i["weeks"], i["at"][:10]) for i in items] == [(2, "2026-09-15"), (2, "2026-10-06")]


def test_sessions_that_did_not_count_build_no_streak():
    assert feed.streak_milestones([_session("09-08", "rejected"), _session("09-15", "rejected")]) == []


def test_newest_first_and_capped():
    items = [{"at": "2026-09-01T00:00:00+00:00"}, {"at": "2026-09-03T00:00:00+00:00"},
             {"at": "2026-09-02T00:00:00"}]
    assert [i["at"][:10] for i in feed.newest_first(items, 2)] == ["2026-09-03", "2026-09-02"]
