"""Key moments: the report's coaching points as replay cards (core.report.coaching_moments)."""

import json

from core.report import coaching_moments


def _report(trusted=True):
    return {
        "video": {"focus_fighter": "A"}, "setup": {"start_seconds": 0.0},
        "integrity": {"identity_evidence_trusted": trusted},
        "key_moments": [],
        "coaching": {
            "A": {
                "strengths": [{"title": "Centre: ahead of your opponent", "detail": "You held the middle.",
                               "evidence_times": [40.0, 12.5]}],
                "improvements": [
                    {"title": "Work on: Guard 12%", "detail": "Hands dropped after exchanges.",
                     "evidence_times": [28.0, 41.0, 72.0, 90.0, 101.0]},
                    {"title": "Work on: Balance 60%", "detail": "Off balance after kicks.", "evidence_times": []},
                ],
            },
            "B": {"strengths": [], "improvements": [{"title": "Work on: Pressure", "detail": "x", "evidence_times": [5.0]}]},
        },
    }


def test_one_card_per_point_with_its_moments_in_time_order():
    cards = coaching_moments(_report(), ["A"], "full", None)
    assert [(c["kind"], c["title"]) for c in cards] == [("keep", "Centre: ahead of your opponent"),
                                                        ("fix", "Guard 12%")]
    assert cards[0]["times"] == [12.5, 40.0]
    assert cards[1]["times"] == [28.0, 41.0, 72.0, 90.0]           # at most four


def test_points_without_moments_are_left_to_the_report():
    titles = [c["title"] for c in coaching_moments(_report(), ["A"], "full", None)]
    assert "Balance 60%" not in titles


def test_nothing_when_the_identity_check_failed():
    assert coaching_moments(_report(trusted=False), ["A", "B"], "full", None) == []


def test_the_plan_cut_matches_the_report():
    compact = coaching_moments(_report(), ["A"], "compact", 1)
    assert [c["kind"] for c in compact] == ["fix"]                  # no strengths on compact
    partial = coaching_moments(_report(), ["A"], "expanded", 1)
    assert [c["kind"] for c in partial] == ["keep", "fix"]


def test_only_the_fighters_asked_for():
    assert {c["fighter"] for c in coaching_moments(_report(), ["A", "B"], "full", None)} == {"A", "B"}
    assert {c["fighter"] for c in coaching_moments(_report(), ["B"], "full", None)} == {"B"}


def test_the_replay_page_shows_the_cards(tmp_path, monkeypatch):
    import app.main as web
    from browser_client import BrowserClient

    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    monkeypatch.setattr(web, "_authorized_job", lambda request, job_id: {"ruleset": "K1"})
    monkeypatch.setattr(web, "_require_completed_artifact", lambda job_id, name: path)
    monkeypatch.setattr(web, "refresh_identity_integrity", lambda report: report)
    # The page rebuilds coaching from the metrics; this report states it directly.
    monkeypatch.setattr(web, "_apply_report_annotations", lambda report, annotations: None)
    monkeypatch.setattr(web, "_request_plan", lambda request: {"report_tier": "full", "coaching_items": None})
    with BrowserClient(web.app) as client:
        page = client.get("/replay/abcdef123456").text
    assert 'id="moments"' in page
    assert "Fix next" in page and "Keep doing" in page
    assert 'data-time="28.0"' in page and ">0:28<" in page
    assert "Work on: Guard" not in page                             # the prefix is the badge's job


def test_no_cards_section_when_there_are_no_moments(tmp_path, monkeypatch):
    import app.main as web
    from browser_client import BrowserClient

    report = _report(trusted=False)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(web, "_authorized_job", lambda request, job_id: {"ruleset": "K1"})
    monkeypatch.setattr(web, "_require_completed_artifact", lambda job_id, name: path)
    monkeypatch.setattr(web, "refresh_identity_integrity", lambda report: report)
    # The page rebuilds coaching from the metrics; this report states it directly.
    monkeypatch.setattr(web, "_apply_report_annotations", lambda report, annotations: None)
    with BrowserClient(web.app) as client:
        page = client.get("/replay/abcdef123456").text
    assert 'id="moments"' not in page
