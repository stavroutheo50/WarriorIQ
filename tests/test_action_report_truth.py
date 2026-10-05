import json
from pathlib import Path

import pytest

from action_fixtures import validated_classifier
from core.analyzer import _live_event_payload, _provisional_stats
from core.report import build_report, write_report
from core.report_visuals import build as build_visuals
from core.types import AnalysisRequest, RoundSpec, StrikeEvent


def _event(accepted=False, outcome="clean", at=2):
    return StrikeEvent(
        "A", "B", 1, at * 30 - 2, at * 30, at * 30 + 2,
        at - .1, at, at + .1, "left_round_kick", "kick", "left_leg",
        outcome=outcome, target="body", confidence=.95, contact_confidence=.95,
        model_source="warrioriq_temporal_model" if accepted else "temporal_rules",
        evidence={"peak_attacker_conf": [1.] * 17, "foot_lift_torsos": 1.4,
                  "temporal_decision": {"status": "strike" if accepted else "uncertain",
                                        "label": "left_round_kick", "confidence": .95 if accepted else .2}},
    )


def _report(accepted=False):
    report = json.loads((Path(__file__).parent / "fixtures/report_sample.json").read_text(encoding="utf-8"))
    report["classifier"] = validated_classifier()
    report["events"] = [_event(accepted).to_dict()]
    report["key_moments"] = list(report["events"])
    report["illegal_moves"] = []
    report["integrity"]["action_metrics_trusted"] = accepted
    report["tracking"] = {"fighter_A_coverage": .99, "fighter_B_coverage": .99}
    return report


@pytest.mark.parametrize("status", ["uncertain", "unavailable", "no_action", "missing"])
def test_global_model_trust_cannot_promote_fallback_live_events(status):
    event = _event()
    if status == "missing":
        event.evidence.pop("temporal_decision")
    else:
        event.evidence["temporal_decision"]["status"] = status
    item = _live_event_payload([event], "K1", True)[0]
    assert item["verification"] == "observed"
    assert item["technique"] is None and item["limb"] is None
    assert item["target"] is None and item["outcome"] == "unclassified"


def test_supported_model_action_keeps_existing_live_behavior():
    item = _live_event_payload([_event(True)], "K1", True)[0]
    assert item["verification"] == "verified"
    assert item["technique"] == "left_round_kick"
    assert item["outcome"] == "landed"


def test_mixed_live_feed_cannot_publish_verified_totals_or_combinations():
    feed = _live_event_payload([_event(True), _event(False, at=3)], "K1", True)
    stats = _provisional_stats(feed, {"A": 100, "B": 100}, 100, True, 10)
    assert not stats["action_labels_available"]
    assert stats["fighters"]["A"]["landed"] is None
    assert stats["fighters"]["A"]["combinations"] is None


def test_new_report_does_not_trust_candidate_metrics_despite_validated_model():
    sample = _report()
    report = build_report(
        AnalysisRequest("unused.mp4", [1, 1, 10, 10], [20, 1, 30, 10]), "unused.mp4",
        [RoundSpec(1, 0, 60)], [_event()], [], sample["metrics"], sample["tracking"],
        sample["performance"], sample["classifier"],
    )
    assert not report["integrity"]["action_metrics_trusted"]
    assert report["key_moments"] == []
    assert report["illegal_moves"] == []
    assert report["integrity"]["coaching_evidence_mode"] == "pose_only"


def test_saved_report_rechecks_each_action_and_revokes_old_statistics():
    from app.main import _apply_report_annotations

    report = _report()
    report["integrity"]["action_metrics_trusted"] = True  # stale stored flag
    report["statistics"] = {"action_labels_available": True, "fighters": {"A": {"landed": 99}}}
    report["event_feed"] = _live_event_payload([_event(True)], "K1", True)  # stale verified feed
    _apply_report_annotations(report, [])
    assert not report["integrity"]["action_metrics_trusted"]
    assert report["key_moments"] == []
    assert not report["statistics"]["action_labels_available"]
    assert report["statistics"]["fighters"]["A"]["landed"] is None
    assert all(item["verification"] != "verified" for item in report["event_feed"])


def test_untrusted_visuals_keep_pose_not_hit_maps_or_combinations():
    report = _report()
    visuals = build_visuals(report)
    assert any(row["key"] == "guard" for row in visuals["head_to_head"])
    assert all(row["key"] != "landed" for row in visuals["head_to_head"])
    assert visuals["landed"] is None and visuals["taken"] is None
    assert not visuals["timeline"] and not visuals["chains"] and not visuals["defences"]


def test_blocked_kick_is_never_a_landing_even_in_a_trusted_report():
    report = _report(True)
    report["events"] = [_event(True, "blocked").to_dict()]
    visuals = build_visuals(report)
    assert visuals["landed"]["body"] == 0


def test_standalone_candidates_are_not_asserted_as_fight_facts(tmp_path):
    report = _report()
    _, html = write_report(tmp_path, report)
    text = html.read_text(encoding="utf-8")
    assert "unverified" in text.lower()
    assert "that reached" not in text
    assert "each one happened" not in text
    assert "cannot yet tell you reliably" not in text


def test_demotion_preserves_distinct_opposite_hand_candidates():
    from app.main import _apply_report_annotations

    report = _report()
    report["event_feed"] = [
        {"fighter": "A", "family": "punch", "limb": "left_hand", "technique": "jab",
         "time_seconds": 1., "outcome": "landed", "verification": "verified"},
        {"fighter": "A", "family": "punch", "limb": "right_hand", "technique": "cross",
         "time_seconds": 1.3, "outcome": "landed", "verification": "verified"},
    ]
    _apply_report_annotations(report, [])
    assert report["statistics"]["fighters"]["A"]["attempts"] == 2
    assert len(report["event_feed"]) == 2
    assert report["statistics"]["fighters"]["A"]["combinations"] is None


def test_completed_human_review_uses_corrections_not_original_candidates(tmp_path):
    from app.main import _apply_report_annotations

    report = _report()
    candidate = report["events"][0]
    corrected = dict(candidate, fighter="B", outcome="blocked", contact_time=2.)
    _apply_report_annotations(report, [{"event_time": 2., "predicted": candidate, "corrected": corrected}],
                              human_review_complete=True)
    assert report["statistics"]["action_labels_available"]
    assert report["statistics"]["fighters"]["B"]["blocked"] == 1
    assert report["statistics"]["fighters"]["A"]["landed"] == 0
    visuals = build_visuals(report)
    assert visuals["action_ready"]
    assert sum(visuals["landed"].values()) == 0
    assert sum(visuals["taken"].values()) == 0
    _, html = write_report(tmp_path, report)
    assert "<th>Outcome</th>" in html.read_text(encoding="utf-8")


def test_reviewed_clean_contact_matches_landed_statistics_and_replay():
    from app.main import _apply_report_annotations, _build_replay_chapters

    report = _report()
    candidate = report["events"][0]
    corrected = dict(candidate, outcome="clean", contact_time=2.)
    _apply_report_annotations(report, [{"event_time": 2., "predicted": candidate, "corrected": corrected}],
                              human_review_complete=True)
    assert report["statistics"]["fighters"]["A"]["landed"] == 1
    chapters, mode = _build_replay_chapters(report, "A", outcome_filter="landed")
    assert mode == "verified_actions"
    assert len(chapters) == 1
    assert chapters[0]["time"] == 2.
    assert chapters[0]["lead_seconds"] == 1.
