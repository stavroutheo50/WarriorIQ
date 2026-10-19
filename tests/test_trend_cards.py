"""Fight Camp: every measure, fight by fight (core/progress_insights.trend_cards)."""

from core.progress_insights import build_progress


def _record(job_id, created, guard, pressure=None, numbers=None, identity=None):
    metrics = {"pose_coverage": .85, "guard_index": guard, "guard_definition": "hands_up_share/3"}
    if pressure is not None:
        metrics["pressure_index"] = pressure
    if numbers is not None:
        metrics["numbers"] = numbers
    report = {"video": {"analysis_target": "A", "focus_fighter": "A"}, "setup": {"ruleset": "K1"},
              "metrics": {"A": metrics}, "coaching": {}, "training_plan": {}}
    if identity is not None:
        report["integrity"] = identity
    return {"job_id": job_id, "created_at": created, "fighter_id": 1, "report": report}


def _cards(records):
    return {card["key"]: card for card in build_progress(records, "A")["cards"]}


def test_a_card_per_measure_with_two_fights_and_the_change_since_the_first():
    cards = _cards([_record("a", "2026-08-01", .20, pressure=.10),
                    _record("b", "2026-08-08", .30, pressure=.30),
                    _record("c", "2026-08-15", .45, pressure=.25)])
    guard = cards["guard"]
    assert (guard["first"], guard["latest"], guard["change"], guard["fights"]) == ("20%", "45%", "+25 percentage points", 3)
    assert guard["direction"] == "up"                     # hands up more is better
    assert (cards["pressure"]["first"], cards["pressure"]["latest"]) == ("55", "62")   # out of 100
    assert cards["pressure"]["change"] == "+7 on the 0-100 scale"   # the printed figures, 62 - 55
    assert cards["pressure"]["direction"] == "neutral"   # style, not quality
    assert len(guard["line"].split()) == 3


def test_one_fight_is_not_a_trend():
    assert _cards([_record("a", "2026-08-01", .20)]) == {}


def test_a_fight_whose_identity_failed_is_left_out():
    cards = _cards([_record("a", "2026-08-01", .20),
                    _record("bad", "2026-08-05", .90, identity={"identity_evidence_trusted": False}),
                    _record("b", "2026-08-08", .25)])
    assert cards["guard"]["fights"] == 2
    assert cards["guard"]["latest"] == "25%"


def test_shares_come_from_the_movement_split_and_distance():
    numbers = {"forward_seconds": 30.0, "backward_seconds": 10.0, "circling_seconds": 10.0, "standing_seconds": 50.0,
               "range_seconds": {"close": 20.0, "middle": 60.0, "long": 20.0}}
    cards = _cards([_record("a", "2026-08-01", .2, numbers=numbers),
                    _record("b", "2026-08-08", .2, numbers=dict(numbers, forward_seconds=50.0))])
    assert cards["forward"]["first"] == "30%"
    assert cards["forward"]["latest"] == "42%"            # 50 of 120 seconds
    assert cards["close"]["change"] == "no change"
    assert cards["close"]["direction"] == "flat"
