"""Step-by-step drills on the report's "Fix this first" card (2026-11-26)."""

from core.coaching import POSE_DIMENSIONS
from core.drills import DRILL_STEPS, drill_steps
from core.report import fix_first


def test_every_pose_drill_has_a_step_by_step_version():
    for _key, _label, _reference, drill, _prescription in POSE_DIMENSIONS:
        how = drill_steps(f"Fighter A · {drill}")
        assert how is not None, drill
        assert [step["label"] for step in how["steps"]] == ["Set up", "Do it", "Clean rep", "Reset"]
        assert how["stance"] and how["solo"].startswith("No partner today")
    assert set(DRILL_STEPS) == {row[3] for row in POSE_DIMENSIONS}


def test_unknown_drills_have_no_steps():
    assert drill_steps("Two-to-four strike chain drill") is None


def _report():
    return {
        "coaching": {"A": {
            "improvements": [{"title": "Work on: Guard", "detail": "Hands low.", "metric": "guard_index",
                              "evidence_times": [80.0, 12.5, 40.0]}],
            "drills": [{"name": "Fighter A · Guard-return audit", "prescription": "4 x 90 sec", "metric": "guard_index"}],
        }},
    }


def test_the_card_carries_every_moment_and_the_steps():
    card = fix_first(_report(), "A")
    assert [m["seconds"] for m in card["moments"]] == [12.5, 40.0, 80.0]
    assert card["moment_seconds"] == 12.5 and card["more_moments"] == 2
    assert card["drill"]["how"]["steps"][0]["label"] == "Set up"


def test_a_plan_without_training_keeps_the_drill_off():
    assert fix_first(_report(), "A", training_items=0)["drill"] is None
