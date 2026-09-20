import copy

import numpy as np
import pytest

from core.contact import classify_contact
from core.types import StrikeEvent


def _event(*, guard=True, confidence=True, wrist_x=200):
    defender = np.asarray([
        [200, 100], [195, 98], [205, 98], [190, 102], [210, 102],
        [185, 140], [215, 140], [180, 160], [220, 160],
        [200, 100] if guard else [150, 190],
        [200, 100] if guard else [250, 190],
        [190, 220], [210, 220], [190, 270], [210, 270],
        [190, 320], [210, 320],
    ], dtype=np.float32)
    attacker = defender.copy()
    attacker[10] = [wrist_x, 100]
    samples = []
    for frame in (1, 2, 3):
        sample = {
            "frame": frame, "time": frame / 10,
            "attacker_keypoints": attacker.tolist(),
            "opponent_keypoints": defender.tolist(),
            "opponent_box": [160, 80, 300, 330],
        }
        if confidence:
            sample.update(attacker_conf=[.99] * 17, opponent_conf=[.99] * 17)
        samples.append(sample)
    return StrikeEvent(
        "A", "B", 1, 1, 2, 3, .1, .2, .3,
        "cross", "punch", "right_hand", confidence=.94,
        evidence={"contact_samples": samples},
    )


def _assert_uncertain(event):
    result = classify_contact(event)
    assert result.outcome == "uncertain"
    assert not result.landed
    assert result.contact_confidence == 0


def test_unobserved_defender_is_not_a_verified_block():
    event = _event()
    for sample in event.evidence["contact_samples"]:
        sample["opponent_conf"] = [0.0] * 17
    _assert_uncertain(event)


@pytest.mark.parametrize("value", [0.0, .1, float("nan"), float("inf")])
def test_unreliable_attacking_endpoint_cannot_establish_contact(value):
    event = _event()
    for sample in event.evidence["contact_samples"]:
        sample["attacker_conf"][10] = value
    _assert_uncertain(event)


@pytest.mark.parametrize("missing_wrists", [(9, 10), (9,)])
def test_unobserved_guard_does_not_turn_into_a_clean_hit(missing_wrists):
    event = _event(guard=False)
    for sample in event.evidence["contact_samples"]:
        for wrist in missing_wrists:
            sample["opponent_conf"][wrist] = .1
    _assert_uncertain(event)


def test_unreliable_target_cannot_be_reported_as_a_miss():
    event = _event()
    for sample in event.evidence["contact_samples"]:
        sample["opponent_conf"][:5] = [0.0] * 5
    _assert_uncertain(event)


@pytest.mark.parametrize("guard,wrist_x,outcome", [(True, 200, "blocked"), (False, 200, "clean"), (False, 80, "missed")])
@pytest.mark.parametrize("confidence", [False, True])
def test_observed_outcomes_and_legacy_no_confidence_are_preserved(guard, wrist_x, outcome, confidence):
    event = _event(guard=guard, confidence=confidence, wrist_x=wrist_x)
    result = classify_contact(event)
    assert result.outcome == outcome
    assert result.landed == (outcome == "clean")


def test_rejected_trajectory_does_not_fall_back_to_unfiltered_peak():
    event = _event()
    peak = copy.deepcopy(event.evidence["contact_samples"][1])
    event.evidence.update(
        peak_attacker_keypoints=peak["attacker_keypoints"],
        peak_opponent_keypoints=peak["opponent_keypoints"],
        peak_opponent_box=peak["opponent_box"],
    )
    for sample in event.evidence["contact_samples"]:
        sample["opponent_conf"] = [0.0] * 17
    _assert_uncertain(event)


@pytest.mark.parametrize("side,index", [("attacker_keypoints", 10), ("opponent_keypoints", 0)])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_contact_points_do_not_establish_contact(side, index, value):
    event = _event()
    for sample in event.evidence["contact_samples"]:
        if side == "opponent_keypoints":
            for head in range(5):
                sample[side][head] = [value, 100]
        else:
            sample[side][index] = [value, 100]
    _assert_uncertain(event)
