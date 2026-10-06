"""The identity manager remembers who stands beside the fighters (SETTINGS.bystander_memory)."""

import numpy as np
import pytest

from core.config import SETTINGS
from core.identity import IdentityManager
from core.types import PersonObservation


def _person(track_id, x1, x2, appearance):
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(100, 290, 17)
    return PersonObservation(track_id=track_id, box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
                             confidence=0.95, keypoints=kp, keypoint_conf=np.ones((17,), dtype=np.float32),
                             appearance=np.asarray(appearance, dtype=np.float32))


def _colour(*weights):
    vector = np.zeros((64,), dtype=np.float32)
    for index, weight in weights:
        vector[index] = weight
    return vector


A_LOOK, B_LOOK = _colour((1, 1.0)), _colour((2, 1.0))
# The referee: dark like both fighters, close to neither.
REF_LOOK = _colour((1, 0.6), (2, 0.6), (3, 0.5))


@pytest.fixture
def memory_on():
    before = SETTINGS.bystander_memory
    object.__setattr__(SETTINGS, "bystander_memory", True)
    yield
    object.__setattr__(SETTINGS, "bystander_memory", before)


def _bout(manager, frames, referee_x=300, start=1):
    for frame in range(start, start + frames):
        manager.update([_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK),
                        _person(9, referee_x, referee_x + 100, REF_LOOK)], frame)


def test_a_third_person_beside_both_held_fighters_is_learnt(memory_on):
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, SETTINGS.bystander_min_sightings)
    assert manager._bystanders[9]["sightings"] == SETTINGS.bystander_min_sightings


def test_somebody_far_from_the_fighters_is_not_learnt(memory_on):
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, 20, referee_x=1400)
    assert 9 not in manager._bystanders


def test_nothing_is_learnt_with_the_setting_off():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, 20)
    assert manager._bystanders == {}


def test_a_known_bystander_does_not_take_a_lost_fighters_box(memory_on):
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, SETTINGS.bystander_min_sightings + 2)
    # A's detection is gone and the referee steps into A's place.
    a, b = manager.update([_person(9, 110, 210, REF_LOOK), _person(2, 400, 500, B_LOOK)], 100)
    assert a is None
    assert b is not None and b.track_id == 2
    assert manager.rejections.get("looks_like_a_known_bystander", 0) >= 1


def test_a_fighter_who_inherits_the_bystanders_track_is_still_followed(memory_on):
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, SETTINGS.bystander_min_sightings + 2)
    # After an occlusion the tracker hands track 9 to fighter A himself.
    a, _ = manager.update([_person(9, 110, 210, A_LOOK), _person(2, 400, 500, B_LOOK)], 100)
    assert a is not None and a.track_id == 9


def test_the_fighter_mask_does_not_reopen_a_known_bystander(memory_on):
    """A fighter hidden behind the referee leaves a mask the referee overlaps."""
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    _bout(manager, SETTINGS.bystander_min_sightings + 2)
    referee = _person(9, 110, 210, REF_LOOK)
    a, _ = manager.update([referee, _person(2, 400, 500, B_LOOK)], 100,
                          sam_guidance={"A": referee.box.copy(), "B": None})
    assert a is None


def test_somebody_who_looks_like_a_fighter_is_never_learnt(memory_on):
    """If A's box is on the wrong man, the real A beside him must not be learnt."""
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0)
    manager.a.identity_confidence = manager.b.identity_confidence = 1.0
    wrong_a = _person(1, 100, 200, REF_LOOK)                  # A's box drifted onto the referee
    real_a = _person(5, 210, 310, A_LOOK)
    b = _person(2, 400, 500, B_LOOK)
    for _ in range(SETTINGS.bystander_min_sightings + 5):
        manager._learn_bystanders([wrong_a, real_a, b], wrong_a, b)
    assert 5 not in manager._bystanders
