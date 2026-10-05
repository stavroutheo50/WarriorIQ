"""A fighter does not shrink to a fraction of their size between sightings.

Measured on a handheld pankration bout (identity_pankration ma640), most
frames where a fighter's box sat on someone else, that someone was a
spectator or the next mat's bout behind them - a fifth of the fighter's area.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np

from core import identity
from core.identity import IdentityManager
from core.types import PersonObservation


def _person(track_id, x1, x2, appearance_index, top=80, bottom=310):
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 2, x2 - 2, 17)
    kp[:, 1] = np.linspace(top + 10, bottom - 10, 17)
    appearance = np.zeros((64,), dtype=np.float32)
    appearance[appearance_index] = 1.0
    return PersonObservation(track_id=track_id, box=np.asarray([x1, top, x2, bottom], dtype=np.float32),
                             confidence=0.95, keypoints=kp, keypoint_conf=np.ones((17,), dtype=np.float32),
                             appearance=appearance)


def _settings(on):
    return mock.patch.object(identity, "SETTINGS", SimpleNamespace(**{
        **{name: getattr(identity.SETTINGS, name) for name in identity.SETTINGS.__dataclass_fields__},
        "identity_size_gate": on}))


def _scene(on):
    """Five normal sightings, then A's fighter is missed and a small person
    with A's look stands just behind where A was."""
    with _settings(on):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        for frame in range(1, 7):
            manager.update([_person(1, 100 + frame, 200 + frame, 1), _person(2, 400, 500, 2)], frame)
        small = _person(1, 130, 150, 1, top=150, bottom=210)          # same track id, a fifth the area
        return manager.update([small, _person(2, 400, 500, 2)], 7)


def test_with_the_gate_a_small_stranger_is_not_taken_for_the_fighter():
    a, b = _scene(on=True)
    assert a is None                                               # missing beats the wrong person
    assert b is not None


def test_without_the_gate_the_old_behaviour_is_unchanged():
    a, _ = _scene(on=False)
    assert a is not None


def test_a_crouch_is_not_refused():
    # Shorter but wider: about two thirds of the area.
    with _settings(True):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        for frame in range(1, 7):
            manager.update([_person(1, 100, 200, 1), _person(2, 400, 500, 2)], frame)
        a, _ = manager.update([_person(1, 90, 220, 1, top=170, bottom=310), _person(2, 400, 500, 2)], 7)
    assert a is not None
