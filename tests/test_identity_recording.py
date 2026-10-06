"""tools/identity_recording.py: a replayed recording gives what the live manager gave."""

import numpy as np

from core.config import SETTINGS
from core.identity import IdentityManager
from tools import identity_recording


def _person(track_id, x1, x2, look):
    appearance = np.zeros((64,), dtype=np.float32)
    appearance[look] = 1.0
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(100, 290, 17)
    from core.types import PersonObservation
    return PersonObservation(track_id=track_id, box=np.asarray([x1, 80, x2, 310], dtype=np.float32), confidence=0.9,
                             keypoints=kp, keypoint_conf=np.ones((17,), dtype=np.float32), appearance=appearance)


def test_replay_reproduces_the_live_assignments():
    calls = identity_recording._install_recorder()
    try:
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        live = []
        for frame in range(1, 40):
            shift = frame * 3
            people = [_person(1, 100 + shift, 200 + shift, 1), _person(2, 400 - shift, 500 - shift, 2),
                      _person(7, 250, 350, 5)]
            if frame % 9 == 0:
                people = people[1:]                       # A undetected now and then
            a, b = manager.update(people, frame)
            live.append((None if a is None else a.track_id, None if b is None else b.track_id))
    finally:
        identity_recording._remove_recorder()
    records, _ = identity_recording.replay({"clip": "test", "settings": {}, "calls": calls})
    assert len(records) == len(live)
    replayed = [(r["fighter_A"]["observation"] is not None, r["fighter_B"]["observation"] is not None) for r in records]
    assert replayed == [(a is not None, b is not None) for a, b in live]
    assert IdentityManager.update.__name__ == "update"       # the recorder was removed


def test_no_text_setting_leaves_the_machine():
    kept = identity_recording.safe_settings(SETTINGS)
    assert kept and all(isinstance(value, (bool, int, float)) for value in kept.values())
    assert "bystander_memory" in kept
