import numpy as np

from core.identity import IdentityManager
from core.referee import looks_like_a_clothed_official, torso_skin_share
from core.types import PersonObservation

UPRIGHT = [10.0, 10.0, 40.0, 90.0]      # 30 wide, 80 tall
LYING = [10.0, 10.0, 90.0, 40.0]        # on the canvas


def _person(track_id, torso_skin, box=UPRIGHT):
    return PersonObservation(track_id=track_id, box=np.asarray(box, dtype=np.float32),
                             confidence=0.8, referee_prob=0.01, torso_skin=torso_skin)


def test_bare_fighters_turn_the_filter_on_and_an_upright_clothed_person_is_refused():
    manager = IdentityManager(_person(1, 0.8), _person(2, 0.7), 0, source_fps=30.0)
    assert manager.clothed_official_filter is True
    before = manager.rejections.get("clothed_official", 0)
    manager._score(manager.a, _person(7, 0.05))
    assert manager.rejections.get("clothed_official", 0) == before + 1


def test_a_bare_fighter_is_not_refused():
    manager = IdentityManager(_person(1, 0.8), _person(2, 0.7), 0, source_fps=30.0)
    manager._score(manager.a, _person(1, 0.6))
    assert manager.rejections.get("clothed_official", 0) == 0


def test_a_fighter_on_the_ground_is_not_refused_even_when_the_crop_shows_little_skin():
    # Measured: tangled fighters on the canvas scored 0.13-0.26.
    manager = IdentityManager(_person(1, 0.8), _person(2, 0.7), 0, source_fps=30.0)
    manager._score(manager.a, _person(1, 0.13, box=LYING))
    assert manager.rejections.get("clothed_official", 0) == 0


def test_stands_down_when_a_selected_fighter_is_clothed():
    # Singlets, sports bras, doboks, gis: the fighters themselves would read
    # as clothed, so the rule must not run at all.
    manager = IdentityManager(_person(1, 0.8), _person(2, 0.2), 0, source_fps=30.0)
    assert manager.clothed_official_filter is False
    manager._score(manager.a, _person(7, 0.05))
    assert manager.rejections.get("clothed_official", 0) == 0


def test_stands_down_when_skin_was_not_measured():
    manager = IdentityManager(_person(1, None), _person(2, 0.7), 0, source_fps=30.0)
    assert manager.clothed_official_filter is False


def test_upright_and_covered_is_what_counts():
    assert looks_like_a_clothed_official(_person(7, 0.05)) is True
    assert looks_like_a_clothed_official(_person(7, 0.05, box=LYING)) is False
    assert looks_like_a_clothed_official(_person(7, 0.6)) is False
    assert looks_like_a_clothed_official(_person(7, None)) is False


def test_torso_skin_share_reads_skin_and_not_a_black_shirt():
    skin_bgr = (120, 150, 200)          # a typical skin tone in BGR
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :] = skin_bgr
    assert torso_skin_share(frame, [20, 0, 80, 100]) > 0.9
    frame[:, :] = (15, 15, 15)          # black shirt
    assert torso_skin_share(frame, [20, 0, 80, 100]) < 0.05
    assert torso_skin_share(None, [20, 0, 80, 100]) is None


def test_the_selection_frame_decides_not_the_start_frame():
    # Measured: the analysis started where A's box had missed onto somebody in
    # a shirt, so the filter judged from the start frame never switched on.
    manager = IdentityManager(_person(1, 0.0), _person(2, 0.7), 0, source_fps=30.0)
    assert manager.clothed_official_filter is False
    manager.decide_clothed_official_filter(0.63, 0.85)      # where the fighters were picked
    assert manager.clothed_official_filter is True
    manager.decide_clothed_official_filter(0.63, 0.2)       # a fighter in a top
    assert manager.clothed_official_filter is False
    manager.decide_clothed_official_filter(None, 0.85)
    assert manager.clothed_official_filter is False
