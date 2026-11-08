"""Hand-offs taken back off the suspicious count by the fighter's own kit (2026-10-23).

A hand-held phone fight with a referee and ~17 people in shot failed "could
not follow you" at 14.9 suspicious hand-offs a minute for B, while the two
fighters' kits were very different (similarity 0.144) and the identity
manager never confused them. core/kit.py HandoffKitCheck.
"""

from __future__ import annotations

import numpy as np

from core.kit import HandoffKitCheck, region_colours

RED, BLUE, BLACK, WHITE, GREY = (40, 40, 200), (200, 60, 40), (20, 20, 20), (235, 235, 235), (128, 128, 128)


def _frame(*people):
    """A grey frame with people painted as (box, top colour, shorts colour)."""
    image = np.full((400, 800, 3), 128, dtype=np.uint8)
    for (x1, y1, x2, y2), top, shorts in people:
        height = y2 - y1
        image[y1:y1 + int(0.48 * height), x1:x2] = top
        image[y1 + int(0.48 * height):y2, x1:x2] = shorts
    return image


A_BOX, B_BOX = (100, 100, 160, 300), (400, 100, 460, 300)


def _check():
    selection = _frame((A_BOX, RED, BLACK), (B_BOX, BLUE, WHITE))
    return HandoffKitCheck({"A": region_colours(selection, A_BOX), "B": region_colours(selection, B_BOX)})


def test_the_fighter_picked_up_anywhere_in_their_own_kit_is_confirmed():
    check = _check()
    later = _frame(((600, 120, 660, 320), BLUE, WHITE))
    assert check.confirms(later, (600, 120, 660, 320), "B")
    assert check.confirmed == {"A": 0, "B": 1}


def test_the_opponent_or_someone_else_is_not():
    check = _check()
    opponent = _frame(((600, 120, 660, 320), RED, BLACK))
    stranger = _frame(((600, 120, 660, 320), GREY, GREY))
    assert not check.confirms(opponent, (600, 120, 660, 320), "B")
    assert not check.confirms(stranger, (600, 120, 660, 320), "B")


def test_matching_one_region_is_not_enough():
    # B's top on someone in A's shorts: every region that tells them apart must agree.
    check = _check()
    half = _frame(((600, 120, 660, 320), BLUE, BLACK))
    assert not check.confirms(half, (600, 120, 660, 320), "B")


def test_kit_alike_or_a_tiny_box_confirms_nothing():
    selection = _frame((A_BOX, BLUE, WHITE), (B_BOX, BLUE, WHITE))
    alike = HandoffKitCheck({"A": region_colours(selection, A_BOX), "B": region_colours(selection, B_BOX)})
    assert not alike.enabled
    assert not alike.confirms(selection, B_BOX, "B")
    check = _check()
    small = _frame(((600, 120, 620, 160), BLUE, WHITE))
    assert not check.confirms(small, (600, 120, 620, 160), "B")
    # 55 px: under the report's "too small to compare" line, still read here.
    smallish = _frame(((600, 120, 620, 175), BLUE, WHITE))
    assert check.confirms(smallish, (600, 120, 620, 175), "B")
    assert not HandoffKitCheck(None).confirms(small, B_BOX, "B")


def test_a_confirmed_hand_off_comes_off_the_suspicious_count():
    from test_identity_swap import _person
    from core.identity import IdentityManager

    manager = IdentityManager(_person(1, 100, 200, 1.0), _person(2, 400, 500, 0.0), 0)
    manager._commit(manager.a, _person(1, 100, 200, 1.0), 1, 1.0)
    manager._commit(manager.a, _person(7, 600, 700, 1.0), 2, 1.0, recovered=True)
    assert manager.suspicious_handoffs["A"] == 1
    assert [name for name, _ in manager.handoffs_to_check] == ["A"]
    manager.confirm_handoff("A")
    manager.confirm_handoff("A")
    assert manager.suspicious_handoffs["A"] == 0


def _scored(manager, state, person):
    before = dict(manager.rejections)
    manager._score(state, person)
    return {k: v - before.get(k, 0) for k, v in manager.rejections.items() if v != before.get(k, 0)}


def test_a_fighter_in_their_own_kit_is_not_refused_as_the_referee():
    """Owner, 2026-11-02: sparring in a T-shirt and track trousers read as the
    referee's uniform (0.68-0.99) and the real fighter was refused hundreds of
    times with no referee present. Their own kit overrules the probe."""
    from test_identity_swap import _person
    from core.identity import IdentityManager

    check = _check()
    manager = IdentityManager(_person(1, 100, 200, 1.0), _person(2, 400, 500, 0.0), 0)
    manager.kit_reference = check
    seen = region_colours(_frame(((600, 120, 660, 320), BLUE, WHITE)), (600, 120, 660, 320))
    fighter = _person(9, 400, 500, 0.0)
    fighter.referee_prob, fighter.kit_colours = 0.9, seen
    assert "referee" not in _scored(manager, manager.b, fighter)
    assert manager.official_kit_overrides["B"] == 1
    # The same probe score on someone in neither fighter's kit is still refused,
    stranger = _person(10, 400, 500, 0.0)
    stranger.referee_prob = 0.9
    stranger.kit_colours = region_colours(_frame(((600, 120, 660, 320), WHITE, BLACK)), (600, 120, 660, 320))
    assert _scored(manager, manager.b, stranger).get("referee") == 1
    # and so is anyone at all when there is no kit reference to check against.
    manager.kit_reference = None
    assert _scored(manager, manager.b, fighter).get("referee") == 1


def test_a_fighter_the_probe_already_reads_high_is_not_refused_as_the_official():
    """S01 (2026-11-08): the picked boxer, grey T-shirt and dark trousers,
    scored 0.25 at selection and 0.26-0.59 afterwards, so his own track was
    classed an official. A candidate must read clearly more official than the
    fighter did at selection to be refused."""
    from test_identity_swap import _person
    from core.config import SETTINGS
    from core.identity import IdentityManager

    picked = _person(1, 100, 200, 1.0)
    picked.referee_prob = 0.25
    manager = IdentityManager(picked, _person(2, 400, 500, 0.0), 0)
    assert manager.a.anchor_referee_prob == 0.25 and not manager.a.anchor_is_referee
    level = 0.25 + SETTINGS.official_margin_over_fighter
    same_man = _person(1, 100, 200, 1.0)
    same_man.referee_prob = 0.55
    manager._track_referee[1] = __import__("collections").deque([0.55] * 4)
    manager.official_tracks.add(1)
    assert not ({"official", "referee"} & set(_scored(manager, manager.a, same_man)))
    # Someone the probe rates clearly above him is still refused.
    official = _person(5, 100, 200, 1.0)
    official.referee_prob = level + 0.05
    assert _scored(manager, manager.a, official).get("referee") == 1


def test_a_fighter_the_probe_reads_as_a_fighter_is_judged_as_before():
    from test_identity_swap import _person
    from core.config import SETTINGS
    from core.identity import IdentityManager

    picked = _person(1, 100, 200, 1.0)
    picked.referee_prob = 0.02
    manager = IdentityManager(picked, _person(2, 400, 500, 0.0), 0)
    assert manager._official_level(manager.a) == SETTINGS.min_referee_probability
    official = _person(5, 100, 200, 1.0)
    official.referee_prob = SETTINGS.min_referee_probability
    assert _scored(manager, manager.a, official).get("referee") == 1
