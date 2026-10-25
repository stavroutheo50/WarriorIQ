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
