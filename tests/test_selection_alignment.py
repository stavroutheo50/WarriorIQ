"""A failed identity check at the chosen moment says so (owner, 2026-11-13).

When every other identity check passed but, arriving at the moment the person
picked, the two followed fighters were not on the boxes they drew, the report
fell through to the vague "could not stay certain" with no cause. It now names
what happened and what to do.

(A frame-matching step that tried to move browser-captured picks onto the
analysis's own frame was tried the same day and removed: on real phone footage
the right frame was not reliably distinguishable from its neighbours, and on a
still broadcast shot it moved a pick to a different moment.)
"""

from __future__ import annotations

from core.report import identity_failure


def test_a_failed_check_at_the_chosen_moment_says_so():
    tracking = {"fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9, "identity_seed_confirmed": False}
    failure = identity_failure(tracking)
    assert failure is not None and failure["cause"] == "seed_moved" and failure["repick"] is True
    assert "where you tapped them" in failure["headline"]
