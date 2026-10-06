"""tools/show_identity_misses.py picks out exactly the benchmark's "other person" frames."""

import numpy as np

from tools.show_identity_misses import misses, sheet

A, B, REF = [0, 0, 100, 200], [300, 0, 400, 200], [150, 0, 250, 200]


def _record(frame, a, b):
    return {"source_frame": frame,
            "fighter_A": {"observation": None if a is None else {"box": a}},
            "fighter_B": {"observation": None if b is None else {"box": b}}}


def test_only_somebody_else_counts_not_swaps_or_gaps():
    truth = {"frames": [
        {"source_frame": 10, "time_seconds": 1, "phase": "standing", "A": A, "B": B},   # right
        {"source_frame": 20, "time_seconds": 2, "phase": "standing", "A": A, "B": B},   # A on the referee
        {"source_frame": 30, "time_seconds": 3, "phase": "standing", "A": A, "B": B},   # swapped
        {"source_frame": 40, "time_seconds": 4, "phase": "standing", "A": A, "B": B},   # B missing
        {"source_frame": 50, "time_seconds": 5, "phase": "ground", "pair": A},
    ]}
    records = [_record(10, A, B), _record(20, REF, B), _record(30, B, A), _record(40, A, None), _record(50, REF, REF)]
    found = misses(records, truth)
    assert [(m["fighter"], m["frame"]["source_frame"]) for m in found] == [("A", 20)]
    assert found[0]["followed"] == REF


def test_the_sheet_is_a_grid_of_tiles():
    tiles = [np.zeros((300, 480, 3), np.uint8), np.zeros((280, 480, 3), np.uint8)]
    assert sheet(tiles).shape == (300, 480 * 3, 3)
