"""Movement scored against the who-is-who answer keys (tools/movement_benchmark.py, 2026-11-25)."""

from tools import movement_benchmark as mb
from tools.scoreboard import movement_verdict


def _record(frame, a, b):
    return {"source_frame": frame, "fighter_A": {"observation": {"box": a} if a else None},
            "fighter_B": {"observation": {"box": b} if b else None}}


TRUTH = {"frames": [
    {"time_seconds": 0.0, "source_frame": 0, "phase": "standing", "A": [100, 100, 200, 400], "B": [500, 100, 600, 400]},
    # A walks 150 px toward B (half their height): a clear move.
    {"time_seconds": 4.0, "source_frame": 100, "phase": "standing", "A": [250, 100, 350, 400], "B": [500, 100, 600, 400]},
    {"time_seconds": 8.0, "source_frame": 200, "phase": "ground", "pair": [200, 200, 600, 400]},
]}


def test_a_track_on_the_marks_is_right_in_place_and_direction():
    records = [_record(f["source_frame"], f.get("A"), f.get("B")) for f in TRUTH["frames"][:2]]
    result = mb.score(records, TRUTH)
    assert result["place"] == {"right": 4, "frames": 4}            # the ground frame is not scored
    assert result["direction"] == {"right": 1, "moves": 1}         # only A clearly moved


def test_a_fighter_tracked_going_the_wrong_way_or_missing_is_wrong():
    records = [_record(0, [100, 100, 200, 400], [500, 100, 600, 400]),
               _record(100, [20, 100, 120, 400], None)]            # A tracked backing off; B lost
    result = mb.score(records, TRUTH)
    assert result["place"] == {"right": 2, "frames": 4}
    assert result["direction"] == {"right": 0, "moves": 1}


def test_a_swap_puts_both_fighters_in_the_wrong_place():
    records = [_record(0, [500, 100, 600, 400], [100, 100, 200, 400])]
    assert mb.score(records, {"frames": TRUTH["frames"][:1]})["place"] == {"right": 0, "frames": 2}


def test_the_scoreboard_row_takes_the_weaker_of_the_two():
    row = movement_verdict({"right": 96, "frames": 100}, {"right": 9, "moves": 10})
    assert row["score"] == 0.9 and row["passes"] is False
    assert movement_verdict({"right": 0, "frames": 0}, {"right": 0, "moves": 0})["score"] is None


def test_frames_outside_the_analysed_span_are_not_scored():
    """The phone clips are analysed for 55 s and marked for 2 minutes: counting
    the rest as misses read 63% where 92% of analysed frames were right."""
    records = [_record(0, [100, 100, 200, 400], [500, 100, 600, 400])]        # nothing near frame 100
    result = mb.score(records, TRUTH)
    assert result["place"] == {"right": 2, "frames": 2}
    assert result["direction"] == {"right": 0, "moves": 0}
