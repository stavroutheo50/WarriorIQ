from core.grappling import frame_position, seconds_of, summary, takedowns
from core.official_stats import _clock, official_rounds


def _pose(shoulders_y, hips_y, ankles_y, x=100.0, lying=False):
    keypoints = [[x, 0.0]] * 17
    keypoints = [list(p) for p in keypoints]
    if lying:
        # Flat: shoulders and hips level, side by side.
        keypoints[5], keypoints[6] = [x - 40, hips_y], [x - 40, hips_y + 4]
        keypoints[11], keypoints[12] = [x, hips_y], [x, hips_y + 4]
        keypoints[15], keypoints[16] = [x + 40, hips_y], [x + 40, hips_y + 4]
    else:
        keypoints[5], keypoints[6] = [x - 8, shoulders_y], [x + 8, shoulders_y]
        keypoints[11], keypoints[12] = [x - 6, hips_y], [x + 6, hips_y]
        keypoints[15], keypoints[16] = [x - 6, ankles_y], [x + 6, ankles_y]
    return {"box": [x - 50, shoulders_y - 10, x + 50, max(ankles_y, hips_y) + 10],
            "confidence": 0.9, "keypoints": keypoints, "keypoint_conf": [0.9] * 17}


STANDING_A = _pose(40, 80, 140, x=60)
STANDING_B = _pose(40, 80, 140, x=220)
A_ON_TOP = _pose(50, 80, 96, x=100)            # kneeling over
B_FLAT = _pose(100, 100, 100, x=100, lying=True)
B_ON_TOP = _pose(50, 80, 96, x=100)
A_FLAT = _pose(100, 100, 100, x=100, lying=True)


def _record(t, a, b, rnd=1):
    return {"time_seconds": t, "round_number": rnd, "selected_round": True,
            "fighter_A": {"observation": a}, "fighter_B": {"observation": b}}


def _second(t, a, b, frames=4):
    return [_record(t + i / frames, a, b) for i in range(frames)]


def test_standing_apart_is_not_ground():
    assert frame_position(_record(0, STANDING_A, STANDING_B)) == {"ground": False, "top": None}


def test_fighter_kneeling_over_a_flat_opponent_is_on_top():
    assert frame_position(_record(0, A_ON_TOP, B_FLAT)) == {"ground": True, "top": "A"}
    assert frame_position(_record(0, A_FLAT, B_ON_TOP)) == {"ground": True, "top": "B"}


def test_unseen_fighter_or_hidden_hips_say_nothing():
    assert frame_position(_record(0, A_ON_TOP, None)) is None
    hidden = dict(B_FLAT, keypoint_conf=[0.1] * 17)
    assert frame_position(_record(0, A_ON_TOP, hidden)) is None


def test_one_frame_does_not_decide_a_second():
    records = [_record(5.0, A_ON_TOP, B_FLAT)]
    assert seconds_of(records) == {}


def test_frames_outside_selected_rounds_are_ignored():
    records = [dict(r, selected_round=False) for r in _second(5, A_ON_TOP, B_FLAT)]
    assert seconds_of(records) == {}


def test_control_time_and_takedown_are_credited_to_the_top_fighter():
    records = []
    for t in range(0, 10):
        records += _second(t, STANDING_A, STANDING_B)
    for t in range(10, 30):
        records += _second(t, A_FLAT, B_ON_TOP)
    line = summary(records)[1]
    assert line["ground_seconds"] == 20
    assert line["control_seconds"] == {"A": 0, "B": 20}
    assert line["takedowns"] == {"A": 0, "B": 1}


def test_a_long_ground_spell_is_one_takedown():
    records = []
    for t in range(0, 10):
        records += _second(t, STANDING_A, STANDING_B)
    for t in range(10, 120):
        records += _second(t, A_FLAT, B_ON_TOP)
    assert len(takedowns(seconds_of(records))) == 1


def test_going_to_ground_without_standing_first_is_not_a_takedown():
    records = []
    for t in range(0, 20):
        records += _second(t, A_FLAT, B_ON_TOP)
    assert takedowns(seconds_of(records)) == []


def test_official_control_time_is_read_in_seconds():
    assert _clock("4:30") == 270
    assert _clock("0:00") == 0
    assert _clock("---") == 0
    row = {"EVENT": "E", "BOUT": "X vs. Y", "ROUND": "Round 2", "FIGHTER": "X", "KD": "0",
           "SIG.STR.": "5 of 9", "TOTAL STR.": "6 of 10", "TD": "1 of 3", "CTRL": "2:54",
           "HEAD": "3 of 7", "BODY": "1 of 1", "LEG": "1 of 1",
           "DISTANCE": "4 of 8", "CLINCH": "1 of 1", "GROUND": "0 of 0"}
    line = official_rounds([row], bout="X vs. Y", fighter="X")[2]
    assert line.control_seconds == 174
    assert line.takedowns.landed == 1
    assert line.by_position["distance"].attempted == 8


def test_strikes_count_only_in_seconds_both_fighters_stand():
    from core.grappling import standing_seconds, strikes_while_standing
    records = _second(3, STANDING_A, STANDING_B) + _second(8, A_FLAT, B_ON_TOP)
    assert standing_seconds(records) == {3}
    events = [{"peak_time": 3.4, "fighter": "A"}, {"peak_time": 8.2, "fighter": "B"},
              {"peak_time": 12.0, "fighter": "A"}]
    assert strikes_while_standing(events, records) == [events[0]]


def test_one_fighter_unseen_is_not_standing():
    from core.grappling import standing_seconds
    assert standing_seconds(_second(3, STANDING_A, None)) == set()
