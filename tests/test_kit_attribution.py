from types import SimpleNamespace

import numpy as np

from core.kit_attribution import KitAttribution, shorts_dark_share


def _strike(fighter, seconds):
    return SimpleNamespace(fighter=fighter, peak_time=seconds, technique="jab")


def _watch(selection_seconds, frames):
    """frames: (seconds, shot, A's dark share, B's dark share)."""
    watch = KitAttribution(selection_seconds)
    for seconds, shot, a, b in frames:
        watch.add(seconds, shot, a, b)
    return watch


def _steady(start, end, a, b, shot=0, step=0.2):
    return [(round(start + i * step, 3), shot, a, b) for i in range(int((end - start) / step))]


def test_shadow_mode_never_changes_the_events():
    frames = _steady(0, 10, 0.05, 0.8) + _steady(10, 12, 0.8, 0.05, shot=1) + _steady(12, 20, 0.05, 0.8, shot=2)
    watch = _watch(2.0, frames)
    events = [_strike("A", 11.0)]
    result = watch.proposals(events)
    assert result["mode"] == "shadow"
    assert events[0].fighter == "A"


def test_flags_a_strike_thrown_while_the_boxes_were_swapped():
    # B wears the dark shorts. For two seconds after a cut, A's box is on the
    # dark-shorted fighter: a strike credited to A then belongs to B.
    frames = _steady(0, 10, 0.05, 0.8) + _steady(10, 12, 0.8, 0.05, shot=1) + _steady(12, 20, 0.05, 0.8, shot=2)
    result = _watch(2.0, frames).proposals([_strike("A", 5.0), _strike("A", 11.0), _strike("B", 15.0)])
    assert result["active"] is True
    assert result["darker_shorts_at_selection"] == "B" == result["darker_shorts_by_majority"]
    assert [(f["seconds"], f["credited"], f["proposed"]) for f in result["would_recredit"]] == [(11.0, "A", "B")]


def test_silent_when_the_selection_reading_and_the_majority_disagree():
    # Measured on a real round: hands on the hips at selection read the wrong
    # way round. Acting on it would re-credit correct strikes.
    frames = _steady(0, 4, 0.8, 0.05) + _steady(4, 30, 0.05, 0.8)
    result = _watch(2.0, frames).proposals([_strike("A", 10.0)])
    assert result["active"] is False
    assert result["would_recredit"] == []


def test_silent_when_both_fighters_wear_similar_shorts():
    frames = _steady(0, 20, 0.7, 0.75)
    result = _watch(2.0, frames).proposals([_strike("A", 10.0)])
    assert result["active"] is False


def test_a_camera_cut_bounds_the_frames_that_judge_a_strike():
    # The strike sits right after a cut; the swapped frames before the cut
    # must not judge it.
    frames = _steady(0, 9.9, 0.05, 0.8) + _steady(9.9, 10.0, 0.8, 0.05, shot=1, step=0.05) \
        + _steady(10.0, 20, 0.05, 0.8, shot=2)
    result = _watch(2.0, frames).proposals([_strike("A", 10.1)])
    assert result["would_recredit"] == []


def test_no_selection_time_means_no_proposals():
    result = _watch(None, _steady(0, 20, 0.05, 0.8)).proposals([_strike("A", 5.0)])
    assert result["active"] is False


def test_dark_share_reads_the_shorts_band_from_the_pose():
    frame = np.full((200, 200, 3), 230, np.uint8)          # light background
    frame[100:140, 80:120] = 10                              # black shorts
    keypoints = np.zeros((17, 2), np.float32)
    keypoints[11], keypoints[12] = (85, 100), (115, 100)     # hips
    keypoints[13], keypoints[14] = (85, 180), (115, 180)     # knees
    conf = np.full(17, 0.9, np.float32)
    assert shorts_dark_share(frame, keypoints, conf) > 0.9
    frame[100:140, 80:120] = 220                             # white shorts
    assert shorts_dark_share(frame, keypoints, conf) < 0.1
    conf[13] = 0.1                                           # knee not seen
    assert shorts_dark_share(frame, keypoints, conf) is None
