from types import SimpleNamespace

import numpy as np

from core.kit_attribution import (KitAttribution, box_shorts_dark_share, darker_at_selection_boxes,
                                  shorts_dark_share)


def _strike(fighter, seconds):
    return SimpleNamespace(fighter=fighter, opponent="B" if fighter == "A" else "A",
                           peak_time=seconds, technique="jab", evidence={})


def _watch(selection_seconds, frames, selection_darker=None):
    """frames: (seconds, shot, A's dark share, B's dark share)."""
    watch = KitAttribution(selection_seconds, selection_darker)
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


def test_applied_it_recredits_the_flagged_strike_and_nothing_else():
    frames = _steady(0, 10, 0.05, 0.8) + _steady(10, 12, 0.8, 0.05, shot=1) + _steady(12, 20, 0.05, 0.8, shot=2)
    events = [_strike("A", 5.0), _strike("A", 11.0), _strike("B", 15.0)]
    result = _watch(2.0, frames, selection_darker="B").proposals(events, apply=True)
    assert result["mode"] == "active" and result["recredited"] == 1
    assert [(e.fighter, e.opponent) for e in events] == [("A", "B"), ("B", "A"), ("B", "A")]
    assert events[1].evidence["recredited_by_shorts"] is True
    assert "recredited_by_shorts" not in events[0].evidence


def test_the_drawn_boxes_decide_even_when_the_tracked_boxes_were_swapped():
    # Measured on a rerun of UFC 326 round 1: the tracker had the boxes the
    # wrong way round around the selection frame and on most of the round, so
    # the old rule read the shorts backwards and would have moved 18 correct
    # strikes. The person's own boxes do not have that problem.
    frames = _steady(0, 4, 0.8, 0.05) + _steady(4, 30, 0.8, 0.05) + _steady(30, 32, 0.05, 0.8)
    events = [_strike("A", 10.0), _strike("A", 31.0)]
    result = _watch(2.0, frames, selection_darker="B").proposals(events, apply=True)
    assert result["selection_reading"] == "drawn_boxes"
    assert [e.fighter for e in events] == ["B", "A"]


def test_selection_boxes_need_one_dark_and_one_light_pair_of_shorts():
    assert darker_at_selection_boxes(0.05, 0.73) == "B"
    assert darker_at_selection_boxes(0.63, 0.0) == "A"
    assert darker_at_selection_boxes(0.62, 0.97) is None      # both dark (amateur bout)
    assert darker_at_selection_boxes(0.25, 0.48) is None      # too close (UFC round 3)
    assert darker_at_selection_boxes(None, 0.8) is None


def test_box_band_reads_the_shorts_of_an_upright_fighter():
    frame = np.full((200, 200, 3), 230, np.uint8)
    frame[100:140, 40:160] = 10                               # dark band at hip height
    assert box_shorts_dark_share(frame, [40, 0, 160, 200]) > 0.9
    frame[100:140, 40:160] = 220
    assert box_shorts_dark_share(frame, [40, 0, 160, 200]) < 0.1
    assert box_shorts_dark_share(None, [40, 0, 160, 200]) is None
