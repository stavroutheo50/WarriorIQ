"""Crossings and the person's answers about them (owner, 2026-11-13).

Two fighters in the same kit were mixed up "every time they cross paths".
The analysis records each crossing, asks about the least certain few, and a
later run re-locks A and B where the person answered.
"""

from __future__ import annotations

import numpy as np

from core.crossings import CrossingWatch, anchors_from_answers
from core.identity import IdentityManager
from core.types import PersonObservation

FPS = 25.0


def _box(x: float) -> list[float]:
    return [x, 100.0, x + 60.0, 300.0]           # 200 px tall


def _walk(watch: CrossingWatch, a_xs, b_xs, start_frame=0, confidence=1.0):
    for step, (ax, bx) in enumerate(zip(a_xs, b_xs)):
        frame = start_frame + step
        watch.observe(frame / FPS, frame, _box(ax), _box(bx), confidence, confidence)


def test_a_crossing_is_recorded_with_a_settled_frame_to_ask_on():
    watch = CrossingWatch()
    # Apart, then together for a second, then apart and settled.
    _walk(watch, [100] * 10 + [300] * 25 + [100] * 30, [600] * 10 + [330] * 25 + [600] * 30)
    assert len(watch.crossings) == 1
    crossing = watch.crossings[0]
    assert abs(crossing.entered_seconds - 10 / FPS) < 1e-6 and abs(crossing.left_seconds - 35 / FPS) < 1e-6
    assert crossing.ask_frame is not None and crossing.ask_seconds >= crossing.left_seconds + 0.4
    assert crossing.a_box == [100.0, 100.0, 160.0, 300.0] and crossing.b_box == [600.0, 100.0, 660.0, 300.0]


def test_a_brush_shorter_than_a_quarter_second_is_not_a_crossing():
    watch = CrossingWatch()
    _walk(watch, [100] * 10 + [300] * 3 + [100] * 30, [600] * 10 + [330] * 3 + [600] * 30)
    assert watch.crossings == []


def test_questions_are_the_least_certain_spread_out_and_capped():
    watch = CrossingWatch()
    frame = 0
    for index in range(12):
        confidence = 0.2 if index in (3, 7) else 0.9
        _walk(watch, [100] * 10 + [300] * 25 + [100] * 30, [600] * 10 + [330] * 25 + [600] * 30,
              start_frame=frame, confidence=confidence)
        frame += 65
    questions = watch.questions(limit=4, min_gap=3.0)
    assert len(questions) == 4
    assert [q["ask_seconds"] for q in questions] == sorted(q["ask_seconds"] for q in questions)
    lowest = {round(c.ask_seconds, 3) for c in watch.crossings if c.lowest_confidence < 0.5}
    assert lowest <= {q["ask_seconds"] for q in questions}


def test_answers_become_anchors_and_a_swap_flips_the_boxes():
    questions = [{"ask_seconds": 4.0, "ask_frame": 100, "a_box": _box(100), "b_box": _box(600)},
                 {"ask_seconds": 9.0, "ask_frame": 225, "a_box": _box(120), "b_box": _box(500)},
                 {"ask_seconds": 14.0, "ask_frame": 350, "a_box": _box(140), "b_box": _box(400)}]
    anchors = anchors_from_answers(questions, {"100": "A", "225": "B"})
    assert [a["frame"] for a in anchors] == [100, 225]
    assert anchors[0]["a_box"] == _box(100) and not anchors[0]["swapped"]
    assert anchors[1]["a_box"] == _box(500) and anchors[1]["b_box"] == _box(120) and anchors[1]["swapped"]


def _person(track_id, x):
    box = np.asarray(_box(x), dtype=np.float32)
    return PersonObservation(track_id=track_id, box=box, confidence=0.9,
                             appearance=np.ones(48, dtype=np.float32) / 48)


def test_an_answer_re_locks_a_and_b_on_the_people_named():
    manager = IdentityManager(_person(1, 100), _person(2, 600), source_frame=0, source_fps=FPS)
    # Later the tracker has A on the person at 600 - the person says that is B.
    people = [_person(7, 590), _person(8, 110)]
    a, b = manager.apply_person_assignment(people, _box(100), _box(600), source_frame=50)
    assert a is not None and b is not None
    assert manager.a.current_track_id == 8 and manager.b.current_track_id == 7
    assert manager.a.identity_confidence == 1.0


def test_an_answer_that_lands_on_one_person_changes_nothing():
    manager = IdentityManager(_person(1, 100), _person(2, 600), source_frame=0, source_fps=FPS)
    people = [_person(7, 100)]
    assert manager.apply_person_assignment(people, _box(100), _box(105), source_frame=50) == (None, None)
    assert manager.a.current_track_id == 1 and manager.b.current_track_id == 2


def test_a_crossing_already_answered_is_not_asked_again():
    watch = CrossingWatch()
    _walk(watch, [100] * 10 + [300] * 25 + [100] * 30, [600] * 10 + [330] * 25 + [600] * 30)
    frame = watch.crossings[0].ask_frame
    assert watch.questions(min_gap=0.0)
    assert watch.questions(min_gap=0.0, answered_frames=(frame + 5,), near_frames=10) == []


def _apart(watch: CrossingWatch, frames: int, start_frame: int = 0):
    for step in range(frames):
        frame = start_frame + step
        watch.observe(frame / FPS, frame, _box(100), _box(600))


def test_one_question_after_each_ground_spell_asked_first():
    # Owner, 2026-11-16: identity was lost most on the ground. The MMA ground
    # model's seconds say when; the question goes on the first settled frame after.
    watch = CrossingWatch()
    _apart(watch, 30 * 25)
    assert watch.apply_ground([5, 6, 7, 8, 10, 11, 20, 21, 22]) == 2
    asked = sorted(watch.crossings, key=lambda c: c.ask_seconds)
    assert [c.reason for c in asked] == ["ground", "ground"]
    assert 12.4 <= asked[0].ask_seconds < 12.5 and 23.4 <= asked[1].ask_seconds < 23.5
    assert all(c.lowest_confidence == 0.0 for c in asked)
    assert [q["reason"] for q in watch.questions(limit=2, min_gap=1.0)] == ["ground", "ground"]


def test_no_question_is_asked_inside_a_ground_spell():
    watch = CrossingWatch()
    frame = 0
    for _ in range(6):                                         # a crossing every 2.6 s
        _walk(watch, [100] * 10 + [300] * 25 + [100] * 30, [600] * 10 + [330] * 25 + [600] * 30,
              start_frame=frame)
        frame += 65
    before = [c.ask_seconds for c in watch.crossings]
    watch.apply_ground(range(4, 12))
    after = [c for c in watch.crossings if c.reason == "crossed"]
    assert len(after) < len(before)
    assert not any(4 <= c.ask_seconds < 12 for c in after)


def test_a_short_spell_or_no_ground_changes_nothing():
    watch = CrossingWatch()
    _apart(watch, 25 * 10)
    assert watch.apply_ground([3, 4]) == 0 and watch.apply_ground([]) == 0
    assert watch.crossings == []


def test_after_the_ground_waits_until_they_are_apart():
    watch = CrossingWatch()
    for step in range(25 * 20):
        seconds = step / FPS
        b = 330 if seconds < 14 else 600                       # still close until 14 s
        watch.observe(seconds, step, _box(300 if seconds < 14 else 100), _box(b))
    watch.apply_ground(range(5, 10))
    ground = [c for c in watch.crossings if c.reason == "ground"]
    assert len(ground) == 1 and ground[0].ask_seconds >= 14


def test_a_two_second_hole_in_the_ground_seconds_does_not_end_the_spell():
    # Ma268, 2026-11-16: the model said ground at 155 and 158 s but not 156-157;
    # ending the spell at 155 asked while they were still down.
    watch = CrossingWatch()
    _apart(watch, 25 * 30)
    watch.apply_ground([5, 6, 7, 8, 11, 12])
    ground = [c for c in watch.crossings if c.reason == "ground"]
    assert len(ground) == 1 and ground[0].ask_seconds >= 13.4
