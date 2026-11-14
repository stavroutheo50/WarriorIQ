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
