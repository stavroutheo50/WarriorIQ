"""Where the two fighters crossed, and which crossings to ask the person about.

Owner, 2026-11-13: in fights where both fighters wear the same kit and gear,
the tracker mixed them up "every time they cross paths", and the score and
strike counts followed the wrong person. No colour can settle that, so the
analysis now records each crossing - the two fighters close enough to be
confused, then apart again - and the few it was least sure about are put to
the person as "tap your fighter" on a still taken just after it. Each answer
becomes an identity anchor (core/types.py IdentityAnchor) for the next run of
the analysis, which re-locks A and B at that moment (IdentityManager
.apply_person_assignment).

Pure bookkeeping on boxes and confidences: no image, no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Centres closer than this many body heights: one fighter in front of the
# other, close enough to be confused. Measured on sparring clips (Pexels
# 4761806, Mixkit 40954/40957): two fighters at striking range stand 0.55-0.8
# body heights apart, and dip to 0.13-0.3 when they cross or clinch.
TOGETHER_BODY_HEIGHTS = 0.35
# And apart again once back at striking range (a gap between the two
# thresholds, so a pair hovering at the line is not counted as many crossings).
APART_BODY_HEIGHTS = 0.55
# A touch shorter than this is not a crossing worth asking about.
MIN_TOGETHER_SECONDS = 0.25
# The question is asked on a frame this long after they separate, when both
# are clear of each other again.
SETTLE_SECONDS = 0.4
# At most this many questions per fight, at least this far apart.
MAX_QUESTIONS = 6
MIN_GAP_SECONDS = 8.0


def _centre_gap_in_heights(a_box, b_box) -> float:
    a_height = float(a_box[3] - a_box[1])
    b_height = float(b_box[3] - b_box[1])
    height = max(1.0, (a_height + b_height) / 2.0)
    dx = (a_box[0] + a_box[2] - b_box[0] - b_box[2]) / 2.0
    dy = (a_box[1] + a_box[3] - b_box[1] - b_box[3]) / 2.0
    return float((dx * dx + dy * dy) ** 0.5) / height


@dataclass
class Crossing:
    entered_seconds: float
    left_seconds: float
    # The frame the question is asked on, and the boxes A and B had there.
    ask_seconds: float = 0.0
    ask_frame: int | None = None
    a_box: list[float] | None = None
    b_box: list[float] | None = None
    # The lowest identity confidence either fighter had while they were together.
    lowest_confidence: float = 1.0

    def as_dict(self) -> dict:
        return {"entered_seconds": round(self.entered_seconds, 3), "left_seconds": round(self.left_seconds, 3),
                "ask_seconds": round(self.ask_seconds, 3), "ask_frame": self.ask_frame,
                "a_box": self.a_box, "b_box": self.b_box,
                "lowest_confidence": round(self.lowest_confidence, 3)}


@dataclass
class CrossingWatch:
    """Fed every analysed frame; collects the crossings."""

    crossings: list[Crossing] = field(default_factory=list)
    _together_since: float | None = None
    _lowest: float = 1.0
    _pending: Crossing | None = None

    def observe(self, seconds: float, source_frame: int, a_box, b_box,
                a_confidence: float = 1.0, b_confidence: float = 1.0) -> None:
        if self._pending is not None and seconds >= self._pending.left_seconds + SETTLE_SECONDS:
            if a_box is not None and b_box is not None and _centre_gap_in_heights(a_box, b_box) > APART_BODY_HEIGHTS:
                self._pending.ask_seconds = float(seconds)
                self._pending.ask_frame = int(source_frame)
                self._pending.a_box = [round(float(v), 1) for v in a_box]
                self._pending.b_box = [round(float(v), 1) for v in b_box]
                self.crossings.append(self._pending)
                self._pending = None
        if a_box is None or b_box is None:
            return
        gap = _centre_gap_in_heights(a_box, b_box)
        confidence = min(float(a_confidence), float(b_confidence))
        if self._together_since is None:
            if gap < TOGETHER_BODY_HEIGHTS:
                self._together_since = float(seconds)
                self._lowest = confidence
                self._pending = None                # a new crossing supersedes an unsettled one
            return
        self._lowest = min(self._lowest, confidence)
        if gap > APART_BODY_HEIGHTS:
            if seconds - self._together_since >= MIN_TOGETHER_SECONDS:
                self._pending = Crossing(entered_seconds=self._together_since, left_seconds=float(seconds),
                                         lowest_confidence=self._lowest)
            self._together_since = None

    def questions(self, limit: int = MAX_QUESTIONS, min_gap: float = MIN_GAP_SECONDS,
                  answered_frames: tuple[int, ...] = (), near_frames: int = 0) -> list[dict]:
        """The crossings worth asking about: least certain first, spread out, in time order.

        A crossing within `near_frames` of a frame the person already answered
        about is not asked again: the answer pinned who is who there.
        """
        chosen: list[Crossing] = []
        for crossing in sorted(self.crossings, key=lambda c: (c.lowest_confidence, -(c.left_seconds - c.entered_seconds))):
            if crossing.ask_frame is None:
                continue
            if any(abs(crossing.ask_frame - int(frame)) <= near_frames for frame in answered_frames):
                continue
            if all(abs(crossing.ask_seconds - other.ask_seconds) >= min_gap for other in chosen):
                chosen.append(crossing)
            if len(chosen) >= limit:
                break
        return [c.as_dict() for c in sorted(chosen, key=lambda c: c.ask_seconds)]

    def summary(self, answered_frames: tuple[int, ...] = (), near_frames: int = 0) -> dict:
        return {"count": len(self.crossings),
                "questions": self.questions(answered_frames=answered_frames, near_frames=near_frames)}


def anchors_from_answers(questions: list[dict], answers: dict) -> list[dict]:
    """Identity anchors from the person's answers.

    `answers` maps a question's ask_frame (as text) to "A" when the box the
    analysis called A was their fighter, or "B" when it was the other one -
    the analysis had them swapped there. Each answer pins A and B at that
    frame; a confirmed one is kept too, so a later run cannot drift there.
    """
    anchors = []
    for question in questions:
        answer = str(answers.get(str(question.get("ask_frame")), "")).upper()
        if answer not in {"A", "B"} or question.get("a_box") is None or question.get("b_box") is None:
            continue
        a_box, b_box = question["a_box"], question["b_box"]
        if answer == "B":
            a_box, b_box = b_box, a_box
        anchors.append({"seconds": float(question["ask_seconds"]), "frame": int(question["ask_frame"]),
                        "a_box": list(a_box), "b_box": list(b_box), "swapped": answer == "B"})
    return sorted(anchors, key=lambda anchor: anchor["frame"])
