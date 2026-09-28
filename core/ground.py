"""Moments a fighter went down: knockdowns, slips and takedowns.

Seen from the pose alone, frame by frame: somebody near the fighters whose
torso has tipped over, or whose hips are down at their feet. That is a
"someone is down" signal, not a "fighter A is down" one, on purpose.

Measured on 17 real bouts (MMA, pankration, kickboxing, Muay Thai; about
90 minutes) against 120 seconds judged by eye without seeing the rule's
answer: when this says someone is down it was right 97% of the time, and it
saw about 55% of the seconds in which someone really was down. A knockdown
or takedown lasts several seconds, so most of them are still caught.

Why no fighter is named: the identity tracker keeps both fighters well while
they stand, but on one fully analysed pankration bout it lost one or both
for most of every ground spell, so "who went down" would have been a guess.
The report lists the moments, links each to the replay and lets the owner
say who went down. Nothing here reaches the score.

Tried and rejected: a general person detector's "wider than tall" boxes.
Two fighters in a clinch make one wide box, and fewer than half its flags
were right.
"""

from __future__ import annotations

import math

from core.video import round_at_time

# Joint indices (COCO): shoulders, hips, ankles.
_SHOULDERS, _HIPS, _ANKLES = (5, 6), (11, 12), (15, 16)
_JOINT_CONFIDENCE = 0.4
_PERSON_CONFIDENCE = 0.35
# Torso further than this from upright reads as lying.
_LYING_DEGREES = 55.0
# Hips within this share of the torso length above the feet read as sitting
# or on a knee.
_SITTING_TORSO_SHARE = 0.45
# Only people about the fighters' size and near them: the crowd, the far
# side of the hall and a seated judge at the edge are not fighters.
_MIN_SIZE_SHARE = 0.35
_NEAR_BODY_LENGTHS = 1.5
# And on the same floor: the lowest point of someone who went down stays
# about where a fighter's feet are. Spectators sitting cross-legged behind
# the fighters were the false moments on a real bout; their lowest point is
# well up the picture from the fighters' feet.
_SAME_FLOOR_BODY_LENGTHS = 0.35
# Lying is looser: nobody watches a fight lying down, and a camera that zooms
# in as a fighter goes down puts their lowest point well below where the
# fighter's feet were a moment before.
_LYING_FLOOR_BODY_LENGTHS = 0.6
# Sitting or kneeling counts only for the fighter's own box (the tracked
# fighter is one of the people seen, so the overlap is near total), or for
# someone beside the place a fighter was last seen while the tracker has lost
# them - which is what usually happens when a fighter goes down.
_SAME_PERSON_IOU = 0.5

# A second counts as down when this many analysed frames in it are down;
# one frame alone is too often a pose that collapsed for a frame.
MIN_DOWN_FRAMES_PER_SECOND = 2
# A moment is a new one after this long without anyone down, so a ground
# spell with a few missed seconds stays one moment.
GAP_SECONDS = 6.0
# And needs this many down seconds to be listed at all.
MIN_DOWN_SECONDS = 2

NOTE = ("Found automatically and not checked by a person. On real fights, "
        "when WarriorIQ marks a moment like this someone really was down "
        "almost every time, but it misses some. It cannot yet tell which "
        "fighter went down, so it asks you.")


def _joint(keypoints, conf, index):
    if keypoints is None or conf is None or index >= len(keypoints):
        return None
    if float(conf[index]) <= _JOINT_CONFIDENCE:
        return None
    return float(keypoints[index][0]), float(keypoints[index][1])


def pose_down(keypoints, keypoint_conf) -> str | None:
    """"lying", "sitting", or None when upright or the joints are not visible."""
    shoulders = [_joint(keypoints, keypoint_conf, i) for i in _SHOULDERS]
    hips = [_joint(keypoints, keypoint_conf, i) for i in _HIPS]
    if None in shoulders or None in hips:
        return None
    sx, sy = (shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2
    hx, hy = (hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2
    if hy <= sy:
        return "lying"      # hips above the shoulders: upside down or flat
    if math.degrees(math.atan2(abs(hx - sx), hy - sy)) > _LYING_DEGREES:
        return "lying"
    torso = math.hypot(hx - sx, hy - sy)
    feet = [point[1] for point in (_joint(keypoints, keypoint_conf, i) for i in _ANKLES) if point]
    if feet and max(feet) - hy < _SITTING_TORSO_SHARE * torso:
        return "sitting"
    return None


def _diagonal(box) -> float:
    return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _iou(a, b) -> float:
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    union = ((float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
             + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter)
    return inter / union if union > 0 else 0.0


def _centre(box) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2


class DownWatch:
    """Fed every analysed frame; lists the moments somebody went down."""

    def __init__(self) -> None:
        self._last_boxes: dict[str, list[float]] = {}
        self._seconds: dict[int, dict] = {}

    def observe(self, seconds: float, people, fighter_a, fighter_b) -> None:
        current = {}
        for name, fighter in (("A", fighter_a), ("B", fighter_b)):
            if fighter is not None and getattr(fighter, "box", None) is not None:
                current[name] = self._last_boxes[name] = [float(v) for v in fighter.box]
        second = self._seconds.setdefault(int(seconds), {"frames": 0, "down": 0, "first_down": None})
        second["frames"] += 1
        if self._someone_down(people, current):
            second["down"] += 1
            if second["first_down"] is None:
                second["first_down"] = float(seconds)

    def _someone_down(self, people, current: dict[str, list[float]]) -> bool:
        if not people or not self._last_boxes:
            return False
        biggest = max(_diagonal(p.box) for p in people)
        for person in people:
            if float(getattr(person, "confidence", 0.0)) <= _PERSON_CONFIDENCE:
                continue
            if _diagonal(person.box) < _MIN_SIZE_SHARE * biggest:
                continue
            posture = pose_down(person.keypoints, person.keypoint_conf)
            if posture == "lying":
                if any(self._beside(person.box, box, _LYING_FLOOR_BODY_LENGTHS)
                       for box in self._last_boxes.values()):
                    return True
            elif posture == "sitting":
                # Seated spectators and kneeling officials sit too, often
                # right behind the fighters, so sitting counts only for
                # someone who is where a fighter is: the tracked fighter
                # itself, or beside where the tracker lost one.
                for name, last in self._last_boxes.items():
                    if name in current:
                        if _iou(person.box, current[name]) >= _SAME_PERSON_IOU:
                            return True
                    elif self._beside(person.box, last):
                        return True
        return False

    @staticmethod
    def _beside(box, anchor, floor: float = _SAME_FLOOR_BODY_LENGTHS) -> bool:
        height = max(1.0, float(anchor[3]) - float(anchor[1]))
        (cx, cy), (ax, ay) = _centre(box), _centre(anchor)
        return (math.hypot(cx - ax, cy - ay) <= _NEAR_BODY_LENGTHS * height
                and abs(float(box[3]) - float(anchor[3])) <= floor * height)

    def moments(self, rounds=None) -> list[dict]:
        """Each moment: when it started, which round, how many seconds down.

        With `rounds`, a moment outside every selected round (a break, a
        round not asked for) is dropped, like the strikes there.
        """
        down = sorted(s for s, v in self._seconds.items() if v["down"] >= MIN_DOWN_FRAMES_PER_SECOND)
        spells: list[list[int]] = []
        for second in down:
            if spells and second - spells[-1][-1] <= GAP_SECONDS:
                spells[-1].append(second)
            else:
                spells.append([second])
        out = []
        for spell in spells:
            if len(spell) < MIN_DOWN_SECONDS:
                continue
            start = self._seconds[spell[0]]["first_down"]
            spec = round_at_time(rounds, start) if rounds is not None else None
            if rounds is not None and (spec is None or not spec.selected):
                continue
            out.append({"seconds": round(start, 3), "round": spec.number if spec else None,
                        "down_seconds": len(spell), "last_seconds": float(spell[-1])})
        return out

    def summary(self, rounds=None) -> dict:
        return {"moments": self.moments(rounds), "note": NOTE, "attributed": False}
