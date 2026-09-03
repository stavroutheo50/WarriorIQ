from __future__ import annotations

import cv2
from collections import deque

import numpy as np

from core.config import SETTINGS
from core.types import FighterState, PersonObservation


def box_area(box) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def normalized_distance(a, b) -> float:
    if a is None or b is None:
        return 999.0
    ca, cb = box_center(a), box_center(b)
    d = float(np.linalg.norm(ca - cb))
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    scale = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
    return d / scale


def size_similarity(a, b) -> float:
    aa, bb = box_area(a), box_area(b)
    if aa <= 0 or bb <= 0:
        return 0.0
    return min(aa, bb) / max(aa, bb)


def appearance_hist(frame: np.ndarray, box) -> np.ndarray | None:
    """Compact clothing/appearance descriptor.

    We intentionally use torso-biased HSV histograms because they are cheap
    enough for every analyzed frame. The identity manager treats this only as
    evidence, never as an absolute identity decision.
    """
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Bias toward upper/mid body and avoid gloves/shorts dominating.
    tx1 = int(max(0, x1 + 0.15 * bw))
    tx2 = int(min(w, x2 - 0.15 * bw))
    ty1 = int(max(0, y1 + 0.15 * bh))
    ty2 = int(min(h, y1 + 0.65 * bh))
    if tx2 - tx1 < 8 or ty2 - ty1 < 8:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.reshape(-1).astype(np.float32)


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size != b.size:
        return 0.5
    return float(max(0.0, min(1.0, cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL) * 0.5 + 0.5)))


def pose_signature(keypoints: np.ndarray | None, box) -> np.ndarray | None:
    if keypoints is None or box is None or len(keypoints) < 17:
        return None
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    sig = np.zeros((17, 2), dtype=np.float32)
    sig[:, 0] = (keypoints[:17, 0] - x1) / bw
    sig[:, 1] = (keypoints[:17, 1] - y1) / bh
    return sig.reshape(-1)


def pose_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.5
    diff = float(np.mean(np.abs(a - b)))
    return float(max(0.0, min(1.0, 1.0 - diff)))


def ema(old: np.ndarray | None, new: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if new is None:
        return old
    if old is None or old.shape != new.shape:
        return new.copy()
    return (alpha * old + (1.0 - alpha) * new).astype(np.float32)


class IdentityManager:
    """Owns stable WarriorIQ identities A/B independent of tracker IDs.

    A BoT-SORT ID is treated as temporary evidence. When it disappears or
    becomes implausible, candidates are scored using motion, IoU, size,
    appearance and pose. Ambiguous recovery returns None rather than assigning
    a referee/coach. This implements the project's rule: missing briefly is
    better than tracking the wrong human.
    """

    def __init__(self, initial_a: PersonObservation, initial_b: PersonObservation, source_frame: int = 0, source_fps: float = 30.0):
        self.a = FighterState(
            name="A",
            current_track_id=initial_a.track_id,
            last_box=initial_a.box.copy(),
            last_keypoints=None if initial_a.keypoints is None else initial_a.keypoints.copy(),
            appearance=None if initial_a.appearance is None else initial_a.appearance.copy(),
            pose_signature=pose_signature(initial_a.keypoints, initial_a.box),
            anchor_appearance=None if initial_a.appearance is None else initial_a.appearance.copy(),
            anchor_pose=pose_signature(initial_a.keypoints, initial_a.box),
            identity_confidence=1.0,
            last_seen_source_frame=source_frame,
        )
        self.b = FighterState(
            name="B",
            current_track_id=initial_b.track_id,
            last_box=initial_b.box.copy(),
            last_keypoints=None if initial_b.keypoints is None else initial_b.keypoints.copy(),
            appearance=None if initial_b.appearance is None else initial_b.appearance.copy(),
            pose_signature=pose_signature(initial_b.keypoints, initial_b.box),
            anchor_appearance=None if initial_b.appearance is None else initial_b.appearance.copy(),
            anchor_pose=pose_signature(initial_b.keypoints, initial_b.box),
            identity_confidence=1.0,
            last_seen_source_frame=source_frame,
        )
        # Recent path of every tracked person, so a candidate can be asked the
        # one question that separates a fighter from the people around the mat:
        # did you move? Measured across every fight so far, fighters cover 24 to
        # 65 body lengths a minute; a man standing at the mat edge covered 9.6.
        self._track_history: dict[int, deque] = {}
        self.source_fps = max(1.0, float(source_fps))

    def _remember_positions(self, people: list[PersonObservation], source_frame: int) -> None:
        for person in people:
            if person.track_id is None or person.box is None:
                continue
            history = self._track_history.setdefault(int(person.track_id), deque(maxlen=450))
            box = person.box
            history.append((
                source_frame,
                float((box[0] + box[2]) / 2.0),
                float((box[1] + box[3]) / 2.0),
                max(1.0, float(box[3] - box[1])),
            ))

    def _recent_travel(self, track_id: int | None, fps: float) -> float | None:
        """Body lengths a minute over this track's recent history.

        None when there is not enough of a look to judge - a track seen for a
        moment must never be refused for standing still.
        """
        if track_id is None or fps <= 0:
            return None
        history = self._track_history.get(int(track_id))
        if not history or len(history) < 20:
            return None
        span_frames = history[-1][0] - history[0][0]
        if span_frames <= 0:
            return None
        seconds = span_frames / fps
        if seconds < 3.0:
            return None
        distance = sum(
            float(np.hypot(b[1] - a[1], b[2] - a[2]))
            for a, b in zip(history, list(history)[1:])
        )
        body = sorted(item[3] for item in history)[len(history) // 2]
        return distance / body / (seconds / 60.0)

    @staticmethod
    def _by_track_id(people: list[PersonObservation], track_id: int | None) -> PersonObservation | None:
        if track_id is None:
            return None
        for person in people:
            if person.track_id == track_id:
                return person
        return None

    @staticmethod
    def _predicted_box(state: FighterState) -> np.ndarray | None:
        if state.last_box is None:
            return None
        if state.velocity is None:
            return state.last_box
        predicted = state.last_box.copy()
        predicted[[0, 2]] += float(state.velocity[0])
        predicted[[1, 3]] += float(state.velocity[1])
        return predicted

    def _score(self, state: FighterState, candidate: PersonObservation, keep_id_bonus: bool = True) -> float:
        reference = self._predicted_box(state)
        distance = normalized_distance(reference, candidate.box)
        if state.last_box is not None and distance > SETTINGS.max_normalized_jump:
            return -999.0
        iou = box_iou(reference, candidate.box)
        position = 1.0 / (1.0 + distance)
        size = size_similarity(state.last_box, candidate.box)
        appearance = appearance_similarity(state.appearance, candidate.appearance)
        candidate_sig = candidate.pose_signature if candidate.pose_signature is not None else pose_signature(candidate.keypoints, candidate.box)
        pose = pose_similarity(state.pose_signature, candidate_sig)
        det = float(candidate.confidence)
        # Similarity to the originally selected fighter, which never updates.
        # Without this every term above is relative to the previous frame, so a
        # gradual slide onto another person in the ring is invisible: each step
        # looks like a small, plausible move.
        anchor = appearance_similarity(state.anchor_appearance, candidate.appearance)
        anchor_pose_match = pose_similarity(state.anchor_pose, candidate_sig)
        score = (
            0.18 * iou
            + 0.22 * position
            + 0.10 * size
            + 0.14 * appearance
            + 0.10 * pose
            + 0.06 * det
            + 0.14 * anchor
            + 0.06 * anchor_pose_match
        )
        # A candidate that matches where the fighter should be but looks nothing
        # like the fighter the user picked is the exact shape of a referee walking
        # through. Refusing is correct here: this manager's rule is that missing
        # briefly beats tracking the wrong human.
        if state.anchor_appearance is not None and candidate.appearance is not None:
            if anchor < SETTINGS.min_anchor_appearance_similarity:
                state.switches_rejected += 1
                return -999.0
        # Refuse to move onto somebody who has been standing still. Coaches,
        # the referee and the officials at the table all sit near the action
        # and all look plausible for a frame; what none of them do is cover
        # ground. Fighters measured 24 to 65 body lengths a minute across every
        # bout so far, a man at the mat edge 9.6.
        #
        # One-directional on purpose. This can block a switch to a new track; it
        # can never drop the track already being followed, so a fighter who
        # pauses between exchanges is never given away.
        #
        # Applied during a recovery too, which it was not before.
        #
        # Skipping it there was meant to let a fighter who paused between
        # exchanges be re-acquired, and it did - along with everyone sitting
        # down. Drawing the tracked boxes back onto a bout showed fighter A
        # lost at 25 seconds and locked onto a spectator in a chair for the
        # remaining two and a half minutes, while coverage reported 96%,
        # because coverage counts accepted observations and cannot tell a
        # fighter from a seated man where a fighter used to be. Every number
        # published for that fighter described the spectator.
        #
        # The reason the gate was added no longer holds: it was compensating
        # for a six-second history window, since widened to thirty, so a brief
        # pause no longer erases the travel that proves somebody is fighting.
        # Same bout with this applied during recovery: 1 of 8 sampled frames
        # correct becomes 5 of 8, and attempts rise from 9 and 8 to 47 and 42.
        # Coverage falls from 96% to 81% because it now declines to follow
        # furniture, which is this manager's own rule - missing briefly beats
        # tracking the wrong human.
        if (
            candidate.track_id is not None
            and candidate.track_id != state.current_track_id
        ):
            travel = self._recent_travel(candidate.track_id, self.source_fps)
            if travel is not None and travel < SETTINGS.min_switch_travel_per_minute:
                state.switches_rejected += 1
                return -999.0
        if keep_id_bonus and candidate.track_id is not None and candidate.track_id == state.current_track_id:
            score += SETTINGS.track_id_bonus
        return float(score)

    def _commit(self, state: FighterState, obs: PersonObservation, source_frame: int, score: float, recovered: bool = False) -> PersonObservation:
        old_center = box_center(state.last_box) if state.last_box is not None else None
        new_center = box_center(obs.box)
        if old_center is not None:
            delta = new_center - old_center
            state.velocity = delta if state.velocity is None else (0.65 * state.velocity + 0.35 * delta).astype(np.float32)
        state.prev_box = None if state.last_box is None else state.last_box.copy()
        state.last_box = obs.box.copy()
        state.last_keypoints = None if obs.keypoints is None else obs.keypoints.copy()
        state.appearance = ema(state.appearance, obs.appearance, SETTINGS.appearance_ema)
        sig = obs.pose_signature if obs.pose_signature is not None else pose_signature(obs.keypoints, obs.box)
        state.pose_signature = ema(state.pose_signature, sig, SETTINGS.pose_ema)
        if recovered and obs.track_id != state.current_track_id:
            state.recovery_count += 1
        state.current_track_id = obs.track_id
        state.identity_confidence = float(max(0.0, min(1.0, score)))
        state.missing_frames = 0
        state.last_seen_source_frame = source_frame
        return obs

    def _recover(self, state: FighterState, people: list[PersonObservation], forbidden_id: int | None) -> tuple[PersonObservation | None, float]:
        scored: list[tuple[float, PersonObservation]] = []
        for candidate in people:
            if forbidden_id is not None and candidate.track_id == forbidden_id:
                continue
            score = self._score(state, candidate, keep_id_bonus=False)
            if score > -100:
                scored.append((score, candidate))
        if not scored:
            return None, 0.0
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else -999.0
        if best_score < SETTINGS.min_reid_score:
            return None, best_score
        if len(scored) > 1 and best_score - second < SETTINGS.min_reid_margin:
            return None, best_score
        return best, best_score

    def _update_one(self, state: FighterState, people: list[PersonObservation], source_frame: int, forbidden_id: int | None) -> PersonObservation | None:
        current = self._by_track_id(people, state.current_track_id)
        if current is not None and current.track_id != forbidden_id:
            score = self._score(state, current, keep_id_bonus=True)
            # A tracker ID can still jump to another human. Keep only if its
            # combined evidence remains plausible.
            if score >= 0.46:
                return self._commit(state, current, source_frame, score, recovered=False)
            state.switches_rejected += 1

        recovered, score = self._recover(state, people, forbidden_id)
        if recovered is not None:
            return self._commit(state, recovered, source_frame, score, recovered=True)

        state.missing_frames += 1
        state.identity_confidence = 0.0
        return None

    def update(
        self,
        people: list[PersonObservation],
        source_frame: int,
        sam_guidance: dict[str, np.ndarray | None] | None = None,
    ) -> tuple[PersonObservation | None, PersonObservation | None]:
        """Resolve A and B together so tracker-ID swaps cannot swap fighters.

        BoT-SORT IDs are useful continuity hints, but combatants cross and
        occlude frequently. Joint assignment prioritizes the selected clothing
        appearance plus motion and rejects an ambiguous A/B permutation.
        """
        self._remember_positions(people, source_frame)
        if not people:
            self.a.missing_frames += 1
            self.b.missing_frames += 1
            self.a.identity_confidence = self.b.identity_confidence = 0.0
            return None, None

        def candidate_score(state: FighterState, obs: PersonObservation) -> float:
            base = self._score(state, obs, keep_id_bonus=False)
            guide = None if sam_guidance is None else sam_guidance.get(state.name)
            if base < -100:
                # A long disappearance can make the motion gate reject the
                # correct reappearance. Strong agreement with the propagated
                # fighter mask is allowed to reopen that gate.
                if guide is None or box_iou(guide, obs.box) < 0.18:
                    return base
                base = 0.36 + 0.28 * box_iou(guide, obs.box)
            appearance = appearance_similarity(state.appearance, obs.appearance)
            # An ID receives only a small bonus when clothing evidence agrees.
            if obs.track_id is not None and obs.track_id == state.current_track_id and appearance >= 0.68:
                base += 0.08
            score = base + 0.20 * (appearance - 0.5)
            if guide is not None:
                overlap = box_iou(guide, obs.box)
                distance = normalized_distance(guide, obs.box)
                # A propagated mask is strong independent evidence, but the
                # detector observation is still required for pose and events.
                score += 0.60 * overlap + 0.18 / (1.0 + distance)
                if overlap < 0.02 and distance > 0.75:
                    score -= 0.55
            return score

        scores_a = [candidate_score(self.a, p) for p in people]
        scores_b = [candidate_score(self.b, p) for p in people]
        assignments: list[tuple[float, int | None, int | None]] = []
        for ai in [None, *range(len(people))]:
            for bi in [None, *range(len(people))]:
                if ai is not None and bi is not None and ai == bi:
                    continue
                sa = 0.0 if ai is None else scores_a[ai]
                sb = 0.0 if bi is None else scores_b[bi]
                if ai is not None and sa < SETTINGS.min_reid_score:
                    continue
                if bi is not None and sb < SETTINGS.min_reid_score:
                    continue
                assignments.append((sa + sb, ai, bi))
        assignments.sort(reverse=True, key=lambda x: x[0])
        _, ai, bi = assignments[0] if assignments else (0.0, None, None)

        # If the opposite permutation is almost equally plausible, do not
        # guess. Missing evidence is safer than silently exchanging A and B.
        if ai is not None and bi is not None:
            swapped = scores_a[bi] + scores_b[ai]
            chosen = scores_a[ai] + scores_b[bi]
            if chosen - swapped < 0.06:
                ai = bi = None

        a_obs = self._commit(self.a, people[ai], source_frame, scores_a[ai], recovered=people[ai].track_id != self.a.current_track_id) if ai is not None else None
        b_obs = self._commit(self.b, people[bi], source_frame, scores_b[bi], recovered=people[bi].track_id != self.b.current_track_id) if bi is not None else None
        for state, obs in ((self.a, a_obs), (self.b, b_obs)):
            if obs is None:
                state.missing_frames += 1
                state.identity_confidence = 0.0
        return a_obs, b_obs

    def needs_recovery(self, state: FighterState, analyzed_index: int) -> bool:
        return (
            SETTINGS.sam_recovery_enabled
            and state.missing_frames >= SETTINGS.missing_before_recovery
            and analyzed_index - state.last_sam_attempt_analyzed_frame >= SETTINGS.sam_cooldown_analyzed_frames
            and state.last_box is not None
        )

    def apply_ai_assignment(self, people: list[PersonObservation], a_index: int, b_index: int, source_frame: int, confidence: float) -> tuple[PersonObservation | None, PersonObservation | None]:
        """Commit a high-confidence joint assignment from the visual referee."""
        if a_index == b_index or not (0 <= a_index < len(people)) or not (0 <= b_index < len(people)):
            return None, None
        a_candidate, b_candidate = people[a_index], people[b_index]
        # The external decision may resolve a crossing, but candidates must
        # still be physically plausible relative to the last known tracks.
        if self._score(self.a, a_candidate, keep_id_bonus=False) < 0.36:
            return None, None
        if self._score(self.b, b_candidate, keep_id_bonus=False) < 0.36:
            return None, None
        a = self._commit(self.a, a_candidate, source_frame, confidence, recovered=True)
        b = self._commit(self.b, b_candidate, source_frame, confidence, recovered=True)
        return a, b

    def apply_external_recovery(self, state: FighterState, recovered_box, people: list[PersonObservation], source_frame: int) -> PersonObservation | None:
        if recovered_box is None:
            return None
        # Match a current detector observation to the externally recovered box.
        best, best_score = None, -999.0
        for person in people:
            overlap = box_iou(recovered_box, person.box)
            dist = normalized_distance(recovered_box, person.box)
            score = 0.72 * overlap + 0.28 * (1.0 / (1.0 + dist))
            if score > best_score:
                best, best_score = person, score
        if best is None or best_score < 0.42:
            return None
        state.sam_recovery_count += 1
        return self._commit(state, best, source_frame, max(0.55, best_score), recovered=True)
