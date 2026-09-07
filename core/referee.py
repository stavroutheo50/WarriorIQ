"""Recognising the official, so he is never mistaken for a fighter.

The identity guards are all comparative: they ask whether this candidate looks
or moves enough like the fighter we committed to. The referee defeats every one
of them honestly. He is on the mat, he is the right size, he is lit the same,
and he moves like an athlete because he is one - he circles the exchange, steps
in, breaks a clinch. There is no threshold on similarity or motion that admits
a tiring fighter and refuses him, which is why two earlier attempts failed:

  - the hue/saturation histogram scores him 0.54-0.67 against the anchor and the
    fighters 0.54-0.79, ranges that overlap almost entirely;
  - a learned embedding looked decisive on fifteen sampled crops and dissolved
    over a full round into one smooth unimodal peak from 0.68 to 0.79, with no
    valley to put a threshold in. See core/reid.py.

So the question has to change from "how similar is he" to "what is he", and be
answered from the one thing that is categorical rather than continuous: he is
in uniform and the fighters are not. On all three tournament recordings the
official wears a bright, near-white shirt over dark trousers, while the athletes
wear saturated singlets and shorts with bare legs.

That uniform is not a coincidence of these three recordings, which is worth
knowing before trusting the feature anywhere else. WAKO Referee Rules, Article
14 "Referee's clothes", requires a "White official IF WAKO shirt with short
sleeves" and "Black trousers without turn-ups" - checked against the document
published at wako.sport/rules-overview on 2026-09-07. So pale-over-dark is the
mandated dress for every WAKO event, and the two measured fractions below are
reading the rule rather than a local habit.

All five federations' rulebooks were read on 2026-09-07 to find where else
that holds. It is a shorter list than it looks:

  * **WAKO** - white shirt, black trousers. Referee Rules, Article 14.
  * **World Boxing** - "White, short-sleeved, button-down collared shirt,
    tucked into trousers" and "Black trousers (no denim)". Competition Rules
    in force November 2024, Rule 6.4.1. The same pale-over-dark structure,
    stated just as plainly, so this feature should transfer to boxing - though
    it has not been tested on boxing footage, and 6.4.4 lets an event mandate
    different attire.
  * **IFMA muaythai** - "dark blue or black trousers, black flat shoes ...
    IFMA Official's shirt" (Rules & Regulations v3.057, 21.12). The trousers
    are specified and **the shirt colour is not**: it is a branded garment the
    document never gives a colour for. Half the feature is confirmed and half
    is unknown.
  * **World Taekwondo** - "the referees and judges shall wear the uniform
    designated by the WT" (Competition Rules, 3.6.1). No colour at all.
  * **MMA** - the ABC Unified Rules say nothing about official dress; it is
    left to each commission.

So the rule is documented for two federations, contradicted by none, and
simply absent for three. Do not read that absence as agreement.

That is four numbers, not five hundred. The dimension is the point. With around
a hundred and forty hand-labelled crops, a probe over a 512-d embedding reaches
0.96 on the frames it trained on and 0.57 on later frames of the same round -
below the 0.69 you get by answering "fighter" every time. It memorises. A 4-d
physically-motivated feature is small enough that the labels available can
actually constrain it.

Measured, on 58 hand-checked crops from one bout: AUC 1.00, 98.3% under
six-fold cross-validation, and the scores are bimodal rather than merely
ordered - 652 of 928 boxes below 0.2 and 206 above 0.8. On two bouts it has
never seen, from different events, the most confident detections are all the
official; the crops that land near the boundary are seated table officials in
light shirts, which is where an uncertain answer belongs.

What it found is worse than the sampled frames had suggested. Of the 411 boxes
the analysis previously accepted as fighter A, 241 - fifty-nine per cent - were
the referee. A's entire output was mostly a measurement of the official.
Switching the filter on takes that to one per cent.

It is on by default. Enabling it alone was not enough and briefly looked like
a regression: fighter B lost 356 frames of the real fighter, because the
official had been acting as a sink for A's homeless slot. Two separate faults
in the identity manager were hiding behind him, both now fixed - an unassigned
fighter scored 0.0, so following the wrong person always beat admitting a gap;
and releasing a fighter from a seated spectator left the positional anchor on
the chair, so the real fighter was then refused as an implausible jump for the
rest of the round. With those repaired, on the reference bout: frames spent on
the referee fall from 242 to 6, while frames spent on somebody who is not the
referee are 686 against 671 - the same amount of real tracking, without the
contamination.

A third fault sat behind the official too: a seated spectator could be
acquired and then held for six seconds, because the stationary guard cannot
judge a track until it has watched one that long. Measured over 1.5 s instead,
seated people reach 0.006 to 0.021 body lengths of spread and the least mobile
fighter 0.059, so a much stricter threshold over a much shorter look settles
the motionless cases early and leaves the patient reading to the marginal ones.

Note the coupling. The anchor repair on its own makes matters worse, not
better: it recovers more readily, and with nothing to refuse the official it
recovers onto him, taking referee frames from 242 up to 298. These two changes
are only correct together.

Safe to leave on across sports because of the stand-down below. In a discipline
whose competitors wear white - a karate gi being the obvious case - both
fighters are seeded on people who score as officials, both anchors are marked,
and the filter switches itself off for that bout rather than refusing everyone.

The limit this buys, stated plainly: this is a per-federation model. It
recognises one specific uniform - the WAKO one quoted above, which World Boxing
also mandates - and a promotion whose officials wear black, which is most MMA
and is not ruled out by any of the five rulebooks, needs its own labels. It
refuses to load rather than guess when its weights are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from core.config import SETTINGS

LOGGER = logging.getLogger("warrioriq.referee")

_probe: dict | None = None
_unavailable = False


def _room_brightness(frame: np.ndarray) -> float:
    """Median V of the whole frame, as a reference for "bright" and "dark".

    Absolute brightness is useless here: the same white shirt reads 90 in a
    corner and 190 under the lights. Every threshold below is expressed
    relative to this instead. Subsampled because the median only needs to be
    approximately right and this runs per frame.
    """
    small = frame[::4, ::4]
    return float(np.median(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]))


def uniform_feature(frame: np.ndarray, box, room: float | None = None) -> np.ndarray:
    """Four numbers describing shirt-over-trousers, read from box fractions.

    Fractions of the box rather than keypoints, because this has to work on the
    low-confidence detections where the pose is least trustworthy - which is
    exactly where identity goes wrong.

    The first two are *fractions of pixels* passing a test, not means. That
    matters more than it sounds: at these box sizes the shirt region is about
    eighteen pixels tall and always contains some mat, hair or opponent, and a
    mean is dragged around by them. Measured on 58 hand-checked crops, the mean
    version separates referee from everyone else with AUC 0.70 and the fraction
    version with AUC 1.00.
    """
    blank = np.zeros(4, dtype=np.float32)
    if frame is None or box is None:
        return blank
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Trim the sides: at these box sizes the edges are mostly mat and opponent.
    cx1, cx2 = int(max(0, x1 + 0.25 * bw)), int(min(w, x2 - 0.25 * bw))
    upper = frame[int(max(0, y1 + 0.18 * bh)):int(min(h, y1 + 0.48 * bh)), cx1:cx2]
    lower = frame[int(max(0, y1 + 0.58 * bh)):int(min(h, y1 + 0.92 * bh)), cx1:cx2]
    if upper.size == 0 or lower.size == 0:
        return blank
    if room is None:
        room = _room_brightness(frame)
    up = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lo = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    return np.array([
        ((up[:, 1] < 70) & (up[:, 2] > room)).mean(),   # shirt pale and brighter than the room
        (lo[:, 2] < room * 0.85).mean(),                # trousers darker than the room
        np.median(up[:, 2]) / max(1.0, np.median(lo[:, 2])),
        np.median(up[:, 1]),                            # a singlet is saturated, a shirt is not
    ], dtype=np.float32)


def _load() -> dict | None:
    """Read the trained weights once; stay silent and off if they are missing."""
    global _probe, _unavailable
    if _unavailable or not SETTINGS.referee_filter_enabled:
        return None
    if _probe is None:
        path = Path(SETTINGS.referee_probe_path)
        if not path.exists():
            _unavailable = True
            LOGGER.warning("referee_probe_missing path=%s filter disabled", path)
            return None
        try:
            data = np.load(path)
            _probe = {k: np.asarray(data[k], dtype=np.float32) for k in ("w", "b", "mu", "sd")}
        except Exception as exc:                                    # noqa: BLE001
            _unavailable = True
            LOGGER.warning("referee_probe_unreadable error=%s", type(exc).__name__)
            return None
    return _probe


def referee_probability(frame: np.ndarray, box) -> float | None:
    """How strongly this crop looks like an official, or None for no opinion.

    None rather than 0.0 so the caller can tell "not a referee" apart from
    "this check did not run", and never reads a disabled filter as evidence.
    """
    scores = referee_probabilities(frame, [box])
    return scores[0] if scores else None


def referee_probabilities(frame: np.ndarray, boxes) -> list[float | None]:
    """Score a whole frame's detections at once.

    Only the room-brightness reference is shared, but that reads the entire
    frame, so doing it once per frame rather than once per person is the
    difference between a negligible cost and a per-detection one.
    """
    if frame is None or boxes is None or len(boxes) == 0:
        return []
    probe = _load()
    if probe is None:
        return [None] * len(boxes)
    room = _room_brightness(frame)
    out: list[float | None] = []
    for box in boxes:
        feature = uniform_feature(frame, box, room=room)
        if not feature.any():
            out.append(None)
            continue
        z = (feature - probe["mu"]) / probe["sd"]
        out.append(float(1.0 / (1.0 + np.exp(-(float(z @ probe["w"]) + float(probe["b"]))))))
    return out


def reset_for_tests() -> None:
    global _probe, _unavailable
    _probe = None
    _unavailable = False
