"""Second-opinion keypoints for the two fighters, from a top-down pose model.

Why this exists, measured on real 480x220 tournament footage rather than
argued. The detector and the keypoint head are one fused model: it regresses
joints from whole-frame features, so a fighter who is small, blurred or in an
unusual configuration collapses toward the prior. On frames 450 and 2056 the
resulting skeleton has the torso squeezed to a sliver and both legs converging
on a single point - while reporting ankle confidence of 0.93 to 0.97. It is not
uncertain, it is confidently wrong, which is why no threshold downstream can
catch it.

RTMPose is top-down. It receives the fighter already cropped and rescaled to
its full 192x256 input, so a person forty pixels tall in the source arrives at
a workable size. On the same two frames it returns shoulders spread across the
torso, both arms with bent elbows, and both legs resolved separately to
distinct ankles.

It outputs seventeen COCO keypoints in the COCO order, which is the same
contract core/metrics.py, core/action.py, core/contact.py and core/defense.py
already read. Nothing downstream needs to know which backend produced them.

The cost is real and linear: 15.7 ms per person, with no batching benefit - the
library loops internally. That is why only the two committed fighters are
refined and never every detection in the frame; refining all four people on a
typical frame costs 63 ms rather than 31 ms and buys nothing, because the
referee's joint positions are not measured by anything.
"""

from __future__ import annotations

import logging

import numpy as np

from core.config import SETTINGS
from core.types import PersonObservation

LOGGER = logging.getLogger("warrioriq.rtm_pose")

_refiner: "_Refiner | None" = None
_unavailable = False


class _Refiner:
    """Holds the loaded model. Built once, on first use, never at import."""

    def __init__(self) -> None:
        from rtmlib import RTMPose

        self.model = RTMPose(
            onnx_model=SETTINGS.rtm_pose_model,
            model_input_size=(192, 256),
            backend="onnxruntime",
            device=SETTINGS.rtm_pose_device,
        )

    def __call__(self, frame, boxes: np.ndarray):
        return self.model(frame, bboxes=boxes)


def _get() -> "_Refiner | None":
    """Load on demand, and give up permanently rather than retrying per frame."""
    global _refiner, _unavailable
    if _unavailable:
        return None
    if _refiner is None:
        try:
            _refiner = _Refiner()
        except Exception as exc:                                    # noqa: BLE001
            _unavailable = True
            LOGGER.warning(
                "rtm_pose_unavailable error=%s detail=%s "
                "falling back to the detector's own keypoints",
                type(exc).__name__, str(exc)[:160],
            )
            return None
    return _refiner


def refine(frame, observations: list[PersonObservation | None]) -> int:
    """Replace the keypoints of these observations in place.

    Boxes, track IDs and appearance are untouched: this changes what the joints
    say, not who the person is. Identity has already been decided by the time
    this runs, deliberately, so a refinement can never move a fighter onto
    somebody else.

    Returns how many were refined, and refines nothing at all if the model is
    switched off or could not be loaded - an analysis without this is the
    analysis WarriorIQ ran before it existed, not a broken one.
    """
    if not SETTINGS.rtm_pose_enabled or frame is None:
        return 0
    targets = [obs for obs in observations if obs is not None and obs.box is not None]
    if not targets:
        return 0
    refiner = _get()
    if refiner is None:
        return 0
    boxes = np.asarray([np.asarray(obs.box, dtype=np.float32) for obs in targets], dtype=np.float32)
    try:
        keypoints, scores = refiner(frame, boxes)
    except Exception as exc:                                        # noqa: BLE001
        LOGGER.warning("rtm_pose_failed error=%s", type(exc).__name__)
        return 0
    refined = 0
    for obs, kp, sc in zip(targets, keypoints, scores):
        points = np.asarray(kp, dtype=np.float32)
        if points.shape[0] < 17:
            continue
        obs.keypoints = points[:17, :2]
        obs.keypoint_conf = np.asarray(sc, dtype=np.float32)[:17]
        refined += 1
    return refined


def reset_for_tests() -> None:
    """Forget the loaded model so a test can exercise the failure path."""
    global _refiner, _unavailable
    _refiner = None
    _unavailable = False
