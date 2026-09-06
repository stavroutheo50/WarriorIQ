"""A learned appearance space, for telling a referee from a fighter.

Identity has been decided by a hue/saturation histogram of the torso, and that
descriptor is at its limit here. Measured against the person originally
selected, on real footage: the referee scores 0.54 to 0.67 and the fighters
0.54 to 0.79. The ranges overlap almost completely, so no threshold separates
them - removing the gate triples the frames spent on the referee, and raising
it collapses fighter B's coverage from 0.56 to 0.21. Adding the brightness
channel shifts both down by the same amount rather than opening a gap.

Two people wearing shorts on the same mat under the same lights are not
separable by colour, which is what a histogram measures.

The same comparison through a learned embedding, on the same frames: referee
0.695 to 0.741, fighters 0.724 to 0.828. A threshold of 0.745 refuses all four
referee crops and keeps ten of the eleven fighter crops. That is a usable gate
where the histogram had none.

The encoder is the one Ultralytics already ships for BoT-SORT's own ReID, so
there is no new model to download, no new dependency, and no second weights
file to keep in step with the detector.
"""

from __future__ import annotations

import logging

import numpy as np

from core.config import SETTINGS

LOGGER = logging.getLogger("warrioriq.reid")

_encoder = None
_unavailable = False


def _get():
    """Load the encoder once, and give up permanently rather than per frame."""
    global _encoder, _unavailable
    if _unavailable or not SETTINGS.reid_enabled:
        return None
    if _encoder is None:
        try:
            from ultralytics.trackers.utils.reid import ReID

            # An empty string is rejected outright; None means 'choose for me'.
            _encoder = ReID(SETTINGS.reid_model, device=SETTINGS.reid_device or None)
        except Exception as exc:                                    # noqa: BLE001
            _unavailable = True
            LOGGER.warning(
                "reid_unavailable model=%s error=%s detail=%s falling back to the colour histogram",
                SETTINGS.reid_model, type(exc).__name__, str(exc)[:160],
            )
            return None
    return _encoder


def embed(frame, boxes: np.ndarray) -> list[np.ndarray | None]:
    """One appearance vector per box, or Nones if the encoder is unavailable.

    Never raises. Without this the identity manager uses the histogram it
    always used, which is a worse analysis and not a broken one.
    """
    if frame is None or boxes is None or not len(boxes):
        return []
    encoder = _get()
    if encoder is None:
        return [None] * len(boxes)
    try:
        return list(encoder(frame, np.asarray(boxes, dtype=np.float32)))
    except Exception as exc:                                        # noqa: BLE001
        LOGGER.warning("reid_failed error=%s", type(exc).__name__)
        return [None] * len(boxes)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity, or None when either side has no embedding.

    None means "no opinion" and the caller falls back to the histogram, rather
    than a neutral 0.5 that would read as weak evidence of a match.
    """
    if a is None or b is None:
        return None
    left = np.asarray(a, dtype=np.float32).ravel()
    right = np.asarray(b, dtype=np.float32).ravel()
    if left.size != right.size or not left.size:
        return None
    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    if scale <= 0.0:
        return None
    return float(np.dot(left, right) / scale)


def reset_for_tests() -> None:
    global _encoder, _unavailable
    _encoder = None
    _unavailable = False
