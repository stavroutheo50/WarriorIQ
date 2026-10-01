"""Find people in a still, and the clearest moment of a fight, on the web host.

The fighter-selection step used to ask the user to scrub the video to a clear
moment first. That is manual work, and on a phone that films HEVC the player
was simply black. The web host has OpenCV but none of the analysis models
(no torch, no ultralytics), so the pose model cannot do this there.

A small COCO detector can: NanoDet-Plus-m (Apache-2.0), 3.8 MB, from the
OpenCV model zoo. It runs through cv2.dnn, which the host already has, at
about 0.1-0.2 s per frame on a CPU. It is fetched on first use rather than
committed, and checked against a fixed hash so a changed file is refused.

Everything here is advisory. When the model cannot be fetched or loaded, the
callers fall back to the frame and the manual drawing they had before.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core.config import MODELS

LOGGER = logging.getLogger("warrioriq.person_detect")

MODEL_URL = os.getenv(
    "WARRIORIQ_PERSON_DETECTOR_URL",
    "https://huggingface.co/opencv/object_detection_nanodet/resolve/main/"
    "object_detection_nanodet_2022nov.onnx",
)
MODEL_SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"
MODEL_PATH = MODELS / "person_detector_nanodet_2022nov.onnx"
# "0" turns the detector off, and the pages fall back to drawing by hand. The
# test suite sets it so a run never reaches the network.
ENABLED = os.getenv("WARRIORIQ_PERSON_DETECTOR", "1").strip() != "0"

_INPUT = 416
_STRIDES = (8, 16, 32, 64)
_REG_MAX = 7
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
_PERSON = 0  # COCO class id

# A person worth offering as a fighter.
MIN_CONFIDENCE = 0.35
MAX_CANDIDATES = 10

_lock = threading.Lock()
_net = None
_unavailable = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_model() -> Path | None:
    if MODEL_PATH.exists() and _sha256(MODEL_PATH) == MODEL_SHA256:
        return MODEL_PATH
    partial = MODEL_PATH.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if _sha256(partial) != MODEL_SHA256:
            LOGGER.warning("person_detector_rejected reason=hash_mismatch")
            partial.unlink(missing_ok=True)
            return None
        partial.replace(MODEL_PATH)
        return MODEL_PATH
    except Exception as exc:  # network, disk, permissions: all mean "not today"
        LOGGER.warning("person_detector_unavailable reason=%s", type(exc).__name__)
        partial.unlink(missing_ok=True)
        return None


def _load():
    """The network, loaded once per process; None when it cannot be."""
    global _net, _unavailable
    if _net is not None or _unavailable or not ENABLED:
        return _net
    path = _fetch_model()
    if path is None:
        _unavailable = True
        return None
    try:
        _net = cv2.dnn.readNet(str(path))
    except Exception as exc:
        LOGGER.warning("person_detector_unavailable reason=%s", type(exc).__name__)
        _unavailable = True
    return _net


def _anchors(stride: int) -> np.ndarray:
    side = _INPUT // stride
    xs, ys = np.meshgrid(np.arange(side) * stride, np.arange(side) * stride)
    return np.column_stack((xs.ravel() + 0.5 * (stride - 1), ys.ravel() + 0.5 * (stride - 1)))


def _decode(outputs) -> list[tuple[list[float], float]]:
    """NanoDet heads -> (box in 416 space, person score).

    Output order differs between OpenCV 4 and 5 (interleaved or grouped), so
    heads are paired by shape: class maps have 80 columns, box maps
    4 * (REG_MAX + 1), and each level's two maps have the same row count.
    """
    outputs = [o[0] if o.ndim == 3 else o for o in outputs]
    scores = {o.shape[0]: o for o in outputs if o.shape[1] == 80}
    boxes = {o.shape[0]: o for o in outputs if o.shape[1] == 4 * (_REG_MAX + 1)}
    project = np.arange(_REG_MAX + 1, dtype=np.float32)
    found_boxes, found_scores = [], []
    for stride in _STRIDES:
        rows = (_INPUT // stride) ** 2
        if rows not in scores or rows not in boxes:
            continue
        person = scores[rows][:, _PERSON]
        keep = person >= MIN_CONFIDENCE
        if not keep.any():
            continue
        dist = boxes[rows][keep].reshape(-1, _REG_MAX + 1)
        dist = np.exp(dist - dist.max(axis=1, keepdims=True))
        dist = (dist / dist.sum(axis=1, keepdims=True)) @ project
        dist = dist.reshape(-1, 4) * stride
        centres = _anchors(stride)[keep]
        found_boxes.append(np.column_stack((
            centres[:, 0] - dist[:, 0], centres[:, 1] - dist[:, 1],
            centres[:, 0] + dist[:, 2], centres[:, 1] + dist[:, 3])))
        found_scores.append(person[keep])
    if not found_boxes:
        return []
    all_boxes = np.clip(np.concatenate(found_boxes), 0, _INPUT)
    all_scores = np.concatenate(found_scores)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in all_boxes]
    picked = cv2.dnn.NMSBoxes(xywh, [float(s) for s in all_scores], MIN_CONFIDENCE, 0.6)
    return [([float(v) for v in all_boxes[i]], float(all_scores[i])) for i in np.ravel(picked)]


def detect_people(frame: np.ndarray) -> list[dict] | None:
    """People in a BGR frame as [{"box": [x1, y1, x2, y2], "confidence"}].

    None means the detector is unavailable, which is not the same as nobody
    being in the frame.
    """
    net = _load()
    if net is None or frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    scale = _INPUT / max(height, width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    top, left = (_INPUT - new_h) // 2, (_INPUT - new_w) // 2
    canvas = np.zeros((_INPUT, _INPUT, 3), np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
    blob = cv2.dnn.blobFromImage((canvas.astype(np.float32) - _MEAN) / _STD)
    try:
        with _lock:  # a cv2.dnn Net is not safe to run from two threads at once
            net.setInput(blob)
            outputs = net.forward(net.getUnconnectedOutLayersNames())
    except Exception as exc:
        LOGGER.warning("person_detector_failed reason=%s", type(exc).__name__)
        return None
    people = []
    for (x1, y1, x2, y2), score in _decode(outputs):
        box = [max(0.0, (x1 - left) / scale), max(0.0, (y1 - top) / scale),
               min(float(width), (x2 - left) / scale), min(float(height), (y2 - top) / scale)]
        if box[2] > box[0] and box[3] > box[1]:
            people.append({"box": box, "confidence": score})
    people.sort(key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
    return people[:MAX_CANDIDATES]


def _shared(a: list[float], b: list[float]) -> float:
    """Overlap as a share of the smaller box."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, smaller)


def pair_score(people: list[dict], height: int) -> tuple[float, tuple[int, int] | None]:
    """How clear a start this frame gives, and which two people make it.

    A clear start is two whole people of similar size, apart from each other,
    with nobody else standing across either of them - the same three things
    the selection page asks a person to look for.
    """
    best, best_pair = 0.0, None
    tall = [i for i, p in enumerate(people) if p["box"][3] - p["box"][1] >= 0.12 * height]
    for n, i in enumerate(tall):
        for j in tall[n + 1:]:
            a, b = people[i]["box"], people[j]["box"]
            if _shared(a, b) > 0.10:
                continue
            ha, hb = a[3] - a[1], b[3] - b[1]
            similar = min(ha, hb) / max(ha, hb)
            if similar < 0.5:
                continue
            score = min(people[i]["confidence"], people[j]["confidence"]) * similar
            # Whole bodies: cut off at the frame edge is a weaker start.
            for box in (a, b):
                if box[1] <= 0.01 * height or box[3] >= 0.99 * height:
                    score *= 0.7
            # Somebody else across either fighter (a referee stepping in).
            for k, other in enumerate(people):
                if k not in (i, j) and other["confidence"] >= 0.4 and (
                        _shared(a, other["box"]) > 0.15 or _shared(b, other["box"]) > 0.15):
                    score *= 0.5
                    break
            # Bigger is easier to follow; a gentle preference only.
            score *= min(1.0, ((ha + hb) / 2) / (0.35 * height)) ** 0.5
            if score > best:
                best, best_pair = score, (i, j)
    return best, best_pair


# Where the analysis starts is the chosen moment: the fighters are identified
# there and followed forwards. So an early clear moment beats a slightly
# clearer late one - a pick at 2:11 would skip the first two minutes.
EARLY_WINDOW_SECONDS = 20.0
EARLY_SHARE = 0.10
EARLY_ENOUGH = 0.6       # the opening's best, as a share of the whole file's best
NEAR_BEST = 0.9          # take the earliest moment this close to the window's best


def _scan(cap, indices, fps: float) -> list[dict] | None:
    found = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        people = detect_people(frame)
        if people is None:
            return None
        score, pair = pair_score(people, frame.shape[0])
        # A frame full of similar-sized people is a busier start than two
        # people alone; a mild preference, so a tournament hall still works.
        tall = sum(1 for p in people if p["box"][3] - p["box"][1] >= 0.12 * frame.shape[0])
        score /= 1.0 + 0.15 * max(0, tall - 2)
        if pair is not None:
            found.append({"frame_index": int(index), "seconds": int(index) / fps, "frame": frame,
                          "people": people, "pair": pair, "score": score})
    return found


def find_clear_moment(video_path: str | Path, samples: int = 12) -> dict | None:
    """The earliest clear moment to start from, or None if none qualifies.

    Returns {"frame_index", "seconds", "frame", "people", "pair", "score"}.
    Looks at the opening first (20 s, or a tenth of a long file) and takes the
    earliest moment near the best one there, unless the opening is much the
    worse view than the rest of the file (sampled skipping the first and last
    3%, where the camera is most often being raised or lowered).
    """
    if _load() is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        if total <= 0:
            return None
        early_end = min(total - 1, int(fps * max(EARLY_WINDOW_SECONDS, EARLY_SHARE * total / fps)))
        early = _scan(cap, np.linspace(0, early_end, samples).astype(int), fps)
        if early is None:
            return None
        rest = _scan(cap, (np.linspace(0.03, 0.97, samples) * (total - 1)).astype(int), fps)
        if rest is None:
            return None
        everywhere = early + rest
        if not everywhere:
            return None
        overall = max(m["score"] for m in everywhere)
        early_best = max((m["score"] for m in early), default=0.0)
        # Stay at the start unless the start is much the worse view: skipping
        # the opening is a cost, a muddled first frame is a bigger one.
        if early and early_best >= EARLY_ENOUGH * overall:
            return min((m for m in early if m["score"] >= NEAR_BEST * early_best),
                       key=lambda m: m["frame_index"])
        return max(everywhere, key=lambda m: m["score"])
    finally:
        cap.release()
