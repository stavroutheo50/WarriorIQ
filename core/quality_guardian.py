from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.types import VideoInfo


def quality_summary(width: int, height: int, fps: float, brightness: float, sharpness: float) -> dict:
    score = 100
    notes: list[str] = []
    if min(width, height) < 480:
        score -= 24
        notes.append("Low resolution may hide gloves, feet and contact details.")
    elif min(width, height) < 720:
        score -= 10
        notes.append("HD footage will provide stronger technique and contact evidence.")
    if fps < 18:
        score -= 20
        notes.append("Low frame rate can lose fast-motion evidence between frames.")
    elif fps < 24:
        score -= 8
        notes.append("A higher frame rate would improve fast-motion timing.")
    if brightness < 38:
        score -= 24
        notes.append("Low light may make fighters and limbs difficult to separate.")
    elif brightness > 218:
        score -= 12
        notes.append("Overexposure may remove clothing and glove detail.")
    if sharpness < 28:
        score -= 24
        notes.append("Motion blur may hide strike endpoints and contact.")
    elif sharpness < 55:
        score -= 10
        notes.append("Some motion blur is present; use a steadier or brighter recording when possible.")
    score = max(0, min(100, score))
    status = "strong" if score >= 85 else "usable" if score >= 70 else "review"
    if not notes:
        notes.append("The sampled frames meet the basic capture-quality checks.")
    return {
        "score": score,
        "status": status,
        "notes": notes,
        "measurements": {
            "resolution": f"{width}×{height}",
            "fps": round(float(fps), 1),
            "brightness": round(float(brightness), 1),
            "sharpness": round(float(sharpness), 1),
        },
    }


def inspect_video_quality(path: str | Path, info: VideoInfo) -> dict:
    capture = cv2.VideoCapture(str(path))
    brightness_values: list[float] = []
    sharpness_values: list[float] = []
    try:
        for ratio in (0.12, 0.50, 0.88):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((info.frame_count - 1) * ratio)))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 640.0 / max(width, height))
            if scale < 1:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray)))
            sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        capture.release()
    brightness = float(np.median(brightness_values)) if brightness_values else 0.0
    sharpness = float(np.median(sharpness_values)) if sharpness_values else 0.0
    result = quality_summary(info.width, info.height, info.fps, brightness, sharpness)
    result["sampled_frames"] = len(brightness_values)
    if not brightness_values:
        result["notes"] = ["WarriorIQ could not sample the video reliably. Choose another file before analysis."]
        result["status"] = "review"
        result["score"] = 0
    return result
