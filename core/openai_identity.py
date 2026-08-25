from __future__ import annotations

import base64
import json
import os
from typing import Any

import cv2
import numpy as np

from core.config import SETTINGS
from core.types import PersonObservation


def _image_url(frame: np.ndarray, max_width: int = 960) -> str:
    image = frame
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, max(1, round(image.shape[0] * scale))))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode identity evidence")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _crop(frame: np.ndarray, box) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    return frame[y1:y2, x1:x2].copy()


class OpenAIIdentityReferee:
    """Optional visual referee for rare, unresolved A/B identity failures."""

    def __init__(self, enabled: bool, reference_frame: np.ndarray, a_box, b_box):
        self.enabled = bool(enabled and os.getenv("OPENAI_API_KEY"))
        self.failure_reason = None if self.enabled else "OpenAI recovery disabled or OPENAI_API_KEY is not configured"
        self.reference_a = _crop(reference_frame, a_box)
        self.reference_b = _crop(reference_frame, b_box)
        self.last_attempt_seconds = -10_000.0
        self.attempts = 0
        self.recoveries = 0

    def recover(self, history: list[np.ndarray], current: np.ndarray, people: list[PersonObservation], seconds: float) -> dict[str, Any] | None:
        if not self.enabled or len(people) < 2 or seconds - self.last_attempt_seconds < SETTINGS.openai_identity_cooldown_seconds:
            return None
        self.last_attempt_seconds = seconds
        self.attempts += 1
        try:
            from openai import OpenAI

            annotated = current.copy()
            for index, person in enumerate(people):
                x1, y1, x2, y2 = [int(v) for v in person.box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 4)
                cv2.putText(annotated, f"C{index}", (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 210, 255), 3)

            content: list[dict[str, Any]] = [{"type": "input_text", "text": (
                "Act only as a fighter identity referee. Image 1 is the original Fighter A portrait; image 2 is Fighter B. "
                "The following images are chronological context before an identity failure. The final image labels current people C0, C1, etc. "
                "Match the two original fighters to current candidates. Never choose a referee, coach, or spectator. Return unresolved when uncertain."
            )}]
            for image in [self.reference_a, self.reference_b, *history[-6:], annotated]:
                if image.size:
                    content.append({"type": "input_image", "image_url": _image_url(image), "detail": "high"})

            response = OpenAI().responses.create(
                model=SETTINGS.openai_identity_model,
                store=False,
                input=[{"role": "user", "content": content}],
                text={"format": {"type": "json_schema", "name": "fighter_identity", "strict": True, "schema": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["recover", "unresolved"]},
                        "fighter_a_candidate": {"type": ["integer", "null"]},
                        "fighter_b_candidate": {"type": ["integer", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["decision", "fighter_a_candidate", "fighter_b_candidate", "confidence", "reason"],
                }}},
            )
            result = json.loads(response.output_text)
            ai, bi = result.get("fighter_a_candidate"), result.get("fighter_b_candidate")
            valid = result.get("decision") == "recover" and float(result.get("confidence", 0)) >= SETTINGS.openai_identity_min_confidence
            valid = valid and isinstance(ai, int) and isinstance(bi, int) and ai != bi and 0 <= ai < len(people) and 0 <= bi < len(people)
            if not valid:
                return None
            self.recoveries += 1
            return result
        except Exception as exc:
            self.failure_reason = f"{type(exc).__name__}: {exc}"
            return None
