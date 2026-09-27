"""Does this video show the sport the athlete chose?

Picking kickboxing and uploading a taekwondo bout gives a report scored under
the wrong rules, and nothing said so. This looks at three still frames, taken
in the browser before the upload starts, and names the sport they show, so
the setup page can say "this looks like taekwondo" while the athlete can
still change it.

Why a hosted vision model and not a local one
---------------------------------------------
Tried first and rejected: CLIP (base and large, whole frame and cropped to
the two fighters) named every clip taekwondo at 89-100%, including both
kickboxing bouts in the regression set - broadcast footage puts two small
figures on a mat, and a mat reads as taekwondo. Shipping that would have told
every kickboxer they chose wrong. Telling a glove from a dobok at that size
takes a model that can reason about what it sees.

Off unless configured
---------------------
Nothing leaves the server unless WARRIORIQ_SPORT_CHECK_PROVIDER names a
provider and that provider's key is set:

    WARRIORIQ_SPORT_CHECK_PROVIDER=anthropic   + ANTHROPIC_API_KEY
    WARRIORIQ_SPORT_CHECK_PROVIDER=openai      + OPENAI_API_KEY

Each provider's SDK is imported only when used, so a host without it simply
reports the check as unavailable. Frames are sent, not stored; turning this
on means video frames go to that provider, which the privacy policy should
say before it is enabled. The answer is advisory: it never blocks an upload.
"""

from __future__ import annotations

import base64
import json
import logging
import os

from core.scoring import SPORTS

log = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai")
MAX_FRAMES = 3
MAX_FRAME_BYTES = 400_000
# Below this the page says nothing: a wrong "you chose the wrong sport" costs
# the athlete's trust in everything after it, a missed one costs a re-upload.
WARN_AT_CONFIDENCE = 0.75

_SPORT_WORDS = {
    "kickboxing": "kickboxing (boxing gloves, kicks allowed; WAKO / K-1 style events)",
    "boxing": "boxing (gloves, punches only, usually a roped ring)",
    "muay_thai": "Muay Thai (Thai shorts, clinch, elbows and knees, usually a ring)",
    "taekwondo": "taekwondo (doboks, no boxing gloves, electronic body protectors)",
    "mma": "MMA (small open-finger gloves, often a cage, takedowns and ground work)",
}

_PROMPT = (
    "These are still frames from one combat-sports video. Which sport is being "
    "fought? Choose from: " + "; ".join(_SPORT_WORDS[s] for s in SPORTS) + ". "
    "Answer 'unclear' if the frames do not show a bout, or if you cannot tell. "
    "Judge from what the fighters wear and the venue, not from how the camera "
    "is placed. Give a confidence between 0 and 1 and one short reason."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "sport": {"type": "string", "enum": [*SPORTS, "unclear"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["sport", "confidence", "reason"],
    "additionalProperties": False,
}


def provider() -> str | None:
    """The configured provider, or None when the check is off."""
    name = os.getenv("WARRIORIQ_SPORT_CHECK_PROVIDER", "").strip().lower()
    if name not in PROVIDERS:
        return None
    key = "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENAI_API_KEY"
    return name if os.getenv(key, "").strip() else None


def decode_frames(frames: list[str]) -> list[bytes]:
    """JPEG data URLs from the browser, checked and decoded. Raises ValueError."""
    if not frames or len(frames) > MAX_FRAMES:
        raise ValueError("send between 1 and %d frames" % MAX_FRAMES)
    out = []
    for frame in frames:
        prefix = "data:image/jpeg;base64,"
        if not isinstance(frame, str) or not frame.startswith(prefix):
            raise ValueError("frames must be JPEG data URLs")
        raw = base64.b64decode(frame[len(prefix):], validate=True)
        if len(raw) > MAX_FRAME_BYTES or raw[:3] != b"\xff\xd8\xff":
            raise ValueError("frame too large or not a JPEG")
        out.append(raw)
    return out


def _clean(result: dict) -> dict | None:
    sport = result.get("sport")
    if sport not in (*SPORTS, "unclear"):
        return None
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence"))))
    except (TypeError, ValueError):
        return None
    return {"sport": sport, "confidence": confidence, "reason": str(result.get("reason") or "")[:200]}


def _ask_anthropic(images: list[bytes]) -> dict | None:
    import anthropic

    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(raw).decode("ascii")}}
               for raw in images]
    content.append({"type": "text", "text": _PROMPT})
    response = anthropic.Anthropic().beta.messages.create(
        model=os.getenv("WARRIORIQ_SPORT_CHECK_MODEL", "claude-opus-5"),
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text) if text else None


def _ask_openai(images: list[bytes]) -> dict | None:
    from openai import OpenAI

    content = [{"type": "input_text", "text": _PROMPT}]
    content += [{"type": "input_image", "detail": "high",
                 "image_url": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")}
                for raw in images]
    response = OpenAI().responses.create(
        model=os.getenv("WARRIORIQ_SPORT_CHECK_MODEL", "gpt-5.6-terra"),
        store=False,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "sport_check", "strict": True, "schema": _SCHEMA}},
    )
    return json.loads(response.output_text)


def detect_sport(images: list[bytes]) -> dict | None:
    """{"sport", "confidence", "reason"}, or None when it cannot say.

    Every failure - no provider, SDK missing, network, refusal, a malformed
    answer - is None, because this is a courtesy check and must never be the
    reason an upload fails.
    """
    name = provider()
    if name is None:
        return None
    try:
        result = _ask_anthropic(images) if name == "anthropic" else _ask_openai(images)
    except Exception as exc:          # noqa: BLE001 - advisory; never breaks an upload
        log.warning("sport check failed (%s): %s", name, type(exc).__name__)
        return None
    return _clean(result) if isinstance(result, dict) else None


def verdict(chosen: str, detected: dict | None) -> dict:
    """What the page should say, if anything."""
    if not detected or detected["sport"] in ("unclear", chosen):
        return {"mismatch": False}
    return {"mismatch": detected["confidence"] >= WARN_AT_CONFIDENCE,
            "detected": detected["sport"], "confidence": detected["confidence"],
            "reason": detected["reason"]}
