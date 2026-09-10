"""A learned appearance space, for telling one person from another.

Identity was decided by a hue/saturation histogram of the torso, and that
descriptor is at its limit here. Measured against the person originally
selected, on real footage: the referee scores 0.54 to 0.67 and the fighters
0.54 to 0.79. The ranges overlap almost completely, so no threshold separates
them. Two people in shorts on the same mat under the same lights are not
separable by colour, which is what a histogram measures.

Two attempts at replacing it failed, and both failures were mine rather than
the method's:

  * The encoder was wrong. The setting said "yolo26m.pt", which is the
    *detector*: Ultralytics routes a .pt through the YOLO predictor and reads
    the second-to-last layer, so what came back described "this is a person"
    rather than "this is which person". Ultralytics ships purpose-trained ReID
    encoders as yolo26{n,s,m,l,x}-reid.onnx, and those go through AutoBackend
    instead - more accurate and far cheaper, since the predictor path runs a
    whole detection per call.

  * The sample was too small. Even with the right encoder, one crop of a
    person sixty pixels tall does not describe them. Measured on the busiest
    bout: single crops separate same-person from different-person at AUC
    0.689. The mean of three crops of the same track, same encoder and same
    frames, reaches 0.996.

So embeddings are pooled over a track before anything is compared, and the
fighter's anchor is pooled over their first few frames rather than taken from
the seed crop alone. Across three bouts the same person then scores a median
of 0.93 to 0.98 and a different person 0.56 to 0.82.

The nano encoder is used on purpose: it separates better than the medium one
(0.996 against 0.989) and costs the same, because the time goes on cropping
rather than on the network. There is not enough detail in a person this small
for the larger model's extra capacity to describe.
"""

from __future__ import annotations

import logging

import numpy as np

from core.config import SETTINGS

LOGGER = logging.getLogger("warrioriq.reid")

_encoder = None
_unavailable = False


def _report_execution_providers(encoder) -> None:
    """Say which providers the session actually got, not which were asked for.

    Ultralytics prints "Using ONNX Runtime x.y.z with CUDAExecutionProvider"
    when it *requests* CUDA. ONNX Runtime then falls back to CPU silently if it
    cannot load the provider, and the encouraging line has already been
    printed. On this machine it does exactly that: onnxruntime-gpu 1.29 wants
    CUDA 13 and the box has 12.8 for torch, so `cublasLt64_13.dll` is missing
    and every embedding is computed on the CPU while the log says otherwise.

    That is survivable - measured 2026-09-07, the nano encoder costs about
    4 ms per person on the CPU, which is *faster* than the 10.4 ms recorded for
    it earlier - and the pose model, which is the expensive half, still runs on
    the GPU through torch. But a log line that names a provider it is not using
    is the kind of thing that costs an afternoon later, so the truth is
    recorded here at load time.

    **Chased properly on 2026-09-10 and closed: leave it on the CPU.** A profile
    put ONNX Runtime at 13.5 s of a 91 s analysis, about 15%, so the CPU
    fallback looked like the cheapest win available. There were two separate
    faults behind it, and both are real:

      1. onnxruntime-gpu 1.29 is built for CUDA 13 while torch here is cu128, so
         `cublasLt64_13.dll` is missing and the provider cannot load.
      2. Underneath that, `device=SETTINGS.reid_device or None` passed **None**
         whenever the setting was empty, which is always by default. ONNX
         Runtime receives `{'device_id': None}`, fails to parse it and falls
         back to the CPU. So even with the right build it would not have used
         the GPU. (Ultralytics wants a torch-style string here: "cuda:0", not
         "0", which it rejects as an invalid device string.)

    Both were fixed and benchmarked, and **the GPU is not faster**:

        cpu      7.79 ms for one person    7.14 ms per person, batch of 5
        cuda:0   9.04 ms for one person    7.28 ms per person, batch of 5

    Slower alone, identical in a batch, and the embeddings are numerically the
    same (0.741 either way on the reference pair). The reason is in this
    docstring already: the time goes on cropping, not on the network, and this
    encoder is nano-sized. Both changes were reverted rather than shipped for
    nothing. Do not spend the afternoon this comment exists to save.
    """
    session = getattr(encoder, "session", None) or getattr(
        getattr(encoder, "model", None), "session", None)
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        return
    active = list(getter())
    if any(name.startswith(("CUDA", "Tensorrt")) for name in active):
        LOGGER.info("reid_encoder_providers %s", ", ".join(active))
        return
    LOGGER.warning(
        "reid_encoder_on_cpu providers=%s - embeddings are not using the GPU, "
        "whatever the Ultralytics line above says", ", ".join(active) or "none",
    )


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
            _report_execution_providers(_encoder)
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


def pool(vectors) -> np.ndarray | None:
    """Mean of several embeddings, renormalised, or None if there are none.

    The whole point of this module in practice. A single crop of a person
    sixty pixels tall carries too little to identify them - measured on real
    footage, one crop separates same-person from different-person at AUC 0.689
    and the mean of three at 0.996, with the same encoder on the same frames.
    """
    usable = [np.asarray(v, dtype=np.float32).ravel() for v in vectors if v is not None]
    if not usable:
        return None
    width = usable[0].size
    usable = [v for v in usable if v.size == width]
    if not usable:
        return None
    mean = np.mean(usable, axis=0)
    scale = float(np.linalg.norm(mean))
    return None if scale <= 0.0 else (mean / scale)


def reset_for_tests() -> None:
    global _encoder, _unavailable
    _encoder = None
    _unavailable = False
