"""Building and identifying the TensorRT pose engine.

A TensorRT engine is compiled for one GPU architecture and one TensorRT
runtime. It is not portable, it is not in the checkout, and it takes minutes to
build - so it has to be built where it will run and kept somewhere that
survives the container. This module is the one place that knows how.

**One dynamic engine, not one per size.** The obvious design is an engine per
inference size - 640 for normal footage, the low-resolution size for small
sources - and it was tried. A fixed-shape engine cannot honour a size it was
not built for, and TensorRT does not refuse: the request was silently served at
640, and a 1920 request returned nothing at all. Measured on real 480x220
tournament footage, that cost every fighter in mid-round frames - the engine
found the referee and the seated spectators and not one athlete, while the same
model as .pt found them. `inference_size()` exists precisely so small sources
are upscaled until athletes clear what pose estimation can resolve, and the
quality controller then moves that size by 32 and 64 pixels a step while a
fight runs. A dynamic engine built at the largest size any of that can ask for
serves all of it; two fixed engines serve neither.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
from pathlib import Path

from core.config import SETTINGS

LOGGER = logging.getLogger("warrioriq.trt")


def export_device() -> int | str:
    """Resolve the configured device the way the tracker does.

    SETTINGS.device defaults to "auto", which PoseTracker turns into a real
    device at load time and the exporter rejects outright, so the export tool
    failed before it exported anything.
    """
    import torch

    requested = str(SETTINGS.device).strip().lower()
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested


def engine_size() -> int:
    """The size the engine is optimised for: the one most analyses run at.

    It used to be the largest size anything could ask for (1600), and
    Ultralytics makes that the profile's *optimum* - so the kernels were tuned
    for the rare small-source footage, every ordinary 640 run used tactics
    picked for 1600, and the profile's maximum came out at 1600 x workspace =
    6400, which only made the build slower and the activations larger.
    """
    return SETTINGS.default_imgsz


# Sizes the engine must accept even though it is not tuned for them. 1280 is
# named because footage between the default and the low-resolution rule is run
# there; the low-resolution size (1600 by default) is what small sources need.
REQUIRED_SIZES = (1280,)


def engine_shape_multiplier() -> int:
    """How many times the optimum the largest accepted input is.

    Ultralytics sets a dynamic profile's maximum to the export size times
    max(2, workspace), so the workspace argument is what sets the ceiling.
    """
    needed = max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz, *REQUIRED_SIZES)
    return max(2, math.ceil(needed / engine_size()))


def engine_max_size() -> int:
    """The largest input the engine accepts."""
    return engine_size() * engine_shape_multiplier()


def gpu_slug(name: str | None = None) -> str:
    """A filename-safe name for the GPU an engine was built on.

    The engine is keyed by this because reusing one across GPU models is not a
    performance question but a correctness one - it either refuses to
    deserialise or, worse, is accepted by a close-enough architecture. A shared
    cache with one filename would hand an A10G engine to an L4.
    """
    if name is None:
        try:
            import torch

            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:                                           # noqa: BLE001
            name = "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return slug or "unknown"


def engine_path_for(models_dir: str | Path, name: str | None = None, size: int | None = None) -> Path:
    """Where this GPU's engine lives, named so two GPU types cannot collide.

    The name carries the optimum and the largest accepted size, so an engine
    built under the old sizing (one number, 1600) is never mistaken for this one.
    """
    optimum = size or engine_size()
    largest = optimum * max(2, math.ceil(engine_max_size() / optimum))
    return Path(models_dir) / f"pose_engine_{gpu_slug(name)}_{optimum}_max{largest}.engine"


def build_pose_engine(destination: str | Path, size: int | None = None,
                      device: int | str | None = None) -> Path:
    """Compile the pose model to TensorRT at `destination`, and return it.

    fp16 roughly halves both the inference time and the activation memory,
    which matters on an 8 GB card that also drives a display: at imgsz 1600 the
    fp32 engine alone holds 3.85 GB, leaving too little for the tracker's ReID
    model and SAM2 recovery beside it.

    TensorRT 11 removed the fp16 builder flag, so the precision has to be baked
    into the ONNX graph by NVIDIA ModelOpt first. That needs onnx>=1.18, whose
    test data cannot be unpacked on Windows unless long paths are enabled.
    Rather than fail the export - which would leave no engine at all and stop
    every analysis - fall back to fp32 and say exactly what is missing.
    """
    from ultralytics import YOLO

    size = int(size or engine_size())
    device = export_device() if device is None else device
    model = YOLO(SETTINGS.pose_model_pt)
    # The workspace is the ceiling as well as the builder's memory pool in GB
    # (see engine_shape_multiplier): optimum 640 with workspace 3 accepts up to
    # 1920, which covers 1280 and the 1600 low-resolution size.
    multiplier = max(2, math.ceil(engine_max_size() / size))

    def build(half: bool):
        return model.export(format="engine", device=device, imgsz=size,
                            half=half, dynamic=True, workspace=multiplier)

    try:
        exported = build(half=True)
        precision = "fp16"
    except Exception as exc:                                        # noqa: BLE001
        LOGGER.warning(
            "trt_fp16_unavailable error=%s detail=%s building fp32 instead "
            "(for fp16: enable Windows long paths, then pip install -U onnx)",
            type(exc).__name__, str(exc)[:160],
        )
        exported = build(half=False)
        precision = "fp32"

    source = Path(str(exported))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    LOGGER.info("trt_engine_built path=%s opt=%s max=%s precision=%s dynamic=True",
                destination, size, size * multiplier, precision)
    return destination


def _failure_marker(engine: Path) -> Path:
    return engine.with_suffix(".failed")


def ensure_pose_engine(models_dir: str | Path, name: str | None = None,
                       retry_failed_after_seconds: float | None = None) -> Path | None:
    """Return this GPU's cached engine, building it once if it is not there.

    Never raises. A build that fails must cost the analysis its speed and not
    its existence: the tracker falls back to the .pt checkpoint, which is the
    same model and the same answers, only slower.

    ``retry_failed_after_seconds`` is for a scale-to-zero worker, where every
    container is a cold start: a build that cannot succeed there (no TensorRT
    in the image, which was the case on Modal until 2026-10-04) still exported
    ONNX first and then failed, on every cold start, before the first frame.
    A failure is recorded beside the engine and not retried until it is that
    old. None retries every time, as a local worker always has.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            LOGGER.info("trt_skipped reason=no_cuda")
            return None
    except Exception:                                               # noqa: BLE001
        return None
    destination = engine_path_for(models_dir, name)
    if destination.exists():
        LOGGER.info("trt_engine_cached path=%s", destination)
        return destination
    marker = _failure_marker(destination)
    if retry_failed_after_seconds is not None and marker.exists():
        import time

        age = time.time() - marker.stat().st_mtime
        if age < retry_failed_after_seconds:
            LOGGER.warning("trt_engine_build_skipped previous_failure=%s age_seconds=%.0f - using the "
                           ".pt checkpoint", marker.read_text(encoding="utf-8")[:200], age)
            return None
    LOGGER.info("trt_engine_missing path=%s building_now=True", destination)
    try:
        built = build_pose_engine(destination)
    except Exception as exc:                                        # noqa: BLE001
        LOGGER.warning(
            "trt_engine_build_failed error=%s detail=%s falling back to the .pt checkpoint",
            type(exc).__name__, str(exc)[:200],
        )
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{type(exc).__name__}: {str(exc)[:500]}", encoding="utf-8")
        except OSError:
            pass
        return None
    marker.unlink(missing_ok=True)
    return built
