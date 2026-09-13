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
    """The one size the engine is built at: the largest anything can ask for."""
    return max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz)


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
    """Where this GPU's engine lives, named so two GPU types cannot collide."""
    return Path(models_dir) / f"pose_engine_{gpu_slug(name)}_{size or engine_size()}.engine"


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

    def build(half: bool):
        return model.export(format="engine", device=device, imgsz=size,
                            half=half, dynamic=True, workspace=4)

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
    LOGGER.info("trt_engine_built path=%s imgsz=%s precision=%s dynamic=True",
                destination, size, precision)
    return destination


def ensure_pose_engine(models_dir: str | Path, name: str | None = None) -> Path | None:
    """Return this GPU's cached engine, building it once if it is not there.

    Never raises. A build that fails must cost the analysis its speed and not
    its existence: the tracker falls back to the .pt checkpoint, which is the
    same model and the same answers, only slower.
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
    LOGGER.info("trt_engine_missing path=%s building_now=True", destination)
    try:
        return build_pose_engine(destination)
    except Exception as exc:                                        # noqa: BLE001
        LOGGER.warning(
            "trt_engine_build_failed error=%s detail=%s falling back to the .pt checkpoint",
            type(exc).__name__, str(exc)[:200],
        )
        return None
