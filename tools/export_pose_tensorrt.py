from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

from core.config import MODELS, SETTINGS


def _export_device() -> int | str:
    """Resolve the configured device the way the tracker does.

    SETTINGS.device defaults to "auto", which PoseTracker turns into a real
    device at load time and the exporter rejects outright, so this tool failed
    before it exported anything.
    """
    import torch

    requested = str(SETTINGS.device).strip().lower()
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested


def main():
    print("Exporting WarriorIQ pose model to TensorRT for the RTX GPU…")
    model = YOLO(SETTINGS.pose_model_pt)
    # The engine has to serve every size inference_size() can ask for, not just
    # one. It was exported fixed at default_imgsz (640) while low-resolution
    # footage asks for low_resolution_imgsz (1280), and a fixed-shape engine
    # cannot honour a size it was not built for: the request was silently
    # served at 640, and a 1920 request returned nothing at all. Measured on
    # real 480x220 tournament footage, that cost every fighter in mid-round
    # frames - the engine found the referee and the seated spectators and not
    # one athlete, while the same model as .pt found them.
    #
    # inference_size() exists precisely to upscale small sources so athletes
    # clear what pose estimation can resolve. A dynamic engine is what lets it
    # actually do that.
    size = max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz)

    def build(half: bool):
        return model.export(
            format="engine",
            device=_export_device(),
            imgsz=size,
            half=half,
            dynamic=True,
            workspace=4,
        )

    # fp16 roughly halves both the inference time and the activation memory,
    # which matters on an 8GB card that also drives a display: at imgsz 1600
    # the fp32 engine alone holds 3.85GB, leaving too little for the tracker's
    # ReID model and SAM2 recovery beside it.
    #
    # TensorRT 11 removed the fp16 builder flag, so the precision has to be
    # baked into the ONNX graph by NVIDIA ModelOpt first. That needs onnx>=1.18,
    # whose test data cannot be unpacked on Windows unless long paths are
    # enabled. Rather than fail the export - which would leave no engine at all
    # and stop every analysis - fall back to fp32 and say exactly what is
    # missing.
    try:
        exported = build(half=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  fp16 unavailable ({type(exc).__name__}: {exc}); building fp32 instead.")
        print("  To get fp16: enable Windows long paths, then `pip install -U onnx`.")
        exported = build(half=False)
    src = Path(str(exported))
    dst = Path(SETTINGS.pose_model_engine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print("TensorRT engine ready:", dst)
    print("Restart WarriorIQ. It will automatically prefer this engine.")


if __name__ == "__main__":
    main()
