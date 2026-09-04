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
    exported = model.export(
        format="engine",
        device=_export_device(),
        imgsz=max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz),
        half=False,
        dynamic=True,
        workspace=4,
    )
    src = Path(str(exported))
    dst = Path(SETTINGS.pose_model_engine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print("TensorRT engine ready:", dst)
    print("Restart WarriorIQ. It will automatically prefer this engine.")


if __name__ == "__main__":
    main()
