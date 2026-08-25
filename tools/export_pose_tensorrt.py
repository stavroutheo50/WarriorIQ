from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

from core.config import MODELS, SETTINGS


def main():
    print("Exporting WarriorIQ pose model to TensorRT for the RTX GPU…")
    model = YOLO(SETTINGS.pose_model_pt)
    exported = model.export(
        format="engine",
        device=SETTINGS.device,
        imgsz=SETTINGS.default_imgsz,
        half=True,
        dynamic=False,
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
