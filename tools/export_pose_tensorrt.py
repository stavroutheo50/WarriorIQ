"""Build the TensorRT pose engine for this machine's GPU.

The recipe lives in core/trt_engine.py so the Modal worker builds an identical
engine inside its container - the same dynamic shape, the same fp16 fallback -
rather than keeping a second copy of it that drifts.
"""

from __future__ import annotations

import logging

from core.config import SETTINGS
from core.trt_engine import build_pose_engine, engine_size, gpu_slug


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Exporting WarriorIQ pose model to TensorRT for {gpu_slug()} at imgsz {engine_size()}…")
    destination = build_pose_engine(SETTINGS.pose_model_engine)
    print("TensorRT engine ready:", destination)
    print("Restart WarriorIQ. It will automatically prefer this engine.")


if __name__ == "__main__":
    main()
