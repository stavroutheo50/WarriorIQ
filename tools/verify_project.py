from __future__ import annotations

import compileall
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use the project-local Ultralytics configuration before importing the package.
import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def check(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)
    print("OK  ", message)


def main():
    print("\nWarriorIQ project verification")
    print("Root:", ROOT)

    required = [
        "run.py",
        "app/main.py",
        "core/analyzer.py",
        "core/identity.py",
        "core/action.py",
        "core/contact.py",
        "core/scoring.py",
        "app/templates/index.html",
        "app/templates/select.html",
        "app/templates/result.html",
    ]
    for relative in required:
        check((ROOT / relative).exists(), f"file exists: {relative}")

    # Compile only WarriorIQ-owned code. Recursing through .venv both wastes
    # time and can fail on third-party packages with Windows path-length limits.
    compile_targets = [ROOT / "app", ROOT / "core", ROOT / "tools", ROOT / "tests"]
    compiled = all(compileall.compile_dir(str(path), quiet=1) for path in compile_targets)
    compiled = compiled and all(
        compileall.compile_file(str(ROOT / name), quiet=1)
        for name in ("run.py", "run_cli.py")
    )
    check(compiled, "all WarriorIQ Python files compile")

    from jinja2 import Environment, FileSystemLoader

    templates_root = ROOT / "app" / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_root)))
    for path in sorted(templates_root.glob("*.html")):
        env.get_template(path.name)
    check(True, "all Jinja templates parse")

    import cv2
    import fastapi
    import pydantic
    import torch
    import ultralytics

    check(True, "analysis backend is available")
    print("Compute:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback (slower)")
    print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
    print("OpenCV:", cv2.__version__)
    print("Ultralytics:", getattr(ultralytics, "__version__", "unknown"))
    print("FastAPI:", fastapi.__version__)
    print("Pydantic:", pydantic.__version__)

    # Synthetic WarriorIQ identity test: a tracker ID may change, but A must
    # remain A and must never take B's ID.
    from core.identity import IdentityManager
    from core.types import PersonObservation

    def person(track_id, x1, x2, appearance_index):
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
        kp[:, 1] = np.linspace(100, 290, 17)
        appearance = np.zeros((64,), dtype=np.float32)
        appearance[appearance_index] = 1.0
        return PersonObservation(
            track_id=track_id,
            box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
            confidence=0.95,
            keypoints=kp,
            keypoint_conf=np.ones((17,), dtype=np.float32),
            appearance=appearance,
        )

    a0, b0 = person(1, 100, 200, 1), person(2, 400, 500, 2)
    manager = IdentityManager(a0, b0, 0)
    a1, b1 = manager.update([person(7, 106, 206, 1), person(2, 394, 494, 2)], 1)
    check(a1 is not None and a1.track_id == 7, "WarriorIQ A survives a BoT-SORT ID change")
    check(b1 is not None and b1.track_id == 2, "WarriorIQ B remains B during A recovery")
    check(a1.track_id != b1.track_id, "one human cannot become both fighters")

    # Import complete application last, after dependency checks.
    import app.main  # noqa: F401
    check(True, "FastAPI application imports")

    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    check(result.wasSuccessful(), "WarriorIQ core unit tests pass")

    print("\nWARRIORIQ PROJECT CHECK PASSED\n")


if __name__ == "__main__":
    main()
