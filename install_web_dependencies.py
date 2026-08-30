from __future__ import annotations

import subprocess
import sys
from pathlib import Path


requirements = Path(__file__).with_name("requirements-web.txt")
if not requirements.is_file():
    raise RuntimeError(f"Missing web dependency file: {requirements}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)],
    check=True,
)
print("WarriorIQ web dependencies installed successfully.")
