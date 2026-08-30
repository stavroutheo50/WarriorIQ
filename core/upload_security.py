from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from core.config import SETTINGS


def scan_upload(path: str | Path) -> dict:
    """Scan one generated upload path with a configured antivirus command.

    ClamAV's clamdscan/clamdscan.exe contract is supported directly: exit 0 is
    clean, 1 is infected, and any other exit code means the scanner failed.
    The command comes only from trusted deployment configuration and is never
    built from the original user filename.
    """
    command = shlex.split(SETTINGS.malware_scan_command, posix=False)
    if not command:
        return {
            "status": "unavailable" if SETTINGS.malware_scan_required else "skipped",
            "clean": not SETTINGS.malware_scan_required,
        }
    try:
        result = subprocess.run(
            [*command, "--no-summary", str(Path(path).resolve())],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "clean": False}
    if result.returncode == 0:
        return {"status": "clean", "clean": True}
    if result.returncode == 1:
        return {"status": "infected", "clean": False}
    return {"status": "unavailable", "clean": False}
