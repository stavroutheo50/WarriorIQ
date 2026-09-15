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


# The container signatures WarriorIQ accepts, as (offset, bytes) pairs. A file
# matching none of them is not a video whatever it is called.
#
#   ISO base media (MP4, MOV, M4V): a box-size word, then the type "ftyp".
#   EBML (MKV, WEBM):               the EBML magic at byte 0.
#   RIFF (AVI):                     "RIFF", a size word, then "AVI ".
_VIDEO_SIGNATURES: tuple[tuple[tuple[int, bytes], ...], ...] = (
    ((4, b"ftyp"),),
    ((0, b"\x1a\x45\xdf\xa3"),),
    ((0, b"RIFF"), (8, b"AVI ")),
)

# Enough to reach the furthest signature offset with room to spare.
_HEADER_BYTES = 16


def looks_like_video(path: str | Path) -> bool:
    """Whether the bytes on disk are one of the containers we accept.

    The upload route checked the filename suffix and nothing else, so an
    eleven-byte text file renamed to .mp4 was accepted as a fight: the drop
    zone turned green, the submit button enabled, and the failure arrived much
    later from the decoder, by which time the reader had been told their fight
    was uploading.

    Deliberately a container check, not a "can this decode" check. Answering
    that means starting a decoder on an untrusted file, which is the thing a
    format check exists to avoid; a file that passes here and still will not
    decode is caught by the probe that follows, with a real error to show.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(_HEADER_BYTES)
    except OSError:
        return False
    return any(
        all(header[at:at + len(magic)] == magic for at, magic in signature)
        for signature in _VIDEO_SIGNATURES
    )
