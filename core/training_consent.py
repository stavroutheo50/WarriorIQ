"""Fights their owners chose to share to improve the analysis (2026-10-27).

The upload page has an optional box, off unless ticked: "Help WarriorIQ get
better". When it is ticked and the analysis finishes, the fight's video, its
report and its tracking file are kept here, apart from the 30-day video
retention, so the strike detector can be trained on real customer footage -
kicks most of all, where the detector is weakest and public footage is
scarce. Nothing here is shown to anyone; it is pulled to the training machine
with the worker token (/api/worker/consented).

Deleting the fight, its video or the account deletes the copy here too
(app.main._remove_fight_files and delete_original_video call forget()).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from core.config import DATASET

LOGGER = logging.getLogger("warrioriq.training_consent")
FOLDER = DATASET / "consented"
# The analysis files worth keeping beside the video: what was detected, and
# the tracks to cut training windows from.
KEPT_ARTIFACTS = ("report.json", "tracking.jsonl")
_JOB_ID = re.compile(r"^[0-9a-f]{8,32}$")


def _folder(job_id: str) -> Path | None:
    if not _JOB_ID.match(str(job_id or "")):
        return None
    return FOLDER / job_id


def _place(source: Path, target: Path) -> None:
    """A hard link where possible: the upload is deleted after 30 days, the link stays."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def keep(job_id: str, video_path: str | os.PathLike, run_directory: str | os.PathLike, details: dict) -> bool:
    """Keep a consented fight's video and analysis files. False when there is nothing to keep."""
    folder = _folder(job_id)
    video = Path(video_path)
    if folder is None or not video.is_file():
        return False
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"video{video.suffix.lower() or '.mp4'}"
        if not target.exists():
            _place(video, target)
        for name in KEPT_ARTIFACTS:
            source = Path(run_directory) / name
            if source.is_file() and not (folder / name).exists():
                _place(source, folder / name)
        (folder / "consent.json").write_text(json.dumps({
            "job_id": job_id, "kept_at_epoch": time.time(), **details}, indent=1), encoding="utf-8")
    except OSError as exc:
        LOGGER.error("consented_fight_not_kept job_id=%s error=%s", job_id, type(exc).__name__)
        shutil.rmtree(folder, ignore_errors=True)
        return False
    return True


def forget(job_id: str) -> None:
    """Delete everything kept for this fight. Safe to call for any fight."""
    folder = _folder(job_id)
    if folder is not None and folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def kept() -> list[dict]:
    """Every kept fight: its consent record and the names and sizes of its files."""
    if not FOLDER.exists():
        return []
    fights = []
    for folder in sorted(FOLDER.iterdir()):
        record = folder / "consent.json"
        if not folder.is_dir() or not record.is_file():
            continue
        try:
            details = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        details["files"] = {path.name: path.stat().st_size for path in sorted(folder.iterdir())
                            if path.is_file() and path.name != "consent.json"}
        fights.append(details)
    return fights


def kept_file(job_id: str, name: str) -> Path | None:
    """One kept file, or None. Only names this module writes are served."""
    folder = _folder(job_id)
    if folder is None or not (name in KEPT_ARTIFACTS or re.match(r"^video\.[a-z0-9]{2,5}$", name)):
        return None
    path = folder / name
    return path if path.is_file() else None
