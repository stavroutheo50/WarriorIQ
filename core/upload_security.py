from __future__ import annotations

import shlex
import subprocess
import json
import shutil
import time
import asyncio
from pathlib import Path

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

from core.config import OUTPUTS, UPLOADS, SETTINGS
from core.db import connection


class UploadCapacityError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


FIGHT_UPLOAD_PATH = "/upload"


def is_fight_upload(scope) -> bool:
    """The one definition of the route that upload admission control guards.

    Both the body-limit middleware and the admission middleware have to agree
    with the router about which request this is. They ran off a literal path
    instead, so an application served under a mount prefix would have let the
    upload reach the handler with no storage lease and no reserved analysis -
    and the handler would have raised AttributeError looking for the job id
    that admission never set. Matching after the prefix keeps the three in step.
    """
    if scope.get("type") != "http" or scope.get("method") != "POST":
        return False
    path = scope.get("path") or ""
    root = (scope.get("root_path") or "").rstrip("/")
    if root and path.startswith(root):
        path = path[len(root):] or "/"
    return path == FIGHT_UPLOAD_PATH


class UploadBodyLimitMiddleware:
    """Bound multipart spooling, including requests with no Content-Length."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not is_fight_upload(scope):
            return await self.app(scope, receive, send)
        limit = min(SETTINGS.max_fight_bytes, SETTINGS.max_upload_bytes) + 1024 * 1024
        headers = dict(scope.get("headers", []))
        try:
            too_large = int(headers.get(b"content-length", b"0")) > limit
        except ValueError:
            too_large = True
        if too_large:
            return await JSONResponse({"detail": "This upload exceeds the maximum allowed size."}, status_code=413)(scope, receive, send)
        received = 0
        exceeded = False
        deadline = time.monotonic() + SETTINGS.upload_timeout_seconds

        async def limited_receive():
            nonlocal received, exceeded
            try:
                message = await asyncio.wait_for(receive(), timeout=max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError as exc:
                raise MultiPartException("The upload took too long. Please retry on a stable connection.") from exc
            received += len(message.get("body", b""))
            if received > limit:
                exceeded = True
                # The multipart parser closes its temporary files on this error.
                raise MultiPartException("This upload exceeds the maximum allowed size.")
            return message

        async def limited_send(message):
            if exceeded and message["type"] == "http.response.start":
                message = {**message, "status": 413}
            await send(message)

        await self.app(scope, limited_receive, limited_send)


def reserve_upload_storage(account_id: int, job_id: str, byte_limit: int) -> None:
    """Serialize admission across processes before receiving or copying a video."""
    now = time.time()
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        for row in con.execute("SELECT job_id FROM upload_leases WHERE expires_epoch < ?", (now,)):
            if not (OUTPUTS / row["job_id"] / "analysis-session.json").exists() and not con.execute(
                "SELECT 1 FROM fights WHERE job_id=?", (row["job_id"],),
            ).fetchone():
                con.execute("DELETE FROM analysis_usage WHERE job_id=?", (row["job_id"],))
        con.execute("DELETE FROM upload_leases WHERE expires_epoch < ?", (now,))
        leases = con.execute("SELECT * FROM upload_leases").fetchall()
        own_leases = [row for row in leases if row["account_id"] == account_id]
        paths: set[Path] = set()
        pending = set()
        for session in OUTPUTS.glob("*/analysis-session.json"):
            try:
                job = json.loads(session.read_text(encoding="utf-8"))
                if job.get("account_id") != account_id:
                    continue
                if job.get("status") in {"selecting", "queued", "running", "preparing"}:
                    pending.add(session.parent.name)
                if job.get("video_path"):
                    paths.add(Path(job["video_path"]).resolve())
            except (OSError, ValueError, TypeError):
                continue
        rows = con.execute(
            "SELECT video_path FROM fights WHERE profile_id=(SELECT profile_id FROM accounts WHERE id=?)",
            (account_id,),
        ).fetchall()
        paths.update(Path(row["video_path"]).resolve() for row in rows if row["video_path"])
        pending.update(row["job_id"] for row in own_leases)
        if len(pending) >= SETTINGS.max_pending_uploads:
            raise UploadCapacityError(429, "Finish your pending fight selections or analyses before uploading another video.")
        used = 0
        for path in paths:
            if path.parent != UPLOADS.resolve():
                continue
            for candidate in {path, path.with_name(f"{path.stem}_web.mp4")}:
                try:
                    used += candidate.stat().st_size
                except FileNotFoundError:
                    pass
        # Room for the original and its browser-compatible derivative.
        own_reserved = sum(row["reserved_bytes"] for row in own_leases)
        if used + own_reserved + byte_limit * 2 > SETTINGS.account_storage_bytes:
            raise UploadCapacityError(413, "Your private video storage is full. Delete an old fight video before uploading another.")
        # A transfer can temporarily occupy spool + original + derivative space.
        reserved = sum(row["reserved_bytes"] for row in leases)
        free = shutil.disk_usage(UPLOADS).free
        if free - reserved - byte_limit * 3 < int(SETTINGS.minimum_free_storage_gb * 1024**3):
            raise UploadCapacityError(507, "Fight uploads are temporarily paused while storage capacity is restored.")
        con.execute("INSERT INTO upload_leases VALUES(?,?,?,?)", (
            job_id, account_id, byte_limit * 3, now + SETTINGS.upload_timeout_seconds + 900,
        ))


def release_upload_storage(job_id: str) -> None:
    with connection() as con:
        con.execute("DELETE FROM upload_leases WHERE job_id=?", (job_id,))


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
