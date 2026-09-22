"""An upload that survives a phone connection, in pieces.

The problem
-----------
A phone films 1080p at 8-17 Mbps, so two minutes of fight is 140-260 MB, and
a single POST of that has nothing to resume from. Whatever goes wrong at any
point in the transfer costs the whole fight.

**How long that transfer takes is not knowable in advance.** Two measurements
of the same path to the live host, within an hour of each other: 183 KiB/s and
3.5 MiB/s. At the first, 260 MB is twenty-three minutes and exceeds
`upload_timeout_seconds`; at the second it is seventy seconds and comfortable.
Neither was taken from a phone on mobile data in a sports hall, which is the
connection this actually has to serve.

That spread is the argument. A design that only works at the fast end fails
for the people most likely to be at the slow end, and no threshold picked from
one sample would have told us which end a given upload is on. Splitting the
transfer removes the question: a dropped connection costs one chunk, and the
bytes already accepted stay accepted.

A body ceiling was the original reason for this work. It turned out to be
WarriorIQ's own `max_upload_bytes` rather than the host's - 134 MiB of request
body reaches the live application, which then answers on its own terms - so
the ceiling is liftable. Resumability is not, and that is what this is for.

The shape
---------
Three calls: `begin` reserves, `chunk` appends, `finish` hands the assembled
file to the pipeline that already exists. Each chunk is its own small request,
so the body ceiling stops being the thing that decides whether a fight can be
uploaded at all.

**The part file's size is the resume cursor.** There is no separate ledger of
which pieces arrived, because a ledger can disagree with the disk. A chunk is
appended only when its offset equals the current size; anything else is
answered with the true offset and the client seeks there. That makes a
resumed upload and a retried chunk the same operation.

**Sequential, not parallel.** The application is served by a2wsgi under
Passenger, which has a small fixed number of workers. Parallel chunks would
occupy all of them and starve the rest of the site for the length of an
upload. Appending in order also means no sparse file and no assembly pass.

**The lease is the session.** `upload_leases` already carries
(job_id, account_id, reserved_bytes, expires_epoch) and is already swept, so
a reservation that outlives one request is what the table was built for.
Every chunk pushes the expiry out, because otherwise a slow upload gets swept
out from under itself at the fifteen-minute mark - which is exactly the
upload this exists to support.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from core.config import SETTINGS, UPLOADS
from core.db import connection
from core.upload_security import FIGHT_VIDEO_EXTENSIONS

# Server-minted, and validated on every request that names one, because it is
# a path component. Never accept a client's idea of a job id.
JOB_ID = re.compile(r"^[0-9a-f]{12}$")

# The partial file and its sidecar. Both are keyed by job id, so the existing
# sweep in core/retention.py - which groups by `owning_stem` - already ages
# them out together with everything else belonging to an abandoned job.
PART_SUFFIX = ".part"
META_SUFFIX = ".part.json"


class ChunkedUploadError(RuntimeError):
    def __init__(self, status: int, detail: str, *, offset: int | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        # On a mismatch the client is told where the file actually ends, so a
        # retry needs no extra round trip to find out.
        self.offset = offset


@dataclass(frozen=True)
class UploadSession:
    job_id: str
    account_id: int
    suffix: str
    declared_bytes: int
    received_bytes: int
    original_name: str
    form: dict

    @property
    def complete(self) -> bool:
        return self.received_bytes >= self.declared_bytes


def part_path(job_id: str) -> Path:
    return UPLOADS / f"{job_id}{PART_SUFFIX}"


def meta_path(job_id: str) -> Path:
    return UPLOADS / f"{job_id}{META_SUFFIX}"


def _require_job_id(job_id: str) -> str:
    if not JOB_ID.match(job_id or ""):
        raise ChunkedUploadError(404, "No such upload.")
    return job_id


def suffix_for(filename: str) -> str:
    """The extension the finished file will carry.

    Checked at `begin` rather than at `finish`, so a format WarriorIQ cannot
    read is refused before a single byte crosses the wire instead of after all
    of them have.
    """
    suffix = Path(filename or "").suffix.lower() or ".mp4"
    if suffix not in FIGHT_VIDEO_EXTENSIONS:
        raise ChunkedUploadError(400, "Unsupported video format.")
    return suffix


def begin(job_id: str, account_id: int, *, filename: str, declared_bytes: int,
          form: dict) -> UploadSession:
    """Open a session. Admission has already run and holds the lease."""
    _require_job_id(job_id)
    suffix = suffix_for(filename)
    ceiling = min(SETTINGS.max_chunked_upload_bytes, SETTINGS.max_fight_bytes)
    if declared_bytes <= 0:
        raise ChunkedUploadError(400, "Tell WarriorIQ how large the video is.")
    if declared_bytes > ceiling:
        raise ChunkedUploadError(
            413,
            f"That video is {declared_bytes / 1048576:.0f} MB. WarriorIQ accepts "
            f"up to {ceiling / 1048576:.0f} MB. Trim it to a single round.")
    session = UploadSession(
        job_id=job_id, account_id=account_id, suffix=suffix,
        declared_bytes=declared_bytes, received_bytes=0,
        original_name=Path(filename or "fight.mp4").name, form=form)
    # Truncate rather than append: a job id is fresh, so anything already at
    # this path is debris from a previous life of the same id and must not
    # become the first bytes of somebody's fight.
    part_path(job_id).write_bytes(b"")
    _write_meta(session)
    return session


def _write_meta(session: UploadSession) -> None:
    meta_path(session.job_id).write_text(json.dumps({
        "job_id": session.job_id,
        "account_id": session.account_id,
        "suffix": session.suffix,
        "declared_bytes": session.declared_bytes,
        "original_name": session.original_name,
        "form": session.form,
        "opened_epoch": time.time(),
    }), encoding="utf-8")


def load(job_id: str, account_id: int) -> UploadSession:
    """The session, or a refusal. Ownership is checked here, every time."""
    _require_job_id(job_id)
    meta = meta_path(job_id)
    try:
        stored = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ChunkedUploadError(404, "No such upload.") from None
    # A job id is unguessable, but "unguessable" is not an authorisation
    # check, and this one decides which account's disk a body lands on.
    if int(stored.get("account_id", -1)) != int(account_id):
        raise ChunkedUploadError(404, "No such upload.")
    part = part_path(job_id)
    return UploadSession(
        job_id=job_id, account_id=account_id,
        suffix=str(stored.get("suffix") or ".mp4"),
        declared_bytes=int(stored.get("declared_bytes") or 0),
        received_bytes=part.stat().st_size if part.exists() else 0,
        original_name=str(stored.get("original_name") or "fight.mp4"),
        form=dict(stored.get("form") or {}))


def append(session: UploadSession, offset: int, body: bytes) -> int:
    """Append one chunk. Returns the new end of the file.

    The offset must be exactly where the file currently ends. A client that is
    behind gets told where to resume; a client that is ahead is refused, because
    writing past the end would leave a hole that later reads as silence.
    """
    part = part_path(session.job_id)
    current = part.stat().st_size if part.exists() else 0
    if offset != current:
        raise ChunkedUploadError(
            409, "That piece does not continue where the upload left off.",
            offset=current)
    if not body:
        return current
    if len(body) > SETTINGS.upload_chunk_bytes:
        raise ChunkedUploadError(413, "That piece is larger than the agreed chunk size.")
    if current + len(body) > session.declared_bytes:
        raise ChunkedUploadError(413, "This upload is larger than it said it would be.")
    _guard_disk(len(body))
    with part.open("ab") as handle:
        handle.write(body)
    return current + len(body)


def _guard_disk(incoming: int) -> None:
    """Never let an upload take the host below its free-space reserve.

    The single-request path checks this per megabyte as it copies; the chunked
    path checks it per chunk, which is the same guarantee at a coarser grain.
    """
    import shutil

    reserve = int(SETTINGS.minimum_free_storage_gb * 1024**3)
    try:
        free = shutil.disk_usage(UPLOADS).free
    except OSError:
        return
    if free - incoming <= reserve:
        raise ChunkedUploadError(
            507, "Fight uploads are temporarily paused while storage capacity "
                 "is restored.")


def finalise(session: UploadSession) -> Path:
    """Turn the finished part file into the video the pipeline expects."""
    part = part_path(session.job_id)
    size = part.stat().st_size if part.exists() else 0
    if size != session.declared_bytes:
        raise ChunkedUploadError(
            400,
            f"The upload is incomplete: {size} of {session.declared_bytes} bytes "
            "arrived.", offset=size)
    destination = UPLOADS / f"{session.job_id}{session.suffix}"
    part.replace(destination)
    meta_path(session.job_id).unlink(missing_ok=True)
    return destination


def discard(job_id: str) -> None:
    """Forget a session. Safe to call when there is nothing to forget."""
    if not JOB_ID.match(job_id or ""):
        return
    part_path(job_id).unlink(missing_ok=True)
    meta_path(job_id).unlink(missing_ok=True)


def extend_lease(job_id: str) -> None:
    """Push an upload's expiry out because it is still being uploaded.

    Without this a transfer slower than `upload_timeout_seconds` is swept
    while it is still running - and a transfer slower than fifteen minutes is
    precisely the one this whole path exists to carry.
    """
    if not JOB_ID.match(job_id or ""):
        return
    expires = time.time() + SETTINGS.upload_timeout_seconds
    with connection() as con:
        con.execute("UPDATE upload_leases SET expires_epoch=? WHERE job_id=?",
                    (expires, job_id))


class StoredUpload:
    """A file already on disk, offered where an UploadFile is expected.

    `finish` needs everything the single-request upload does after the bytes
    land - the container check, the scanner, the normaliser, the decoder, the
    limits, the job row - and that is two hundred lines of route. Rather than
    cut them out and risk the two paths drifting, the assembled file is handed
    to the same route, which recognises this type and skips only the copy it
    would otherwise make.

    Copying instead would mean writing a second 512 MB file on a shared host
    to move it a few inches.
    """

    def __init__(self, path: Path, filename: str):
        self.path = path
        self.filename = filename

    def digest(self) -> str:
        """The same sha256 _save_upload_limited computes while copying."""
        running = hashlib.sha256()
        with self.path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                running.update(chunk)
        return running.hexdigest()
