from __future__ import annotations

import http.client
import json
import re
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from app.state import AnalysisRunLost


class RemoteWorkerError(RuntimeError):
    pass


class RemoteWorkerClient:
    """Small authenticated client for the WarriorIQ web/GPU boundary."""

    def __init__(self, base_url: str, token: str, worker_id: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.worker_id = worker_id

    def _request(self, method: str, path: str, payload: dict | None = None, *, timeout: float = 30.0):
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 409:
                raise AnalysisRunLost(detail) from exc
            raise RemoteWorkerError(f"Worker API returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteWorkerError(f"Worker API connection failed: {type(exc).__name__}") from exc
        return json.loads(body) if body else {}

    def heartbeat(self) -> None:
        self._request("POST", "/api/worker/heartbeat", {"worker_id": self.worker_id})

    def claim(self) -> dict | None:
        response = self._request("POST", "/api/worker/claim", {"worker_id": self.worker_id})
        return response.get("job")

    def progress(self, job_id: str, analysis_run_id: str, patch: dict) -> None:
        self._request("POST", f"/api/worker/jobs/{job_id}/progress", {
            "worker_id": self.worker_id,
            "analysis_run_id": analysis_run_id,
            "patch": patch,
        })

    def failed(self, job_id: str, analysis_run_id: str, error_code: str) -> None:
        self._request("POST", f"/api/worker/jobs/{job_id}/failed", {
            "worker_id": self.worker_id,
            "analysis_run_id": analysis_run_id,
            "error_code": error_code,
        })

    def download_video(self, job: dict, destination: Path) -> None:
        query = urllib.parse.urlencode({
            "worker_id": self.worker_id,
            "analysis_run_id": job["analysis_run_id"],
        })
        request = urllib.request.Request(
            f"{self.base_url}/api/worker/jobs/{job['job_id']}/video?{query}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 409:
                raise AnalysisRunLost(detail) from exc
            raise RemoteWorkerError(f"Video download returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise RemoteWorkerError(f"Video download failed: {type(exc).__name__}") from exc

    def complete(self, job_id: str, analysis_run_id: str, archive_path: Path) -> None:
        boundary = f"warrioriq-{uuid.uuid4().hex}"
        fields = {
            "worker_id": self.worker_id,
            "analysis_run_id": analysis_run_id,
        }
        parts = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )
        file_header = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"archive\"; "
            f"filename=\"worker-result.zip\"\r\nContent-Type: application/zip\r\n\r\n"
        ).encode()
        closing = f"\r\n--{boundary}--\r\n".encode()
        content_length = sum(map(len, parts)) + len(file_header) + archive_path.stat().st_size + len(closing)
        parsed = urllib.parse.urlsplit(self.base_url)
        target = f"{parsed.path.rstrip('/')}/api/worker/jobs/{job_id}/complete"
        connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(
            parsed.hostname, parsed.port,
            timeout=180,
            **({"context": ssl.create_default_context()} if parsed.scheme == "https" else {}),
        )
        try:
            connection.putrequest("POST", target)
            connection.putheader("Authorization", f"Bearer {self.token}")
            connection.putheader("Accept", "application/json")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            for part in parts:
                connection.send(part)
            connection.send(file_header)
            with archive_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(closing)
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")[:500]
            if response.status == 409:
                raise AnalysisRunLost(body)
            if response.status not in {200, 201}:
                raise RemoteWorkerError(f"Artifact upload returned HTTP {response.status}: {body}")
        except (http.client.HTTPException, TimeoutError, OSError) as exc:
            raise RemoteWorkerError(f"Artifact upload failed: {type(exc).__name__}") from exc
        finally:
            connection.close()


def send_magic_packet(mac: str, host: str, port: int = 9, timeout: float = 4.0) -> bool:
    """Wake a sleeping analysis machine over the network.

    The analysis GPU lives on a home connection behind NAT, so the queue cannot
    reach in and the machine cannot poll while it is asleep. A Wake-on-LAN magic
    packet is the one thing a sleeping network card still listens for: six 0xFF
    bytes followed by the target MAC repeated sixteen times, sent as UDP.

    Best effort by design. The fight is already queued durably and the scheduled
    drain still collects it, so a blocked port or a changed address must never
    fail an upload -- it only costs the visitor the wait until the next drain.
    """
    digits = re.sub(r"[^0-9a-fA-F]", "", mac or "")
    if len(digits) != 12 or not host:
        return False
    payload = b"\xff" * 6 + bytes.fromhex(digits) * 16
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(payload, (host, int(port)))
        return True
    except OSError:
        return False


def wake_remote_worker(wake_url: str, token: str, job_id: str, timeout: float = 8.0) -> bool:
    """Ask a scale-to-zero GPU to start and drain the queue.

    Best effort by design. The fight is already queued durably, so a failed or
    slow wake must never fail the uploader's request; the worker still picks the
    fight up on its next start either way.
    """
    if not wake_url:
        return False
    request = urllib.request.Request(
        wake_url,
        data=json.dumps({"job_id": job_id}, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def retry_heartbeat(client: RemoteWorkerClient, attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            client.heartbeat()
            return
        except RemoteWorkerError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(1.0 + attempt)
