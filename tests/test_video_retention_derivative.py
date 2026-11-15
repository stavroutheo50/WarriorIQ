"""Deleting a fight's video deletes its browser-playable copy too (2026-11-15).

The 30-day retention sweep and the "delete original video" button removed
the original but left "<job>_web.mp4" behind. The fight stays saved, so the
abandoned-file sweep protects that copy for ever: footage kept after the
homepage and the privacy policy say it is gone.
"""

from __future__ import annotations

import time

import app.main as web
from app import state
from core import retention


def test_the_retention_sweep_deletes_the_playable_copy_too(monkeypatch, tmp_path):
    uploads, outputs = tmp_path / "uploads", tmp_path / "outputs"
    uploads.mkdir()
    outputs.mkdir()
    for module in (web, retention):
        monkeypatch.setattr(module, "UPLOADS", uploads)
        monkeypatch.setattr(module, "OUTPUTS", outputs)
    monkeypatch.setattr(state, "OUTPUTS", outputs)
    monkeypatch.setattr(state, "_jobs", {})
    original = uploads / "keptfight01.mp4"
    copy = uploads / "keptfight01_web.mp4"
    other = uploads / "otherfight1_web.mp4"
    for path in (original, copy, other):
        path.write_bytes(b"footage")
    expired = {"job_id": "keptfight01", "profile_id": 1, "video_path": str(original)}
    marked = []
    monkeypatch.setattr(web, "list_expired_fight_videos", lambda: [expired])
    monkeypatch.setattr(web, "mark_fight_video_deleted", lambda job_id, profile_id: marked.append(job_id))
    monkeypatch.setattr(web, "record_security_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(web, "list_all_fight_storage",
                        lambda: [{"job_id": "keptfight01"}, {"job_id": "otherfight1"}])
    monkeypatch.setattr(web, "_last_guest_cleanup", 0)
    monkeypatch.setattr(web, "_last_saved_video_cleanup", 0)

    web._run_periodic_maintenance(now=time.monotonic() + 7200)

    assert marked == ["keptfight01"]
    assert not original.exists() and not copy.exists()
    assert other.exists()                    # another saved fight's copy is untouched
