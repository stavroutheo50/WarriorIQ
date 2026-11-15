"""Profile photos and videos are served only to whoever may see the profile (2026-11-15).

They used to be saved in app/static/profile/ and served by the public static
mount at guessable names, whatever the profile's visibility - a minor's too.
"""

from __future__ import annotations

import uuid

import pytest

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session

JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


def _account(minor: bool = False):
    created = db.create_account(f"media-{uuid.uuid4().hex[:10]}@example.test", hash_password("Strong-Local-Password"))
    if minor:
        db.set_guardian_approval_status(int(created["id"]), "accepted")
    return created


def _client(created=None):
    client = BrowserClient(web.app)
    if created:
        client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return client


@pytest.fixture
def media(tmp_path, monkeypatch):
    new, legacy = tmp_path / "profile_media", tmp_path / "legacy"
    new.mkdir()
    legacy.mkdir()
    monkeypatch.setattr(web, "PROFILE_MEDIA", new)
    monkeypatch.setattr(web, "LEGACY_PROFILE_MEDIA", legacy)
    return new, legacy


def _with_photo(folder, created, stored_prefix, visibility="public"):
    profile_id = int(created["profile_id"])
    name = f"profile_{profile_id}.jpg"
    (folder / name).write_bytes(JPEG)
    db.update_profile(profile_id, "Fighter", f"{stored_prefix}{name}", None)
    db.set_public_profile(profile_id, f"h{uuid.uuid4().hex[:10]}", visibility, "", "")
    return f"/profile-media/{name}"


def test_a_public_profile_photo_is_served_to_anyone(media):
    owner = _account()
    url = _with_photo(media[0], owner, "/profile-media/")
    answer = _client().get(url)
    assert answer.status_code == 200 and answer.content == JPEG
    assert "no-store" in answer.headers["cache-control"]


def test_a_private_profile_photo_is_served_to_its_owner_and_accepted_followers_only(media):
    owner, stranger, follower = _account(), _account(), _account()
    url = _with_photo(media[0], owner, "/profile-media/", visibility="private")
    assert _client(owner).get(url).status_code == 200
    assert _client().get(url).status_code == 404
    assert _client(stranger).get(url).status_code == 404
    db.request_follow(int(follower["profile_id"]), int(owner["profile_id"]), accepted=False)
    assert _client(follower).get(url).status_code == 404               # pending is not accepted
    db.accept_follow(int(follower["profile_id"]), int(owner["profile_id"]))
    assert _client(follower).get(url).status_code == 200


def test_a_minors_photo_is_never_served_to_anyone_else(media):
    owner, stranger = _account(minor=True), _account()
    url = _with_photo(media[0], owner, "/profile-media/")
    assert _client(owner).get(url).status_code == 200
    assert _client(stranger).get(url).status_code == 404
    assert _client().get(url).status_code == 404


def test_an_older_upload_is_served_from_where_it_was_through_the_checked_route(media):
    owner, stranger = _account(), _account()
    url = _with_photo(media[1], owner, "/static/profile/", visibility="private")
    assert _client(owner).get(url).status_code == 200
    assert _client(stranger).get(url).status_code == 404


def test_the_old_public_address_and_unknown_names_serve_nothing(media):
    owner = _account()
    url = _with_photo(media[0], owner, "/profile-media/")
    name = url.rsplit("/", 1)[-1]
    client = _client(owner)
    assert client.get(f"/static/profile/{name}").status_code == 404
    assert client.get(f"/static/./profile/{name}").status_code == 404
    assert client.get("/profile-media/profile_999999999.jpg").status_code == 404
    assert client.get("/profile-media/..%2Fwarrioriq.sqlite3").status_code == 404
    assert client.get("/static/glass.css").status_code == 200         # the rest of /static still works


def test_the_pages_link_to_the_checked_route():
    assert web._profile_media_url("/static/profile/profile_7.jpg") == "/profile-media/profile_7.jpg"
    assert web._profile_media_url("/profile-media/profile_7.jpg") == "/profile-media/profile_7.jpg"
    assert web._profile_media_url(None) is None


def test_a_new_upload_is_saved_outside_the_public_folder(media):
    import cv2
    import numpy as np

    owner = _account()
    client = _client(owner)
    client.get("/profile")                                              # sets the form token cookie
    ok, encoded = cv2.imencode(".jpg", np.full((40, 40, 3), 120, np.uint8))
    assert ok
    saved = client.post("/profile", data={"display_name": "Fighter"},
                        files={"photo": ("me.jpg", encoded.tobytes(), "image/jpeg")},
                        headers={"X-CSRF-Token": client.cookies.get("warrioriq_csrf")}, follow_redirects=False)
    assert saved.status_code == 303, saved.text
    name = f"profile_{owner['profile_id']}.jpg"
    assert (media[0] / name).is_file() and not (media[1] / name).exists()
    assert db.get_profile(int(owner["profile_id"]))["photo_path"] == f"/profile-media/{name}"
    assert client.get(f"/profile-media/{name}").status_code == 200


def test_the_real_static_folder_never_serves_a_legacy_file_by_any_path():
    """Written where the old uploads really are, so the mount itself is tested."""
    from pathlib import Path

    folder = Path(web.ROOT) / "app" / "static" / "profile"
    existed = folder.exists()
    folder.mkdir(exist_ok=True)
    name = f"profile_{uuid.uuid4().int % 10**9}.jpg"
    (folder / name).write_bytes(JPEG)
    try:
        client = _client()
        for url in (f"/static/profile/{name}", f"/static/%2e%2e/static/profile/{name}",
                    f"/static/%2E%2E/static/profile/{name}", f"/static/./profile/{name}",
                    f"/static/profile/./{name}", f"/static/%2e/profile/{name}"):
            assert client.get(url).status_code == 404, url
            assert client.head(url).status_code == 404, url
        assert client.get("/static/glass.css").status_code == 200
    finally:
        (folder / name).unlink(missing_ok=True)
        if not existed:
            folder.rmdir()
