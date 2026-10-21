"""Owner's requests and the bug they reported, 20 Oct 2026.

- a large (chunked) upload named the opponent "annotation=str required=False ..."
- the share button only says Share, and the share box has an X
- the rules are asked next to the video, not under "More options"
- update-worker updates a Git copy with Git
"""

from __future__ import annotations

import inspect
from pathlib import Path

from browser_client import BrowserClient

import app.main as web
from core import db as database

ROOT = Path(__file__).resolve().parents[1]


def test_a_chunked_upload_passes_every_form_field_to_the_upload():
    source = inspect.getsource(web.chunked_upload_finish)
    call = source[source.index("await upload("):]
    fields = [name for name, parameter in inspect.signature(web.upload).parameters.items()
              if name not in {"request", "background", "video"}]
    missing = [name for name in fields if f"{name}=" not in call]
    assert not missing, missing


def test_the_leaked_default_is_never_a_name():
    from fastapi import Form

    assert database.clean_fight_name(Form("")) is None
    assert database.clean_fight_name("annotation=str required=False alias='opponent_name'") is None
    assert database.clean_fight_name("  Nikos  ") == "Nikos"


def test_a_stored_leaked_name_is_cleared_on_start():
    import uuid

    from core.auth import hash_password

    account = database.create_account(f"own20-{uuid.uuid4().hex[:8]}@example.com", hash_password("Strong-Local-Password"))
    profile_id = int(database.get_account_by_email(account["email"])["profile_id"])
    job = uuid.uuid4().hex[:12]
    database.save_fight(job, profile_id, "a.mp4", "", "", "sparring", "K1", "A", {}, "2999-01-01T00:00:00+00:00")
    with database.connection() as con:
        con.execute("UPDATE fights SET opponent_name=?, event_name=? WHERE job_id=?",
                    ("annotation=str required=False", "annotation=str", job))
    database._schema_ready_for = None    # as on the next start of the app
    database.init_db()
    with database.connection() as con:
        row = con.execute("SELECT opponent_name, event_name FROM fights WHERE job_id=?", (job,)).fetchone()
    assert tuple(row) == (None, None)


def test_the_share_button_just_says_share_and_the_box_has_an_x():
    page = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    assert "Share to story" not in page and "Share to your story" not in page
    assert "data-share-card-open>Share</button>" in page
    assert 'class="share-card-x" type="button" data-share-card-close aria-label="Close"' in page
    script = (ROOT / "app" / "static" / "share_card.js").read_text(encoding="utf-8")
    assert 'querySelectorAll("[data-share-card-close]")' in script
    assert "event.target === dialog" in script


def test_the_rules_are_asked_next_to_the_video():
    import uuid

    client = BrowserClient(web.app)
    client.post("/signup", data={"email": f"own20-{uuid.uuid4().hex[:8]}@example.com",
                                 "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_confirmed": "true"}, follow_redirects=False)
    text = client.get("/analyze/kickboxing").text
    assert text.index('id="fightVideo"') < text.index('id="fightRuleset"') < text.index('name="rights_confirmed"')
    assert "rulesSummary" not in text


def test_update_worker_uses_git_in_a_git_copy():
    script = (ROOT / "deploy" / "update-worker.ps1").read_text(encoding="utf-8")
    assert "if (Test-Path (Join-Path $root '.git'))" in script
    assert "git -C $root merge --ff-only --quiet origin/main" in script
    assert "git -C $root stash push" in script


def test_the_replay_goes_full_screen_with_the_skeletons():
    page = (ROOT / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
    video = page[page.index('<video id="video"'):page.index("</video>")]
    # An iPhone plays inline (its own player has no skeletons), and the video's
    # own full-screen button, which shows it without them, is hidden.
    assert "playsinline" in video and 'controlslist="nofullscreen"' in video
    assert 'id="replayFs"' in page
    assert "wrap.requestFullscreen" in page and "wrap.classList.add('replay-theatre')" in page
    assert "playFullButton.onclick=()=>{enterBig();playAt(" in page
    css = (ROOT / "app" / "static" / "components.css").read_text(encoding="utf-8")
    assert ".replay-wrap.replay-theatre{position:fixed;inset:0" in css


def test_the_phone_menu_is_grouped_and_marks_the_page_you_are_on():
    base = (ROOT / "app" / "templates" / "base.html").read_text(encoding="utf-8")
    menu = base[base.index('<div class="mobile-menu" id="mobileMenu"'):base.index("<main id=")]
    for heading in ("Your fights", "Account", "Help"):
        assert f'<p class="mobile-menu-heading">{heading}</p>' in menu
    assert '<a data-nav="camp" href="/camp">Fight Camp<small>Your progress over time</small></a>' in menu
    assert ">Plans and billing<" not in menu      # it sat beside "Plans"; now "Billing"
    assert "document.querySelectorAll(`[data-nav=\"${active}\"]`)" in base
    css = (ROOT / "app" / "static" / "product.css").read_text(encoding="utf-8")
    assert ".mobile-menu{max-height:calc(100dvh - 80px);overflow-y:auto" in css


def test_each_kickboxing_ruleset_says_what_it_allows():
    from core.scoring import RULESETS, ruleset_hint

    assert ruleset_hint("K1") == "Full power, low kicks allowed, knees allowed."
    assert ruleset_hint("POINT_FIGHTING") == "Light contact, stops after each point, no low kicks, no knees."
    # Built from the profile's own flags, so it cannot claim what scoring does not do.
    for key, profile in RULESETS.items():
        hint = ruleset_hint(key)
        if profile.sport != "kickboxing":
            assert hint == ""
            continue
        assert ("knees allowed" in hint) == profile.allow_knee
        assert ("low kicks allowed" in hint) == profile.allow_low_kick
    page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
    assert 'id="rulesetHint"' in page and "Not sure? Leave" in page


def test_the_six_numbers_say_good_or_bad_only_from_the_catalogue():
    from core.metric_catalog import BY_KEY

    page = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    for key in ("footwork_body_lengths_per_second", "pressure_index", "ring_center_control",
                "guard_index", "balance_index", "pose_coverage"):
        assert f'verdict("{key}"' in page
    assert BY_KEY["guard_index"].reading(0.70)["standing"] == "doing_well"
    assert BY_KEY["ring_center_control"].reading(0.77)["standing"] == "no_reference"
    assert "Style, not a score" in page and "numbers.state != 'unverified'" in page
