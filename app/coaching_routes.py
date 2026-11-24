"""Coaching and gyms: a coach or a gym sees an athlete's fights, with the athlete's consent.

Moved out of app/main.py unchanged (2026-11-24). Names shared with the rest of
the application are read from app.main when a request arrives (``m.name``), not
copied in at import, so anything that replaces one of them on app.main - the
tests do this - reaches this code too.

The routes are registered straight onto app.main's application when app.main
runs this module, at the spot they used to occupy, so the route table is
exactly what it was (see app/admin_routes.py for why not an APIRouter).
"""

from __future__ import annotations

from fastapi import Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import app.main as m

app = m.app


# ---- Coaching: a coach sees an athlete's fights with their consent ----------
#
# Owner's brief, 2026-11-14: "Coaches must only access athletes who have
# explicitly granted them permission." A coach invites an athlete by email; the
# athlete, signed in to their own account, accepts; from then on the coach can
# read that athlete's reports (read-only), and add notes and training
# objectives, until either of them ends it. Under-18 athletes cannot accept
# yet: that needs a parent or guardian's consent, which this does not collect.

COACH_INVITE_GONE = "This invitation has expired, was withdrawn or was already answered. Ask the coach for a new one."


def _is_coach_account(profile: dict | None) -> bool:
    return str((profile or {}).get("account_type") or "athlete") == "coach"


def _coach_seats(request: Request, coach_profile_id: int) -> dict:
    """Athletes linked or invited, against the plan's roster seats."""
    used = len(m.coach_links_for_coach(coach_profile_id))
    return m.roster_capacity(m._request_plan(request), used)


@app.get("/coaching", response_class=HTMLResponse)
def coaching_page(request: Request, notice: str = "", error: str = ""):
    return _coaching_response(request, notice=notice, error=error)


def _coaching_response(request: Request, *, invite: str = "", notice: str = "", error: str = ""):
    """The coaching page; `invite` only from the invitation form's own response, never an address."""
    profile_id = m._profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/coaching", status_code=303)
    profile = m.get_profile(profile_id) or {}
    is_coach = _is_coach_account(profile)
    return m.templates.TemplateResponse(
        request=request, name="coaching.html",
        context={"request": request, "is_coach": is_coach,
                 "my_coaches": m.coach_links_for_athlete(profile_id),
                 "notes": m.list_coach_notes(profile_id),
                 "athletes": m.coach_links_for_coach(profile_id) if is_coach else [],
                 # Athletes this person can see because they coach in a gym.
                 "gym_athletes": m.gym_athletes_for_coach(profile_id),
                 "seats": _coach_seats(request, profile_id) if is_coach else None,
                 # The invitation link, shown once, right after it is made.
                 "invite_url": (f"{m._public_base(request).rstrip('/')}/coaching/invite/{invite}"
                                if invite and m.re.fullmatch(r"[A-Za-z0-9_-]{20,100}", invite) else None),
                 "notice": notice[:200], "error": error[:200]})


@app.post("/coaching/invite", dependencies=[Depends(m.require_csrf)])
def coaching_invite(request: Request, email: str = Form(...)):
    m._enforce_rate_limit(request, "coach-invite", 20, 3600)
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account:
        return RedirectResponse("/login?next=/coaching", status_code=303)
    if not _is_coach_account(m.get_profile(profile_id)):
        return RedirectResponse("/coaching?" + m.urlencode(
            {"error": "Set your account type to Coach on your athlete profile to invite athletes."}), status_code=303)
    email = email.strip().lower()
    if not m.re.fullmatch(r"[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}", email):
        return RedirectResponse("/coaching?" + m.urlencode({"error": "Enter the athlete's email address."}),
                                status_code=303)
    if email == str(account.get("email") or "").lower():
        return RedirectResponse("/coaching?" + m.urlencode({"error": "That is your own email address."}),
                                status_code=303)
    if not _coach_seats(request, profile_id)["can_add"]:
        return RedirectResponse("/coaching?" + m.urlencode(
            {"error": "Every athlete seat on your plan is taken. End a link or a pending invitation first."}),
            status_code=303)
    token = m.session_token()
    m.create_coach_invite(profile_id, email, m.token_digest(token))
    coach_name = (m.get_profile(profile_id) or {}).get("display_name") or "A coach"
    link = f"{m._public_base(request).rstrip('/')}/coaching/invite/{token}"
    sent = True
    try:
        m.deliver_email(email, f"{coach_name} wants to coach you on WarriorIQ",
                      f"{coach_name} has invited you to share your WarriorIQ fights with them.\n\n"
                      f"If you accept, they can read your fight reports (not change them) and leave notes and "
                      f"training objectives. You can end it at any time from Coaching in your account.\n\n"
                      f"See the invitation: {link}\n\nIt expires in 14 days. If you do not know this person, "
                      f"ignore this email.")
    except m.EmailNotSent:
        sent = False
    m.record_security_event("coach_invite_created", account_id=int(account["id"]), resource_type="coach_link",
                          metadata={"emailed": sent})
    notice = ("Invitation emailed. You can also send them this link yourself:" if sent
              else "Email is not set up, so send them this link yourself:")
    # The link is shown once, on the page that answers the form. It never
    # travels in an address: the front-end server's access log keeps the whole
    # request line, query string included (security review, PR #233).
    return _coaching_response(request, invite=token, notice=notice)


@app.get("/coaching/invite/{token}", response_class=HTMLResponse)
def coaching_invite_page(request: Request, token: str):
    link = m.coach_invite_by_token(m.token_digest(token))
    if not link:
        raise HTTPException(404, COACH_INVITE_GONE)
    profile_id = m._profile_id(request)
    if profile_id is None:
        return RedirectResponse(f"/login?next=/coaching/invite/{token}", status_code=303)
    coach = m.get_profile(int(link["coach_profile_id"])) or {}
    minor = m.social.is_minor_account(m.get_account_by_profile(profile_id))
    return m.templates.TemplateResponse(
        request=request, name="coaching_invite.html",
        context={"request": request, "token": token, "coach_name": coach.get("display_name") or "A coach",
                 "coach_handle": coach.get("handle"), "own_invite": int(link["coach_profile_id"]) == profile_id,
                 "minor": minor})


@app.post("/coaching/invite/{token}/answer", dependencies=[Depends(m.require_csrf)])
def coaching_invite_answer(request: Request, token: str, answer: str = Form(...)):
    m._enforce_rate_limit(request, "coach-invite-answer", 20, 3600)
    link = m.coach_invite_by_token(m.token_digest(token))
    if not link:
        raise HTTPException(404, COACH_INVITE_GONE)
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account:
        return RedirectResponse(f"/login?next=/coaching/invite/{token}", status_code=303)
    accept = answer == "accept"
    if accept and m.social.is_minor_account(account):
        raise HTTPException(403, "Under-18 accounts cannot share their fights with a coach yet.")
    if not m.answer_coach_invite(int(link["id"]), profile_id, accept):
        raise HTTPException(404, COACH_INVITE_GONE)
    m.record_security_event("coach_invite_accepted" if accept else "coach_invite_declined",
                          account_id=int(account["id"]), resource_type="coach_link", resource_id=str(link["id"]))
    notice = "Your coach can now see your fights." if accept else "Invitation declined."
    return RedirectResponse("/coaching?" + m.urlencode({"notice": notice}), status_code=303)


@app.post("/coaching/links/{link_id}/end", dependencies=[Depends(m.require_csrf)])
def coaching_end_link(request: Request, link_id: int):
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account or not m.end_coach_link(link_id, profile_id):
        raise HTTPException(404)
    m.record_security_event("coach_link_ended", account_id=int(account["id"]), resource_type="coach_link",
                          resource_id=str(link_id))
    return RedirectResponse("/coaching?" + m.urlencode({"notice": "Done. That link is ended."}), status_code=303)


def _linked_athlete(request: Request, athlete_profile_id: int) -> tuple[int, dict]:
    """The coach's id and the athlete's profile, or a 404 when they are not linked."""
    coach_id = m._profile_id(request)
    if coach_id is None or not m.coach_can_view(coach_id, athlete_profile_id):
        raise HTTPException(404)
    return coach_id, m.get_profile(athlete_profile_id) or {}


@app.get("/coaching/athletes/{athlete_profile_id}", response_class=HTMLResponse)
def coaching_athlete_page(request: Request, athlete_profile_id: int, notice: str = ""):
    coach_id, athlete = _linked_athlete(request, athlete_profile_id)
    fights = m.list_fights(athlete_profile_id)
    for fight in fights:
        fight["choice_label"] = m.fight_choice_label(fight.get("ruleset"), fight.get("created_at"),
                                                   fight.get("fight_type"), fight.get("fighter_name"))
        fight["report_failed"] = m._report_failed(fight["job_id"])
    objectives = [item for item in m.list_assignments(athlete_profile_id)
                  if item.get("assigned_by_profile_id") == coach_id]
    return m.templates.TemplateResponse(
        request=request, name="coaching_athlete.html",
        context={"request": request, "athlete": athlete, "athlete_id": athlete_profile_id, "fights": fights,
                 "notes": m.list_coach_notes(athlete_profile_id, coach_id), "objectives": objectives,
                 "notice": notice[:200]})


@app.get("/coaching/reports/{job_id}", response_class=HTMLResponse)
def coaching_report(request: Request, job_id: str):
    fight = m.get_fight(job_id)
    if not fight:
        raise HTTPException(404)
    coach_id, athlete = _linked_athlete(request, int(fight["profile_id"]))
    return m._read_only_report(request, job_id, "This report is not available.",
                             coach_view={"athlete_id": int(fight["profile_id"]),
                                         "athlete_name": athlete.get("display_name") or "Athlete"})


@app.post("/coaching/athletes/{athlete_profile_id}/notes", dependencies=[Depends(m.require_csrf)])
def coaching_add_note(request: Request, athlete_profile_id: int, body: str = Form(...), job_id: str = Form("")):
    m._enforce_rate_limit(request, "coach-notes", 60, 3600)
    coach_id, _ = _linked_athlete(request, athlete_profile_id)
    text = body.strip()
    if not text:
        return RedirectResponse(f"/coaching/athletes/{athlete_profile_id}", status_code=303)
    fight = m.get_fight(job_id) if job_id else None
    if job_id and (not fight or int(fight["profile_id"]) != athlete_profile_id):
        raise HTTPException(404)
    m.add_coach_note(coach_id, athlete_profile_id, text, job_id or None)
    return RedirectResponse(f"/coaching/athletes/{athlete_profile_id}?" + m.urlencode({"notice": "Note saved."}),
                            status_code=303)


@app.post("/coaching/athletes/{athlete_profile_id}/objectives", dependencies=[Depends(m.require_csrf)])
def coaching_add_objective(request: Request, athlete_profile_id: int, title: str = Form(...), detail: str = Form("")):
    m._enforce_rate_limit(request, "coach-objectives", 60, 3600)
    coach_id, _ = _linked_athlete(request, athlete_profile_id)
    title = title.strip()[:120]
    if not title:
        return RedirectResponse(f"/coaching/athletes/{athlete_profile_id}", status_code=303)
    m.add_assignment(athlete_profile_id, title, detail.strip()[:1000], assigned_by_profile_id=coach_id)
    return RedirectResponse(f"/coaching/athletes/{athlete_profile_id}?" + m.urlencode(
        {"notice": "Objective added. It shows in their Fight Camp missions."}), status_code=303)


# ---- Gyms: coaches and athletes under one organisation ---------------------
#
# Owner's brief, 2026-11-14. The owner invites coaches and athletes by email;
# each joins only by accepting while signed in. An athlete who joins lets the
# gym's coaches read their reports (app.main._linked_athlete via
# core.db.coach_can_view) until they leave or are removed.

GYM_INVITE_GONE = "This gym invitation has expired, was withdrawn or was already answered. Ask the gym for a new one."


def _gym_seats(request: Request, org_id: int) -> dict:
    """Athletes in the gym, active or invited, against the owner's plan."""
    athletes = [m for m in m.organisation_members(org_id) if m["role"] == "athlete"]
    return m.roster_capacity(m._request_plan(request), len(athletes))


@app.get("/gym", response_class=HTMLResponse)
def gym_page(request: Request, notice: str = "", error: str = ""):
    return _gym_response(request, notice=notice, error=error)


def _gym_response(request: Request, *, invite: str = "", notice: str = "", error: str = ""):
    """The gym page; `invite` only from the invitation form's own response, never an address."""
    profile_id = m._profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/gym", status_code=303)
    owned = m.organisation_owned_by(profile_id)
    return m.templates.TemplateResponse(
        request=request, name="gym.html",
        context={"request": request, "owned": owned,
                 "members": m.organisation_members(int(owned["id"])) if owned else [],
                 "seats": _gym_seats(request, int(owned["id"])) if owned else None,
                 "memberships": [m for m in m.memberships_of(profile_id) if m["role"] != "owner"],
                 "invite_url": (f"{m._public_base(request).rstrip('/')}/gym/invite/{invite}"
                                if invite and m.re.fullmatch(r"[A-Za-z0-9_-]{20,100}", invite) else None),
                 "notice": notice[:200], "error": error[:200]})


@app.post("/gym/create", dependencies=[Depends(m.require_csrf)])
def gym_create(request: Request, name: str = Form(...)):
    m._enforce_rate_limit(request, "gym-create", 5, 3600)
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account:
        return RedirectResponse("/login?next=/gym", status_code=303)
    if m.organisation_owned_by(profile_id):
        return RedirectResponse("/gym", status_code=303)
    name = name.strip()
    if not name:
        return RedirectResponse("/gym?" + m.urlencode({"error": "Give your gym a name."}), status_code=303)
    m.create_organisation(profile_id, name)
    m.record_security_event("gym_created", account_id=int(account["id"]), resource_type="organisation")
    return RedirectResponse("/gym?" + m.urlencode({"notice": "Gym created. Invite your coaches and athletes."}),
                            status_code=303)


@app.post("/gym/invite", dependencies=[Depends(m.require_csrf)])
def gym_invite(request: Request, email: str = Form(...), role: str = Form("athlete")):
    m._enforce_rate_limit(request, "gym-invite", 40, 3600)
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account:
        return RedirectResponse("/login?next=/gym", status_code=303)
    owned = m.organisation_owned_by(profile_id)
    if not owned:
        raise HTTPException(404)
    role = role if role in {"coach", "athlete"} else "athlete"
    email = email.strip().lower()
    if not m.re.fullmatch(r"[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}", email) or email == str(account.get("email") or "").lower():
        return RedirectResponse("/gym?" + m.urlencode({"error": "Enter the person's email address."}), status_code=303)
    # A coach place reads other people's reports, so it is safe only when the
    # invited person can prove the address, which needs email to work.
    if role == "coach" and m.email_settings_problem():
        return RedirectResponse("/gym?" + m.urlencode(
            {"error": "Coach invitations need email to be set up, so the coach can verify their address. "
                      "Invite them as an athlete for now, or set up email first."}), status_code=303)
    if role == "athlete" and not _gym_seats(request, int(owned["id"]))["can_add"]:
        return RedirectResponse("/gym?" + m.urlencode(
            {"error": "Every athlete seat on your plan is taken. Remove an athlete or an invitation first."}),
            status_code=303)
    token = m.session_token()
    m.create_org_invite(int(owned["id"]), role, email, m.token_digest(token))
    link = f"{m._public_base(request).rstrip('/')}/gym/invite/{token}"
    what = ("If you join as an athlete, the gym's coaches can read your fight reports (not change them). "
            if role == "athlete" else "As a coach you can read the reports of the gym's athletes. ")
    sent = True
    try:
        m.deliver_email(email, f"Join {owned['name']} on WarriorIQ",
                      f"{owned['name']} has invited you to join their gym on WarriorIQ as "
                      f"{'a coach' if role == 'coach' else 'an athlete'}.\n\n{what}"
                      f"You can leave at any time from Gym in your account.\n\nSee the invitation: {link}\n\n"
                      f"It expires in 14 days. If you do not know this gym, ignore this email.")
    except m.EmailNotSent:
        sent = False
    m.record_security_event("gym_invite_created", account_id=int(account["id"]), resource_type="organisation",
                          resource_id=str(owned["id"]), metadata={"role": role, "emailed": sent})
    notice = ("Invitation emailed. You can also send them this link yourself:" if sent
              else "Email is not set up, so send them this link yourself:")
    # Shown on this response, never in an address (see coaching_invite).
    return _gym_response(request, invite=token, notice=notice)


@app.get("/gym/invite/{token}", response_class=HTMLResponse)
def gym_invite_page(request: Request, token: str):
    invite = m.org_invite_by_token(m.token_digest(token))
    if not invite:
        raise HTTPException(404, GYM_INVITE_GONE)
    profile_id = m._profile_id(request)
    if profile_id is None:
        return RedirectResponse(f"/login?next=/gym/invite/{token}", status_code=303)
    account = m._account(request) or {}
    return m.templates.TemplateResponse(
        request=request, name="gym_invite.html",
        context={"request": request, "token": token, "org_name": invite["org_name"], "role": invite["role"],
                 "own_invite": int(invite["owner_profile_id"]) == profile_id,
                 "wrong_account": invite["role"] == "coach" and str(account.get("email") or "").strip().lower()
                 != str(invite.get("invite_email") or "").strip().lower(),
                 "minor": invite["role"] == "athlete" and m.social.is_minor_account(m.get_account_by_profile(profile_id))})


@app.post("/gym/invite/{token}/answer", dependencies=[Depends(m.require_csrf)])
def gym_invite_answer(request: Request, token: str, answer: str = Form(...)):
    m._enforce_rate_limit(request, "gym-invite-answer", 20, 3600)
    invite = m.org_invite_by_token(m.token_digest(token))
    if not invite:
        raise HTTPException(404, GYM_INVITE_GONE)
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account:
        return RedirectResponse(f"/login?next=/gym/invite/{token}", status_code=303)
    accept = answer == "accept"
    if accept and int(invite["owner_profile_id"]) == profile_id:
        raise HTTPException(409, "This is your own gym.")
    if accept and invite["role"] == "athlete" and m.social.is_minor_account(account):
        raise HTTPException(403, "Under-18 accounts cannot join a gym as an athlete yet.")
    # A coach's place reads other people's reports, so only the person it was
    # sent to can take it: a forwarded or leaked link is not enough. (Joining
    # as an athlete shares only the joiner's own fights.)
    if accept and invite["role"] == "coach" and \
            str(account.get("email") or "").strip().lower() != str(invite.get("invite_email") or "").strip().lower():
        raise HTTPException(403, "This coach invitation was sent to a different email address. "
                                 "Sign in with that account to accept it.")
    # And proven theirs: anyone can sign up with an address they do not own
    # while verification is optional, so matching text is not proof.
    if accept and invite["role"] == "coach" and not account.get("email_verified_at"):
        raise HTTPException(403, "Verify your email address first (Settings), then accept this coach invitation.")
    if not m.answer_org_invite(int(invite["id"]), profile_id, accept):
        raise HTTPException(404, GYM_INVITE_GONE)
    m.record_security_event("gym_invite_accepted" if accept else "gym_invite_declined",
                          account_id=int(account["id"]), resource_type="organisation",
                          resource_id=str(invite["org_id"]), metadata={"role": invite["role"]})
    notice = f"You joined {invite['org_name']}." if accept else "Invitation declined."
    return RedirectResponse("/gym?" + m.urlencode({"notice": notice}), status_code=303)


@app.post("/gym/members/{member_id}/end", dependencies=[Depends(m.require_csrf)])
def gym_end_membership(request: Request, member_id: int):
    profile_id = m._profile_id(request)
    account = m._account(request)
    if profile_id is None or not account or not m.end_org_membership(member_id, profile_id):
        raise HTTPException(404)
    m.record_security_event("gym_membership_ended", account_id=int(account["id"]), resource_type="org_member",
                          resource_id=str(member_id))
    return RedirectResponse("/gym?" + m.urlencode({"notice": "Done."}), status_code=303)
