"""Delete the QA pass's test fighters and their fights from ONE account.

Audit 2026-10-14 (#32). A dry run unless --confirm is given: it first lists
exactly what it would delete, and touches nothing else.

    python tools/delete_qa_test_data.py --email owner@example.com
    python tools/delete_qa_test_data.py --email owner@example.com --confirm

What it matches, and nothing more:

  * the account with that exact email (required; one account per run);
  * fighters in that account whose name is EXACTLY one of QA_FIGHTER_NAMES
    (or the --name values given instead), compared case-insensitively after
    collapsing whitespace - never a substring, so "QA Test Boxer 2" or a real
    athlete called "Qasim" is never caught;
  * the fights filed under those fighters, with their report folder and
    uploaded video (the same files the site's own Delete removes).

Everything else in the account - other fighters, unfiled fights, settings,
billing - is left alone. Each deletion is written to security_events.

Run it against production only after reviewing the dry run, and keep a backup
of the database first. QA belongs in its own account from now on:
tools/seed_qa_account.py creates one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The names the QA pass created (audit 2026-10-14, #32), exactly as entered.
QA_FIGHTER_NAMES = (
    "QA Test Boxer",
    "QA Test MMA",
    "QA Test MT",
    "QA Test TKD",
    "QA <img src=x onerror=alert(1)> \"O'Neil\" & Ωmega",
)


def _key(name: str) -> str:
    return " ".join(str(name or "").split()).casefold()


def plan(email: str, names: tuple[str, ...] = QA_FIGHTER_NAMES) -> dict:
    """What a run would delete. Reads only."""
    from core import db

    account = db.get_account_by_email(email)
    if not account:
        raise SystemExit(f"No account with the email {email!r}. Nothing to do.")
    profile_id = int(account["profile_id"])
    wanted = {_key(name) for name in names}
    fighters = [f for f in db.list_fighters(profile_id, include_archived=True) if _key(f["name"]) in wanted]
    ids = {int(f["id"]) for f in fighters}
    fights = [f for f in db.list_fights(profile_id) if f.get("fighter_id") is not None and int(f["fighter_id"]) in ids]
    return {"account": account, "profile_id": profile_id, "fighters": fighters, "fights": fights}


def _fight_files(fight: dict) -> list[Path]:
    """The report folder and the uploaded video - only inside the data roots."""
    from core.config import OUTPUTS, UPLOADS

    files = []
    job_dir = (OUTPUTS / fight["job_id"]).resolve()
    if job_dir.parent == OUTPUTS.resolve() and job_dir.exists():
        files.append(job_dir)
    if fight.get("video_path"):
        video = Path(fight["video_path"]).resolve()
        if video.parent == UPLOADS.resolve():
            files.append(video)
    return files


def execute(found: dict) -> None:
    import shutil

    from core import db
    from core.video import remove_derivative

    account_id = int(found["account"]["id"])
    for fight in found["fights"]:
        files = _fight_files(fight)
        db.delete_fight(fight["job_id"])
        for path in files:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                remove_derivative(path)
                path.unlink(missing_ok=True)
        db.record_security_event("qa_test_data_deleted", account_id=account_id,
                                 resource_type="fight", resource_id=fight["job_id"])
    removed = db.delete_fighters(found["profile_id"], [int(f["id"]) for f in found["fighters"]])
    for fighter in removed:
        db.record_security_event("qa_test_data_deleted", account_id=account_id,
                                 resource_type="fighter", resource_id=str(fighter["id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True, help="the one account to clean")
    parser.add_argument("--name", action="append", default=[],
                        help="an exact fighter name to delete instead of the built-in QA list (repeatable)")
    parser.add_argument("--confirm", action="store_true", help="actually delete; without it nothing changes")
    args = parser.parse_args(argv)

    found = plan(args.email, tuple(args.name) or QA_FIGHTER_NAMES)
    print(f"Account: {found['account']['email']} (profile {found['profile_id']})")
    print(f"Fighters to delete ({len(found['fighters'])}):")
    for fighter in found["fighters"]:
        print(f"  #{fighter['id']:<6} {fighter['name']!r}{'  (archived)' if fighter.get('archived') else ''}")
    print(f"Fights to delete ({len(found['fights'])}):")
    for fight in found["fights"]:
        print(f"  {fight['job_id']:<14} {str(fight.get('created_at') or '')[:19]}  file={fight.get('original_name')!r}")
        for path in _fight_files(fight):
            print(f"      {path}")
    if not (found["fighters"] or found["fights"]):
        print("Nothing matches. Nothing was changed.")
        return 0
    if not args.confirm:
        print("\nDry run: nothing was changed. Run again with --confirm to delete exactly the rows above.")
        return 0
    execute(found)
    print(f"\nDeleted {len(found['fighters'])} fighter(s) and {len(found['fights'])} fight(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
