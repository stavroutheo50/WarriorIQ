"""Create a dedicated QA account with the QA test fighters (audit 2026-10-14, #32).

QA used to run in the owner's production account, so its test fighters
("QA Test Boxer" and the markup-in-a-name fighter) sat in a real roster. QA now
runs in an account of its own, created here:

    WARRIORIQ_QA_PASSWORD=... python tools/seed_qa_account.py --email qa@example.com

The password is read from WARRIORIQ_QA_PASSWORD and never printed or stored
anywhere but as a hash. Running it again is safe: an existing account keeps
its password and the fighters are reused, not duplicated. It refuses an
account holding any fight or any other fighter, so it cannot be pointed at a
real athlete's workspace by mistake.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.delete_qa_test_data import QA_FIGHTER_NAMES  # noqa: E402


def seed(email: str, password: str | None) -> dict:
    from core import db
    from core.auth import hash_password

    account = db.get_account_by_email(email)
    if account is None:
        if not password:
            raise SystemExit("Set WARRIORIQ_QA_PASSWORD (10+ characters) to create the QA account.")
        account = db.create_account(email, hash_password(password))
    profile_id = int(account["profile_id"])
    qa_names = {" ".join(name.split()).casefold() for name in QA_FIGHTER_NAMES}
    others = [f for f in db.list_fighters(profile_id, include_archived=True)
              if " ".join(f["name"].split()).casefold() not in qa_names]
    if others or db.list_fights(profile_id):
        raise SystemExit(f"{email} already holds fighters or fights; QA needs an account of its own.")
    fighters = [db.create_fighter(profile_id, name) for name in QA_FIGHTER_NAMES]
    return {"account": account, "fighters": [f for f in fighters if f]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True)
    args = parser.parse_args(argv)
    result = seed(args.email, os.getenv("WARRIORIQ_QA_PASSWORD"))
    print(f"QA account: {result['account']['email']} (profile {result['account']['profile_id']})")
    for fighter in result["fighters"]:
        print(f"  fighter #{fighter['id']}: {fighter['name']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
