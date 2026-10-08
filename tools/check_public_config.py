"""Fail when a public legal page would print a placeholder instead of a fact.

QA, 2026-10-07: /subprocessors named the email provider "smtp"; no host,
operator or analysis provider was named; contacts were a Gmail address.
Run against the deployment's environment, as part of its build:

    python tools/check_public_config.py

Exits 1 and lists every problem, or exits 0 with "OK".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from core.legal import public_config_problems

    problems = public_config_problems()
    if not problems:
        print("OK  public legal configuration names real providers and contacts")
        return 0
    print("Public legal configuration is incomplete:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
