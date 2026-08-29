from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_validation import audit_dataset_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit WarriorIQ temporal sequences before training or release evaluation."
    )
    parser.add_argument("--development", default="dataset/sequences")
    parser.add_argument("--untouched-test", default="dataset/untouched_test")
    parser.add_argument(
        "--require",
        choices=("none", "experimental", "release"),
        default="none",
        help="Return a non-zero exit code when the selected readiness gate is not met.",
    )
    args = parser.parse_args()

    audit = audit_dataset_split(Path(args.development), Path(args.untouched_test))
    # Fingerprints are used internally for leakage checks. They add noise to a
    # human-readable audit and reveal nothing useful to the operator.
    public = json.loads(json.dumps(audit))
    public["development"].pop("sequence_fingerprints", None)
    public["untouched_test"].pop("sequence_fingerprints", None)
    print(json.dumps(public, indent=2))

    if args.require == "experimental" and not audit["development"]["experimental_train_ready"]:
        raise SystemExit(2)
    if args.require == "release" and not audit["release_data_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
