# End-to-end fight regression evidence

WarriorIQ validates fight actions with human-reviewed labels; it never generates placeholder ground truth.

1. Review real analysis candidates in the existing evidence-review workflow.
2. Run `python tools/build_end_to_end_regression.py` to freeze the current reviewed set.
3. Run `python tools/promote_end_to_end_validation.py --annotations dataset/regression/private/manifest.json` to measure the release gate.
4. Use `--promote` only after the printed gate passes on an untouched, representative set.

The private manifest and source videos remain outside Git. Its hashes make later edits or asset substitutions detectable. A small local set is useful for regression checks but must not be presented as production validation until every release threshold passes.
