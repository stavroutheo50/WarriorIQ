"""Examine a strike model and write the per-sport verdict that can switch counts on.

    python tools/strike_exam.py --checkpoint models/warrioriq_temporal_best.pt \\
        --windows dataset/exam/tkd_kick3_test dataset/exam/boxingvi_test \\
        --official dataset/public/ufc_stats/ufc_fight_stats.csv \\
        --ufc-manifest dataset/exam/ufc_bouts.json \\
        --write

What it checks, and the bar, are in core/strike_exam.py. Two kinds of human
evidence, both required for a family of a sport to pass:

* ``--windows``: directories of held-out sequences (the .npz layout in
  dataset/README.md) that people labelled completely. A directory whose fight
  ids the model trained on, or was selected on (its validation fights), is
  refused, and so is any ``auto_`` fight: WarriorIQ's own labels are never an
  answer key.
* ``--ufc-manifest``: whole UFC bouts already analysed **by this same
  checkpoint**, each with the official per-round statistics row to compare
  against. A result made by any other model is refused.

    [{"result": "outputs/<job>", "bout": "A vs. B", "fighter_a": "A", "fighter_b": "B",
      "event": "optional event name"}]

Without ``--write`` it only prints. ``--write`` replaces
dataset/strike_exam_verdict.json; commit that file to switch counts on, and
deploy the same checkpoint to the worker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.strike_exam import BAR, CountCheck, VERDICT_PATH, decide, not_an_answer_key, window_checks  # noqa: E402

SPORTS_WITH_OFFICIAL_COUNTS = ("mma",)


class ExamRefused(RuntimeError):
    pass


def _fight_id(path: Path, data) -> str:
    return str(np.asarray(data["fight_id"]).item()) if "fight_id" in data else path.stem.split("__", 1)[0]


def load_windows(directories, excluded_fights: set[str]):
    """(x, true class index, fight id, source dir) for every held-out window; refuses leaks."""
    rows = []
    for directory in directories:
        directory = Path(directory)
        files = sorted(directory.glob("*.npz"))
        if not files:
            raise ExamRefused(f"{directory}: no .npz sequences")
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                fight = _fight_id(path, data)
                refused = not_an_answer_key(fight)
                if refused:
                    raise ExamRefused(f"{path}: {fight} is {refused}")
                if fight in excluded_fights:
                    raise ExamRefused(f"{path}: fight {fight} was used to train or select this model")
                rows.append((np.asarray(data["x"], dtype=np.float32), int(data["y"]), fight, directory.name))
    return rows


def load_model(checkpoint: Path):
    import torch

    from core.temporal_model import ACTION_CLASSES, build_temporal_network, checkpoint_sha256

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if list(payload.get("classes", ACTION_CLASSES)) != ACTION_CLASSES:
        raise ExamRefused("checkpoint classes do not match this build")
    model = build_temporal_network(payload.get("architecture", "gru_v1"), int(payload.get("input_dim", 102)),
                                   len(ACTION_CLASSES))
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()
    seen = set(payload.get("training_fights") or []) | set(payload.get("held_out_fights") or [])
    return model, seen, checkpoint_sha256(checkpoint)


def predict(model, rows, batch: int = 256) -> list[int]:
    """Argmax class per window.

    No probability threshold: in a fight a low-confidence call on a candidate
    the rules proposed still becomes a counted attempt (marked uncertain), so
    the exam counts it too rather than flattering precision.
    """
    import torch

    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            x = torch.from_numpy(np.stack([row[0] for row in rows[start:start + batch]]))
            out.extend(int(i) for i in model(x).argmax(dim=-1).tolist())
    return out


def count_check(manifest_path: Path, official_csv: Path, model_sha: str, line: str = "total") -> CountCheck:
    from core.official_stats import RoundLine, compare, load_official, official_rounds, warrioriq_rounds

    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = load_official(official_csv)
    key = "attempted_vs_total" if line == "total" else "attempted_vs_significant"
    rounds = []
    for entry in entries:
        result = Path(entry["result"])
        report = json.loads((result / "report.json").read_text(encoding="utf-8"))
        made_by = ((report.get("classifier") or {}).get("temporal_checkpoint_sha256"))
        if made_by != model_sha:
            raise ExamRefused(f"{result}: analysed by another model ({made_by}), not the one being examined")
        events = json.loads((result / "events.json").read_text(encoding="utf-8"))
        events = events.get("events", events) if isinstance(events, dict) else events
        # Rounds the analysis covered. A fighter with no strikes counted in one
        # of them was counted as zero - an undercount the exam must see, not a
        # round compare() would call missing and leave out.
        analysed = {int(r["number"]) for r in report.get("rounds") or [] if r.get("number") is not None}
        for side, name in (("A", entry["fighter_a"]), ("B", entry["fighter_b"])):
            official = official_rounds(rows, bout=entry["bout"], fighter=name, event=entry.get("event"))
            if not official:
                raise ExamRefused(f"no official rounds for {name} in {entry['bout']}")
            ours = warrioriq_rounds(events, side)
            for number in analysed:
                ours.setdefault(number, RoundLine())
            for row in compare(ours, official)["rounds"]:
                rounds.append((entry["bout"], int(row[key]["warrioriq"]), int(row[key]["official"])))
    return CountCheck("mma", tuple(rounds), f"ufc_stats:{line}")


def run(args) -> dict:
    from core.scoring import sport_counted_families
    from core.temporal_model import ACTION_CLASSES

    model, seen_fights, model_sha = load_model(Path(args.checkpoint))
    rows = load_windows(args.windows, seen_fights)
    guesses = predict(model, rows)
    pairs = [(ACTION_CLASSES[row[1]], ACTION_CLASSES[guess]) for row, guess in zip(rows, guesses)]
    sources: dict[str, set] = {}
    for row in rows:
        from core.strike_exam import family_of

        family = family_of(ACTION_CLASSES[row[1]])
        if family:
            sources.setdefault(family, set()).add(row[3])
    windows = window_checks(pairs, sources)
    counts = {}
    if args.ufc_manifest:
        if not args.official:
            raise ExamRefused("--ufc-manifest needs --official")
        counts["mma"] = count_check(Path(args.ufc_manifest), Path(args.official), model_sha, args.official_line)
    plural = {"punches": "punch", "kicks": "kick", "knees": "knee"}

    def scored(sport):
        return [plural[f] for f in sport_counted_families(sport) if f in plural]

    sports = sorted(set(args.sports or ()) | set(counts))
    return decide(windows, counts, sports, checkpoint_sha256=model_sha, scored_families=scored)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", nargs="+", required=True,
                        help="held-out, completely labelled sequence directories")
    parser.add_argument("--official", help="ufc_fight_stats.csv (dataset/public/ufc_stats)")
    parser.add_argument("--ufc-manifest", help="JSON list of analysed UFC bouts (see module docstring)")
    parser.add_argument("--official-line", choices=("total", "significant"), default="total",
                        help="which official strikes-attempted line to compare with (default: total)")
    parser.add_argument("--sports", nargs="*", default=list(SPORTS_WITH_OFFICIAL_COUNTS),
                        help="sports to write a verdict for (a sport without a count check cannot pass)")
    parser.add_argument("--write", action="store_true", help=f"write {VERDICT_PATH.relative_to(ROOT)}")
    parser.add_argument("--out", default=str(VERDICT_PATH))
    args = parser.parse_args(argv)
    try:
        verdict = run(args)
    except ExamRefused as refusal:
        print(f"Exam refused: {refusal}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2))
    passed = {sport: entry["passed_families"] for sport, entry in verdict["sports"].items()}
    print(f"\nBar: {BAR}\nPassed: {passed or 'nothing'}", file=sys.stderr)
    if args.write:
        Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
