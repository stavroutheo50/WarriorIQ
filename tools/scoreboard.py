"""One table for how accurate the analysis is, against what people marked (2026-11-23).

    python tools/scoreboard.py                 # score what is on this machine
    python tools/scoreboard.py --run-identity  # first re-analyse the identity clips with the current code
    python tools/scoreboard.py --save          # also keep this result and show the change since the last one
    python tools/scoreboard.py --json

The owner's goal is 95% for every analysis. That is a measurement, not a
setting, so every change to the engine is judged by this table rather than by
eye. It joins the scorers that already exist and adds nothing of its own to
how anything is scored:

  Who is who   tools/identity_benchmark.py on dataset/regression/identity_*/,
               frames a person marked. Target: 95% of frames right, no swaps.
  Ground time  tools/fit_ground_time.py's leave-one-bout-out score on
               dataset/regression/grappling_pankration/. Target: 95% of seconds.
  Strikes      tools/benchmark_dense_strikes.py on every fight with every
               strike labelled. Each counted strike matched to a labelled one
               (same fighter, within the benchmark's window). Target: 95%
               precision and 95% recall, and each fighter's total within 10%
               (or 2 strikes) - on fights a PERSON labelled. Fights with only
               machine labels are shown, but never count toward the target:
               Claude's own strike labels found 41 of a person's 111 strikes.
  Movement     not measured yet: nothing marked to compare it with.

What it needs is not in Git (videos, pose tracks of identifiable people): a
part with nothing on this machine says so instead of showing a number. Run it
on the analysis PC, where the clips are.

Identity is scored from tracks made by --run-identity (or an earlier run) in
dataset/public/scoreboard_runs/<clip>/; ground time reads the pose and picture
caches tools/fit_ground_time.py keeps beside each bout's video.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = 0.95
TOTAL_TOLERANCE = 0.10          # a fighter's strike total within 10% ...
TOTAL_TOLERANCE_MIN = 2         # ... or within 2 strikes, whichever is larger
RUNS = PROJECT_ROOT / "dataset" / "public" / "scoreboard_runs"
HISTORY = PROJECT_ROOT / "dataset" / "public" / "scoreboard_history"


# --- who is who -----------------------------------------------------------------

def identity_section(run: bool = False) -> dict:
    from tools import run_identity_benchmark as runner
    from tools.identity_benchmark import score

    keys = ("right", "swapped", "partial", "other", "missing")
    clips, total = {}, {**{key: 0 for key in keys}, "frames": 0}
    for path in runner.TRUTH:
        truth = json.loads(path.read_text(encoding="utf-8"))
        out_dir = RUNS / path.stem
        if run:
            video = _identity_video(runner, truth)
            if video is None:
                clips[path.stem] = {"status": "video not on this machine"}
                continue
            os.environ.update({"WARRIORIQ_HARD_REALTIME": "0", "WARRIORIQ_UPLOADS_DIR": str(RUNS / "uploads"),
                               "WARRIORIQ_OUTPUTS_DIR": str(RUNS / "outputs"),
                               "WARRIORIQ_DB_PATH": str(RUNS / "scoreboard.sqlite3")})
            runner._analyse(video, truth, out_dir, runner.PHONE_SECONDS)
        tracking = out_dir / "tracking.jsonl"
        if not tracking.exists():
            clips[path.stem] = {"status": "not analysed here (use --run-identity)"}
            continue
        marked = dict(truth)
        if "file" in truth:                       # a phone clip: scored on its first PHONE_SECONDS, as measured before
            limit = float(truth["start_seconds"]) + runner.PHONE_SECONDS
            marked["frames"] = [f for f in truth["frames"] if float(f["time_seconds"]) <= limit]
        records = [json.loads(line) for line in tracking.read_text(encoding="utf-8").splitlines() if line]
        result = score(records, marked)
        row = {key: result["A"][key] + result["B"][key] for key in keys}
        row["frames"] = result["A"]["frames"] + result["B"]["frames"]
        clips[path.stem] = row
        for key in row:
            total[key] += row[key]
    return {"clips": clips, "total": total, **identity_verdict(total)}


def _identity_video(runner, truth: dict) -> Path | None:
    if "video" in truth:                          # the owner's fights: only where the file is
        found = [PROJECT_ROOT / v for v in truth["video"] if (PROJECT_ROOT / v).is_file()]
        return found[0] if found else None
    return runner._download(truth)                # public clips: fetched from archive.org


def identity_verdict(total: dict) -> dict:
    frames = total.get("frames", 0)
    if not frames:
        return {"score": None, "passes": None, "note": "no marked frames scored on this machine"}
    right = (total["right"] + total.get("partial", 0)) / frames
    return {"score": round(right, 4), "passes": right >= TARGET and total["swapped"] == 0,
            "note": f"{total['right'] + total.get('partial', 0)}/{frames} frames right, {total['swapped']} swapped, "
                    f"{total['other']} on another person, {total['missing']} missing"}


# --- ground time ------------------------------------------------------------------

def ground_section() -> dict:
    from tools import fit_ground_time

    bouts = fit_ground_time.available_bouts()
    if len(bouts) < 2:
        return {"bouts": {}, "score": None, "passes": None,
                "note": "needs at least two labelled bouts' videos in dataset/public/grappling_clips/"}
    rows = fit_ground_time.leave_one_out(bouts)
    return {"bouts": rows, **ground_verdict(rows["TOTAL"])}


def ground_verdict(total: dict) -> dict:
    if not total.get("seconds"):
        return {"score": None, "passes": None, "note": "no labelled seconds"}
    share = total["right"] / total["seconds"]
    return {"score": round(share, 4), "passes": share >= TARGET,
            "note": f"{total['right']}/{total['seconds']} seconds right (each bout judged by a model fitted "
                    f"without it); labels by Claude, not yet checked by a person"}


# --- strikes ----------------------------------------------------------------------

def strike_section() -> dict:
    from tools import benchmark_dense_strikes as dense

    fights = dense.dense_fights()
    if not fights:
        return {"fights": {}, "person": None, "machine": None, "score": None, "passes": None,
                "note": "no fight with every strike labelled and its pose track on this machine"}
    rows = dense.run(None)
    person = {f.name for f in fights if dense.labelled_by_person(f)}
    return strike_summary({name: row for name, row in rows.items() if name != "TOTAL"}, person)


def _within(counted: int, labelled: int) -> bool:
    return abs(counted - labelled) <= max(TOTAL_TOLERANCE_MIN, TOTAL_TOLERANCE * labelled)


def strike_summary(rows: dict, person: set[str]) -> dict:
    """Pooled precision and recall, and fighter totals, kept apart by who labelled the fight."""
    def pool(names):
        names = [n for n in names if n in rows]
        if not names:
            return None
        real = sum(rows[n]["real"] for n in names)
        counted = sum(rows[n]["counted"] for n in names)
        labelled = sum(rows[n]["labelled"] for n in names)
        totals = [(rows[n][f"counted_{f}"], rows[n][f"labelled_{f}"]) for n in names for f in ("A", "B")]
        return {"fights": len(names), "labelled": labelled, "counted": counted, "real": real,
                "precision": round(real / counted, 4) if counted else None,
                "recall": round(real / labelled, 4) if labelled else None,
                "totals_within": sum(1 for c, l in totals if _within(c, l)), "totals": len(totals)}

    by_person = pool(sorted(person))
    by_machine = pool(sorted(set(rows) - person))
    out = {"fights": {name: {**row, "labelled_by": "person" if name in person else "machine"}
                      for name, row in rows.items()},
           "person": by_person, "machine": by_machine}
    if not by_person:
        out.update(score=None, passes=None,
                   note="no fight labelled by a person on this machine; machine labels are not an answer key")
        return out
    precision, recall = by_person["precision"] or 0, by_person["recall"] or 0
    totals_share = by_person["totals_within"] / by_person["totals"] if by_person["totals"] else 0
    out.update(score=round(min(precision, recall), 4),
               passes=precision >= TARGET and recall >= TARGET and totals_share >= TARGET,
               note=f"precision {precision:.0%}, recall {recall:.0%}, fighter totals within 10% "
                    f"{by_person['totals_within']}/{by_person['totals']} - {by_person['fights']} person-labelled "
                    f"fight(s), {by_person['labelled']} strikes")
    return out


# --- the table --------------------------------------------------------------------

def build(run_identity: bool = False) -> dict:
    board = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "target": TARGET}
    for name, make in (("identity", lambda: identity_section(run_identity)), ("ground", ground_section),
                       ("strikes", strike_section)):
        try:
            board[name] = make()
        except Exception as exc:                  # noqa: BLE001 - one broken part must not hide the others
            board[name] = {"score": None, "passes": None, "note": f"could not score: {type(exc).__name__}: {exc}"}
    board["movement"] = {"score": None, "passes": None,
                         "note": "not measured yet: needs marked positions to compare with"}
    try:
        from core.build_info import analysis_fingerprint
        board["analysis_fingerprint"] = analysis_fingerprint()
    except Exception:                             # noqa: BLE001
        pass
    return board


LABELS = {"identity": "Who is who", "ground": "Ground time", "strikes": "Strikes", "movement": "Movement"}


def render(board: dict, previous: dict | None = None) -> str:
    lines = [f"WarriorIQ analysis scoreboard - target {board['target']:.0%} - {board['measured_at']}"
             + (f" - analysis code {board['analysis_fingerprint']}" if board.get("analysis_fingerprint") else ""), ""]
    for key, label in LABELS.items():
        part = board[key]
        score = part.get("score")
        status = "PASS" if part.get("passes") else ("not measured" if part.get("passes") is None else "below")
        shown = f"{score:.1%}" if score is not None else "-"
        change = ""
        if previous and previous.get(key, {}).get("score") is not None and score is not None:
            delta = score - previous[key]["score"]
            change = f" ({delta:+.1%} since {previous['measured_at'][:10]})"
        lines.append(f"  {label:12} {shown:>7}  {status:12} {part.get('note', '')}{change}")
    machine = (board.get("strikes") or {}).get("machine")
    if machine:
        lines.append(f"  {'':12} {'':>7}  {'':12} machine-labelled fights, not counted: precision "
                     f"{machine['precision'] or 0:.0%}, recall {machine['recall'] or 0:.0%} on {machine['fights']}")
    return "\n".join(lines)


def _last_saved() -> dict | None:
    saved = sorted(HISTORY.glob("*.json")) if HISTORY.is_dir() else []
    return json.loads(saved[-1].read_text(encoding="utf-8")) if saved else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-identity", action="store_true",
                        help="re-analyse the identity clips with the current code first (slow; uses the GPU)")
    parser.add_argument("--save", action="store_true", help="keep this result in " + str(HISTORY.relative_to(PROJECT_ROOT)))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    previous = _last_saved()
    board = build(args.run_identity)
    print(json.dumps(board, indent=2) if args.json else render(board, previous))
    if args.save:
        HISTORY.mkdir(parents=True, exist_ok=True)
        stamp = board["measured_at"].replace(":", "").replace("-", "")
        (HISTORY / f"{stamp}.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
