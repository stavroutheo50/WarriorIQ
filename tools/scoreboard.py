"""One table for how accurate the analysis is, against what people marked (2026-11-23).

    python tools/scoreboard.py                 # score what is on this machine
    python tools/scoreboard.py --fetch         # first get what is missing (see fetch() below)
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
  Movement     tools/movement_benchmark.py on the same tracks and answer keys
               as "Who is who": each fighter's floor point within a quarter
               of their height of the marked one, and the direction of each
               clear move toward or away from the opponent. Target: 95% of
               both. Time in the centre is not measured (no ring is marked).

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


# --- getting what is missing ------------------------------------------------------

HUMAN_STRIKE_LABELS = PROJECT_ROOT / "dataset" / "regression" / "human_strikes" / "2026-11-10.labels.json"


def fetch() -> None:
    """Everything the ground and strike rows need that can be made on this machine.

    1. The five labelled pankration bouts, from archive.org (CC BY 2.0), into
       dataset/public/grappling_clips/<bout>.mp4 - the item's own MP4, as the
       labels' README says.
    2. The pose tracks of the stock clips in dataset/stock_strikes/
       (tools/make_stock_strikes.py: downloads each clip and runs the real
       analysis, so it uses the GPU).
    3. The owner's labeller's marks on those clips as answer keys
       (tools/import_human_strikes.py), so they count as person-labelled.
    Each step skips what is already there.
    """
    import subprocess
    import urllib.request

    from tools import fit_ground_time

    fit_ground_time.CLIPS.mkdir(parents=True, exist_ok=True)
    for labels in sorted(fit_ground_time.LABELS.glob("*.json")):
        target = fit_ground_time.CLIPS / f"{labels.stem}.mp4"
        if target.exists():
            continue
        item = json.loads(labels.read_text(encoding="utf-8"))["archive_item"]
        with urllib.request.urlopen(f"https://archive.org/metadata/{item}", timeout=60) as response:
            files = [f for f in json.load(response).get("files", []) if str(f.get("name", "")).lower().endswith(".mp4")]
        own = [f for f in files if f["name"] == f"{item}.mp4"]
        pick = own[0] if own else max(files, key=lambda f: int(f.get("size") or 0))
        print(f"  downloading {labels.stem} ({int(pick.get('size') or 0) // 1_000_000} MB)", flush=True)
        partial = target.with_suffix(".part")
        url = f"https://archive.org/download/{item}/{urllib.request.quote(pick['name'])}"
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(target)

    stock = PROJECT_ROOT / "dataset" / "stock_strikes"
    missing = sorted(p.name for p in stock.iterdir()
                     if p.is_dir() and (p / "source.json").exists() and not (p / "track.jsonl.gz").exists())
    if missing:
        print(f"  rebuilding pose tracks for {len(missing)} stock clip(s); this runs the analysis", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "make_stock_strikes.py"), *missing],
                       cwd=PROJECT_ROOT, check=True)
    if HUMAN_STRIKE_LABELS.exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "import_human_strikes.py"),
                        str(HUMAN_STRIKE_LABELS)], cwd=PROJECT_ROOT, check=True)


# --- movement --------------------------------------------------------------------

def movement_section() -> dict:
    """Where each fighter stood and which way they moved (tools/movement_benchmark.py).

    Read from the same tracks and answer keys as "Who is who": each fighter's
    box is marked there by a person every few seconds (2026-11-25).
    """
    from tools import run_identity_benchmark as runner
    from tools.movement_benchmark import score

    place, direction, clips = {"right": 0, "frames": 0}, {"right": 0, "moves": 0}, {}
    for path in runner.TRUTH:
        tracking = RUNS / path.stem / "tracking.jsonl"
        if not tracking.exists():
            continue
        records = [json.loads(line) for line in tracking.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(records, json.loads(path.read_text(encoding="utf-8")))
        clips[path.stem] = result
        for key in place:
            place[key] += result["place"][key]
        for key in direction:
            direction[key] += result["direction"][key]
    return movement_verdict(place, direction, clips)


def movement_verdict(place: dict, direction: dict, clips: dict | None = None) -> dict:
    if not place["frames"]:
        return {"score": None, "passes": None, "clips": clips or {},
                "note": "no clip analysed here yet (use --run-identity)"}
    place_share = place["right"] / place["frames"]
    direction_share = direction["right"] / direction["moves"] if direction["moves"] else None
    shares = [share for share in (place_share, direction_share) if share is not None]
    score = round(min(shares), 4)
    return {"score": score, "passes": all(share >= TARGET for share in shares), "clips": clips or {},
            "place": place, "direction": direction,
            "note": (f"right place {place['right']}/{place['frames']} ({place_share:.0%}), right direction "
                     + (f"{direction['right']}/{direction['moves']} ({direction_share:.0%})" if direction_share is not None
                        else "no clear moves") + f" - {len(clips or {})} clips; time in the centre not measured")}


# --- the table --------------------------------------------------------------------

def build(run_identity: bool = False) -> dict:
    board = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "target": TARGET}
    for name, make in (("identity", lambda: identity_section(run_identity)), ("ground", ground_section),
                       ("strikes", strike_section)):
        try:
            board[name] = make()
        except Exception as exc:                  # noqa: BLE001 - one broken part must not hide the others
            board[name] = {"score": None, "passes": None, "note": f"could not score: {type(exc).__name__}: {exc}"}
    try:
        board["movement"] = movement_section()
    except Exception as exc:                      # noqa: BLE001
        board["movement"] = {"score": None, "passes": None, "note": f"could not score: {type(exc).__name__}: {exc}"}
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
    parser.add_argument("--fetch", action="store_true",
                        help="first download the ground-time bouts and rebuild the labelled strike clips' tracks")
    parser.add_argument("--run-identity", action="store_true",
                        help="re-analyse the identity clips with the current code first (slow; uses the GPU)")
    parser.add_argument("--save", action="store_true", help="keep this result in " + str(HISTORY.relative_to(PROJECT_ROOT)))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    previous = _last_saved()
    if args.fetch:
        fetch()
    board = build(args.run_identity)
    print(json.dumps(board, indent=2) if args.json else render(board, previous))
    if args.save:
        HISTORY.mkdir(parents=True, exist_ok=True)
        stamp = board["measured_at"].replace(":", "").replace("-", "")
        (HISTORY / f"{stamp}.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
