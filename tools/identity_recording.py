"""Record what the identity manager saw on the benchmark clips, and replay it.

    python tools/identity_recording.py record                 # every benchmark clip
    python tools/identity_recording.py record --clips ma604
    python tools/identity_recording.py replay                 # score the recordings as they are
    python tools/identity_recording.py replay --set bystander_memory=true

Testing an identity change on the analysis PC costs a full benchmark run per
idea, and an analysis on another machine is not a substitute: on a CPU without
the pose refiner and the recovery step the tracker renumbered people so often
that ma604 scored 20 right of 50 against the PC's 92% (2026-10-06).

So this records, on the PC, every call the analysis makes on the
IdentityManager - the people detected on each frame with their boxes,
keypoints and colours, and the settings in force - into
dataset/public/identity_recordings/<clip>.pkl.gz. Replaying a recording
through the current core/identity.py takes seconds, on any machine, and scores
it against the frames a person marked, so an identity change can be tried
against the PC's own detections before the PC is asked for a confirming run.

A replay is close to the PC, not identical to it: the analysis also searches
where the manager expects a fighter, so a manager that decides differently
would have been shown slightly different crops. The confirming run decides.
The recordings are pickles; replay only ones made by this tool.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_identity_benchmark import PHONE_SECONDS, TRUTH, WORK, _download  # noqa: E402

RECORDINGS = WORK / "identity_recordings"
RECORDED_METHODS = ("__init__", "adopt_anchor", "prime_track_history", "update",
                    "apply_external_recovery", "apply_ai_assignment")


def safe_settings(settings) -> dict:
    """The numeric and on/off settings only.

    The identity manager reads nothing else, and the full settings hold
    secrets (API keys, the mail password, the session secret) that must never
    leave the machine in a recording.
    """
    return {key: value for key, value in dataclasses.asdict(settings).items()
            if isinstance(value, (bool, int, float))}


_ORIGINALS: dict = {}


def _remove_recorder() -> None:
    from core import identity

    for name, original in _ORIGINALS.items():
        setattr(identity.IdentityManager, name, original)
    _ORIGINALS.clear()


def _install_recorder() -> list:
    """Wrap IdentityManager so every call is appended to the returned list."""
    from core import identity

    calls: list = []
    for name in RECORDED_METHODS:
        original = _ORIGINALS.setdefault(name, getattr(identity.IdentityManager, name))

        def wrapper(self, *args, _name=name, _original=original, **kwargs):
            # A FighterState is recorded by name: replay hands the replayed
            # manager's own state back.
            recorded = ((args[0].name,) + tuple(args[1:])
                        if _name in ("adopt_anchor", "apply_external_recovery") else args)
            calls.append((_name, recorded, kwargs))
            return _original(self, *args, **kwargs)

        setattr(identity.IdentityManager, name, wrapper)
    return calls


def _record_child(job: dict) -> None:
    """Run one analysis in this process and write its recording (subprocess side)."""
    calls = _install_recorder()
    from core import analyzer
    from core.config import SETTINGS
    from core.types import AnalysisRequest

    analyzer.analyze(AnalysisRequest(
        video_path=job["video"], fighter_a_box=job["a"], fighter_b_box=job["b"],
        start_seconds=job["start"], selection_seconds=job["start"], end_seconds=job["end"], round_count=1,
        round_duration_seconds=job["end"] - job["start"], output_dir=job["out"], persist_result=False))
    out = Path(job["recording"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as handle:
        pickle.dump({"clip": job["clip"], "settings": safe_settings(SETTINGS), "calls": calls}, handle)
    print(f"recorded {len(calls)} identity calls to {out}", flush=True)


def record(clips: list[str] | None, phone_seconds: float = PHONE_SECONDS) -> list[Path]:
    runs = WORK / "identity_runs"
    os.environ.update({
        "WARRIORIQ_HARD_REALTIME": "0",
        "WARRIORIQ_UPLOADS_DIR": str(runs / "uploads"),
        "WARRIORIQ_OUTPUTS_DIR": str(runs / "outputs"),
        "WARRIORIQ_DB_PATH": str(runs / "benchmark.sqlite3"),
    })
    written = []
    for path in TRUTH:
        if clips and path.stem not in clips:
            continue
        truth = json.loads(path.read_text(encoding="utf-8"))
        video = _download(truth)
        start = float(truth["start_seconds"])
        end = (float(truth["end_seconds"]) if "file" not in truth
               else min(float(truth["end_seconds"]), start + phone_seconds))
        target = RECORDINGS / f"{path.stem}.pkl.gz"
        job = {"clip": path.stem, "video": str(video), "a": truth["fighter_a_box"], "b": truth["fighter_b_box"],
               "start": start, "end": end, "out": str(runs / "recorded" / path.stem), "recording": str(target)}
        code = "import json, sys\nfrom tools.identity_recording import _record_child\n_record_child(json.loads(sys.argv[1]))\n"
        print(f"recording {path.stem}", flush=True)
        subprocess.run([sys.executable, "-c", code, json.dumps(job)], cwd=PROJECT_ROOT, env=os.environ, check=True)
        written.append(target)
    return written


def load(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def replay(recording: dict) -> tuple[list[dict], object]:
    """Feed a recording through the current IdentityManager. Returns tracking records and the manager."""
    from core import identity

    manager = None
    out: dict[int, dict] = {}
    for name, args, kwargs in recording["calls"]:
        if name == "__init__":
            manager = identity.IdentityManager(*args, **kwargs)
        elif name == "adopt_anchor":
            manager.adopt_anchor(getattr(manager, args[0].lower()), *args[1:], **kwargs)
        elif name == "prime_track_history":
            manager.prime_track_history(*args, **kwargs)
        elif name == "update":
            a, b = manager.update(*args, **kwargs)
            frame = int(args[1] if len(args) > 1 else kwargs["source_frame"])
            out[frame] = {"A": a, "B": b}
        elif name == "apply_external_recovery":
            obs = manager.apply_external_recovery(getattr(manager, args[0].lower()), *args[1:], **kwargs)
            frame = int(args[3] if len(args) > 3 else kwargs["source_frame"])
            if obs is not None and frame in out:
                out[frame][args[0]] = obs
        elif name == "apply_ai_assignment":
            manager.apply_ai_assignment(*args, **kwargs)
    records = [{"source_frame": frame,
                "fighter_A": {"observation": None if v["A"] is None else {"box": [float(x) for x in v["A"].box]}},
                "fighter_B": {"observation": None if v["B"] is None else {"box": [float(x) for x in v["B"].box]}}}
               for frame, v in sorted(out.items())]
    return records, manager


def _apply_settings(values: dict) -> None:
    from core.config import SETTINGS

    for key, value in values.items():
        if hasattr(SETTINGS, key):
            object.__setattr__(SETTINGS, key, value)


def _parse(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def score_recording(recording: dict, overrides: dict | None = None) -> tuple[dict, object]:
    """Replay with the PC's settings (plus overrides) and score against the marked frames."""
    from tools.identity_benchmark import score

    _apply_settings(recording["settings"])
    _apply_settings(overrides or {})
    truth_path = next(p for p in TRUTH if p.stem == recording["clip"])
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    records, manager = replay(recording)
    if "file" in truth:                                   # a phone clip: only what was analysed
        last = records[-1]["source_frame"] if records else 0
        truth = {**truth, "frames": [f for f in truth["frames"] if int(f["source_frame"]) <= last]}
    result = score(records, truth)
    keys = ("right", "swapped", "partial", "other", "missing", "frames")
    return {key: result["A"][key] + result["B"][key] for key in keys}, manager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--clips", nargs="*")
    rep = sub.add_parser("replay")
    rep.add_argument("--clips", nargs="*")
    rep.add_argument("--set", nargs="*", default=[], help="SETTINGS overrides, name=value")
    args = parser.parse_args(argv)
    if args.mode == "record":
        record(args.clips)
        return 0
    overrides = {k: _parse(v) for k, v in (item.split("=", 1) for item in args.set)}
    totals: dict[str, int] = {}
    for path in sorted(RECORDINGS.glob("*.pkl.gz")):
        clip = path.name.split(".")[0]
        if args.clips and clip not in args.clips:
            continue
        result, manager = score_recording(load(path), overrides)
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + value
        print(f"{clip:6} right {result['right']}/{result['frames']}  partial {result['partial']}  "
              f"other {result['other']}  swapped {result['swapped']}  missing {result['missing']}  "
              f"refusals {dict(sorted(manager.rejections.items()))}")
    if totals:
        print(f"TOTAL  right {totals['right']}/{totals['frames']}  partial {totals['partial']}  "
              f"other {totals['other']}  swapped {totals['swapped']}  missing {totals['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
