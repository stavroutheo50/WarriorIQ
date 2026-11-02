"""Turn densely labelled fights into judged punch proposals for the punch check.

The video punch check (core/strike_verifier.py, fitted by
tools/fit_strike_check.py) learned from four boxing fights only, so on Muay
Thai it scored kick wind-ups, arm swings and clinch work as punches: on the
labelled MT980 stretch 20 of 25 counted strikes were false, most of them with
a punch probability between 0.5 and 0.9. The dense benchmark fights
(tools/benchmark_dense_strikes.py) already say, for every strike thrown, who
threw it and when - which is all a "yes"/"no" on a punch proposal needs.

    python tools/judge_from_dense_labels.py

For each dense fight with a video, every punch proposal the action path makes
(the same proposals the check scores in the analysis) is answered:

  yes      a sure labelled punch by the same fighter within MATCH_SECONDS;
  no       nothing labelled for that fighter within NOTHING_SECONDS - a motion
           the labeller saw was not a strike by them;
  unsure   anything else: an unsure label, a kick or knee by the same fighter
           (a strike, but not a punch), or a proposal inside an unreadable
           span or outside the labelled stretch. Not used for fitting.

Fights whose proposals were already judged one by one (V16L and the other
boxing fights) are skipped rather than added twice.

Written to dataset/public/strike_judged/<fight>D/ (judged.json, source.json and
the fight's track), which is outside Git like every other pose track. The
folder name starts with the fight's own name, so the benchmark's held-out head
(--held-out) never scores a fight with weights fitted on it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import benchmark_dense_strikes as dense  # noqa: E402
from tools.fit_strike_check import ROOT as JUDGED  # noqa: E402

MATCH_SECONDS = dense.MATCH_SECONDS
# Further than this from any strike of theirs, a proposal is a motion that was
# not a strike. Wider than the match window, so a slightly mistimed real punch
# is never answered "no".
NOTHING_SECONDS = 0.6


def answer(fighter: str, t: float, truth: dict) -> str:
    start, end = truth["covered"]
    if not start <= t <= end or dense._inside(t, truth.get("unreadable") or []):
        return "unsure"
    mine = [s for s in truth["strikes"] if s["fighter"] == fighter]
    close = [s for s in mine if abs(s["t"] - t) <= MATCH_SECONDS]
    if any(s.get("sure", True) and s.get("family") == "punch" for s in close):
        return "yes"
    if not any(abs(s["t"] - t) <= NOTHING_SECONDS for s in mine):
        return "no"
    return "unsure"


def judge(fight: Path) -> dict | None:
    source = fight / "source.json"
    if not source.exists():
        return None
    truth = json.loads((fight / "strikes.json").read_text(encoding="utf-8"))
    video = dense._local_video(fight)
    proposals = [e for e in dense._thrown(fight / "track.jsonl.gz") if e.family == "punch"]
    rows = []
    for event in proposals:
        peak = float(event.evidence.get("action_peak_time", event.peak_time))
        rows.append({"fighter": event.fighter, "limb": event.limb, "peak": round(peak, 3), "family": "punch",
                     "answer": answer(event.fighter, float(event.peak_time), truth),
                     "source": "dense_labels"})
    out = JUDGED / f"{fight.name}D"
    out.mkdir(parents=True, exist_ok=True)
    (out / "judged.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "source.json").write_text(json.dumps({"video": str(video.relative_to(PROJECT_ROOT))}), encoding="utf-8")
    shutil.copyfile(fight / "track.jsonl.gz", out / "track.jsonl.gz")
    return {"fight": fight.name, "yes": sum(r["answer"] == "yes" for r in rows),
            "no": sum(r["answer"] == "no" for r in rows), "unsure": sum(r["answer"] == "unsure" for r in rows)}


def judged_by_hand(fight: Path) -> bool:
    """A fight whose proposals were already answered one by one (V16L, ...)."""
    return any(p.name.startswith(fight.name) and p.name != f"{fight.name}D"
               for p in JUDGED.iterdir() if (p / "judged.json").exists())


def main() -> int:
    for fight in dense.dense_fights():
        if judged_by_hand(fight):
            continue
        result = judge(fight)
        if result:
            print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
