"""Measure who identity actually followed, and show the frames that prove it.

Three separate measurement bugs in one afternoon produced three confident and
wrong conclusions about this pipeline, and none of them were bugs in the
pipeline:

  * two analyses in one process, so the second started on a card the first had
    not released and looked like a regression;
  * an A/B whose "off" arm was not off, because it patched a function the code
    no longer called;
  * seeds from a rule, which put fighter B on a seated coach for a whole bout -
    and then reported 0.728 coverage of him.

The last one is the worst, because the number moved in the *right* direction
while the tracking got worse. A man in a chair is trivially easy to keep
covered. Coverage alone cannot tell you anything, which is why this tool
always renders what it tracked.

So this fixes the whole class by construction:

  * seeds come from tools/verified_seeds.json, confirmed by eye;
  * the stride is pinned, because it is otherwise chosen off a wall clock and
    the same video can be sampled two different ways (see core/machine_profile);
  * one analysis per process, enforced by this being a script and not a loop;
  * and a contact sheet is written every time, so the next person can check
    rather than trust.

    tools/measure_identity.py --fight 5736 --stride 3
    tools/measure_identity.py --fight b883 --stride 3 --frames 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS_PATH = Path(__file__).resolve().parent / "verified_seeds.json"


def load_fight(name: str) -> dict:
    fights = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["fights"]
    if name not in fights:
        raise SystemExit("unknown fight %r - have %s" % (name, ", ".join(sorted(fights))))
    return fights[name]


def trace_run(fight: dict, stride: int) -> tuple[list, dict]:
    """One analysis, recording where each fighter was on every analysed frame."""
    os.environ["WARRIORIQ_FORCE_STRIDE"] = str(int(stride))

    from core import analyzer
    from core.identity import IdentityManager, box_iou
    from core.types import AnalysisRequest

    video = str(PROJECT_ROOT / fight["video"])
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    capture.release()
    duration = float(total) / max(1.0, float(fps))
    start_seconds = float(fight["seed_frame"]) / max(1.0, float(fps))

    trace: list[dict] = []
    original = IdentityManager.update

    def traced(self, people, source_frame, sam_guidance=None):
        before = dict(self.blocked_recovery)
        # Was anybody detected where this fighter was expected to be? That is
        # the question that splits the two failures, and counting heads does
        # not answer it: on 1.mp4 a seated coach is detected all bout with a
        # referee probability of 0.05, so "two people were available" reads as
        # an identity failure when the second fighter was never detected at
        # all. Measured, that mislabelled 207 of 276 frames.
        seen_where_expected = {}
        for side, state in (("a", self.a), ("b", self.b)):
            predicted = self._predicted_box(state)
            seen_where_expected[side] = None if predicted is None else any(
                box_iou(predicted, p.box) >= 0.3 for p in people)
        a, b = original(self, people, source_frame, sam_guidance=sam_guidance)
        after = dict(self.blocked_recovery)
        trace.append({
            "frame": int(source_frame),
            "a_box": None if a is None else [float(v) for v in a.box],
            "b_box": None if b is None else [float(v) for v in b.box],
            "b_refusal": self.b.last_refusal,
            "a_refusal": self.a.last_refusal,
            "seen_a": seen_where_expected["a"],
            "seen_b": seen_where_expected["b"],
            "blocked": [k for k, v in after.items() if v > before.get(k, 0)],
        })
        return a, b

    IdentityManager.update = traced
    try:
        report = analyzer.analyze(AnalysisRequest(
            video_path=video,
            fighter_a_box=[float(v) for v in fight["fighter_a"]],
            fighter_b_box=[float(v) for v in fight["fighter_b"]],
            ruleset="K1", fight_type="competition", round_count=1,
            start_seconds=start_seconds,
            round_duration_seconds=max(1.0, duration - start_seconds),
            job_id="measure_identity", profile_id=1, persist_result=False))
    finally:
        IdentityManager.update = original
    return trace, report.get("tracking", {})


def contact_sheet(fight: dict, trace: list, out: Path, count: int) -> None:
    """The frames themselves, evenly spaced, with whatever was tracked drawn on.

    Evenly spaced rather than chosen: picking the frames where it did well is
    how a tool flatters itself.
    """
    video = str(PROJECT_ROOT / fight["video"])
    picks = [trace[i] for i in np.linspace(0, len(trace) - 1, count).astype(int)] if trace else []
    capture = cv2.VideoCapture(video)
    tiles = []
    for entry in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, entry["frame"])
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        scale = max(1, int(round(900.0 / max(1, frame.shape[1]))))
        big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for key, colour, tag in (("a_box", (0, 220, 0), "A"), ("b_box", (0, 80, 255), "B")):
            box = entry.get(key)
            if not box:
                continue
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            cv2.rectangle(big, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(big, tag, (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        missing = " ".join(t for t, k in (("A?", "a_box"), ("B?", "b_box"))
                           if not entry.get(k))
        cv2.putText(big, "f%d %s" % (entry["frame"], missing), (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(big)
    capture.release()
    if not tiles:
        return
    width = min(t.shape[1] for t in tiles)
    height = min(t.shape[0] for t in tiles)
    tiles = [t[:height, :width] for t in tiles]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fight", required=True, help="key in tools/verified_seeds.json")
    parser.add_argument("--stride", type=int, default=3,
                        help="pinned, because an unpinned stride is chosen off a clock")
    parser.add_argument("--frames", type=int, default=8, help="tiles on the contact sheet")
    parser.add_argument("--out", default=None, help="where to write the sheet")
    args = parser.parse_args()

    fight = load_fight(args.fight)
    print("fight %s  seeds A=%s B=%s at frame %d  stride %d" % (
        args.fight, fight["fighter_a"], fight["fighter_b"],
        fight["seed_frame"], args.stride), flush=True)
    print("  %s" % fight["note"], flush=True)

    trace, tracking = trace_run(fight, args.stride)
    held_a = sum(1 for t in trace if t["a_box"])
    held_b = sum(1 for t in trace if t["b_box"])
    print("\nanalysed %d frames" % len(trace))
    print("  fighter A held %4d (%.1f%%)   report says %.3f" % (
        held_a, 100.0 * held_a / max(1, len(trace)),
        tracking.get("fighter_A_coverage", 0.0)))
    print("  fighter B held %4d (%.1f%%)   report says %.3f" % (
        held_b, 100.0 * held_b / max(1, len(trace)),
        tracking.get("fighter_B_coverage", 0.0)))

    for side in ("A", "B"):
        boxes = [t["%s_box" % side.lower()] for t in trace if t["%s_box" % side.lower()]]
        if not boxes:
            continue
        xs = [(b[0] + b[2]) / 2.0 for b in boxes]
        # A fighter crosses the mat. Something that never moves is furniture,
        # and this is the number that says so before coverage flatters it.
        print("  fighter %s travelled x %.0f..%.0f (%.0f px)%s" % (
            side, min(xs), max(xs), max(xs) - min(xs),
            "   <-- SUSPECT: barely moved" if max(xs) - min(xs) < 60 else ""))

    blocked: dict[str, int] = {}
    for entry in trace:
        for reason in entry["blocked"]:
            blocked[reason] = blocked.get(reason, 0) + 1
    if blocked:
        print("  recovery blocked by: %s" % ", ".join(
            "%s=%d" % kv for kv in sorted(blocked.items(), key=lambda kv: -kv[1])[:5]))

    # The split that decides who owns the problem, and the only one that has
    # ever held up here. A fighter missing while somebody was detected where
    # they were expected is identity losing a fighter it could see. A fighter
    # missing while nothing was detected there is not an identity failure at
    # all - there was nothing to assign, and no threshold in core/identity.py
    # can invent a detection.
    #
    # Measured on 1.mp4: of 241 frames where B was lost, nothing was detectable
    # at his position at any confidence down to 0.03 on 130 of them. Lowering
    # new_track_thresh from 0.30 to 0.20 would have recovered 8. This is a
    # detector-and-footage problem wearing an identity problem's clothes.
    for side in ("A", "B"):
        key = "%s_box" % side.lower()
        missing = [t for t in trace if not t[key]]
        if not missing:
            continue
        unseen = sum(1 for t in missing if t["seen_%s" % side.lower()] is False)
        seen = sum(1 for t in missing if t["seen_%s" % side.lower()] is True)
        print("  fighter %s missing on %d frames: %d had nobody detected where "
              "they were expected (not an identity failure), %d had somebody "
              "there and lost them anyway" % (side, len(missing), unseen, seen))

    out = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / (
        "identity_%s_stride%d.png" % (args.fight, args.stride))
    contact_sheet(fight, trace, out, args.frames)
    print("\ncontact sheet: %s" % out)
    print("Look at it. Coverage has been wrong about this before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
