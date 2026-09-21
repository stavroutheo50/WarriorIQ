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


def trace_run(fight: dict, stride: int, sam_model: str | None = None,
              sam_backend: str | None = None) -> tuple[list, dict]:
    """One analysis, recording where each fighter was on every analysed frame.

    `sam_model` swaps the SAM2 checkpoint and `sam_backend` swaps the model
    that runs the sweep entirely ("sam2" or "edgetam"). Both are set here rather
    than by the caller because SETTINGS is a frozen dataclass whose defaults
    evaluate when core.config is first imported, three lines below - an
    environment variable set after that import is silently ignored, and the
    run would quietly measure the default while claiming to measure the swap.
    That is the same shape as the two measurement bugs in this module's own
    docstring: an A/B arm that was not actually switched on.
    """
    os.environ["WARRIORIQ_FORCE_STRIDE"] = str(int(stride))
    if sam_model:
        os.environ["WARRIORIQ_SAM_MODEL"] = sam_model
    if sam_backend:
        os.environ["WARRIORIQ_SAM_BACKEND"] = sam_backend

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
        # How much the track each fighter is being held on has actually moved
        # lately. A fighter crosses the ring; a cornerman outside the ropes and
        # a spectator in the foreground do not. This is the same measurement
        # _release_if_furniture makes before letting an identity go, read here
        # every frame instead of only at the six-second mark - so a bystander
        # held for a second and a half is visible rather than forgiven.
        spread = {}
        for side, state in (("a", self.a), ("b", self.b)):
            spread[side] = self._recent_spread(state.current_track_id, self.source_fps)
        trace.append({
            "frame": int(source_frame),
            "a_box": None if a is None else [float(v) for v in a.box],
            "b_box": None if b is None else [float(v) for v in b.box],
            "b_refusal": self.b.last_refusal,
            "a_refusal": self.a.last_refusal,
            "seen_a": seen_where_expected["a"],
            "seen_b": seen_where_expected["b"],
            "spread_a": spread["a"],
            "spread_b": spread["b"],
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


def score_against_labels(trace: list, labels_path: Path) -> None:
    """How often the box was on the right person, which coverage cannot say.

    Five outcomes, not two, because "held nothing" is right or wrong depending
    on whether the fighter was there to hold. Lumping those together is how a
    tracker that quietly follows a spectator scores well.
    """
    from core.identity import box_iou

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    by_frame = {t["frame"]: t for t in trace}
    unfilled = 0
    tally = {"right person": 0, "wrong person": 0, "missed a visible fighter": 0,
             "held a non-fighter": 0, "correctly held nothing": 0,
             "tracked past a missed detection": 0, "missed an undetected fighter": 0}
    for entry in payload.get("frames", []):
        tracked = by_frame.get(entry["frame"])
        if tracked is None:
            continue
        for side in ("a", "b"):
            answer = entry.get("fighter_%s" % side)
            if answer == "FILL":
                unfilled += 1
                continue
            box = tracked["%s_box" % side]
            if answer == "unboxed":
                # Visible, but the detector missed them. Holding a box here is
                # the pipeline doing its job - SAM2 guidance and guided pose
                # recovery exist for exactly this - so it cannot be scored as
                # holding a non-fighter. It cannot be scored as right either,
                # because there is no box to compare against.
                tally["tracked past a missed detection" if box is not None
                      else "missed an undetected fighter"] += 1
                continue
            if answer in (None, "absent"):
                tally["correctly held nothing" if box is None else "held a non-fighter"] += 1
                continue
            try:
                truth = entry["candidates"][int(answer)]
            except (TypeError, ValueError, IndexError):
                unfilled += 1
                continue
            if box is None:
                tally["missed a visible fighter"] += 1
            elif box_iou(np.asarray(truth, dtype=np.float32),
                         np.asarray(box, dtype=np.float32)) >= 0.5:
                tally["right person"] += 1
            else:
                tally["wrong person"] += 1

    judged = sum(tally.values())
    if not judged:
        print("\n  no labels filled in yet - %s still has FILL in it" % labels_path.name)
        return
    print("\n  scored against %d labelled answers%s:" % (
        judged, " (%d still unfilled)" % unfilled if unfilled else ""))
    for name, count in tally.items():
        print("     %-26s %4d  (%.1f%%)" % (name, count, 100.0 * count / judged))
    correct = (tally["right person"] + tally["correctly held nothing"]
               + tally["tracked past a missed detection"])
    print("     %-26s %4d  (%.1f%%)" % ("CORRECT", correct, 100.0 * correct / judged))
    print("  This is the number coverage was never able to give.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fight", required=True, help="key in tools/verified_seeds.json")
    parser.add_argument("--stride", type=int, default=3,
                        help="pinned, because an unpinned stride is chosen off a clock")
    parser.add_argument("--frames", type=int, default=8, help="tiles on the contact sheet")
    parser.add_argument("--out", default=None, help="where to write the sheet")
    parser.add_argument("--labels", default=None,
                        help="a filled-in file from tools/label_identity.py, to score against")
    parser.add_argument("--sam-model", default=None,
                        help="SAM2 checkpoint to use, e.g. facebook/sam2.1-hiera-tiny")
    parser.add_argument("--sam-backend", default=None, choices=["sam2", "edgetam"],
                        help="which model runs the continuous sweep")
    parser.add_argument("--trace-out", default=None,
                        help="write the per-frame trace as JSON, so a later "
                             "question about the gaps needs no second run")
    args = parser.parse_args()

    fight = load_fight(args.fight)
    print("fight %s  seeds A=%s B=%s at frame %d  stride %d" % (
        args.fight, fight["fighter_a"], fight["fighter_b"],
        fight["seed_frame"], args.stride), flush=True)
    print("  sam_model %s  backend %s" % (
        args.sam_model or "default", args.sam_backend or "default"), flush=True)
    print("  %s" % fight["note"], flush=True)

    trace, tracking = trace_run(fight, args.stride, args.sam_model, args.sam_backend)
    if args.trace_out:
        # A run costs about eighty seconds. Every "why was it missing there?"
        # asked afterwards used to cost another one, so the trace is worth
        # keeping whenever somebody asks for it.
        destination = Path(args.trace_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(
            {"fight": args.fight, "stride": args.stride,
             "tracking": {k: v for k, v in tracking.items() if not isinstance(v, (list, dict))},
             "frames": trace}, indent=1, default=str), encoding="utf-8")
        print("trace: %s" % destination)
    probe = cv2.VideoCapture(str(PROJECT_ROOT / fight["video"]))
    fps_hint = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
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

    # Held, but held on what? Coverage counts a frame the same whether the box
    # is on a fighter or on a cornerman outside the ropes, and this project has
    # been fooled by that three times: 98% on a seated spectator, 0.728 on a
    # seated coach, 0.994 on HD footage where the sheet showed a cornerman and
    # then a spectator's head.
    #
    # **There is no number here, and that is the finding.** Two automatic
    # measures were tried and both failed, in opposite directions:
    #
    #   * the manager's own _recent_spread, the reading the furniture guard
    #     uses. It will not answer about a track without history, and a freshly
    #     latched bystander is exactly that track - it judged 74 of 164 frames
    #     on the HD bout and reported 0% on a clip with two bystanders visible
    #     by eye. Blind precisely where it was needed.
    #
    #   * displacement of the tracked box over a second, which needs no
    #     history. It flagged 43% of fighter A's windows on the same clip,
    #     including frames where the sheet shows A correctly on the fighter.
    #     Muay Thai fighters circle and clinch; standing still for a second is
    #     ordinary fighting, not a bystander.
    #
    # Distinguishing "held a fighter" from "held someone who looks like one"
    # requires knowing which is which, and that is ground truth, not a proxy.
    # Until a labelled sample exists the contact sheet below is the instrument -
    # it is what caught all three cases above. Look at it.

    if args.labels:
        score_against_labels(trace, Path(args.labels))

    out = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / (
        "identity_%s_stride%d.png" % (args.fight, args.stride))
    contact_sheet(fight, trace, out, args.frames)
    print("\ncontact sheet: %s" % out)
    print("Look at it. Coverage has been wrong about this before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
