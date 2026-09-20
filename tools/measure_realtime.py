"""Does the analysis finish inside the video's own length, and does it know?

The product's stated promise - written into plan_for_budget's first line - is
that an analysis never takes longer than the video it is analysing. This times
a real run against that promise and prints what the planner BELIEVED alongside
what actually happened, because the interesting failure is the two disagreeing.

    python tools/measure_realtime.py <video> <seconds> [--sam-off]
                                    [--sam-model=<hf id>] [--stride=<n>]

<seconds> is passed as end_seconds, so it genuinely bounds the analysed span.
The ratio is still computed from what the run reports it covered, never from
what was asked for.

--sam-model swaps the SAM2 checkpoint for one run, which is the cheapest
speed experiment available: `facebook/sam2.1-hiera-tiny` is 39M parameters
against the small default, and SAM2 is over half of a short round's wall time.
build_sam2_video_predictor_hf takes the repo id straight through, so nothing
but this string changes.

--stride pins WARRIORIQ_FORCE_STRIDE, and **two runs cannot be compared
without it.** The planner otherwise picks the stride off a wall clock, so the
same fight and the same code give fighter B 0.26 on one run and 0.73 on the
next. An A/B without a pinned stride measures the clock, not the change.

Both are read before core.config is imported, which is not a style choice:
SETTINGS is a frozen dataclass whose defaults evaluate at that import, so an
environment variable set afterwards is silently ignored. --sam-off has always
done it this way; these follow it.

It reports:

    realtime ratio   wall seconds per video second. <= 1.0 keeps the promise.
    budget_reason    what the planner decided: on_track, sampling_reduced,
                     cannot_meet_budget_above_quality_floor, stride_pinned.
    budget_expected  what the planner PREDICTED. If this is True and the ratio
                     is above 1.0, the planner's cost model is wrong and the
                     report is telling the user something false.
    cost_source      configured / profile / measured - which number it planned
                     from. "profile" means a value stored from an earlier run
                     on this machine, which is what makes runs reproducible and
                     is also what goes stale when something else is using the
                     GPU.

Free VRAM is printed first, because a card that is already occupied is the
single largest cause of a run missing this budget, and nothing in the product
logs it today.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _flag(name: str) -> str | None:
    """Read `--name=value` from argv, or None.

    argparse would reject the positional-then-flags shape this tool has always
    had, and rewriting the interface would break whatever is already calling it.
    """
    prefix = f"{name}="
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return argument[len(prefix):].strip() or None
    return None


def _analysed_span(path: Path) -> float:
    """Seconds of video the run actually covered, read from its own trace."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])["time_seconds"]
        last = json.loads(lines[-1])["time_seconds"]
        return max(0.0, float(last) - float(first))
    except Exception:                                            # noqa: BLE001
        return 0.0


def vram() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA"
        free, total = torch.cuda.mem_get_info()
        return (f"{free/2**30:.2f} GB free of {total/2**30:.2f} GB "
                f"({(total-free)/total*100:.0f}% already in use)")
    except Exception as exc:                                     # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    video = sys.argv[1]
    seconds = float(sys.argv[2])
    sam_off = "--sam-off" in sys.argv
    sam_model = _flag("--sam-model")
    stride = _flag("--stride")

    # Everything that changes SETTINGS has to happen here, above the core
    # import below. See the note in the module docstring.
    import os
    if sam_off:
        os.environ["WARRIORIQ_SAM_RECOVERY"] = "0"
        os.environ["WARRIORIQ_SAM_CONTINUOUS"] = "0"
    if sam_model:
        os.environ["WARRIORIQ_SAM_MODEL"] = sam_model
    if stride:
        os.environ["WARRIORIQ_FORCE_STRIDE"] = str(int(stride))

    print(f"VRAM before: {vram()}", flush=True)
    # Printed so a saved run says what produced it. Two ratios compared without
    # these three lines beside them are not evidence of anything.
    print(f"sam_model  : {sam_model or 'default (see WARRIORIQ_SAM_MODEL)'}")
    print(f"stride     : {stride or 'ADAPTIVE - not comparable between runs'}")
    print(f"sam        : {'off' if sam_off else 'on'}", flush=True)

    import cv2

    from core import analyzer
    from core.types import AnalysisRequest

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Boxes matter for identity, not for speed. This measures speed, so they are
    # two plausible halves of the frame rather than a verified seed - do not
    # read any accuracy number off this run.
    a = [w * 0.22, h * 0.28, w * 0.46, h * 0.92]
    b = [w * 0.70, h * 0.28, w * 0.92, h * 0.92]

    start = time.perf_counter()
    report = analyzer.analyze(AnalysisRequest(
        video_path=video,
        fighter_a_box=[float(v) for v in a],
        fighter_b_box=[float(v) for v in b],
        ruleset="WT", fight_type="competition", round_count=1,
        start_seconds=0.0, round_duration_seconds=seconds,
        # end_seconds is what actually bounds the work.
        #
        # round_duration_seconds does NOT, and that is deliberate rather than a
        # bug: build_round_schedule extends the last round to the end of the
        # file whenever every round is selected, because a nine minute bout
        # entered as 3x2min once had three of its nine minutes silently thrown
        # away. Rounds decide where the round lines fall, not how much footage
        # is worth looking at. So a request for 70 seconds of a five minute file
        # analysed all 302.6 of it, correctly, and this tool reported 5.59x for
        # a run that was 1.29x.
        end_seconds=seconds,
        job_id="measure_realtime", profile_id=1, persist_result=False))
    wall = time.perf_counter() - start

    tracking = report.get("tracking") or {}
    performance = report.get("performance") or {}

    # Measure against what was ACTUALLY analysed, not what was asked for.
    #
    # round_duration_seconds does not bound the analysed segment the way it
    # looks like it should: a request for 70 seconds of a 5 minute file
    # analysed the whole 302.6 seconds. Dividing the wall time by the REQUESTED
    # duration reported 5.59x for a run that was really 1.29x - a tool that
    # overstates by 4x is worse than no tool, and it was my own arithmetic, not
    # the product's.
    analysed = float(tracking.get("recording", {}).get("analysed_seconds") or 0.0)
    if not analysed:
        analysed = _analysed_span(ROOT / "outputs" / "measure_realtime" / "tracking.jsonl")
    if not analysed:
        analysed = seconds
        print("  (could not read the analysed span; falling back to the request)")
    seconds = analysed
    ratio = wall / max(1e-6, seconds)
    print()
    print("=" * 58)
    print(f"  video analysed      {seconds:.0f}s at {fps:.2f}fps ({w}x{h})")
    print(f"  wall time           {wall:.0f}s")
    print(f"  REALTIME RATIO      {ratio:.2f}x   "
          f"({'KEEPS' if ratio <= 1.0 else 'MISSES'} the <= 1.0 promise)")
    print()
    # The budget fields live under report["performance"], not ["tracking"].
    for key in ("budget_plan", "budget_met_expected", "budget_cost_source",
                "budget_cost_planned_seconds", "budget_cost_observed_seconds",
                "planned_stride", "final_analysis_fps", "final_imgsz",
                "vram_free_at_start"):
        if key in performance:
            print(f"  {key:30} {performance[key]}")
    for key in ("analyzed_frames", "sam_continuous_frames"):
        if key in tracking:
            print(f"  {key:30} {tracking[key]}")
    print(f"  VRAM after          {vram()}")
    print("=" * 58)
    if performance.get("budget_met_expected") and ratio > 1.0:
        print("  !! The planner predicted it would meet the budget and it did")
        print("     not. Its cost model is wrong, and the report published that")
        print("     prediction to the user as though it were true.")
    Path(ROOT / "outputs").mkdir(exist_ok=True)
    out = ROOT / "outputs" / "realtime_probe.json"
    out.write_text(json.dumps({"wall_seconds": wall, "video_seconds": seconds,
                               "realtime_ratio": ratio, "tracking": tracking,
                               "performance": performance},
                              indent=1, default=str), encoding="utf-8")
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
