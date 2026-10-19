"""Processing time divided by video length, for several clip lengths at once.

    python tools/benchmark_realtime.py                      # 24 s, 60 s and 180 s reference clips
    python tools/benchmark_realtime.py --lengths 24 60      # chosen lengths
    python tools/benchmark_realtime.py --video fight.mp4 --lengths 24 60 180

The product promise (QA, 2026-10-14, #1) is that an analysis takes no longer
than the video it analyses: a ratio of 1.0 or less. This runs the real analysis
pipeline - the same `core.analyzer.analyze` a worker runs - on each length and
prints the ratio, plus where the time went (report["performance"]["stage_seconds"]).

Without --video the clips are generated from Ultralytics' bundled two-person
photo (zidane.jpg), gently panned, the same reference tests/test_realtime_budget.py
uses. They measure speed only: the boxes are not a verified seed and no
accuracy number should be read off this run.

The ratio is a GPU promise. On a machine without CUDA this still runs and
prints honest numbers, which will be above 1.0; the exit status is 1 whenever
any ratio is above 1.0, so it can gate a deploy on the worker's own hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reference_clip(path: Path, seconds: float, fps: int = 25) -> tuple[Path, list[float], list[float]]:
    """A panned two-person clip, and boxes around the two people on its first frame."""
    import cv2
    import numpy as np
    import ultralytics

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(int(round(seconds * fps))):
        shift = np.float32([[1, 0, int(10 * np.sin(index / 6))], [0, 1, 0]])
        writer.write(cv2.warpAffine(image, shift, (width, height), borderMode=cv2.BORDER_REFLECT))
    writer.release()
    # zidane.jpg is 1280x720: the two men stand left and right of centre.
    return path, [120.0, 40.0, 700.0, 720.0], [700.0, 40.0, 1150.0, 720.0]


def video_boxes(video: str) -> tuple[list[float], list[float]]:
    import cv2

    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return [w * 0.22, h * 0.28, w * 0.46, h * 0.92], [w * 0.70, h * 0.28, w * 0.92, h * 0.92]


def run_once(video: str, seconds: float, box_a: list[float], box_b: list[float], job_id: str) -> dict:
    from core import analyzer
    from core.types import AnalysisRequest

    started = time.perf_counter()
    report = analyzer.analyze(AnalysisRequest(
        video_path=video, fighter_a_box=box_a, fighter_b_box=box_b,
        ruleset="K1", fight_type="sparring", round_count=1, start_seconds=0.0,
        round_duration_seconds=max(10.0, seconds), end_seconds=seconds,
        job_id=job_id, profile_id=1, persist_result=False))
    wall = time.perf_counter() - started
    performance = report.get("performance") or {}
    analysed = float(((report.get("tracking") or {}).get("recording") or {}).get("analysed_seconds") or seconds)
    return {"video_seconds": round(analysed, 2), "wall_seconds": round(wall, 2),
            "ratio": round(wall / max(1e-6, analysed), 3),
            "stage_seconds": performance.get("stage_seconds"),
            "final_analysis_fps": performance.get("final_analysis_fps"),
            "final_imgsz": performance.get("final_imgsz")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="A real fight video; without it reference clips are generated")
    parser.add_argument("--lengths", type=float, nargs="+", default=[24.0, 60.0, 180.0])
    parser.add_argument("--json", type=Path, help="Also write the results here")
    args = parser.parse_args(argv)

    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no CUDA)"
    except Exception:                                               # noqa: BLE001
        device = "unknown"
    print(f"device: {device}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="wiq-bench-") as scratch:
        for seconds in args.lengths:
            if args.video:
                video, (box_a, box_b) = args.video, video_boxes(args.video)
            else:
                path, box_a, box_b = reference_clip(Path(scratch) / f"reference_{int(seconds)}s.mp4", seconds)
                video = str(path)
            result = run_once(video, seconds, box_a, box_b, job_id=f"benchmark_{int(seconds)}s")
            rows.append(result)
            print(f"{result['video_seconds']:>7.1f}s video  {result['wall_seconds']:>8.1f}s processing  "
                  f"ratio {result['ratio']:.2f}  ({'within' if result['ratio'] <= 1.0 else 'OVER'} video length)",
                  flush=True)
            print(f"           stages: {result['stage_seconds']}", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"device": device, "results": rows}, indent=1), encoding="utf-8")
    return 0 if all(row["ratio"] <= 1.0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
