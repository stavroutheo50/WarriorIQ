"""Repeat a real local analysis without publishing results or changing source footage."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StageTimers:
    """Exclusive wall costs for synchronous pipeline calls, local to this tool."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.rows = {}
        self.stack = []

    def wrap(self, function, label):
        @wraps(function)
        def measured(*args, **kwargs):
            frame = [self.clock(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = self.clock() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                row = self.rows.setdefault(label, {"calls": 0, "exclusive_seconds": 0.0})
                row["calls"] += 1
                row["exclusive_seconds"] += max(0, elapsed - frame[1])
        return measured

    def install(self, context, analyzer):
        from core.pose_tracker import PoseTracker
        from core.sam_recovery import SamRecovery
        from core.edgetam_recovery import EdgeTamRecovery
        targets = [
            (analyzer, "get_video_info", "video_metadata"),
            (analyzer, "get_pose_tracker", "pose_model_load"),
            (analyzer, "probe_video", "video_preflight"),
            (PoseTracker, "warmup", "pose_warmup"),
            (SamRecovery, "track_segment", "identity_guidance"),
            (EdgeTamRecovery, "track_segment", "identity_guidance"),
            (PoseTracker, "track", "pose_detection_tracking"),
            (PoseTracker, "recover_from_guidance", "pose_recovery"),
            (analyzer, "refine_fighter_pose", "pose_refinement"),
            (analyzer.ActionEngine, "update", "action_detection"),
            (analyzer, "classify_contact", "contact_classification"),
            (analyzer.MetricsAccumulator, "finalize", "statistics"),
            (analyzer, "build_report", "report_construction"),
            (analyzer, "write_report", "report_write"),
        ]
        for owner, name, label in targets:
            context.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), label)))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    paths = sorted((ROOT / "core").glob("*.py"))
    paths += [ROOT / "tools/benchmark_analysis.py", ROOT / "tools/verified_seeds.json"]
    paths += sorted((ROOT / "models").glob("*.yaml"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def timing_summary(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("At least one completed run is required")
    values = [float(run["wall_seconds"]) for run in runs]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Run durations must be positive and finite")
    warm = values[1:]
    return {
        "first_run_seconds": values[0],
        "warm_runs": len(warm),
        "warm_median_seconds": statistics.median(warm) if warm else None,
        "warm_min_seconds": min(warm) if warm else None,
        "warm_max_seconds": max(warm) if warm else None,
        "outputs_identical_across_repeats": all(
            run["output_hashes"] == runs[0]["output_hashes"] for run in runs),
    }


def profile_calls(profiler: cProfile.Profile) -> list[dict]:
    """Cumulative call costs overlap; they must never be added as stage totals."""
    selected = {
        "video.py": {"get_video_info"},
        "preflight.py": {"probe"},
        "analyzer.py": {"get_pose_tracker"},
        "pose_tracker.py": {"warmup", "track", "recover_from_guidance"},
        "sam_recovery.py": {"track_segment", "_load"},
        "rtm_pose.py": {"refine"},
        "action.py": {"update"},
        "contact.py": {"classify_contact"},
        "report.py": {"build_report", "write_report"},
    }
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        module = Path(filename).name
        if name in selected.get(module, set()) or "VideoCapture" in name:
            rows.append({"function": f"{module}:{name}", "calls": calls,
                         "self_seconds": own, "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fight", required=True, help="Key in tools/verified_seeds.json")
    parser.add_argument("--ruleset", required=True, help="Explicit, fixed analysis ruleset")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--runs", type=int, default=4, help="First run plus repeated warm-worker runs")
    parser.add_argument("--profile", action="store_true", help="Adds profiling overhead; do not compare with unprofiled timings")
    parser.add_argument("--output", type=Path, required=True, help="New private directory; must not already exist")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 2 or args.stride < 1 or args.runs < 1:
        parser.error("Use finite seconds >= 2, stride >= 1 and runs >= 1")
    seeds = json.loads((ROOT / "tools/verified_seeds.json").read_text(encoding="utf-8"))["fights"]
    if args.fight not in seeds:
        parser.error("Unknown fight key")
    seed = seeds[args.fight]
    video = ROOT / seed["video"]
    if not video.is_file():
        parser.error("The selected footage is not present locally")
    output = args.output.resolve()
    private_root = (ROOT / "dataset/regression/private").resolve()
    if not output.is_relative_to(private_root) or output == private_root:
        parser.error("Store benchmark outputs in a new dataset/regression/private/ subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    # Keep models available but isolate every runtime write from user jobs.
    for key, name in (("WARRIORIQ_DB_PATH", "scratch.sqlite3"),
                      ("WARRIORIQ_UPLOADS_DIR", "uploads"),
                      ("WARRIORIQ_OUTPUTS_DIR", "outputs"),
                      ("WARRIORIQ_MACHINE_PROFILE", "machine_profile.json")):
        os.environ[key] = str(output / name)
    os.environ["WARRIORIQ_FORCE_STRIDE"] = str(args.stride)
    os.environ["HF_HUB_OFFLINE"] = "1"
    before = source_hashes()
    import_start = time.perf_counter()
    import torch
    from core import analyzer
    from core.config import SETTINGS
    from core.types import AnalysisRequest
    from core.video import get_video_info
    import_seconds = time.perf_counter() - import_start
    info = get_video_info(str(video))
    start = seed["seed_frame"] / info.fps
    end = min(start + args.seconds, info.duration)
    if end - start < 2:
        parser.error("Not enough footage remains after the verified selection frame")
    model_paths = sorted((ROOT / "models").glob("*.engine"))
    model_paths += sorted((ROOT / "models").glob("*.pt"))
    model_paths += sorted((ROOT / "models").glob("*.onnx"))
    model_paths += sorted(ROOT.glob("*.pt"))
    versions = {}
    for package in ("torch", "ultralytics", "onnxruntime-gpu", "rtmlib", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "schema": "warrioriq.runtime_benchmark.v1", "status": "running",
        "fight": args.fight, "ruleset": args.ruleset,
        "video_sha256": file_hash(video), "start_seconds": start, "end_seconds": end,
        "source_video": {"fps": info.fps, "width": info.width, "height": info.height},
        "seed": seed, "stride": args.stride, "profiled": args.profile,
        "module_import_seconds": import_seconds,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": before,
        "model_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in model_paths},
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "versions": versions,
        "quality_settings": {name: getattr(SETTINGS, name) for name in (
            "default_imgsz", "low_resolution_imgsz", "target_tracking_fps", "min_tracking_fps",
            "max_tracking_fps", "rtm_pose_enabled", "rtm_pose_device", "sam_recovery_enabled")},
        "upload_seconds": None, "queue_seconds": None, "recognition_accuracy": None,
        "limitations": [
            "Local engine timing excludes upload, server queue and browser delivery.",
            "First run is a fresh process, not a cleared operating-system/disk cache.",
            "Warm repeats retain worker caches and memory, matching a long-lived worker.",
            "Coverage is not identity accuracy. Outputs are predictions, not ground truth.",
            "Profile cumulative call times overlap; never add them as stage totals.",
            "Stage timers use exclusive host wall time; GPU work may finish at a later synchronization.",
        ], "runs": [],
    }

    def save():
        (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for index in range(args.runs):
            run_dir = output / f"run-{index}"
            request = AnalysisRequest(
                video_path=str(video), fighter_a_box=seed["fighter_a"], fighter_b_box=seed["fighter_b"],
                start_seconds=start, end_seconds=end, round_count=1,
                round_duration_seconds=max(10, end-start), ruleset=args.ruleset,
                job_id=f"runtime-{args.fight}-{index}", persist_result=False,
                openai_identity_recovery=False, output_dir=str(run_dir))
            progress = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profiler = cProfile.Profile() if args.profile else None
            stages = StageTimers()
            began = time.perf_counter()

            def on_progress(payload):
                if not progress or payload["stage"] != progress[-1]["stage"]:
                    progress.append({"stage": payload["stage"], "at_seconds": time.perf_counter()-began})
                    print(f"Run {index + 1}/{args.runs}: {payload['stage']}", flush=True)

            if profiler:
                profiler.enable()
            try:
                with ExitStack() as context:
                    stages.install(context, analyzer)
                    report = analyzer.analyze(request, on_progress)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                if profiler:
                    profiler.disable()
            elapsed = time.perf_counter() - began
            row = {"run": index, "wall_seconds": elapsed, "phase_notifications": progress,
                   "stages": stages.rows,
                   "other_seconds": max(0, elapsed - sum(item["exclusive_seconds"] for item in stages.rows.values())),
                   "performance": report["performance"], "tracking": report["tracking"],
                   "output_hashes": {name: file_hash(run_dir / name) for name in ("events.json", "tracking.jsonl")}}
            if profiler:
                row["profile_calls"] = profile_calls(profiler)
                profiler.dump_stats(str(output / f"run-{index}.prof"))
                with (output / f"run-{index}-profile.txt").open("w", encoding="utf-8") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
            result["runs"].append(row)
            save()
            print(f"Run {index + 1}: {elapsed:.3f}s for {end-start:.3f}s footage", flush=True)
        result["summary"] = timing_summary(result["runs"])
        result["source_unchanged_during_run"] = before == source_hashes()
        result["status"] = "complete" if result["source_unchanged_during_run"] else "invalid_source_changed"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        save()
    print(json.dumps(result.get("summary", {"status": result["status"]})), flush=True)


if __name__ == "__main__":
    main()
