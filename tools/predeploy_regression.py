"""End-to-end regression suite to run before every deploy (QA, 2026-10-04, item 22).

    python tools/predeploy_regression.py dataset/predeploy/manifest.json [--clips DIR] [--only id,id]

Run it on the machine that will analyse fights (the GPU worker image), from the
commit about to be deployed. Every clip in the manifest is analysed with the
real pipeline - core.analyzer.analyze, exactly what a worker runs - and held to
the product's rules (core/predeploy_checks.py):

  * the whole video is analysed;
  * no kick or knee where the legs were not visible;
  * processing time no longer than the video (models loaded first, as in a
    warm worker; a cold container's start-up is measured separately by
    deploy/modal_worker.py drain_queue);
  * one verdict per report.

Clip kinds and what each must show:

  fight      two fighters: all four rules.
  waist_up   two fighters, legs out of shot: also no kick or knee at all.
  rotated    a sideways phone clip: the upload's orientation check must ask
             for the turn in the manifest, then it is analysed upright.
  solo       one person: a solo report, whole video, no score.
  no_people  nobody in shot: completes, and claims nothing about anyone.
  corrupt    not a decodable video: refused with a clear error, no crash.

Exit code 0 when every clip passes, 1 otherwise. Clips are not in git (they
are people's fights); the manifest names them and --clips says where they are.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _request(clip: dict, video: Path, workdir: Path):
    from core.types import AnalysisRequest

    return AnalysisRequest(
        video_path=str(video), fighter_a_box=list(clip.get("fighter_a_box") or []),
        fighter_b_box=list(clip.get("fighter_b_box") or []),
        analysis_target="A" if clip["kind"] == "solo" else "BOTH", focus_fighter="A",
        ruleset=clip.get("ruleset", "K1"), start_seconds=0.0,
        selection_seconds=float(clip.get("selection_seconds", 0.0)),
        round_count=1, round_duration_seconds=float(clip.get("round_seconds", 10_000.0)),
        break_duration_seconds=0.0, solo=clip["kind"] == "solo",
        output_dir=str(workdir / clip["id"]), persist_result=False, job_id=clip["id"])


def run_clip(clip: dict, clips_dir: Path, workdir: Path, timing: bool = True) -> list[str]:
    from core import analyzer, predeploy_checks as checks

    video = clips_dir / clip["path"]
    if not video.is_file():
        return [f"clip missing: {video}"]
    kind = clip["kind"]
    if kind == "corrupt":
        try:
            analyzer.analyze(_request(clip, video, workdir))
        except Exception as exc:                                    # noqa: BLE001 - the refusal is the pass
            return [] if str(exc).strip() else ["refused without saying why"]
        return ["a corrupt file was analysed as if it were a video"]
    if kind == "rotated":
        import cv2

        from core.orientation import needed_turn
        from core.person_detect import detect_people

        ok, frame = cv2.VideoCapture(str(video)).read()
        turn = needed_turn(frame if ok else None, detect_people)
        if turn != int(clip.get("expected_turn", 90)):
            return [f"orientation check asked for {turn} degrees, expected {clip.get('expected_turn', 90)}"]
        upright = workdir / f"{clip['id']}-upright{video.suffix}"
        shutil.copy(video, upright)
        from core.orientation import tag_rotation
        from core.video import _ffmpeg_exe

        if not tag_rotation(upright, turn, _ffmpeg_exe()):
            return ["could not turn the clip upright"]
        video = upright
    started = time.perf_counter()
    report = analyzer.analyze(_request(clip, video, workdir))
    wall = time.perf_counter() - started
    events_path = workdir / clip["id"] / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.is_file() else []
    failures = checks.whole_video(report) + checks.one_verdict(report)
    if timing:
        failures += checks.within_video_length(wall, report)
    if kind in ("fight", "waist_up", "rotated"):
        failures += checks.no_kicks_without_legs(events, legs_in_shot=kind != "waist_up")
    if kind == "solo" and report.get("mode") != "solo":
        failures.append("a solo clip did not produce a solo report")
    if kind == "no_people":
        trusted = (report.get("integrity") or {}).get("identity_evidence_trusted")
        if trusted and any(e.get("family") for e in events):
            failures.append("strikes attributed on a clip with nobody in it")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--clips", help="directory holding the clips (default: beside the manifest)")
    parser.add_argument("--only", default="")
    parser.add_argument("--skip-timing", action="store_true",
                        help="a dry run on a machine without the GPU; never for a deploy decision")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    clips_dir = Path(args.clips) if args.clips else Path(args.manifest).parent
    only = {item for item in args.only.split(",") if item}
    from core import analyzer

    analyzer.get_pose_tracker()          # loaded once per worker, not once per fight
    failed = 0
    with tempfile.TemporaryDirectory(prefix="predeploy-") as temporary:
        for clip in manifest["clips"]:
            if only and clip["id"] not in only:
                continue
            problems = run_clip(clip, clips_dir, Path(temporary), timing=not args.skip_timing)
            failed += bool(problems)
            print(("PASS " if not problems else "FAIL ") + clip["id"]
                  + "".join(f"\n     - {problem}" for problem in problems), flush=True)
    print(f"{failed} of {len(manifest['clips'])} clips failed" if failed else "all clips passed")
    if args.skip_timing:
        print("timing was NOT checked (--skip-timing): this run cannot approve a deploy")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
