from __future__ import annotations

import uuid

import cv2

from core.analyzer import analyze
from core.scoring import RULESETS
from core.types import AnalysisRequest
from core.video import get_video_info, read_frame


def select_box(title, frame):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise RuntimeError(f"{title} selection cancelled")
    return [float(x), float(y), float(x + w), float(y + h)]


def main():
    path = input("Fight video path: ").strip().strip('"')
    info = get_video_info(path)
    print(f"Video: {info.duration:.1f}s · {info.fps:.2f} FPS · {info.width}x{info.height}")
    fight_type = (input("Competition or sparring [competition]: ").strip().lower() or "competition")
    start = float(input("Fight/round 1 start second [0]: ") or "0")
    count = int(input("Round count [3]: ") or "3")
    round_seconds = float(input("Round duration seconds [120]: ") or "120")
    break_seconds = float(input("Break duration seconds [60]: ") or "60")
    selected_text = input("Rounds to analyze [ALL]: ").strip()
    selected = None if not selected_text or selected_text.upper() == "ALL" else [int(x.strip()) for x in selected_text.split(",") if x.strip().isdigit()]
    end_text = input(f"Optional hard end second [scheduled/full]: ").strip()
    end = None if not end_text else float(end_text)
    frame = read_frame(path, int(round(start * info.fps)))
    print("Draw Fighter A and press ENTER. Then draw Fighter B.")
    a = select_box("WarriorIQ - Fighter A", frame)
    b = select_box("WarriorIQ - Fighter B", frame)
    target = input("Analyze 1=A, 2=B, 3=Both [3]: ").strip() or "3"
    target = {"1": "A", "2": "B", "3": "BOTH"}.get(target, "BOTH")
    print("Rulesets:", ", ".join(RULESETS.keys()))
    ruleset = input("Ruleset [K1]: ").strip() or "K1"
    job_id = uuid.uuid4().hex[:12]

    def progress(patch):
        if "percent" in patch:
            print(
                f"{patch.get('percent', 0):5.1f}% | "
                f"{patch.get('speed', 0):.2f}x realtime | "
                f"ETA {patch.get('eta_seconds', 0):.0f}s | "
                f"A {patch.get('fighter_a_confidence', 0):.2f} | "
                f"B {patch.get('fighter_b_confidence', 0):.2f} | "
                f"{patch.get('message', '')}"
            )

    report = analyze(
        AnalysisRequest(
            video_path=path,
            fighter_a_box=a,
            fighter_b_box=b,
            original_name=path.split("\\")[-1].split("/")[-1],
            analysis_target=target,
            fight_type=fight_type,
            ruleset=ruleset,
            start_seconds=start,
            round_count=count,
            round_duration_seconds=round_seconds,
            break_duration_seconds=break_seconds,
            selected_rounds=selected,
            end_seconds=end,
            job_id=job_id,
            profile_id=1,
        ),
        progress,
    )
    print("\nDONE")
    print("Analysis:", round(report["performance"]["analysis_seconds"], 1), "s")
    print("Video segment:", round(report["performance"]["segment_duration_seconds"], 1), "s")
    print("Speed:", round(report["performance"]["realtime_speed"], 2), "x realtime")
    print("Within budget:", report["performance"]["within_video_length_budget"])
    print("Report folder: outputs/" + job_id)


if __name__ == "__main__":
    main()
