"""Rebuild app/static/hero-fight.mp4 and hero-fight.jpg, the homepage clip.

Source: Mixkit stock video 40979, "Two women fighting mixed martial arts",
https://mixkit.co/free-stock-video/two-women-fighting-mixed-martial-arts-40979/
under the Mixkit Stock Video Free License (https://mixkit.co/license/#videoFree),
both checked on 2026-11-01. Download the 720p file and pass its path:

    python tools/make_hero_clip.py 40979-720.mp4

The overlay is the analyser's own pose model and tracker settings
(SETTINGS.pose_model_pt, SETTINGS.tracker, SETTINGS.detection_conf,
SETTINGS.default_imgsz), drawn as a box and a skeleton per fighter. Strike
names are deliberately not drawn (owner, 2026-11-01): they were wrong too
often to put on the homepage. Nothing is added or corrected by hand; a
fighter the model did not see in a frame is simply not drawn.

The window is the one stretch (3.75 s to 10.8 s) where both fighters keep a
single unbroken track: fighter A is lost behind B at 2.5-3.7 s and comes
back under a new id, and after 10.8 s they are on the ground. The script
prints every track's span so a different window or tracker version can be
checked before the ids below are trusted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from core.config import SETTINGS  # noqa: E402

START, END = 90, 259          # frames at 24 fps: 3.75 s to 10.8 s
POSTER_SECONDS = 2.33         # into the cut: A's kick, both fighters in view
FIGHTERS = {9: ("A", (255, 194, 140)), 2: ("B", (112, 100, 255))}  # BGR of #8cc2ff (blue), #ff6470 (red)
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
SEEN = .5                     # keypoint confidence below which nothing is drawn
STATIC = ROOT / "app" / "static"


def track(source: Path) -> list[list[dict]]:
    from ultralytics import YOLO

    model = YOLO(SETTINGS.pose_model_pt)
    cap = cv2.VideoCapture(str(source))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker=SETTINGS.tracker, imgsz=SETTINGS.default_imgsz,
                             conf=SETTINGS.detection_conf, classes=[0], verbose=False)[0]
        people = []
        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, kp, kc in zip(result.boxes.xyxy.tolist(), result.boxes.id.int().tolist(),
                                        result.keypoints.xy.tolist(), result.keypoints.conf.tolist()):
                people.append({"id": tid, "box": box, "kp": kp, "kc": kc})
        frames.append(people)
    return frames


def print_spans(frames: list[list[dict]]) -> None:
    spans: dict[int, list[int]] = {}
    for index, people in enumerate(frames):
        for person in people:
            spans.setdefault(person["id"], []).append(index)
    for tid, seen in sorted(spans.items(), key=lambda item: item[1][0]):
        print(f"track {tid}: frames {seen[0]}-{seen[-1]} ({len(seen)} seen)")


def draw(frame, person: dict, tag: str, colour) -> None:
    x1, y1, x2, y2 = map(int, person["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
    points, seen = person["kp"], person["kc"]
    for a, b in EDGES:
        if seen[a] > SEEN and seen[b] > SEEN:
            cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])), colour, 3, cv2.LINE_AA)
    for k in range(5, 17):
        if seen[k] > SEEN:
            cv2.circle(frame, tuple(map(int, points[k])), 5, (255, 255, 255), -1, cv2.LINE_AA)
    tx, ty = max(x1, 0), max(y1, 30)  # the tag stays inside the frame
    cv2.rectangle(frame, (tx, ty - 30), (tx + 30, ty), (20, 20, 20), -1)
    cv2.rectangle(frame, (tx, ty - 30), (tx + 30, ty), colour, 1)
    cv2.putText(frame, tag, (tx + 7, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    source = Path(argv[1])
    frames = track(source)
    print_spans(frames)
    draft = STATIC / "hero-fight-draft.mp4"
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(draft), cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), size)
    for index in range(START, END):
        ok, frame = cap.read()
        if not ok:
            break
        for person in frames[index]:
            if person["id"] in FIGHTERS:
                draw(frame, person, *FIGHTERS[person["id"]])
        out.write(frame)
    out.release()
    clip = STATIC / "hero-fight.mp4"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(draft), "-an", "-c:v", "libx264",
                        "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "slow", "-b:v", "2300k",
                        "-maxrate", "2800k", "-bufsize", "4600k", "-movflags", "+faststart", str(clip)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(POSTER_SECONDS), "-i", str(clip),
                        "-frames:v", "1", "-q:v", "4", str(STATIC / "hero-fight.jpg")], check=True)
    finally:
        draft.unlink(missing_ok=True)
    print(clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
