"""Render every detection on one frame, numbered, so a person can pick the two.

There is no automatic rule for "which two of these are the fighters". Every one
tried here has failed: the two largest non-referee detections on frame 0 picks
a seated coach on two of the three library fights, and on a multi-mat hall
there are several bouts running at once so the question does not even have one
right answer without a person saying which bout they mean.

So this does not decide. It draws the candidates with numbers on them and
prints their boxes, and a human reads off the two indices. Those go into
tools/verified_seeds.json with a note saying who is who, and from then on
tools/measure_identity.py can measure that fight repeatably.

    tools/pick_seeds.py --video fights/my_bout.mp4 --frame 900
    tools/pick_seeds.py --video fights/my_bout.mp4 --frame 900 --zoom 160,20,340,170

Pick a frame where the two fighters are apart. Mid-exchange they merge into one
detection, and a seed drawn round both of them is a seed round neither.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale", type=int, default=0,
                        help="0 picks a scale that makes a small source readable")
    parser.add_argument("--zoom", default=None,
                        help="x1,y1,x2,y2 in source pixels, to crop in on a crowded frame")
    args = parser.parse_args()

    from core import preflight
    from core.pose_tracker import PoseTracker

    video = str(PROJECT_ROOT / args.video) if not Path(args.video).is_absolute() else args.video
    if not Path(video).exists():
        raise SystemExit("no such video: %s" % video)

    capture = cv2.VideoCapture(video)
    ok, first = capture.read()
    if not ok:
        raise SystemExit("could not read the first frame of %s" % video)
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise SystemExit("could not read frame %d (the file has %d)" % (args.frame, total))

    tracker = PoseTracker()
    tracker.warmup(first)
    imgsz = preflight.probe(video, tracker.model).recommended_inference_size
    people = tracker.track(frame, imgsz)

    print("%s  %dx%d  %.0f fps  %d frames" % (
        Path(video).name, frame.shape[1], frame.shape[0], fps, total))
    print("frame %d (%.1fs), inference size %d, %d detections\n" % (
        args.frame, args.frame / max(1.0, fps), imgsz, len(people)))
    print("  idx  box                      area   referee?")
    for index, person in enumerate(people):
        x1, y1, x2, y2 = [float(v) for v in person.box]
        referee = float(getattr(person, "referee_prob", 0.0) or 0.0)
        print("  %3d  [%4.0f,%4.0f,%4.0f,%4.0f]  %6.0f   %.2f%s" % (
            index, x1, y1, x2, y2, (x2 - x1) * (y2 - y1), referee,
            "  <- treated as the referee" if referee >= 0.5 else ""))

    view = frame
    offset_x = offset_y = 0
    if args.zoom:
        try:
            zx1, zy1, zx2, zy2 = [int(v) for v in args.zoom.split(",")]
        except ValueError:
            raise SystemExit("--zoom wants x1,y1,x2,y2")
        view = frame[max(0, zy1):zy2, max(0, zx1):zx2]
        offset_x, offset_y = max(0, zx1), max(0, zy1)
        if view.size == 0:
            raise SystemExit("--zoom selects nothing")

    scale = args.scale or max(2, int(round(1600.0 / max(1, view.shape[1]))))
    big = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    for index, person in enumerate(people):
        x1, y1, x2, y2 = [float(v) for v in person.box]
        referee = float(getattr(person, "referee_prob", 0.0) or 0.0)
        colour = (0, 165, 255) if referee >= 0.5 else (0, 255, 255)
        cv2.rectangle(big, (int((x1 - offset_x) * scale), int((y1 - offset_y) * scale)),
                      (int((x2 - offset_x) * scale), int((y2 - offset_y) * scale)), colour, 2)
        cv2.putText(big, str(index),
                    (int((x1 - offset_x) * scale), max(18, int((y1 - offset_y) * scale) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    out = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / (
        "pick_%s_f%d.png" % (Path(video).stem[:20], args.frame))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), big)
    print("\nwrote %s" % out)
    print("Open it, read off the two indices that are the fighters, and add them to")
    print("tools/verified_seeds.json. If neither is clearly a fighter, try another frame -")
    print("do not settle for the largest boxes, which is how a coach ends up as fighter B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
