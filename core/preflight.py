"""Look at a video before analysing it, and say what will and will not work.

Two jobs, one measurement.

**Choosing the inference size.** What governs detection is not the source
resolution, it is how tall the subject ends up in the network's input. Measured
on fight 3 (568x320, fighter ~76 px), sweeping only the inference size:

    fighter px in net    people found    max conf
            69               0.3           0.37
            86               0.7           0.29
           103               1.3           0.46
           120               3.2           0.49
           137               6.7           0.70
           171               9.3           0.81
           214              15.0           0.85
           240              16.7           0.87

Monotonic, and still climbing at 240. Below ~120 the detector finds almost
nobody; from ~190 it finds everybody.

The rule this replaces asked the wrong question. `inference_size()` looks at the
source's long edge and hands anything at or above 960 the default 640 - so a
phone video gets *downscaled*, and the better the camera the worse the
treatment. A 1080p clip of a fighter 200 px tall puts them at 200 x 640/1920 =
**67 px** in the network, which is the dead row of that table. Every fight this
project has analysed is under 960 and took the other branch, so the path every
phone upload would take has never once been exercised.

Aiming for a target subject height fixes that. It is applied as a **floor, not
a replacement**: `QualityController` takes the larger of this and the old rule.
Trying it as a replacement cost coverage on the footage the old rule already
suited - fight 3 went from 0.4451 to 0.3973 when 1600 became 1440 - because the
curve above is still climbing at 240 px, so 200 is a size worth guaranteeing
rather than a size worth settling at. The saving available on an already-close
subject is therefore deliberately not taken, since nothing here has measured
that it is safe to take.

**Telling the filmer what went wrong.** The same probe answers the questions a
person can act on - are you close enough, how many people are in shot, is the
camera steady, is it sideways. For an audience of children filming each other
on phones, saying "you were too far away, stand at the mat edge next time" at
upload is worth more than a report full of refusals five minutes later.

What this deliberately does **not** do is predict whether the analysis will
succeed. It reports what it measured and how that compares with footage the
system has actually been run on. Only three fights have ever been measured
end to end, all of them wide tournament video at 60-80 px, so there is no
evidence here for what good footage does - and a confident green light would be
exactly the kind of invented certainty the rest of this codebase exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import SETTINGS

# Where the detector becomes reliable, from the table above. 214 px measured
# 15.0 people and 0.85 confidence; 171 measured 9.3 and 0.81. 200 sits in that
# band, and the curve is flat enough there that missing by 20 px either way
# costs little.
TARGET_SUBJECT_PX = 200

# The probe must not fall into the hole it exists to detect, so it looks at a
# size that works on the smallest footage on record and, failing that, larger.
PROBE_SIZES = (1600, 2048)

# Below this the subject cannot be brought up to target even at the largest
# inference size we are willing to run, so the shortfall is reported rather
# than silently accepted.
MAX_INFERENCE_SIZE = 2048
MIN_INFERENCE_SIZE = 640

# The three fights this project has measured end to end. Quoted so a filmer is
# compared against something real rather than an invented standard.
REFERENCE_SUBJECT_PX = (60, 77, 79)
REFERENCE_PEOPLE_IN_FRAME = (13, 18)

# A fighter filling about a third of the picture's height is what "close
# enough" means when the resolution is adequate. Below this the framing is the
# problem; at or above it and still short of pixels, the resolution is.
WELL_FRAMED_SHARE = 0.30

# Under two seconds there is no movement to measure: the action window used
# throughout core/ is 0.6-1.5s, so a shorter clip cannot hold one exchange.
MIN_USABLE_SECONDS = 2.0
# A limb on a person filling a 256px-tall frame is a few pixels across. This is
# not the "too far away" case - it is the case where no framing would save it.
MIN_USABLE_LONG_EDGE = 256


@dataclass
class Preflight:
    """What was measured, what follows from it, and what nobody knows."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    portrait: bool = False

    subject_height_px: float = 0.0
    subject_share_of_height: float = 0.0
    people_in_frame: float = 0.0
    camera_shift_percent: float = 0.0

    recommended_inference_size: int = SETTINGS.default_imgsz
    subject_px_in_network: float = 0.0

    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    measured: bool = False

    @property
    def can_analyse(self) -> bool:
        return self.measured and not self.blocking

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "can_analyse": self.can_analyse,
            "source": {"width": self.width, "height": self.height,
                       "fps": round(self.fps, 2), "frames": self.frame_count,
                       "portrait": self.portrait},
            "subject_height_px": round(self.subject_height_px, 1),
            "subject_share_of_height": round(self.subject_share_of_height, 3),
            "people_in_frame": round(self.people_in_frame, 1),
            "camera_shift_percent": round(self.camera_shift_percent, 2),
            "recommended_inference_size": self.recommended_inference_size,
            "subject_px_in_network": round(self.subject_px_in_network, 0),
            "blocking": list(self.blocking),
            "warnings": list(self.warnings),
            "advice": list(self.advice),
        }


def inference_size_for_subject(long_edge: int, subject_px: float) -> int:
    """Inference size that puts a subject of this height near the target.

    Rounded to a multiple of 32 because the detector's stride requires it, and
    clamped: below MIN there is no point running at all, above MAX the cost
    stops being worth the remaining accuracy on the curve.
    """
    if long_edge <= 0 or subject_px <= 0:
        return SETTINGS.default_imgsz
    wanted = TARGET_SUBJECT_PX * float(long_edge) / float(subject_px)
    stepped = int(round(wanted / 32.0) * 32)
    return max(MIN_INFERENCE_SIZE, min(MAX_INFERENCE_SIZE, stepped))


def _global_shift(previous: np.ndarray, current: np.ndarray) -> float:
    """How far the whole picture moved between two frames, in pixels.

    Phase correlation on a small greyscale copy: it answers "did the camera
    move" without tracking anything, which is what is wanted here - a handheld
    phone and a tripod differ by this number and by very little else.
    """
    a = cv2.cvtColor(cv2.resize(previous, (160, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(current, (160, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy)) / 160.0 * 100.0


def probe(video_path: str, model, start_seconds: float = 0.0,
          end_seconds: float | None = None, samples: int = 8) -> Preflight:
    """Measure a video. `model` is a loaded pose model; nothing is loaded here.

    Sampling rather than reading everything: this runs before the user has
    agreed to wait for anything, so it has to cost a second, not a minute.
    Sampling is honest for "how big is the subject" and "how many people are
    about", which barely change through a round. It would not be honest for a
    rate - see the note in project-coverage-can-mean-failure about eight frames
    undercounting a failure by an order of magnitude.
    """
    report = Preflight()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        report.blocking.append("This file could not be opened as a video.")
        return report

    report.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    report.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    report.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    report.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    report.portrait = report.height > report.width
    long_edge = max(report.width, report.height)

    if report.width <= 0 or report.height <= 0 or report.frame_count <= 0:
        report.blocking.append("This file has no readable video frames.")
        capture.release()
        return report
    if report.fps <= 0:
        report.fps = 30.0
        report.warnings.append("The frame rate could not be read; assuming 30 fps.")

    # Say the real reason. Tested with a one-frame file and a 64x48 file: both
    # fell through to "no people could be found... the camera is too far away",
    # which is true but is not why, and sends the filmer off to fix the wrong
    # thing. Duration and frame size are knowable before any model runs, so
    # they are answered before any model runs.
    duration = report.frame_count / report.fps
    if duration < MIN_USABLE_SECONDS:
        report.blocking.append(
            f"This video is only {duration:.1f} seconds long. A round needs at "
            f"least {MIN_USABLE_SECONDS:.0f} seconds of continuous footage to "
            "measure anything.")
        capture.release()
        return report
    if long_edge < MIN_USABLE_LONG_EDGE:
        report.blocking.append(
            f"This video is {report.width}x{report.height}, which is too small "
            "to make out a fighter's arms and legs however close the camera "
            "was. Send the original file rather than a shrunken copy.")
        capture.release()
        return report

    first = int(max(0.0, start_seconds) * report.fps)
    last = int((end_seconds if end_seconds else report.frame_count / report.fps) * report.fps)
    last = min(max(last, first + 1), report.frame_count - 1)
    picks = np.linspace(first, last, num=max(2, samples)).astype(int)

    frames = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        report.blocking.append("No frames could be decoded from this file.")
        return report

    # Camera movement, between neighbouring samples. These are seconds apart, so
    # this measures pans and drift rather than per-frame shake; a number for
    # true shake would need consecutive frames and is not what a filmer can act
    # on anyway.
    shifts = [_global_shift(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    report.camera_shift_percent = float(np.median(shifts)) if shifts else 0.0

    heights, counts = [], []
    for size in PROBE_SIZES:
        heights, counts = [], []
        for frame in frames:
            result = model.predict(frame, imgsz=size, conf=SETTINGS.detection_conf,
                                   classes=[0], verbose=False)[0]
            if result.boxes is None or not len(result.boxes):
                counts.append(0)
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            counts.append(len(boxes))
            # The tallest few, not the average: the subject of a fight video is
            # nearer the camera than the room behind them, and the median over
            # everybody in a busy hall describes the hall.
            tall = np.sort(boxes[:, 3] - boxes[:, 1])[-3:]
            heights.extend(float(h) for h in tall)
        if heights:
            break

    if not heights:
        report.blocking.append(
            "No people could be found in this video. If it is a fight, the "
            "camera is too far away or the picture is too dark to use.")
        return report

    report.measured = True
    report.subject_height_px = float(np.median(heights))
    report.subject_share_of_height = report.subject_height_px / max(1, report.height)
    report.people_in_frame = float(np.median(counts))
    report.recommended_inference_size = inference_size_for_subject(
        long_edge, report.subject_height_px)
    report.subject_px_in_network = (
        report.subject_height_px * report.recommended_inference_size / max(1, long_edge))

    _judge(report)
    return report


def _judge(report: Preflight) -> None:
    """Turn the measurements into things a person can do something about."""
    reference_low, reference_high = min(REFERENCE_SUBJECT_PX), max(REFERENCE_SUBJECT_PX)

    if report.subject_px_in_network < 120:
        # The dead rows of the calibration table: 0.3 to 3.2 people found.
        report.blocking.append(
            "The fighters are too small in this video to analyse - about "
            f"{report.subject_height_px:.0f} pixels tall. Film from closer to "
            "the mat, or zoom in, so one fighter fills about a third of the "
            "height of the screen.")
    elif report.subject_px_in_network < 170:
        report.warnings.append(
            "The fighters are small, so tracking will drop out often and "
            "punches will not be counted.")
        report.advice.append("Stand closer to the mat next time, or zoom in.")

    # A ceiling, not a band. Fight 1's fighters are 59 px - a pixel under the
    # smallest reference - and a range test let the worst-performing footage
    # this project owns through with nothing said about it. Anything no bigger
    # than the reference footage is at least as hard as the reference footage.
    if report.subject_height_px <= reference_high:
        report.warnings.append(
            f"The fighters are about {report.subject_height_px:.0f} pixels tall, "
            f"no bigger than the tournament footage ({reference_low}-{reference_high} px) "
            "this system was measured on. On that footage it followed a fighter "
            "for under half the round, could not count punches at all, and "
            "missed a knockdown.")
        # Two different causes with two different fixes, and telling them apart
        # matters. Fight 2's fighters already fill 34% of the picture and are
        # still only 75 px tall, because the picture is 220 px tall: standing
        # closer would not help at all. Advising "get closer" on the strength of
        # pixel height alone is wrong for exactly the case a phone user hits
        # when a clip has been shrunk on its way to us.
        if report.subject_share_of_height < WELL_FRAMED_SHARE:
            report.advice.append(
                "One fighter should fill about a third of the height of the "
                f"picture. Here they fill about {100 * report.subject_share_of_height:.0f}%, "
                "so film from closer to the mat or zoom in.")
        else:
            report.advice.append(
                f"The framing is fine - one fighter fills {100 * report.subject_share_of_height:.0f}% "
                f"of the picture - but the video is only {report.height} pixels "
                "tall, so there is not enough detail. Send the original file "
                "from the phone rather than a copy shared through a messaging "
                "app, and record at 1080p or better.")

    if report.people_in_frame >= min(REFERENCE_PEOPLE_IN_FRAME):
        report.warnings.append(
            f"About {report.people_in_frame:.0f} people are in shot. On footage "
            "this busy, the system put the wrong person in the box in roughly "
            "one clip in eight.")
        report.advice.append(
            "Frame just the one mat you care about, so other bouts and the "
            "crowd stay out of the picture.")

    if report.portrait:
        report.advice.append(
            "This video is taller than it is wide. Turning the phone sideways "
            "fits both fighters in and leaves them bigger in the picture.")

    if report.camera_shift_percent >= 6.0:
        report.warnings.append(
            "The camera moves a lot between shots of the round.")
        report.advice.append(
            "Rest the phone on something, or hold it with both elbows tucked in.")

    if report.fps and report.fps < 24:
        report.warnings.append(
            f"This video runs at {report.fps:.0f} frames per second. Fast kicks "
            "and punches fall between frames below about 24.")

    if not report.blocking and not report.warnings:
        # Deliberately not "this will work". Nothing here has been validated on
        # footage better than the three wide tournament fights, so the honest
        # statement is about what was checked, not about the outcome.
        report.advice.append(
            "Nothing obviously wrong with this recording. That is not a promise "
            "the analysis will be accurate - only that the picture is good "
            "enough to try.")
