from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from core.config import SETTINGS
from ultralytics import YOLO
from core.identity import appearance_hist, pose_signature
from core.referee import referee_probabilities
from core.reid import embed
from core.types import PersonObservation

LOGGER = logging.getLogger("warrioriq.pose")


# Cropping to the action does not work on this footage, and it was measured
# rather than assumed, because the prize looks obvious: over full bouts of both
# real recordings the fighters occupy only 57% of the width and 58% of the
# height, and every frame pays for the rest.
#
# The difficulty is knowing which 57% before the fight has been analysed.
#
#   * From the seed boxes the coach drew: too small. The measured span over a
#     round is 4.3x the width of the seed union on fight 1, so a crop padded
#     generously enough for fight 3 still ended at x=390 while the fighters
#     reached x=400 - clipping fighter B for part of the round, which costs the
#     coverage the crop was meant to buy.
#   * From a motion heat map over the segment: cannot discriminate. On fight 3
#     the crowd, the officials and the scoreboard move as much as the mat does,
#     and the motion region is the entire frame at every threshold that
#     contains the fighters.
#
# Where it can be made safe it is not worth having: the best honest case was a
# long edge of 0.80 on fight 1, which is pose cost 0.64, and pose is around 17%
# of a run - six percent end to end, against the risk of losing a fighter who
# runs wide. Spending the same crop on resolution instead would give subjects
# 1.25x more pixels, which is real but small.
#
# This would work on tightly-framed footage with a still crowd. It does not
# work here.


def inference_size(source_width: int, source_height: int) -> int:
    """Choose the detector input size for a source resolution.

    A fighter in a wide ringside shot is only a few dozen pixels tall. Running a
    small source at the standard size barely upscales it, so the athletes stay
    below what pose estimation resolves and only the nearest person is found.
    Large sources keep the tuned default, which already downscales them.
    """
    long_edge = max(int(source_width or 0), int(source_height or 0))
    if 0 < long_edge < SETTINGS.low_resolution_edge:
        return max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz)
    return SETTINGS.default_imgsz


def _device_name() -> str:
    """The card this is running on, or "cpu". Never raises: it only keys a file."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:                       # noqa: BLE001 - naming, not running
        pass
    return "cpu"


class QualityController:
    """Adaptive analysis quality while respecting the <= video-length target."""

    def __init__(self, source_fps: float, source_width: int = 0, source_height: int = 0,
                 measured_imgsz: int | None = None):
        self.source_fps = max(1.0, float(source_fps))
        self.target_fps = min(self.source_fps, SETTINGS.target_tracking_fps)
        self.min_fps = min(self.source_fps, SETTINGS.min_tracking_fps)
        self.max_fps = min(self.source_fps, SETTINGS.max_tracking_fps)
        self.stride = max(1, round(self.source_fps / self.target_fps))
        # A size measured from how tall the subject actually is beats one
        # guessed from the source resolution - see core/preflight.py for the
        # calibration and for why the resolution rule is backwards on anything
        # a phone produces. `inference_size` remains the fallback for when
        # there was no probe: a fixture, a still, a file the probe could not
        # read.
        #
        # **Never below what the old rule would have chosen.** The measured
        # size aims a subject at 200 px in the network, and on fight 3 that is
        # 1440 against the old rule's 1600 - which cost coverage: A 0.4451 ->
        # 0.3973, B 0.1987 -> 0.1598, one run each and the pipeline is
        # deterministic. The calibration curve is still climbing at 240 px, so
        # 200 is a floor worth guaranteeing and not a ceiling worth enforcing.
        # Taking the larger of the two means this change can only add size
        # where the old rule under-served - which is the high-resolution case
        # it was written for - and can never take it away from the footage the
        # old rule already suited.
        rule_imgsz = inference_size(source_width, source_height)
        self.base_imgsz = self.imgsz = max(int(measured_imgsz or 0), rule_imgsz)
        self.mode = "balanced"
        self.last_adjust = 0
        # Budget planning state. See plan_for_budget.
        self.planned = False
        self.planned_stride: int | None = None
        self.budget_reason = "not_planned"
        self.budget_expected_met: bool | None = None
        # Where the frame cost came from, so a run that planned differently
        # from another on the same machine can be explained rather than
        # argued about. See _frame_cost.
        self.budget_cost_source = "not_planned"
        # What a frame ACTUALLY cost during calibration, next to what the plan
        # was made with. These are usually the same number. When they are not,
        # the machine is not the machine its stored profile describes, and the
        # gap is the only warning anything gets - see plan_for_budget.
        self.budget_cost_planned: float | None = None
        self.budget_cost_observed: float | None = None
        # Which card this is, for the stored profile's key: the same footage
        # costs a different amount on different hardware, and a profile shared
        # between them would plan from somebody else's machine.
        self.device_name = _device_name()

    @property
    def effective_fps(self) -> float:
        return self.source_fps / self.stride

    def _frame_cost(self, measured_seconds: float) -> float:
        """What to plan with: a configured cost, a stored one, or this run's.

        Preference order matters. An explicitly configured cost is somebody
        stating a fact about their hardware and wins outright. A stored cost is
        this machine's own earlier measurement and is what makes two runs of
        one video plan identically. Measuring here is the last resort, and the
        measurement is written down so it is the last resort only once.
        """
        if SETTINGS.frame_cost_seconds > 0:
            self.budget_cost_source = "configured"
            return float(SETTINGS.frame_cost_seconds)
        from core import machine_profile

        stored = machine_profile.frame_cost(self.device_name, self.imgsz)
        if stored:
            self.budget_cost_source = "profile"
            return stored
        self.budget_cost_source = "measured"
        # Snapped before use, not merely before storing, so this first run
        # plans from the same number every later run will read back.
        snapped = machine_profile.record_frame_cost(
            self.device_name, self.imgsz, measured_seconds)
        return snapped or measured_seconds

    def plan_for_budget(self, analyzed_frames: int, processed_seconds: float,
                        elapsed_seconds: float, segment_duration: float) -> None:
        """Choose one sampling stride that finishes inside the video's length.

        The product's stated target is that analysis never takes longer than the
        video. Nothing enforced it: `hard_realtime_budget` was read nowhere at
        all, and `adaptive_quality` - the only real mechanism - is off by
        default and rightly so, because adapting continuously to momentary
        machine load makes identical fights follow different frame paths.

        So this plans **once**. It measures the true cost of a frame on this
        machine over a short calibration, works out how many frames the
        remaining budget affords, and fixes the stride for the rest of the run.
        One decision, recorded in the report, rather than a controller chasing
        load for the whole analysis.

        It will not go below `min_tracking_fps`. A round sampled too sparsely
        cannot hold a strike - the action windows in core/ are 0.6-1.5s - so
        buying the budget past that point buys a number and loses the analysis.
        When the floor is not enough, the stride stops there and
        `budget_expected_met` goes False, which the report publishes. Measured
        on this machine: 480x220 footage with a 59px subject needs imgsz 1632,
        costs 0.194s per analysed frame, and cannot finish in real time at 10
        fps whatever this method does. Saying so is the honest outcome.

        **Reproducibility, and why the cost no longer comes from this run.**
        Planning reads a clock, and a clock moves with machine load. The
        earlier version measured the cost of a frame inside the run, and the
        warning here used to say that was stable only by luck. It was: on
        fight 1, the same file and commit planned stride 3 on some runs and
        stride 4 on others, and that flip moved fighter B's coverage from
        0.260 to 0.728. Measured across all three fights the flip is worth
        +0.468, -0.138 and 0.009 - opposite directions, so there is no safer
        stride to prefer, only a requirement that the same input plans the
        same way.

        So the frame cost now comes from core/machine_profile.py: measured on
        the first analysis that needs it, rounded to a bucket, written down,
        and read back by every later run on that machine. The clock is read
        once in the life of a machine rather than once per analysis, and
        `budget_cost_source` in the report says which happened.
        """
        if self.planned or analyzed_frames < 40 or elapsed_seconds <= 0:
            return
        self.planned = True
        if SETTINGS.force_tracking_stride > 0:
            self.stride = int(SETTINGS.force_tracking_stride)
            self.planned_stride = self.stride
            self.budget_reason = "stride_pinned"
            self.budget_cost_source = "pinned"
            self.budget_expected_met = None
            return

        observed = elapsed_seconds / max(1, analyzed_frames)
        per_frame = self._frame_cost(observed)
        self.budget_cost_planned = float(per_frame)
        self.budget_cost_observed = float(observed)
        # Plan from the stored cost, predict from the observed one.
        #
        # These are the same number on a machine behaving as its profile says,
        # and the stride below must keep coming from the stored value or two
        # runs of one video stop planning identically - which is the whole
        # reason the profile exists. But when the machine is slower than its
        # profile RIGHT NOW, planning from the profile and then reporting that
        # the budget will be met publishes something false: measured on this
        # machine, a profile of 0.083 s/frame against an actual 2.4 s/frame
        # still produced budget_expected_met = True while the run missed its
        # deadline roughly thirty-fold, and the report said it had met it.
        #
        # So the stride stays deterministic and the PREDICTION becomes honest.
        # A run that cannot meet the budget now says so, whatever the stored
        # number believes, and carries both costs so the gap is visible instead
        # of being argued about later.
        #
        # 1.5x is deliberately loose. Frame cost varies run to run with what
        # else the machine is doing, and a prediction that flickers to False on
        # ordinary noise is a worse lie than the one it replaces.
        divergence = observed / max(1e-9, per_frame)
        cost_is_stale = self.budget_cost_source in {"configured", "profile"} and divergence > 1.5
        remaining_video = max(0.0, segment_duration - processed_seconds)
        remaining_budget = segment_duration - elapsed_seconds
        if remaining_video <= 0:
            self.budget_reason = "already_finished"
            self.budget_expected_met = True
            return
        if remaining_budget <= 0:
            affordable = 0.0
        else:
            affordable = remaining_budget / per_frame

        wanted = remaining_video * self.source_fps / max(1, self.stride)
        if wanted <= affordable:
            self.planned_stride = self.stride
            if cost_is_stale and wanted * observed > max(0.0, remaining_budget):
                # Affordable by the stored cost, not affordable by this run's.
                # Keep the stride; drop the promise.
                self.budget_reason = "machine_slower_than_profile"
                self.budget_expected_met = False
                return
            self.budget_reason = "on_track"
            self.budget_expected_met = True
            return

        needed = remaining_video * self.source_fps / max(1e-6, affordable)
        floor_stride = max(1, int(self.source_fps / max(1e-6, self.min_fps)))
        chosen = max(1, min(int(needed) + 1, floor_stride))
        self.stride = chosen
        self.planned_stride = chosen
        self.mode = "deadline"
        affordable_at_chosen = remaining_video * self.source_fps / max(1, chosen)
        self.budget_expected_met = affordable_at_chosen <= affordable
        self.budget_reason = (
            "sampling_reduced" if self.budget_expected_met
            else "cannot_meet_budget_above_quality_floor")
        # Same check on the reduced-sampling path: a stride chosen against a
        # stored cost that this run is not achieving does not meet the budget
        # either, and saying it does is the failure this guards against.
        if self.budget_expected_met and cost_is_stale:
            if affordable_at_chosen * observed > max(0.0, remaining_budget):
                self.budget_expected_met = False
                self.budget_reason = "machine_slower_than_profile"

    def maybe_adjust(self, analyzed_index: int, processed_seconds: float, elapsed_seconds: float) -> tuple[int, int, str]:
        if not SETTINGS.adaptive_quality or elapsed_seconds < 2.0 or analyzed_index - self.last_adjust < 60:
            return self.stride, self.imgsz, self.mode

        speed = processed_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        fps = self.effective_fps

        if speed < 0.92:
            # Behind budget: reduce expensive inference before compromising
            # identity safeguards.
            fps = max(self.min_fps, fps * 0.82)
            self.imgsz = max(SETTINGS.min_imgsz, self.imgsz - 64)
            self.mode = "deadline"
        elif speed < 1.08:
            fps = max(self.min_fps, fps * 0.92)
            self.imgsz = max(SETTINGS.min_imgsz, self.imgsz - 32)
            self.mode = "economy"
        elif speed > 1.65:
            fps = min(self.max_fps, fps * 1.12)
            # Recover toward this source's own size, not the global default, so
            # a low-resolution fight is not permanently capped at 640.
            self.imgsz = min(self.base_imgsz, self.imgsz + 32)
            self.mode = "high"
        elif speed > 1.20:
            self.mode = "balanced"
        else:
            self.mode = "balanced"

        self.stride = max(1, round(self.source_fps / max(self.min_fps, fps)))
        self.last_adjust = analyzed_index
        return self.stride, self.imgsz, self.mode


class PoseTracker:
    def __init__(self):
        requested = str(SETTINGS.device).strip().lower()
        self.device = (0 if torch.cuda.is_available() else "cpu") if requested == "auto" else (int(requested) if requested.isdigit() else requested)
        self.uses_cuda = self.device != "cpu" and torch.cuda.is_available()
        engine_path = Path(SETTINGS.pose_model_engine)
        model_path = str(engine_path) if self.uses_cuda and engine_path.exists() else SETTINGS.pose_model_pt
        self.model_path = model_path
        # Whether TensorRT is carrying this run is the difference between a
        # fight finishing inside its own length and not, and every way of
        # losing it was silent: a path that does not exist, a CPU device, an
        # engine built for another GPU. Say which one is loaded, out loud, once.
        if model_path.endswith(".engine"):
            LOGGER.info("pose_backend=tensorrt engine=%s", model_path)
        elif not self.uses_cuda:
            LOGGER.warning("pose_backend=pytorch_cpu model=%s - no CUDA device", model_path)
        else:
            LOGGER.warning(
                "pose_backend=pytorch model=%s - no TensorRT engine at %s, so this "
                "analysis runs slower than it needs to", model_path, engine_path,
            )
        self.model = YOLO(model_path)
        self._focus_model = None
        self._warmed = False

    def warmup(self, frame) -> None:
        if self._warmed:
            return
        try:
            _ = self.model.predict(
                frame,
                device=self.device,
                imgsz=SETTINGS.default_imgsz if self.uses_cuda else min(416, SETTINGS.default_imgsz),
                conf=SETTINGS.detection_conf,
                classes=[0],
                verbose=False,
            )
        except Exception as exc:                                    # noqa: BLE001
            if not self.model_path.endswith(".engine"):
                raise
            # TensorRT engines are tied to compatible NVIDIA runtimes. Fall
            # back to the original PyTorch checkpoint on another device. This
            # is the second silent way to lose TensorRT and the harder one to
            # spot: the engine file is right there, it just will not run here.
            LOGGER.warning(
                "pose_backend=pytorch_fallback engine=%s rejected error=%s detail=%s - "
                "the engine exists but this runtime cannot load it, most likely built "
                "for a different GPU or TensorRT version",
                self.model_path, type(exc).__name__, str(exc)[:200],
            )
            self.model_path = SETTINGS.pose_model_pt
            self.model = YOLO(self.model_path)
            self._focus_model = None
            _ = self.model.predict(
                frame,
                device=self.device,
                imgsz=SETTINGS.default_imgsz if self.uses_cuda else min(416, SETTINGS.default_imgsz),
                conf=SETTINGS.detection_conf,
                classes=[0],
                verbose=False,
            )
        self._warmed = True

    def reset_tracking(self) -> None:
        """Best-effort reset of Ultralytics tracker state between fights.

        The YOLO model is cached so we do not pay model-loading cost for every
        fight, but BoT-SORT state must never leak from one uploaded video into
        the next. Ultralytics internals can vary by version, so this deliberately
        uses feature checks rather than depending on one private layout.
        """
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return

        trackers = getattr(predictor, "trackers", None)
        if trackers is not None:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass

        # Some Ultralytics tracker callbacks use vid_path to decide whether a
        # source changed. Clearing it prevents a cached ndarray source from
        # inheriting identity state from the previous analysis.
        if hasattr(predictor, "vid_path"):
            try:
                current = predictor.vid_path
                predictor.vid_path = [None] * len(current) if isinstance(current, (list, tuple)) else None
            except Exception:
                pass

    def track(self, frame, imgsz: int | None = None) -> list[PersonObservation]:
        size = int(imgsz or SETTINGS.default_imgsz)
        results = self.model.track(
            frame,
            persist=True,
            tracker=SETTINGS.tracker,
            device=self.device,
            imgsz=size,
            conf=SETTINGS.detection_conf,
            classes=[0],
            verbose=False,
        )
        people = self.parse(results[0], frame)
        # One batched pass for the whole frame, so the identity manager can
        # compare learned appearance instead of a colour histogram. See
        # core/reid.py for why the histogram is not enough here.
        if people:
            boxes = np.asarray([p.box for p in people], dtype=np.float32)
            vectors = embed(frame, boxes)
            # And who the official is, which is a categorical question the
            # comparative guards cannot answer. See core/referee.py.
            verdicts = referee_probabilities(frame, boxes)
            # Both helpers return one entry per box, including on failure,
            # so a mismatch here would be a bug rather than a bad frame.
            for person, vector, verdict in zip(people, vectors, verdicts, strict=True):
                person.reid = vector
                person.referee_prob = verdict
        return people

    def recover_from_guidance(
        self,
        frame: np.ndarray,
        guidance: dict[str, np.ndarray] | None,
        existing: list[PersonObservation],
    ) -> list[PersonObservation]:
        """Run focused pose inference where SAM sees a fighter YOLO missed."""
        if not guidance:
            return []
        from core.identity import box_iou

        height, width = frame.shape[:2]
        requests: list[tuple[str, np.ndarray, int, int, np.ndarray]] = []
        for name in ("A", "B"):
            guide = guidance.get(name)
            if guide is None or any(box_iou(guide, person.box) >= 0.12 for person in existing):
                continue
            x1, y1, x2, y2 = map(float, guide)
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx1 = max(0, int(x1 - 0.45 * bw))
            cy1 = max(0, int(y1 - 0.30 * bh))
            cx2 = min(width, int(x2 + 0.45 * bw))
            cy2 = min(height, int(y2 + 0.30 * bh))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.shape[0] >= 20 and crop.shape[1] >= 12:
                requests.append((name, guide, cx1, cy1, crop))
        if not requests:
            return []

        if self._focus_model is None:
            # A separate predictor is essential: predict() on the persistent
            # tracking model changes its internal source geometry and breaks
            # BoT-SORT camera-motion state on the next full frame.
            #
            # It must NOT be the TensorRT engine, though, and that was costing
            # far more than the recovery it buys. Loading the .engine a second
            # time creates a second IExecutionContext, and the context - not the
            # 86 MiB of weights - is 3.5 GB. Measured in the logs of every run
            # today: the card goes 3582 MiB after the tracking model and 7164
            # MiB once this one loads, on an 8151 MiB card.
            #
            # What that starves is SAM2. An earlier session measured SAM2 at
            # 0.176 s/frame with ONE engine resident and 2.71 GB free, and noted
            # it cost nothing. With two there is under 1 GB left, so SAM2 spills
            # to system RAM over PCIe - which Windows does silently instead of
            # erroring - and a twenty second clip did not finish in forty-five
            # minutes with the GPU pinned at 100%. Runs with SAM2 disabled
            # completed; runs with it enabled did not. That is the whole
            # difference.
            #
            # This path runs on a handful of frames in a round, on crops, at
            # imgsz 384. PyTorch weights are far quicker than that needs and
            # cost a few hundred MB instead of 3.5 GB. The engine stays the
            # fallback for a checkout that has no .pt.
            focus_path = self.model_path
            pt_path = Path(SETTINGS.pose_model_pt)
            if self.model_path.endswith(".engine") and pt_path.exists():
                focus_path = str(pt_path)
            LOGGER.info("focus_backend=%s model=%s",
                        "tensorrt" if focus_path.endswith(".engine") else "pytorch",
                        focus_path)
            self._focus_model = YOLO(focus_path)
        # One crop per call rather than one batched call over the list.
        #
        # The batch was not "guaranteed one result per request" as the comment
        # here used to claim: the crops are cut to each fighter and so have
        # different shapes, and Ultralytics returned fewer results than inputs,
        # dying inside its own predictor at `self.results[i].speed = {` with
        # `IndexError: list index out of range`. That took the whole analysis
        # with it, which is why WARRIORIQ_SAM_CONTINUOUS=true crashed on the
        # first fighter it tried to recover - a setting core/config.py invites
        # the reader to turn on.
        #
        # This path only runs where SAM sees a fighter YOLO missed, a handful of
        # frames in a round, so looping costs nothing worth protecting.
        results = [
            self._focus_model.predict(
                request[4],
                device=self.device,
                imgsz=384 if self.uses_cuda else 320,
                conf=max(0.10, SETTINGS.detection_conf * 0.65),
                classes=[0],
                verbose=False,
            )[0]
            for request in requests
        ]
        recovered: list[PersonObservation] = []
        for (name, guide, offset_x, offset_y, crop), result in zip(requests, results, strict=True):
            candidates = self.parse(result, crop)
            best = None
            best_score = -1.0
            for candidate in candidates:
                candidate.box[[0, 2]] += offset_x
                candidate.box[[1, 3]] += offset_y
                if candidate.keypoints is not None:
                    valid = (candidate.keypoints[:, 0] > 0) & (candidate.keypoints[:, 1] > 0)
                    candidate.keypoints[valid, 0] += offset_x
                    candidate.keypoints[valid, 1] += offset_y
                score = 0.75 * box_iou(guide, candidate.box) + 0.25 * candidate.confidence
                if score > best_score:
                    best, best_score = candidate, score
            if best is not None and best_score >= 0.18:
                best.track_id = -1001 if name == "A" else -1002
                best.appearance = appearance_hist(frame, best.box)
                best.pose_signature = pose_signature(best.keypoints, best.box)
                recovered.append(best)
        return recovered

    @staticmethod
    def parse(result, frame) -> list[PersonObservation]:
        people: list[PersonObservation] = []
        if result.boxes is None or len(result.boxes) == 0:
            return people

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.detach().cpu().numpy().astype(int)

        keypoints_xy = None
        keypoints_conf = None
        if result.keypoints is not None:
            if result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

        for i, box in enumerate(boxes):
            kp = np.asarray(keypoints_xy[i], dtype=np.float32) if keypoints_xy is not None else None
            obs = PersonObservation(
                track_id=int(ids[i]) if ids is not None and i < len(ids) else None,
                box=np.asarray(box, dtype=np.float32),
                confidence=float(confs[i]),
                keypoints=kp,
                keypoint_conf=np.asarray(keypoints_conf[i], dtype=np.float32) if keypoints_conf is not None else None,
            )
            obs.appearance = appearance_hist(frame, obs.box)
            obs.pose_signature = pose_signature(kp, obs.box)
            people.append(obs)
        return people


def find_initial_people(manual_a, manual_b, people: list[PersonObservation], frame=None):
    from core.identity import box_iou, normalized_distance

    def match(selection, candidates, excluded=None):
        available = [person for person in candidates if person is not excluded]
        if not available:
            return None, -1.0
        ranked = sorted(
            available,
            key=lambda person: (box_iou(selection, person.box), -normalized_distance(selection, person.box)),
            reverse=True,
        )
        best = ranked[0]
        overlap = box_iou(selection, best.box)
        # The first identity decision is irreversible enough to seed SAM2 and
        # every later pose/action result.  Centre distance alone is unsafe in a
        # fight because the referee often stands immediately beside or between
        # the selected fighters.  A detector candidate therefore has to agree
        # with a meaningful area of the user's box; otherwise the exact manual
        # selection becomes the temporary anchor.
        return (best, overlap) if overlap >= SETTINGS.min_initial_iou else (None, overlap)

    def manual_anchor(selection):
        """Create a temporary identity anchor when first-frame detection misses.

        The user-confirmed box is stronger evidence than aborting the fight.
        A later tracked observation can acquire a real BoT-SORT ID through the
        normal appearance/position recovery path.
        """
        box = np.asarray(selection, dtype=np.float32)
        return PersonObservation(
            track_id=None,
            box=box,
            confidence=0.55,
            keypoints=None,
            keypoint_conf=None,
            appearance=appearance_hist(frame, box) if frame is not None else None,
            # Without this the anchor has no learned appearance, every
            # comparison returns "no opinion", and the learned gate quietly
            # never runs while looking enabled.
            reid=(embed(frame, box.reshape(1, 4)) or [None])[0] if frame is not None else None,
            referee_prob=(
                referee_probabilities(frame, box.reshape(1, 4)) or [None]
            )[0] if frame is not None else None,
            pose_signature=None,
        )

    best_a, iou_a = match(manual_a, people)
    best_b, iou_b = match(manual_b, people, best_a)

    if best_a is None:
        best_a, iou_a = manual_anchor(manual_a), 0.0
    if best_b is None:
        best_b, iou_b = manual_anchor(manual_b), 0.0
    if best_a.track_id is not None and best_b.track_id is not None and best_a.track_id == best_b.track_id:
        raise RuntimeError("Fighter A and B received the same initial tracker ID.")
    return best_a, best_b, float(iou_a), float(iou_b)
