from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("WARRIORIQ_DATA_DIR", str(ROOT))).expanduser().resolve()
UPLOADS = DATA_ROOT / "uploads"
OUTPUTS = DATA_ROOT / "outputs"
MODELS = DATA_ROOT / "models"
DATASET = DATA_ROOT / "dataset"
DB_PATH = DATA_ROOT / "warrioriq.sqlite3"
ULTRALYTICS_CONFIG = DATA_ROOT / ".ultralytics"
HUGGINGFACE_CACHE = DATA_ROOT / ".huggingface"

# Keep Ultralytics settings inside the project. This avoids roaming-profile
# permission failures and keeps the local build's configuration self-contained.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE))

for path in (UPLOADS, OUTPUTS, MODELS, DATASET, ULTRALYTICS_CONFIG, HUGGINGFACE_CACHE):
    path.mkdir(parents=True, exist_ok=True)


def env_secret(name: str, default: str = "") -> str:
    """Read a shared secret, tolerating how hosting panels mangle pasted values.

    A secret is usually copied by hand from a control panel into a .env file or
    back again, and it commonly arrives wrapped: angle brackets left over from a
    placeholder like <token>, or quotes added to be safe. The wrapper is never
    part of the secret, but it makes authentication fail with a bare 401 that
    looks identical to a wrong password, so it is stripped here rather than
    debugged again at every call site.
    """
    value = os.getenv(name, default).strip()
    while len(value) >= 2 and value[0] in "<\"'" and value[-1] in ">\"'":
        value = value[1:-1].strip()
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


IS_RENDER = env_bool("RENDER", False)


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------
    # Core vision stack
    # ------------------------------------------------------------
    # YOLO26m Pose is the default compromise for an RTX 5060 8 GB and
    # the hard <= 1x video-duration budget. If a TensorRT engine exists,
    # WarriorIQ automatically prefers it.
    pose_model_pt: str = os.getenv("WARRIORIQ_POSE_MODEL", "yolo26m-pose.pt")
    pose_model_engine: str = os.getenv(
        "WARRIORIQ_POSE_ENGINE",
        str(MODELS / "yolo26m-pose.engine"),
    )
    tracker: str = os.getenv("WARRIORIQ_TRACKER", str(MODELS / "warrioriq_botsort.yaml"))
    # "auto" selects CUDA when available and CPU everywhere else.
    device: str = os.getenv("WARRIORIQ_DEVICE", "auto")
    default_imgsz: int = int(os.getenv("WARRIORIQ_IMGSZ", "640"))
    min_imgsz: int = int(os.getenv("WARRIORIQ_MIN_IMGSZ", "512"))
    # Fighters in a wide or low-resolution recording occupy very few pixels.
    # Inferring a 480px-wide video at 640 barely upscales it and the athletes
    # stay under the size the detector resolves reliably, so a small source is
    # analysed at a larger inference size instead.
    low_resolution_edge: int = int(os.getenv("WARRIORIQ_LOW_RES_EDGE", "960"))
    # Measured on real 480x220 tournament footage, frame by frame. At 1280 the
    # athletes are detected at 0.24 to 0.32 confidence - at or under BoT-SORT's
    # new_track_thresh of 0.30 - so the tracker never starts a track for them
    # and identity is left choosing between the referee and the spectators. At
    # 1600 the same fighters come in at 0.55 to 0.86 and clear it comfortably.
    # 1920 is worse, not better: the peaks do not improve and it adds a tail of
    # 0.12 to 0.19 detections for the tracker to trip over.
    low_resolution_imgsz: int = int(os.getenv("WARRIORIQ_LOW_RES_IMGSZ", "1600"))
    detection_conf: float = float(os.getenv("WARRIORIQ_DET_CONF", "0.20"))
    # Fighter drawing is fully manual. Candidate detection is only a visual
    # advisory, so free-tier web instances can skip loading YOLO on this page.
    selection_detection_enabled: bool = env_bool("WARRIORIQ_SELECTION_DETECTION", not IS_RENDER)

    # ------------------------------------------------------------
    # Performance target
    # ------------------------------------------------------------
    target_tracking_fps: float = float(os.getenv("WARRIORIQ_TARGET_FPS", "15"))
    min_tracking_fps: float = float(os.getenv("WARRIORIQ_MIN_FPS", "10"))
    max_tracking_fps: float = float(os.getenv("WARRIORIQ_MAX_FPS", "30"))
    # Hold the analysis to the length of the video by planning the sampling
    # stride once, early, from measured throughput. Until 2026-09-10 this
    # setting was defined here and **read nowhere in the codebase** - a switch
    # named for the product's headline performance promise that did nothing,
    # while a two-minute video took five and a half minutes. See
    # QualityController.plan_for_budget: it never samples below
    # min_tracking_fps, and when the floor is not enough it says so in the
    # report rather than quietly returning a worse analysis.
    hard_realtime_budget: bool = env_bool("WARRIORIQ_HARD_REALTIME", True)
    # Pin the sampling stride and skip planning entirely. The governor decides
    # from measured throughput, and measured throughput moves with machine load,
    # so a video whose required stride sits near a rounding boundary could be
    # planned differently on two runs. Everything this project concludes is
    # decided by A/B, and an A/B across a changed frame path is worthless - set
    # this for that work. 0 means plan normally.
    force_tracking_stride: int = int(os.getenv("WARRIORIQ_FORCE_STRIDE", "0"))
    # Fixed sampling is the reproducible default. Adaptive sampling depends on
    # momentary machine load and can make identical fights follow different
    # frame paths; it remains available as an explicit speed opt-in.
    adaptive_quality: bool = env_bool("WARRIORIQ_ADAPTIVE_QUALITY", False)
    progress_interval_frames: int = 40

    # ------------------------------------------------------------
    # Identity manager
    # ------------------------------------------------------------
    # A manual fighter selection must substantially overlap a detector box
    # before that detector identity is allowed to replace the user's box.
    # Low-overlap, center-near candidates are commonly referees.
    min_initial_iou: float = 0.35
    track_id_bonus: float = 0.34
    # How much a candidate must still resemble the fighter the user selected.
    # Real footage puts three to eleven people in frame - referees, corners,
    # crowd - at sizes close enough that position alone cannot separate them.
    # **This gate cannot tell the two fighters apart, and the numbers that said
    # it could were measured on one bout.** That comment read: "the same
    # fighter scores 0.67 at worst and 0.94 typically; two different people
    # score 0.56 typically and 0.48 at the low end" - a clean gap, with this
    # threshold sitting safely under it.
    #
    # Re-measured on 926 comparisons across all six fighter tracks in the three
    # reference fights:
    #
    #     same fighter    median 0.789   min 0.494
    #     other fighter   median 0.683   max 0.964
    #
    # The worst same-fighter score is 0.494, not 0.67 - every track dips below
    # the claimed floor - and **100% of impostor scores land above it**. As a
    # discriminator the histogram is AUC 0.667, which is barely a signal at
    # all. No threshold separates these distributions, so do not go tuning this
    # number expecting one to.
    #
    # What 0.55 actually is: a floor that refuses the crowd, who look nothing
    # like either athlete. It refuses the *correct* fighter in 2-9% of frames
    # as the cost of that. Raising it refuses more of the right person;
    # lowering it admits more of the wrong one. Telling the two athletes apart
    # is not this gate's job and it never managed it - see
    # project-validated-on-three-fights for the pooled-ReID equivalent, which
    # is 0.89-0.90 between the two fighters and no better.
    min_anchor_appearance_similarity: float = float(
        os.getenv("WARRIORIQ_MIN_ANCHOR_SIMILARITY", "0.55"))
    # How alike the two selected fighters may look before per-fighter results
    # stop meaning anything. This is a different question from every other
    # threshold here: not "is this candidate the fighter" but "can these two be
    # told apart at all in this video", and it is answerable in a second, at
    # selection, before anything is analysed.
    #
    # Measured on three bouts, comparing the two chosen fighters to each other:
    # the two where identity holds score 0.613 and 0.640, and the one where it
    # cannot be made to work scores 0.891. The learned embedding is no use for
    # this - it says 0.89 to 0.91 for all three alike - because it is trained
    # to describe a person, and two people in similar kit genuinely are similar.
    # Colour is cruder and, for this one question, better.
    #
    # Deliberately nearer the failing case: a false warning costs a user a
    # re-selection, while a missed one costs them a scorecard that names the
    # wrong fighter.
    max_fighter_pair_similarity: float = float(
        os.getenv("WARRIORIQ_MAX_PAIR_SIMILARITY", "0.78"))
    # A learned appearance space for the same gate. The histogram above cannot
    # separate a referee from a fighter - measured, their ranges overlap almost
    # completely - while an embedding puts the referee at 0.695-0.741 and the
    # fighters at 0.724-0.828. See core/reid.py. Off until measured end to end.
    reid_enabled: bool = env_bool("WARRIORIQ_REID", True)
    # A purpose-trained re-identification encoder, not the detector. The old
    # default here was "yolo26m.pt", which is the detector: Ultralytics routes
    # a .pt through the YOLO predictor and reads the second-to-last layer, so
    # what came back described "this is a person" rather than "this is *which*
    # person". That is why two attempts at a learned appearance gate failed.
    # It was also far slower, because the predictor path runs a full detection
    # per call; the .onnx assets go through AutoBackend at 10.4 ms per person.
    #
    # The nano encoder is used deliberately: measured on real footage it
    # separates better than the medium one (pooled AUC 0.996 against 0.989)
    # and costs the same, since the time goes on cropping rather than on the
    # network. There is not enough detail in a 60-pixel-tall person for the
    # larger model's extra capacity to describe.
    reid_model: str = os.getenv("WARRIORIQ_REID_MODEL", "yolo26n-reid.onnx").strip()
    # Embeddings are pooled over a track before being compared. One crop of a
    # person this small is too noisy to identify anybody: measured on the
    # busiest bout, single crops separate the same person from a different one
    # at AUC 0.689, and the mean of three at 0.996. Same encoder, same frames -
    # the descriptor was never the problem, the sample size was.
    reid_pool_size: int = int(os.getenv("WARRIORIQ_REID_POOL", "6"))
    reid_min_pool: int = int(os.getenv("WARRIORIQ_REID_MIN_POOL", "3"))
    reid_device: str = os.getenv("WARRIORIQ_REID_DEVICE", "").strip()
    # Against pooled embeddings, not single crops. Measured across three bouts:
    # the same person scores a median of 0.93 to 0.98 and a different person
    # 0.56 to 0.82. The tails overlap, so this cannot be set to separate them
    # cleanly and the value is a chosen trade, measured by eye on all three:
    #
    #   0.72  the quiet bout keeps 11/12 and 10/12; the crowded one collapses
    #         to 5/12, barely better than having no gate at all
    #   0.78  the quiet bout 10/12 and 10/12; the crowded one 9/12, with its
    #         wrong boxes down from roughly 800 to 220
    #   0.85  the crowded one reaches 10/12 and loses half its coverage
    #
    # 0.78 is the middle. Raising it buys precision on a busy hall and costs
    # frames everywhere; lowering it gives the crowd back.
    #
    # **Lowering it was tried properly on 2026-09-09 and it is a downgrade at
    # every value tested. Do not try it again.** This gate is the binding
    # constraint on coverage - measured per lost fighter rather than per
    # candidate, it accounts for **61%** of the frames where a fighter is
    # unassigned (753 of ~1,243 on fight 3), far ahead of the referee filter at
    # 9%. So relaxing it looks like the obvious move, and the coverage numbers
    # agree loudly. They are wrong.
    #
    #     threshold   A coverage   B coverage   both held
    #     0.78          0.445        0.199       83 frames
    #     0.70          0.492        0.475      ---
    #     0.60          0.529        0.611      243 frames
    #
    # Twelve gained boxes per run were rendered with the boxes drawn and judged
    # by eye - the only test that has ever caught this:
    #
    #     0.60   8 of 12 right overall, but split by fighter A is 7/8 and B is
    #            1/4: B's tripled coverage is a kneeling official, a coach by
    #            the chairs, a spectator in red.
    #     0.70   B is right in about 2 of 12, and the dominant failure is new -
    #            **B locks onto fighter A**. Frames 896, 1206, 1214, 1222, 1224
    #            and 1240 all put B's box on A's man.
    #
    # That last line is why this stays where it is. Hand labelling all 178 clips
    # at 0.78 found ten referees, eight bystanders, six id-swaps and **zero**
    # A-versus-B confusions; lowering the gate manufactures the one failure the
    # product had never made. An A/B swap is the worst outcome available here,
    # because it silently credits one fighter's work to the other and coverage
    # rises while it happens.
    #
    # The asymmetry is not a tuning accident. Fighter A wears light blue and
    # white; fighter B wears black in a hall where the officials, the coaches
    # and half the crowd wear black. Compared against every other person
    # detected in the opening exchange, 17% of them clear this gate against B's
    # anchor and only 5% against A's. A single global number cannot serve both
    # fighters of the same bout, and the honest fix - if there is one - is a
    # per-fighter gate set from how separable that fighter is from the room,
    # not a different constant. (Those two percentages are single-crop
    # comparisons, while the gate itself compares pooled embeddings, so treat
    # them as the direction of the effect and not its size.)
    min_anchor_reid_similarity: float = float(
        os.getenv("WARRIORIQ_MIN_ANCHOR_REID", "0.78"))
    # A categorical refusal rather than another threshold on similarity. The
    # referee passes every comparative guard honestly, so the only thing that
    # separates him is what he is wearing. See core/referee.py for why this
    # feature is deliberately four numbers wide and per-federation.
    referee_filter_enabled: bool = env_bool("WARRIORIQ_REFEREE_FILTER", True)
    referee_probe_path: str = os.getenv(
        "WARRIORIQ_REFEREE_PROBE", "models/referee_probe.npz").strip()
    # Was 0.50, on the reasoning that "the official scores 0.56 to 0.997 and
    # everyone else 0.003 to 0.091, so the midpoint costs nothing". That gap
    # was real on the 58 crops it was measured on - all from one bout - and it
    # is not the gap on other footage.
    #
    # Re-measured on **178 hand-labelled clips across all three fights**, ten
    # of them the official. The probe still *ranks* beautifully (AUC 0.993),
    # but referees score anywhere from 0.06 to 0.99, so at 0.50 it caught only
    # five of ten. What separates them is not a wide margin, it is that
    # everyone else scores lower still:
    #
    # The gate is applied to **every candidate during tracking**, not just the
    # seed, so the cost of lowering it is real fighter observations refused.
    # Both sides were measured - officials against the ten hand-labelled ones,
    # cost against all 469 tracked fighter boxes across the three fights:
    #
    #     threshold   officials caught   real fighter boxes refused
    #        0.50           5 / 10             1 / 469  (0.2%)
    #        0.16           5 / 10            11 / 469  (2.3%)
    #        0.15           8 / 10            12 / 469  (2.6%)
    #        0.12           8 / 10            15 / 469  (3.2%)
    #        0.05          10 / 10            far more
    #
    # **0.15 was tried on that evidence and reverted.** Do not try it again
    # without reading this.
    #
    # The table above is real but it is the wrong table. Both columns were
    # measured on boxes the tracker had already *chosen* - and this gate does
    # not filter chosen boxes, it filters every candidate offered during
    # tracking. Two full runs of fight 1, identical code, only the threshold
    # different, then the frames where they disagreed rendered and looked at:
    #
    #     89 frames changed. Only 6 of them were the referee being removed.
    #     51 lost their box entirely. Of six sampled by eye, one dropped an
    #     official and three dropped a real fighter.
    #
    # Coverage said it was a wash (A 0.192 -> 0.235, B 0.397 -> 0.360), which
    # is exactly why coverage is not the test. Three athletes lost for one
    # official removed is a bad trade, so the gate stays where it was.
    #
    # What survives from the measurement: the probe *ranks* officials well
    # (AUC 0.993 across three venues) and 0.50 catches only five of ten. The
    # five it misses are not reachable by moving this number - they need a
    # probe trained across venues. See project-detector-measured-on-178-clips.
    min_referee_probability: float = float(
        os.getenv("WARRIORIQ_MIN_REFEREE_PROB", "0.50"))
    # Fighter motion between sampled frames is often larger than generic
    # pedestrian motion. These values still reject distant bystanders, while
    # allowing a fighter to be recovered after a tracker-ID reset.
    min_reid_score: float = 0.48
    min_reid_margin: float = 0.05
    max_normalized_jump: float = 1.65
    max_missing_analyzed_frames: int = 20
    missing_before_recovery: int = 4
    appearance_ema: float = 0.85
    pose_ema: float = 0.82

    # ------------------------------------------------------------
    # SAM recovery
    # ------------------------------------------------------------
    # SAM2 propagates the two user-selected identities through the segment.
    # YOLO pose/ReID must still confirm a person before metrics are accepted.
    sam_recovery_enabled: bool = env_bool("WARRIORIQ_SAM_RECOVERY", not IS_RENDER)
    # Continuous mode propagates SAM2 masks across the whole segment before the
    # main pass. **On by default since 2026-09-10.** It was off, on this:
    #
    #   continuous on   188.8s   coverage A 0.997  B 0.916
    #   continuous off   62.6s   coverage A 0.985  B 0.987
    #
    # Those are 98-99% coverage figures, which is what this project reported
    # while it was following a seated spectator - see
    # project-coverage-can-mean-failure. The comparison that retired this
    # feature was measuring the wrong people, and "fighter B was tracked better
    # without it" was a spectator being held more steadily than an athlete.
    #
    # Turning it on also crashed until today (see recover_from_guidance in
    # core/pose_tracker.py), so nobody could have checked.
    #
    # Re-measured across all three reference fights, one variable, same seeds
    # and windows:
    #
    #     fight   A off -> on      B off -> on       time
    #     fam3    0.198 -> 0.637   0.459 -> 0.671    159 -> 179s
    #     f2      0.182 -> 0.438   0.105 -> 0.213    106 -> 148s
    #     f3      0.445 -> 0.605   0.199 -> 0.544    223 -> 264s
    #
    # Coverage is not the reason. The changed frames were rendered and looked at
    # in **both** directions, which is what the referee-threshold A/B further up
    # this file did and what makes the difference:
    #
    #   * gained boxes are mostly the right fighter - on f3, five of six, one
    #     of them inside the 58.6-59.6s knockdown the old settings missed
    #     entirely
    #   * **every one of six sampled lost boxes was on the wrong person** - two
    #     bystanders and a seated spectator in the chairs. The coverage the old
    #     settings reported was partly crowd, and this drops it
    #   * moved boxes stay on the correct athlete
    #
    # The cost is 13-40% more time, not the doubling above; that figure is as
    # stale as the coverage it sits beside.
    #
    # Known costs, not hidden: degenerate narrow boxes rise from 0% to 6% on
    # fam3 and 3% to 6% on f2 (a mask collapsing to a sliver), fam3 gains one
    # identity confusion where it had none, and one sampled fam3 box landed on
    # the referee. None of that outweighs dropping spectators, but it is the
    # thing to look at next.
    #
    # This does **not** unlock scoring: 0.54-0.67 is still under the 0.85
    # min_tracking_coverage_for_score gate.
    sam_continuous_enabled: bool = env_bool("WARRIORIQ_SAM_CONTINUOUS", True)
    sam_continuous_fps: float = float(os.getenv("WARRIORIQ_SAM_FPS", "4"))
    sam_continuous_max_frames: int = int(os.getenv("WARRIORIQ_SAM_MAX_FRAMES", "360"))
    # Keep the bounded two-minute guidance pass in one memory state. Short
    # chunk resets were faster but could drift when reseeded during a crossing.
    sam_continuous_chunk_frames: int = int(os.getenv("WARRIORIQ_SAM_CHUNK_FRAMES", "360"))
    sam_model_id: str = os.getenv("WARRIORIQ_SAM_MODEL", "facebook/sam2.1-hiera-small")
    sam_buffer_frames: int = int(os.getenv("WARRIORIQ_SAM_BUFFER", "18"))
    sam_cooldown_analyzed_frames: int = 24
    openai_identity_model: str = os.getenv("WARRIORIQ_OPENAI_MODEL", "gpt-5.6-terra")
    openai_identity_min_confidence: float = float(os.getenv("WARRIORIQ_OPENAI_ID_MIN_CONF", "0.82"))
    openai_identity_cooldown_seconds: float = 10.0
    openai_identity_audit_seconds: float = 15.0

    # ------------------------------------------------------------
    # Temporal action recognition
    # ------------------------------------------------------------
    temporal_checkpoint: str = os.getenv(
        "WARRIORIQ_TEMPORAL_MODEL",
        str(MODELS / "warrioriq_temporal_best.pt"),
    )
    action_window: int = 12
    min_event_gap_seconds: float = 0.22
    min_strike_speed_body_lengths_per_s: float = 0.90
    # A limb cannot move this fast. Measured on one real fight, candidate peak
    # speeds ran to 47.6 body lengths a second - roughly 80 m/s - because the
    # trigger read a single frame-to-frame displacement, and one jittery
    # keypoint is indistinguishable from a strike. A human limb peaks around
    # 6-9, so anything past this is tracking noise and the frame is ignored
    # rather than allowed to start or define an action.
    max_plausible_limb_speed_body_lengths_per_s: float = float(
        os.getenv("WARRIORIQ_MAX_LIMB_SPEED", "12.0")
    )
    min_extension_gain: float = 0.07
    temporal_probability_threshold: float = 0.60

    # ------------------------------------------------------------
    # Contact / outcome
    # ------------------------------------------------------------
    # Beyond this separation the opponent simply cannot be reached, so an action
    # fired here was not thrown at them. Measured across four real fights: the
    # two tracked fighters were a median 2.78 body lengths apart when a strike
    # was detected, and only 18% of detections happened inside 1.5 - so most
    # counted "attempts" were thrown at nobody and dragged every accuracy figure
    # down with them. A kick reaches roughly 1.5-2 body lengths; 2.5 is generous.
    max_engagement_body_lengths: float = float(os.getenv("WARRIORIQ_MAX_ENGAGEMENT", "2.5"))
    # The fighters being close is not the same as the strike reaching. Measured
    # on real footage: strikes labelled kick-to-leg finished a median 2.46 body
    # lengths from any target, and punches with no target at all made up a
    # quarter of what was counted - leg movement and feints read as strikes.
    # An arm reaches about 0.5 body lengths and a kick about 0.8, so 1.0 is
    # generous. This removes non-strikes from the denominator; it never turns a
    # miss into a landed strike.
    max_strike_reach_body_lengths: float = float(os.getenv("WARRIORIQ_MAX_STRIKE_REACH", "1.0"))
    # Whether the two people picked are actually the two fighters. Measured on
    # Whether an action may be reported as a strike when the opponent was never
    # seen during it. Measured on real footage: of 81 proposed actions, only 9
    # had the opponent in frame at the peak and 31 never saw them at all, in a
    # bout where the fighters are on the mat together throughout. Those are not
    # strikes that missed, they are actions with no observed target - a fighter
    # shadow-boxing at a gap in the tracking - and they were the bulk of what a
    # human reviewer called "nothing happening".
    require_observed_opponent: bool = env_bool("WARRIORIQ_REQUIRE_OPPONENT", True)
    # three real tournament fights, each analysed twice - once with the real
    # When a strike lands, the fighter receiving it moves too - the guard is
    # driven back, the head turns, an arm is displaced - and the detector reads
    # that as an action by the defender. Measured on three bouts: 72% to 86% of
    # accepted events sit in a pair with the other fighter peaking within a
    # third of a second, which is what that looks like from outside.
    #
    # Real simultaneous exchanges also happen, so the pair is only split when
    # one limb clearly got closer to the opponent than the other. At a margin
    # of 0.2 body lengths that is about half of pairs; the rest are left alone.
    resolve_simultaneous_attribution: bool = env_bool("WARRIORIQ_RESOLVE_ATTRIBUTION", True)
    simultaneous_window_seconds: float = float(
        os.getenv("WARRIORIQ_SIMULTANEOUS_WINDOW", "0.35"))
    attribution_reach_margin: float = float(
        os.getenv("WARRIORIQ_ATTRIBUTION_MARGIN", "0.20"))
    # pair and once with a plausible mistake (a ringside coach, or the referee):
    #
    #   fight       correct pair   mistaken pair
    #   2.mp4          0.76           2.84 (coach)
    #   3.mp4          2.55           3.25 (referee)
    #   0-02-05        1.90           2.71 (coach)
    #
    # The mistaken pair always sits further apart, but correct tops out at 2.55
    # and mistaken starts at 2.71, so there is no room to separate them on
    # distance alone. This threshold therefore sits above every correct pair
    # measured and is paired with "nothing landed at all" below. That catches
    # the unmissable case - a fighter matched with someone who never fights
    # back - and stays silent otherwise. It is deliberately not tuned to catch
    # every mistake: a false accusation on a real fight is the worse error.
    max_median_separation_body_lengths: float = float(
        os.getenv("WARRIORIQ_MAX_MEDIAN_SEPARATION", "2.6")
    )
    # Below this many observed actions there is nothing to judge, and a quiet
    # fight must not be accused of being the wrong two people.
    min_actions_to_judge_selection: int = int(
        os.getenv("WARRIORIQ_MIN_SELECTION_ACTIONS", "12")
    )
    # Fighters move; ringside does not. Measured in body lengths travelled per
    # minute, so it means the same at any camera distance. Across every real run
    # recorded so far - correct pairs and mis-picked ones alike - both tracked
    # people covered 24 to 65 per minute. A live job that had latched onto
    # someone standing at the edge of the mat covered 9.6. This sits below every
    # real observation and well above that, and it catches the case a separation
    # test cannot: two wrong people who happen to stand near each other.
    min_fighter_travel_per_minute: float = float(
        os.getenv("WARRIORIQ_MIN_TRAVEL_PER_MINUTE", "15")
    )
    # Below this, a track is not allowed to take a fighter's identity from
    # another track. Lower than the reporting floor above because refusing a
    # switch is cheap and being wrong about it is not: a fighter resting in a
    # corner should still never be handed to the referee. Only blocks moving
    # onto a new track, never drops the one already held.
    min_switch_travel_per_minute: float = float(
        os.getenv("WARRIORIQ_MIN_SWITCH_TRAVEL", "12")
    )
    # How far a track must roam around its own average position, in body
    # lengths, before it may take a fighter's identity. The travel test above
    # measures path length, which detection noise on a seated spectator quietly
    # accumulates; this one does not accumulate, so a trembling box in the
    # crowd stays small however long it is watched. Only applied after six
    # seconds of continuous history, so a fighter pinned on the ropes is never
    # read as furniture.
    min_switch_spread_body_lengths: float = float(
        os.getenv("WARRIORIQ_MIN_SWITCH_SPREAD", "0.5")
    )
    # A second, much stricter reading of the same measurement over a much
    # shorter look. The 6 s guard above is calibrated to catch marginal cases
    # and so must wait; somebody who has not moved at all is decidable sooner.
    #
    # Calibrated on *rolling* 1.5 s windows, which is what the guard actually
    # reads. Measuring each track's opening 1.5 s instead suggested a clean gap
    # at 0.04 and there is none: a fighter in a clinch or between exchanges is
    # far stiller mid-round than when first seen, and 0.04 refused one in five
    # of their windows. Across a full round, per track: seated spectators and
    # the referee bottom out at 0.003 to 0.031 body lengths, while the least
    # mobile fighter never goes below 0.029. This sits under every fighter
    # window measured and still catches the people who never move at all.
    max_stationary_spread_short: float = float(
        os.getenv("WARRIORIQ_MAX_STATIONARY_SPREAD_SHORT", "0.02"))
    stationary_short_seconds: float = float(
        os.getenv("WARRIORIQ_STATIONARY_SHORT_SECONDS", "1.5"))
    # Seconds of video immediately before the round to run through the tracker
    # for its motion history alone. Without it the guards above cannot judge
    # anybody until the round is already this far along, and a spectator
    # acquired in that window is held until they can. Costs one short pass at
    # the analysis stride; 0 disables it.
    identity_warmup_seconds: float = float(
        os.getenv("WARRIORIQ_IDENTITY_WARMUP_SECONDS", "2.0"))
    # A second opinion on the two fighters' joints, from a top-down pose model.
    # The fused detector regresses keypoints from whole-frame features and
    # collapses on small or unusual bodies - measured on real footage it
    # squeezed a fighter's torso to a sliver and put both legs on one point
    # while reporting 0.93 ankle confidence. RTMPose receives the fighter
    # cropped and rescaled to its full input and gets it right.
    #
    # On, and the cost is smaller than it first looked. 15.7 ms per person with
    # no batching benefit reads like a doubling, but only the two committed
    # fighters are refined and the rest of the pipeline dominates: measured end
    # to end on a 60 s bout, 109.1 s against 122.8 s, or 12.6%. That overruns
    # the 10 s budget this was asked to fit in, and is worth saying plainly -
    # but the detector's own head returns skeletons with the torso squeezed to
    # a sliver and both legs converging on one point while reporting 0.93
    # ankle confidence, and every metric downstream reads those joints.
    #
    # What this does not yet prove is that the techniques it names are more
    # often right. It resolves twice as many distinct techniques - uppercuts,
    # low kicks and knees appear where there were none - which is consistent
    # with better joints and is not evidence of correctness. That needs labels.
    rtm_pose_enabled: bool = env_bool("WARRIORIQ_RTM_POSE", True)
    rtm_pose_device: str = os.getenv("WARRIORIQ_RTM_POSE_DEVICE", "cuda").strip()
    rtm_pose_model: str = os.getenv(
        "WARRIORIQ_RTM_POSE_MODEL",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    ).strip()
    # How often the analysis was looking at one of the two people who actually
    # fought. Set once measured on real runs; a correct run loses its fighters
    # to occlusion and pans, so this only has to separate "followed the fight"
    # from "followed somebody else entirely".
    min_followed_share: float = float(os.getenv("WARRIORIQ_MIN_FOLLOWED_SHARE", "0.35"))
    # Who the pair-finder will consider at all. Lower than the warning floor
    # above on purpose: this only decides who is eligible to be one of the two
    # fighters, and the pairing does the real work. Set equal to the warning
    # floor, whole fights produced no candidate pair and the check said nothing.
    min_candidate_travel_per_minute: float = float(
        os.getenv("WARRIORIQ_MIN_CANDIDATE_TRAVEL", "8")
    )
    contact_threshold_body_lengths: float = 0.24
    likely_contact_threshold_body_lengths: float = 0.33
    contact_confirmation_frames: int = 2
    block_proximity_body_lengths: float = 0.18

    # ------------------------------------------------------------
    # Reports / evidence
    # ------------------------------------------------------------
    min_pose_coverage_for_metric: float = 0.60
    # Below this many observed frames there is genuinely too little to average,
    # and a mean of a handful of frames is noise wearing a number's clothes.
    # This is a sample floor and not a coverage one: a fighter observed for
    # four hundred frames of a round is measurable, and the report says so by
    # carrying the sample size rather than by hiding the result.
    min_metric_samples: int = int(os.getenv("WARRIORIQ_MIN_METRIC_SAMPLES", "60"))
    # A score remains explicitly estimated, but 85% verified coverage for
    # both fighters is sufficient when each counted action also passes the
    # stricter technique/contact evidence thresholds in scoring.py.
    min_tracking_coverage_for_score: float = float(os.getenv("WARRIORIQ_SCORE_COVERAGE", "0.85"))
    # A ten-point-must round is a judgement about who controlled a round, and
    # two detected actions cannot support one. Measured on real 480x220
    # tournament footage: 244 actions detected, 2 of them verified, and the
    # scorer still published a 9-10 round with a named winner. The score is
    # withheld below this; every movement measurement is kept.
    min_verified_actions_for_score: int = int(os.getenv("WARRIORIQ_MIN_SCORING_ACTIONS", "6"))
    evidence_pre_seconds: float = 1.0
    evidence_post_seconds: float = 1.0
    save_tracking_jsonl: bool = True
    save_debug_video: bool = False

    # ------------------------------------------------------------
    # Product
    # ------------------------------------------------------------
    brand_name: str = "WarriorIQ"
    version: str = "1.0"
    default_profile_name: str = "My Athlete"
    payments_enabled: bool = env_bool("WARRIORIQ_PAYMENTS", False)
    # The launch checklist names the operator details still missing - company
    # name, registration number, copyright agent. Useful to whoever is setting
    # WarriorIQ up, and to a visitor it reads as "this company does not exist
    # yet". Off unless the operator deliberately turns it on.
    show_launch_checklist: bool = env_bool("WARRIORIQ_SHOW_LAUNCH_CHECKLIST", False)

    # Social sign-in is opt-in per provider. A provider is exposed only when
    # both credentials and a stable state-signing secret are configured.
    oauth_state_secret: str = os.getenv("WARRIORIQ_OAUTH_STATE_SECRET", "").strip()
    google_client_id: str = os.getenv("WARRIORIQ_GOOGLE_CLIENT_ID", "").strip()
    google_client_secret: str = os.getenv("WARRIORIQ_GOOGLE_CLIENT_SECRET", "").strip()
    facebook_client_id: str = os.getenv("WARRIORIQ_FACEBOOK_CLIENT_ID", "").strip()
    facebook_client_secret: str = os.getenv("WARRIORIQ_FACEBOOK_CLIENT_SECRET", "").strip()
    microsoft_client_id: str = os.getenv("WARRIORIQ_MICROSOFT_CLIENT_ID", "").strip()
    microsoft_client_secret: str = os.getenv("WARRIORIQ_MICROSOFT_CLIENT_SECRET", "").strip()
    github_client_id: str = os.getenv("WARRIORIQ_GITHUB_CLIENT_ID", "").strip()
    github_client_secret: str = os.getenv("WARRIORIQ_GITHUB_CLIENT_SECRET", "").strip()

    # Analysis jobs can run inside the web process for a local/PyCharm build,
    # or be claimed by a separate GPU worker in production.  External mode is
    # deliberately opt-in so existing local installs keep working unchanged.
    analysis_worker_mode: str = os.getenv("WARRIORIQ_WORKER_MODE", "inprocess").strip().lower()
    worker_poll_seconds: float = max(0.2, float(os.getenv("WARRIORIQ_WORKER_POLL_SECONDS", "1.0")))
    worker_lease_seconds: int = max(30, int(os.getenv("WARRIORIQ_WORKER_LEASE_SECONDS", "180")))
    worker_stale_seconds: int = max(60, int(os.getenv("WARRIORIQ_WORKER_STALE_SECONDS", "300")))
    # Remote mode lets a GPU machine claim jobs over HTTPS instead of requiring
    # the web server and worker to share a filesystem. The token must be the
    # same high-entropy secret on both machines and is never sent to browsers.
    # A detached worker keeps the queue durable, so a fight can be accepted while
    # the analysis machine is switched off and claimed when it next connects.
    # Turn this off to refuse uploads whenever no worker is currently online.
    accept_deferred_analysis: bool = env_bool("WARRIORIQ_ACCEPT_DEFERRED_ANALYSIS", True)
    # Optional wake hook for a scale-to-zero GPU. When a fight is queued with no
    # worker online, the web process pings this URL so a serverless GPU starts,
    # drains the queue and shuts down again. Polling a GPU would bill idle time;
    # waking one on demand bills only the analysis itself.
    worker_wake_url: str = os.getenv("WARRIORIQ_WORKER_WAKE_URL", "").strip()
    # Wake-on-LAN for an analysis machine that sleeps between fights. The queue
    # cannot reach a sleeping PC and a sleeping PC cannot poll, so the web
    # process sends a magic packet the network card still listens for. Needs the
    # router to forward this UDP port to the LAN broadcast address. Purely an
    # accelerator: the scheduled drain still collects the fight if it fails.
    wol_mac: str = os.getenv("WARRIORIQ_WOL_MAC", "").strip()
    wol_host: str = os.getenv("WARRIORIQ_WOL_HOST", "").strip()
    wol_port: int = int(os.getenv("WARRIORIQ_WOL_PORT", "9"))
    # How often deploy/drain-queue.ps1 is scheduled to wake the machine. This is
    # the guaranteed ceiling on how long a fight waits when the magic packet
    # never arrives, and it is what the uploader is promised before any real
    # wake latency has been observed.
    wake_drain_interval_seconds: int = int(os.getenv("WARRIORIQ_DRAIN_INTERVAL", "300"))
    worker_remote_url: str = os.getenv("WARRIORIQ_WORKER_REMOTE_URL", "").rstrip("/")
    worker_token: str = env_secret("WARRIORIQ_WORKER_TOKEN")
    worker_artifact_max_bytes: int = max(
        10 * 1024 * 1024,
        int(os.getenv("WARRIORIQ_WORKER_ARTIFACT_MAX_BYTES", str(256 * 1024 * 1024))),
    )
    minimum_free_storage_gb: float = max(0.25, float(os.getenv("WARRIORIQ_MIN_FREE_STORAGE_GB", "2")))
    max_fight_bytes: int = max(
        50 * 1024 * 1024,
        int(os.getenv("WARRIORIQ_MAX_FIGHT_BYTES", str(2 * 1024 * 1024 * 1024))),
    )
    # What the web host will actually accept in one request body, which is not
    # the same thing as what WarriorIQ would allow. Measured against the live
    # server: the body is refused at exactly 130 MiB, three times running, with
    # a 500 rather than a 413 - so an upload ran for a minute and then died
    # with no usable message. max_fight_bytes above says 2 GB and cannot be
    # honoured while this is smaller. Raise this once the host's
    # LimitRequestBody is raised, and the page follows it.
    max_upload_bytes: int = max(
        8 * 1024 * 1024,
        int(os.getenv("WARRIORIQ_MAX_UPLOAD_BYTES", str(130 * 1024 * 1024))),
    )
    max_video_duration_seconds: int = max(60, int(os.getenv("WARRIORIQ_MAX_VIDEO_SECONDS", "10800")))
    max_video_pixels: int = max(640 * 360, int(os.getenv("WARRIORIQ_MAX_VIDEO_PIXELS", str(3840 * 2160))))
    malware_scan_command: str = os.getenv("WARRIORIQ_MALWARE_SCAN_COMMAND", "").strip()
    malware_scan_required: bool = env_bool("WARRIORIQ_MALWARE_SCAN_REQUIRED", False)

    # ------------------------------------------------------------
    # Public-launch and legal identity
    # ------------------------------------------------------------
    # These values intentionally have no made-up defaults. Paid checkout is
    # blocked until the real operator has supplied every launch-critical item.
    policy_version: str = os.getenv("WARRIORIQ_POLICY_VERSION", "2026-08-24")
    public_base_url: str = os.getenv("WARRIORIQ_PUBLIC_BASE_URL", "").rstrip("/")
    operator_name: str = os.getenv("WARRIORIQ_OPERATOR_NAME", "").strip()
    operator_address: str = os.getenv("WARRIORIQ_OPERATOR_ADDRESS", "").strip()
    operator_registration: str = os.getenv("WARRIORIQ_OPERATOR_REGISTRATION", "").strip()
    operator_vat: str = os.getenv("WARRIORIQ_OPERATOR_VAT", "").strip()
    governing_country: str = os.getenv("WARRIORIQ_GOVERNING_COUNTRY", "").strip()
    support_email: str = os.getenv("WARRIORIQ_SUPPORT_EMAIL", "").strip()
    privacy_email: str = os.getenv("WARRIORIQ_PRIVACY_EMAIL", "").strip()
    dmca_email: str = os.getenv("WARRIORIQ_DMCA_EMAIL", "").strip()
    dmca_agent_name: str = os.getenv("WARRIORIQ_DMCA_AGENT_NAME", "").strip()
    minimum_account_age: int = int(os.getenv("WARRIORIQ_MINIMUM_AGE", "18"))
    saved_video_retention_days: int = max(1, int(os.getenv("WARRIORIQ_VIDEO_RETENTION_DAYS", "30")))
    failed_upload_retention_hours: int = max(1, int(os.getenv("WARRIORIQ_FAILED_UPLOAD_RETENTION_HOURS", "24")))
    # Complimentary plan grants as "email:plan_key" pairs, comma separated.
    # Kept in configuration rather than as a row edit on the live database so a
    # grant is visible, reviewable and survives a restore.
    #
    # No default. The owner's own address was hardcoded here, which published a
    # personal email in a public repository and named the account holding a free
    # plan. Grants belong in the server's environment, not in source.
    complimentary_plans: dict[str, str] = field(default_factory=lambda: {
        email.strip().lower(): plan.strip().lower()
        for entry in os.getenv("WARRIORIQ_COMPLIMENTARY_PLANS", "").split(",")
        if entry.strip() and ":" in entry
        for email, plan in [entry.split(":", 1)]
    })
    admin_emails: tuple[str, ...] = tuple(
        email.strip().lower()
        for email in os.getenv("WARRIORIQ_ADMIN_EMAILS", "").split(",")
        if email.strip()
    )
    # Google Analytics measurement ID (G-XXXXXXXXXX). The tag is rendered only
    # for visitors who accept analytics cookies; leaving this empty disables
    # analytics entirely and keeps the strict Content-Security-Policy.
    #
    # **The default below does not exist in Google's system.** Checked
    # 2026-09-09 against the live site: `googletagmanager.com/gtag/js?id=` for
    # this ID returns **HTTP 404**, six times, with and without a Referer. A
    # completely made-up ID (G-ZZZZZZZZZZ) returns 200 and a working 428 KB
    # script, so a 404 is not what an unknown ID looks like - this string is
    # rejected outright. The consequence, confirmed in the browser: gtag is
    # defined, the config command reaches the dataLayer, consent is granted,
    # and **no /g/collect request is ever sent**, which is why every report is
    # empty however many devices are tested.
    #
    # Replace it with the Measurement ID from the GA4 property itself:
    # Admin -> Data streams -> the web stream -> Measurement ID, top right.
    # Set WARRIORIQ_ANALYTICS_ID rather than editing this line, so the value
    # lives with the deployment and not in the repository.
    analytics_measurement_id: str = os.getenv("WARRIORIQ_ANALYTICS_ID", "G-5V5Q4H30LD").strip()
    # Search Console ownership token: the `content` value of the
    # <meta name="google-site-verification"> tag Google offers under
    # "HTML tag" verification. Empty renders no tag, which is the honest
    # default - an empty or invented token fails verification silently and
    # looks identical to not having tried.
    site_verification_token: str = os.getenv("WARRIORIQ_SITE_VERIFICATION", "").strip()
    # Google Tag Manager container. GTM loads whatever tags the container holds,
    # so if a GA4 tag inside it uses the same measurement ID as
    # WARRIORIQ_ANALYTICS_ID, every page view is counted twice. Run one or the
    # other: empty this to use the direct tag, or empty the measurement ID to
    # let the container own analytics.
    gtm_container_id: str = os.getenv("WARRIORIQ_GTM_ID", "GTM-PFCW27J2").strip()
    email_provider: str = os.getenv("WARRIORIQ_EMAIL_PROVIDER", "").strip()
    require_email_verification: bool = env_bool("WARRIORIQ_REQUIRE_EMAIL_VERIFICATION", False)


SETTINGS = Settings()

# Rule labels shown everywhere in the product.
# Grouped by sport for the upload form. The key order here is the order a
# visitor sees, so the discipline most people arrive for stays first.
RULESET_SPORTS = {
    "kickboxing": "Kickboxing",
    "boxing": "Boxing",
    "muay_thai": "Muay Thai",
    "taekwondo": "Taekwondo",
    "mma": "MMA",
}

RULESET_LABELS = {
    "K1": "K-1",
    "LOW_KICK": "Low Kick",
    "FULL_CONTACT": "Full Contact",
    "POINT_FIGHTING": "Point Fighting",
    "LIGHT_CONTACT": "Light Contact",
    "KICK_LIGHT": "Kick Light",
    "BOXING": "Boxing",
    "MUAY_THAI": "Full rules (elbows allowed)",
    "MUAY_THAI_NO_ELBOWS": "No elbows",
    "ITF_TAEKWONDO": "ITF · International Taekwon-Do Federation",
    "WT_TAEKWONDO": "WT · World Taekwondo (Olympic)",
    "MMA": "MMA (standing exchanges)",
}

RULESET_SHORT = {
    "ITF_TAEKWONDO": "ITF",
    "WT_TAEKWONDO": "WT",
    "MUAY_THAI": "Full rules",
    "MUAY_THAI_NO_ELBOWS": "No elbows",
    "MMA": "Standing exchanges",
}

FIGHT_TYPES = ("competition", "sparring")
ANALYSIS_TARGETS = ("A", "B", "BOTH")
