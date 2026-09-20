"""Scale-to-zero GPU worker for WarriorIQ.

Runs the same `worker.py` and the same models as a local GPU machine, so the
analysis is identical. The difference is lifetime: the container starts when a
fight is queued, drains the queue, and exits, so no GPU is billed while idle.

    modal secret create warrioriq \
        WARRIORIQ_WORKER_REMOTE_URL=https://warrioriq.eu \
        WARRIORIQ_WORKER_TOKEN=<the same token as the web server>
    modal deploy deploy/modal_worker.py

Deploying prints the web endpoint URL. Put it in WARRIORIQ_WORKER_WAKE_URL on
the web server and every queued fight will start a GPU run.

NOT YET DEPLOYED, but no longer unchecked. On 2026-09-20 this module was
executed against the installed client (modal 1.5.5) and builds its App cleanly:
both functions register and `wake` is recognised as a web endpoint. Every API
name here was confirmed current against Modal 1.x - `max_containers` (renamed
from concurrency_limit in v0.73.76), `modal.fastapi_endpoint` (renamed from
web_endpoint in v0.73.82), string GPU names, and add_local_dir in place of the
removed Mount.

Three faults were found and fixed in that pass, all of which would have failed
the deploy or the first request rather than degraded quietly:

  * gpu="A10G" - Modal has no such string; the card is "A10".
  * `request: "Request"` with no import - FastAPI resolves route annotations at
    registration, so this raises NameError before serving anything.
  * an ignore list that shipped ~1 GB it did not need, including fights/.

What still needs a real account: the image build (torch + sam2 + onnxruntime-gpu
resolving together on debian_slim), the first TensorRT engine build on an A10,
and one end-to-end fight. Nothing below that line has been run.
"""

from __future__ import annotations

import os

import modal

# At module scope on purpose, not inside `wake`. `from __future__ import
# annotations` above makes every annotation a string, and FastAPI resolves a
# route's strings with get_type_hints() against module globals when the route
# is registered - so a name that was never imported raises NameError at deploy
# time, before anything is served. The image carries fastapi (requirements.txt),
# and so does the machine running `modal deploy`, because the web app needs it.
from fastapi import Request

# models/ is gitignored, so nothing ships with the checkout. The tracker config
# is small enough to carry here; the weights are fetched once into a Volume.
TRACKER_YAML = """tracker_type: botsort
track_high_thresh: 0.28
track_low_thresh: 0.08
new_track_thresh: 0.30
track_buffer: 90
match_thresh: 0.78
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.45
appearance_thresh: 0.72
with_reid: true
model: auto
"""

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # OpenCV needs the GL/glib runtime libraries; ffmpeg decodes the fight video.
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    # Optional RTMPose refinement, added so a remote run measures the same
    # joints as a local one. requirements.txt deliberately selects no ONNX
    # Runtime or RTMLib backend (see README "Optional RTMPose refinement"),
    # and without these two layers core/rtm_pose.py sets _unavailable and logs
    # rtm_pose_unavailable - the analysis still completes, but every fighter
    # keeps the fused model's raw skeleton, which is the confidently-wrong one
    # that module exists to replace. A silent quality difference between local
    # and remote is worth more to avoid than these layers cost.
    #
    # --no-deps is not optional: RTMLib 0.0.16's metadata pulls a second
    # OpenCV distribution and the CPU onnxruntime over the GPU one.
    #
    # THESE TWO LINES ARE THE FIRST THING TO DELETE IF THE IMAGE BUILD FAILS.
    # onnxruntime-gpu needs CUDA/cuDNN libraries that it expects to find from
    # the torch wheel, and that pairing is the least certain part of this
    # image. Dropping them gives a working worker with unrefined joints.
    .pip_install_from_requirements("requirements-rtm-cuda12.txt")
    .run_commands("pip install --no-deps rtmlib==0.0.16")
    .add_local_dir(
        ".",
        "/app",
        # Measured on the working checkout 2026-09-20, because what this list
        # forgets is uploaded on every deploy. The old list caught .venv (9.1
        # GB) and .git (51 MB) and missed roughly a gigabyte besides:
        #
        #     fights/         387 MB   the source videos - the one thing the
        #                              worker is handed by the web server and
        #                              must never carry in its own image
        #     .tmp/           250 MB   pytest scratch dirs, some unreadable
        #     .huggingface/   176 MB   re-fetched onto the Volume anyway,
        #                              since HF_HOME points there
        #     *.engine.backup 181 MB   see the pattern note below
        #     .claude/        6.2 MB   contains an entire second checkout
        #
        # `**/*.engine` did not match `yolo26m-pose.engine.backup-640` or
        # `...backup-fp32`, so 181 MB of engines that cannot deserialise on a
        # Modal GPU shipped anyway. The trailing `*` fixes that.
        #
        # What deliberately STAYS: the root *.pt / *.onnx weights (208 MB),
        # because `pose_model_pt` is a bare filename resolved against cwd and
        # these are not names Ultralytics can fetch; and models/, because
        # `referee_probe_path` defaults to the relative "models/referee_probe.npz"
        # and the referee filter silently disables itself without it.
        #
        # .env and session-secret.txt are excluded as secrets, not as bulk.
        # They are currently harmless - worker.py calls load_dotenv() at
        # override=False, so the Modal secret wins - but a token baked into an
        # image layer is a token you cannot rotate by rotating the secret.
        ignore=["**/.git", "**/.venv", "**/uploads", "**/outputs", "**/dataset",
                "**/__pycache__", "**/*.engine*", "**/warrioriq.sqlite3",
                "**/.tmp", "**/fights", "**/.huggingface", "**/logs",
                "**/.claude", "**/.idea", "**/.pytest_cache", "**/.ruff_cache",
                "**/.env", "**/session-secret.txt", "**/*.log", "**/*.log.*"],
    )
)

# Weights persist between runs. Without this every cold start re-downloads YOLO
# and SAM2, which would dominate both the wait and the bill.
weights = modal.Volume.from_name("warrioriq-weights", create_if_missing=True)

app = modal.App("warrioriq-worker", image=image)


def _prepare_runtime() -> None:
    """Point WarriorIQ at the mounted volume and supply the tracker config."""
    models = f"{DATA_DIR}/models"
    os.makedirs(models, exist_ok=True)
    tracker = f"{models}/warrioriq_botsort.yaml"
    if not os.path.exists(tracker):
        with open(tracker, "w", encoding="utf-8") as handle:
            handle.write(TRACKER_YAML)
    os.environ.update({
        "WARRIORIQ_DATA_DIR": DATA_DIR,
        "WARRIORIQ_WORKER_MODE": "remote",
        "HF_HOME": f"{DATA_DIR}/.huggingface",
        "YOLO_CONFIG_DIR": f"{DATA_DIR}/.ultralytics",
    })


def _prepare_engine() -> None:
    """Build this GPU's TensorRT engine once, then reuse it from the Volume.

    This used to point WARRIORIQ_POSE_ENGINE at a file called `absent.engine`
    to force the fallback to the portable .pt checkpoint, on the reasoning that
    an engine is tied to one GPU model and none could ship in the image. Both
    halves of that are true and the conclusion still cost every remote run its
    TensorRT speed, every time, with nothing in the logs to say so.

    An engine cannot ship, but it can be built here and kept. The filename
    carries the GPU so two container types cannot be handed each other's
    engine - which is not a performance question but a correctness one, since
    a close-enough architecture may accept a foreign engine rather than refuse
    it. The build costs minutes on the first cold start of a given GPU type and
    nothing afterwards; `weights.commit()` at the end of the run persists it.

    **Setting WARRIORIQ_POSE_ENGINE here does nothing, and used to be the whole
    fix.** SETTINGS is a frozen dataclass whose field defaults are evaluated
    when core.config is first imported - and that import has already happened
    by the time this line runs, triggered by importing core.trt_engine two
    lines above. PoseTracker then reads SETTINGS.pose_model_engine, which still
    holds the generic `<data>/models/yolo26m-pose.engine`, finds nothing at that
    path, and falls back to the .pt checkpoint. So the engine was built, the
    minutes were spent, the file was committed to the Volume, and every remote
    run still went without TensorRT - which is precisely the silent failure the
    paragraph above says this function was written to end, reintroduced one
    layer further down.

    No environment variable can fix it after the import, so the built engine is
    published under the name the tracker is already looking for. The GPU-keyed
    file stays the source of truth; the generic name is a pointer refreshed on
    every cold start, so a container that lands on a different GPU type
    overwrites the previous type's pointer before the tracker loads. A marker
    file records which engine the pointer was made from, so the 90 MB copy is
    paid once per GPU type rather than once per fight.
    """
    import logging
    import shutil
    from pathlib import Path

    from core.config import SETTINGS
    from core.trt_engine import ensure_pose_engine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger("warrioriq.modal")

    engine = ensure_pose_engine(f"{DATA_DIR}/models")
    if engine is None:
        log.warning("engine_unavailable - this run uses the .pt checkpoint")
        return

    expected = Path(SETTINGS.pose_model_engine)
    marker = expected.with_suffix(".engine.source")
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == engine.name and expected.exists():
        log.info("engine_pointer_current path=%s source=%s", expected, engine.name)
        return

    expected.parent.mkdir(parents=True, exist_ok=True)
    staged = expected.with_suffix(".engine.staging")
    shutil.copy2(engine, staged)
    # os.replace is atomic, so a container that dies mid-copy cannot leave a
    # truncated engine behind for the next one to try to deserialise.
    os.replace(staged, expected)
    marker.write_text(engine.name, encoding="utf-8")
    log.info("engine_pointer_written path=%s source=%s", expected, engine.name)


def _report_backend() -> str:
    """Say which backend actually loaded, rather than which one was intended.

    There are three separate ways to lose TensorRT here and all of them were
    silent: no engine file, no CUDA device, or an engine this runtime refuses.
    The tracker logs each at load, and this repeats the answer as the function's
    return value so it is visible in Modal's run output without reading logs.
    """
    import logging

    from core.analyzer import get_pose_tracker

    tracker = get_pose_tracker()
    backend = "tensorrt" if str(tracker.model_path).endswith(".engine") else "pytorch"
    logging.getLogger("warrioriq.modal").info(
        "startup_backend=%s model=%s cuda=%s", backend, tracker.model_path, tracker.uses_cuda)
    return f"{backend}:{tracker.model_path}"


@app.function(
    # A10 over T4: the T4 has no usable fp16 tensor throughput for this stack
    # and 16 GB it cannot feed, while the engine is built dynamic at imgsz 1600
    # for small-source footage. Changing this string is safe - the engine cache
    # is keyed by GPU name, so a new GPU type builds its own rather than
    # loading one it cannot deserialise.
    #
    # "A10", not "A10G". Modal's accepted strings are T4, L4, A10, L40S, A100,
    # A100-40GB, A100-80GB, RTX-PRO-6000, H100, H200, B200, B300 - checked
    # against modal.com/docs/guide/gpu on 2026-09-20. "A10G" is the AWS name
    # for the same card and Modal rejects it, which fails the deploy rather
    # than quietly falling back to a CPU container.
    gpu="A10",
    volumes={DATA_DIR: weights},
    secrets=[modal.Secret.from_name("warrioriq")],
    timeout=3600,
    # One container at a time: worker.py claims a single job per loop, and the
    # web server's lease already prevents two workers owning one analysis.
    max_containers=1,
)
def drain_queue() -> int:
    """Claim and analyse every queued fight, then exit so billing stops."""
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")
    _prepare_runtime()
    _prepare_engine()
    print("WarriorIQ pose backend:", _report_backend())

    from worker import run_worker

    # once=True keeps claiming while work remains and returns as soon as the
    # queue is empty, which is exactly the lifetime we want to pay for.
    result = run_worker(once=True)
    weights.commit()
    return result


@app.function(secrets=[modal.Secret.from_name("warrioriq")])
@modal.fastapi_endpoint(method="POST")
def wake(payload: dict, request: Request) -> dict:
    """Endpoint for WARRIORIQ_WORKER_WAKE_URL.

    This URL is public, so it must verify the shared worker token before
    starting anything. Without that check anyone who found the address could
    spawn GPU runs and spend the account's credits.

    Returns as soon as the run is queued. The web server treats the wake as
    best effort, so a slow reply here must never delay someone's upload.
    """
    import hmac

    from fastapi import HTTPException

    expected = os.environ.get("WARRIORIQ_WORKER_TOKEN", "")
    scheme, _, presented = request.headers.get("authorization", "").partition(" ")
    if not expected or scheme.lower() != "bearer" or not hmac.compare_digest(presented, expected):
        raise HTTPException(401, "Worker authentication failed.")

    drain_queue.spawn()
    return {"ok": True, "job_id": payload.get("job_id")}


@app.local_entrypoint()
def main() -> None:
    """`modal run deploy/modal_worker.py` drains the queue once, for testing."""
    print("worker exit code:", drain_queue.remote())
