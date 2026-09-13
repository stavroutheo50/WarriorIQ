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

UNTESTED: written without a Modal account to run it against. Modal's decorator
names have changed across releases, so check the current docs if deploy rejects
something here. The WarriorIQ side of the contract is verified; this file is the
part that needs a real deploy to confirm.
"""

from __future__ import annotations

import os

import modal

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
    .add_local_dir(
        ".",
        "/app",
        ignore=["**/.git", "**/.venv", "**/uploads", "**/outputs", "**/dataset",
                "**/__pycache__", "**/*.engine", "**/warrioriq.sqlite3"],
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
    """
    import logging

    from core.trt_engine import ensure_pose_engine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    engine = ensure_pose_engine(f"{DATA_DIR}/models")
    if engine is not None:
        os.environ["WARRIORIQ_POSE_ENGINE"] = str(engine)


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
    # A10G over T4: the T4 has no usable fp16 tensor throughput for this stack
    # and 16 GB it cannot feed, while the engine is built dynamic at imgsz 1600
    # for small-source footage. Changing this string is safe - the engine cache
    # is keyed by GPU name, so a new GPU type builds its own rather than
    # loading one it cannot deserialise.
    gpu="A10G",
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
def wake(payload: dict, request: "Request") -> dict:  # noqa: F821
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
