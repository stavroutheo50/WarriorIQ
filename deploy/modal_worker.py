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
        # The .engine is a TensorRT build tied to one GPU model. Leaving it unset
        # makes pose_tracker fall back to the portable .pt weights.
        "WARRIORIQ_POSE_ENGINE": f"{models}/absent.engine",
        "HF_HOME": f"{DATA_DIR}/.huggingface",
        "YOLO_CONFIG_DIR": f"{DATA_DIR}/.ultralytics",
    })


@app.function(
    gpu="T4",
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
