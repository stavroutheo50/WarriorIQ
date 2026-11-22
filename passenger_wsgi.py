import os

# The same thread caps as warrioriq_wsgi.py (shared host, 100-process limit
# that counts threads); set before anything imports OpenCV or NumPy.
os.environ.setdefault("WARRIORIQ_DECODE_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from app.startup_guard import guarded


def _load():
    from dotenv import load_dotenv
    from a2wsgi import ASGIMiddleware

    load_dotenv()

    from app.main import app

    application = ASGIMiddleware(app, wait_time=5.0)
    return application


# A failed import serves a WarriorIQ "back in a moment" page and is retried,
# instead of Passenger's error page until the next restart (app/startup_guard.py).
application = guarded(_load)
