import os

# The web host is shared hosting with a small process limit that counts
# threads (100 on the owner's cPanel account). Video decoders and maths
# libraries each start one thread per core of the shared server unless told
# otherwise, and a decoder that cannot start its threads cannot open the
# video: uploads were refused as unreadable (owner, 2026-11-13). Set before
# anything imports them; a value set in the cPanel Python App wins.
os.environ.setdefault("WARRIORIQ_DECODE_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from dotenv import load_dotenv
from a2wsgi import ASGIMiddleware


load_dotenv()

from app.main import app

application = ASGIMiddleware(app, wait_time=5.0)
