from dotenv import load_dotenv
from a2wsgi import ASGIMiddleware


load_dotenv()

from app.main import app

application = ASGIMiddleware(app, wait_time=5.0)
