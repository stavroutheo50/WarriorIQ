grom a2wsgi import ASGIMiddleware
from app.main import app

application = ASGIMiddleware(app)
