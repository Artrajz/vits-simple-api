import os.path

from flask import Flask

from config import ABS_PATH
from tts_app.voice_api.views import init_routes


def create_app():
    app = Flask(__name__)
    app.config.from_pyfile(os.path.join(ABS_PATH, "config.py"))
    
    # Initialize routes
    init_routes(app)

    return app
