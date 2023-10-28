from flask import Flask
from flask_apscheduler import APScheduler

from tts_app import frontend, voice_api

app = Flask(__name__)

app.register_blueprint(frontend, url_prefix='/')
app.register_blueprint(voice_api, url_prefix='/voice')

app.config.from_pyfile("config.py")

# Initialize scheduler
scheduler = APScheduler()
scheduler.init_app(app)
if app.config.get("CLEAN_INTERVAL_SECONDS", 3600) > 0:
    scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))
