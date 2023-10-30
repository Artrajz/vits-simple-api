import os.path

from flask import Flask
from flask_apscheduler import APScheduler
from flask_login import LoginManager
from flask_wtf import CSRFProtect

from tts_app import frontend, voice_api, auth, admin
from tts_app.auth.models import users

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'static'))

app.config.from_pyfile("config.py")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

csrf = CSRFProtect(app)


@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.get_id() == user_id:
            return user
    return None


# Initialize scheduler
scheduler = APScheduler()
scheduler.init_app(app)
if app.config.get("CLEAN_INTERVAL_SECONDS", 3600) > 0:
    scheduler.start()

app.register_blueprint(frontend, url_prefix='/')
app.register_blueprint(voice_api, url_prefix='/voice')
app.register_blueprint(auth, url_prefix='/auth')
app.register_blueprint(admin, url_prefix='/admin')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))
