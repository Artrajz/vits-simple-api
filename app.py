import os.path

from flask import Flask
from flask_apscheduler import APScheduler
from flask_login import LoginManager
from flask_wtf import CSRFProtect

from utils.data_utils import clean_folder
from utils.phrases_dict import phrases_dict_init
from tts_app import frontend, voice_api, auth, admin
from utils.config_manager import global_config

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'static'))

app.config.from_pyfile("config.py")
app.config.update(global_config)

phrases_dict_init()

csrf = CSRFProtect(app)
# 禁用tts api请求的CSRF防护
csrf.exempt(voice_api)

if app.config.get("IS_ADMIN_ENABLED", False):
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'


    @login_manager.user_loader
    def load_user(user_id):
        users = app.config["users"]["admin"]
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
if app.config.get("IS_ADMIN_ENABLED", False):
    app.register_blueprint(auth, url_prefix=app.config.get("ADMIN_ROUTE", "/admin"))
    app.register_blueprint(admin, url_prefix=app.config.get("ADMIN_ROUTE", "/admin"))


def create_folders(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


create_folders([app.config["UPLOAD_FOLDER"],
                app.config["CACHE_PATH"],
                os.path.join(app.config["ABS_PATH"], "Model")
                ])


# regular cleaning
@scheduler.task('interval', id='clean_task', seconds=app.config.get("CLEAN_INTERVAL_SECONDS", 3600),
                misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))
