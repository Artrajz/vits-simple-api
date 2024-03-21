import logging
import os.path

import ngrok
from flask import Flask
from flask_apscheduler import APScheduler
from flask_login import LoginManager
from flask_wtf import CSRFProtect

from utils.data_utils import clean_folder, check_is_none
from utils.phrases_dict import phrases_dict_init
from tts_app.frontend.views import frontend
from tts_app.voice_api.views import voice_api
from tts_app.auth.views import auth
from tts_app.admin.views import admin

from contants import config

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'tts_app', 'static'))

app.config.from_pyfile("config.py")
# app.config.update(config)

phrases_dict_init()

csrf = CSRFProtect(app)
# 禁用tts api请求的CSRF防护
csrf.exempt(voice_api)

if config.system.is_admin_enabled:
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'


    @login_manager.user_loader
    def load_user(user_id):
        admin = config.admin
        if admin.get_id() == user_id:
            return admin
        return None

# Initialize scheduler
scheduler = APScheduler()
scheduler.init_app(app)
if config.system.clean_interval_seconds > 0:
    scheduler.start()

app.register_blueprint(frontend, url_prefix='/')
app.register_blueprint(voice_api, url_prefix='/voice')
if config.system.is_admin_enabled:
    app.register_blueprint(auth, url_prefix=config.system.admin_route)
    app.register_blueprint(admin, url_prefix=config.system.admin_route)


def create_folders(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


create_folders([os.path.join(config.abs_path, config.system.upload_folder),
                os.path.join(config.abs_path, config.system.cache_path), ])


# regular cleaning
@scheduler.task('interval', id='clean_task', seconds=config.system.clean_interval_seconds,
                misfire_grace_time=900)
def clean_task():
    clean_folder(os.path.join(config.abs_path, config.system.upload_folder))
    clean_folder(os.path.join(config.abs_path, config.system.cache_path))


if __name__ == '__main__':
    try:
        if not check_is_none(config.ngrok_config.auth_token):
            listener = ngrok.forward(config.http_service.port, authtoken=config.ngrok_config.auth_token)

            logging.info(f"Ingress established at {listener.url()}")
        else:
            logging.info(f"Not using ngrok.")
    except Exception as e:
        logging.error(f"Not using ngrok. Authtoken error:{e}")

    app.run(host=config.http_service.host, port=config.http_service.port, debug=config.http_service.debug)
