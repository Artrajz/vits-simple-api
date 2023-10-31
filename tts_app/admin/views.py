from flask import Blueprint
from flask_login import login_required

from tts_app.model_manager import model_manager

admin = Blueprint('admin', __name__)

@admin.route('/')
@login_required
def setting():
    return "Hello Admin!"


@admin.route('/get_models_info')
@login_required
def get_models_info():
    return model_manager.get_models_info()