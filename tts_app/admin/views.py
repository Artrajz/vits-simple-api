import logging

from flask import Blueprint, request, render_template, make_response, jsonify
from flask_login import login_required

from tts_app.model_manager import model_manager

admin = Blueprint('admin', __name__)


@admin.route('/')
@login_required
def home():
    return render_template('pages/home.html')


@admin.route('/get_models_info', methods=["GET", "POST"])
@login_required
def get_models_info():
    return model_manager.get_models_info()


@admin.route('/load_model', methods=["GET", "POST"])
@login_required
def load_model():
    if request.method == "GET":
        request_data = request.args
    elif request.method == "POST":
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form

    model_path = request_data.get("model_path")
    config_path = request_data.get("config_path")
    logging.info(f"Loading model\n"
                 f"model_path: {model_path}\n"
                 f"config_path: {config_path}")
    state = model_manager.load_model(model_path, config_path)
    status = "success" if state else "failed"

    return make_response(jsonify({"status": status}), 200)


@admin.route('/unload_model', methods=["GET", "POST"])
@login_required
def unload_model():
    if request.method == "GET":
        request_data = request.args
    elif request.method == "POST":
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form
    model_type = request_data.get("model_type")
    model_id = request_data.get("model_id")

    logging.info(f"Unloading model. model_type: {model_type} model_id: {model_id}")

    state = model_manager.unload_model(model_type, model_id)
    status = "success" if state else "failed"

    return make_response(jsonify({"status": status}), 200)


@admin.route('/get_path', methods=["GET", "POST"])
@login_required
def get_path():
    return model_manager.scan_path()
