import json
import logging
from copy import deepcopy

import torch
from flask import Blueprint, request, render_template, make_response, jsonify
from flask_login import login_required

from tts_app.auth.models import user2str, str2user
from tts_app.model_manager import model_manager
from utils import config_manager
from utils.config_manager import global_config

admin = Blueprint('admin', __name__)


@admin.route('/')
@login_required
def home():
    return render_template('pages/home.html')


@admin.route('/setting')
@login_required
def setting():
    return render_template('pages/setting.html')


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


@admin.route('/get_config', methods=["GET", "POST"])
@login_required
def get_config():
    dict_data = deepcopy(dict(global_config))
    dict_data["DEVICE"] = str(dict_data["DEVICE"])

    dict_data = user2str(dict_data)

    return jsonify(dict_data)


@admin.route('/set_config', methods=["GET", "POST"])
@login_required
def set_config():
    if request.method == "POST":
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form

    dict_data = dict(request_data)
    print(dict_data)
    # dict_data["DEVICE"] = torch.device(dict_data["DEVICE"])
    if dict_data.get("users", None) is not None:
        dict_data = str2user(dict_data)
    print(global_config)
    global_config.update(dict_data)
    config_manager.save_yaml_config(global_config)

    status = "success"
    return make_response(jsonify({"status": status}), 200)
