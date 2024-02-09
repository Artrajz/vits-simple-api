import logging
import os

from flask import Blueprint, request, render_template, make_response, jsonify
from flask_login import login_required

from contants import config
from tts_app.model_manager import model_manager

admin = Blueprint('admin', __name__)


def extract_filename_and_directory(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    directory_name = os.path.basename(directory)
    if not directory:  # 如果文件所在文件夹为空（即在根目录）
        return filename
    else:
        return directory_name + "/" + filename


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
    loaded_models_info = model_manager.get_models_info()
    for models in loaded_models_info.values():
        for model in models:
            if model.get("model_path") is not None:
                model["model_path"] = extract_filename_and_directory(model["model_path"])

            if model.get("config_path") is not None:
                model["config_path"] = extract_filename_and_directory(model["config_path"])

            if model.get("sovits_path") is not None:
                model["sovits_path"] = extract_filename_and_directory(model["sovits_path"])

            if model.get("gpt_path") is not None:
                model["gpt_path"] = extract_filename_and_directory(model["gpt_path"])

    return loaded_models_info


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
    sovits_path = request_data.get("sovits_path")
    gpt_path = request_data.get("gpt_path")

    if model_path is not None and config_path is not None:
        logging.info(f"Loading model model_path: {model_path} config_path: {config_path}")
    else:
        logging.info(f"Loading model sovits_path: {sovits_path} gpt_path: {gpt_path}")

    state = model_manager.load_model(model_path=model_path,
                                     config_path=config_path,
                                     sovits_path=sovits_path,
                                     gpt_path=gpt_path)
    if state:
        status = "success"
        response_code = 200
    else:
        status = "failed"
        response_code = 500

    return make_response(jsonify({"status": status}), response_code)


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
    if state:
        status = "success"
        response_code = 200
    else:
        status = "failed"
        response_code = 500

    return make_response(jsonify({"status": status}), response_code)


@admin.route('/get_path', methods=["GET", "POST"])
@login_required
def get_path():
    return model_manager.scan_unload_path()


@admin.route('/get_config', methods=["GET", "POST"])
@login_required
def get_config():
    return jsonify(config.asdict())


@admin.route('/set_config', methods=["GET", "POST"])
@login_required
def set_config():
    if request.method == "POST":
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form

    # try:
    #     new_config = dict(request_data)
    #     config.update_config(new_config)
    #     status = "success"
    #     code = 200
    # except Exception as e:
    #     status = "failed"
    #     code = 500
    #     logging.error(e)
    new_config = dict(request_data)
    config.update_config(new_config)
    config.save_config(config)
    status = "success"
    code = 200
    return make_response(jsonify({"status": status}), code)


@admin.route('/save_current_model', methods=["GET", "POST"])
@login_required
def save_current_model():
    try:
        models_path = model_manager.get_models_path()
        models = {"models": models_path}

        config.update_config({"tts_config": models})
        config.save_config(config)

        status = "success"
        response_code = 200
    except Exception as e:
        status = "failed"
        response_code = 500
        logging.info(e)

    return make_response(jsonify({"status": status}), response_code)
