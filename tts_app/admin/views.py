import logging
import os
import traceback

from flask import Blueprint, request, render_template, make_response, jsonify
from flask_login import login_required

from config import config, BASE_DIR, save_config_to_yaml
from contants import TTSType
from tts_app.model_manager import model_manager

admin = Blueprint('admin', __name__)


def extract_model_dir(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    directory_name = os.path.basename(directory)
    if not directory:  # 如果文件所在文件夹为空（即在根目录）
        return filename
    else:
        return directory_name + "/" + filename


def redirect_model_dir(path):
    return os.path.join(
        BASE_DIR,
        config.system.data_path,
        config.tts_model_config.models_dir,
        path,
    )


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
                model["model_path"] = extract_model_dir(model["model_path"])

            if model.get("config_path") is not None:
                model["config_path"] = extract_model_dir(model["config_path"])

            if model.get("sovits_path") is not None:
                model["sovits_path"] = extract_model_dir(model["sovits_path"])

            if model.get("gpt_path") is not None:
                model["gpt_path"] = extract_model_dir(model["gpt_path"])

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

    tts_type = request_data.get("tts_type")
    tts_model = {
        "tts_type": tts_type,

    }
    if tts_type == TTSType.GPT_SOVITS:
        sovits_path = request_data.get("sovits_path")
        gpt_path = request_data.get("gpt_path")

        sovits_path = redirect_model_dir(sovits_path)
        gpt_path = redirect_model_dir(gpt_path)

        tts_model.update(
            {
                "sovits_path": sovits_path,
                "gpt_path": gpt_path,
            }
        )
        logging.info(f"Loading model sovits_path: {sovits_path} gpt_path: {gpt_path}")
    else:
        vits_path = request_data.get("vits_path")
        config_path = request_data.get("config_path")

        vits_path = redirect_model_dir(vits_path)
        config_path = redirect_model_dir(config_path)

        tts_model.update(
            {
                "vits_path": vits_path,
                "config_path": config_path,
            }
        )
        logging.info(f"Loading model model_path: {vits_path} config_path: {config_path}")

    state = model_manager.load_model(tts_model)
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

    return jsonify(config.model_dump())


@admin.route('/set_config', methods=["GET", "POST"])
@login_required
def set_config():
    if request.method == "POST":
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form.to_dict()
    else:
        return jsonify({"error": "Unsupported request method"}), 400

    config.update_config(request_data)

    status = "success"
    code = 200
    return make_response(jsonify({"status": status}), code)


@admin.route('/save_current_model', methods=["GET", "POST"])
@login_required
def save_current_model():
    try:
        tts_models = model_manager.get_models_path()

        config.tts_model_config.update_tts_models(tts_models)
        save_config_to_yaml(config)

        status = "success"
        response_code = 200
    except Exception as e:
        status = "failed"
        response_code = 500
        logging.error(traceback.format_exc())

    return make_response(jsonify({"status": status}), response_code)
