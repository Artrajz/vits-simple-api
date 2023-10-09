import os
import time
import uuid

from contants import ModelType
from logger import logger
from flask import Flask, request, send_file, jsonify, make_response, render_template
from werkzeug.utils import secure_filename
from flask_apscheduler import APScheduler
from functools import wraps
from utils.data_utils import save_audio, clean_folder, check_is_none
from utils.load_model import load_model
from io import BytesIO

app = Flask(__name__)
app.config.from_pyfile("config.py")

scheduler = APScheduler()
scheduler.init_app(app)
if app.config.get("CLEAN_INTERVAL_SECONDS", 3600) > 0:
    scheduler.start()

for path in (app.config['LOGS_PATH'], app.config['UPLOAD_FOLDER'], app.config['CACHE_PATH']):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error(f"Unable to create directory {path}: {str(e)}")

# load model
tts = load_model(app.config["MODEL_LIST"])


def require_api_key(func):
    @wraps(func)
    def check_api_key(*args, **kwargs):
        if not app.config.get('API_KEY_ENABLED', False):
            return func(*args, **kwargs)
        else:
            api_key = request.args.get('api_key') or request.headers.get('X-API-KEY')
            if api_key and api_key == app.config['API_KEY']:
                return func(*args, **kwargs)
            else:
                return make_response(jsonify({"status": "error", "message": "Invalid API Key"}), 401)

    return check_api_key


@app.route('/', methods=["GET", "POST"])
def index():
    kwargs = {
        "speakers": tts.voice_speakers,
        "speakers_count": tts.speakers_count,
        "vits_speakers_count": tts.vits_speakers_count,
        "w2v2_speakers_count": tts.w2v2_speakers_count,
        "w2v2_emotion_count": tts.w2v2_emotion_count,
        "bert_vits2_speakers_count": tts.bert_vits2_speakers_count
    }
    return render_template("index.html", **kwargs)


@app.route('/voice/speakers', methods=["GET", "POST"])
def voice_speakers_api():
    return jsonify(tts.voice_speakers)


@app.route('/voice', methods=["GET", "POST"])
@app.route('/voice/vits', methods=["GET", "POST"])
@require_api_key
def voice_vits_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = request_data.get("text", "")
        id = int(request_data.get("id", app.config.get("ID", 0)))
        format = request_data.get("format", app.config.get("FORMAT", "wav"))
        lang = request_data.get("lang", app.config.get("LANG", "auto")).lower()
        length = float(request_data.get("length", app.config.get("LENGTH", 1)))
        noise = float(request_data.get("noise", app.config.get("NOISE", 0.667)))
        noisew = float(request_data.get("noisew", app.config.get("NOISEW", 0.8)))
        max = int(request_data.get("max", app.config.get("MAX", 50)))
        use_streaming = request_data.get('streaming', False, type=bool)
    except Exception as e:
        logger.error(f"[{ModelType.VITS.value}] {e}")
        return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.VITS.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew}")
    logger.info(f"[{ModelType.VITS.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.VITS.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.vits_speakers_count:
        logger.info(f"[{ModelType.VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = tts.voice_speakers[ModelType.VITS.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}), 400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    task = {"text": text,
            "id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "max": max,
            "lang": lang,
            "speaker_lang": speaker_lang}

    if use_streaming:
        audio = tts.stream_vits_infer(task)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        t1 = time.time()
        audio = tts.vits_infer(task)
        t2 = time.time()
        logger.info(f"[{ModelType.VITS.value}] finish in {(t2 - t1):.2f}s")

        if app.config.get("SAVE_AUDIO", False):
            logger.debug(f"[{ModelType.VITS.value}] {fname}")
            path = os.path.join(app.config.get('CACHE_PATH'), fname)
            save_audio(audio.getvalue(), path)

        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/hubert-vits', methods=["POST"])
@require_api_key
def voice_hubert_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            id = int(request.form.get("id"))
            format = request.form.get("format", app.config.get("LANG", "auto"))
            length = float(request.form.get("length", app.config.get("LENGTH", 1)))
            noise = float(request.form.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(request.form.get("noisew", app.config.get("NOISEW", 0.8)))
            use_streaming = request.form.get('streaming', False, type=bool)
        except Exception as e:
            logger.error(f"[{ModelType.HUBERT_VITS.value}] {e}")
            return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.HUBERT_VITS.value}] id:{id} format:{format} length:{length} noise:{noise} noisew:{noisew}")

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    if check_is_none(id):
        logger.info(f"[{ModelType.HUBERT_VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.hubert_speakers_count:
        logger.info(f"[{ModelType.HUBERT_VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    file_type = f"audio/{format}"
    task = {"id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "audio_path": os.path.join(app.config['UPLOAD_FOLDER'], fname)}

    t1 = time.time()
    audio = tts.hubert_vits_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.HUBERT_VITS.value}] finish in {(t2 - t1):.2f}s")

    if app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.HUBERT_VITS.value}] {fname}")
        path = os.path.join(app.config.get('CACHE_PATH'), fname)
        save_audio(audio.getvalue(), path)

    if use_streaming:
        audio = tts.generate_audio_chunks(audio)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/w2v2-vits', methods=["GET", "POST"])
@require_api_key
def voice_w2v2_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = request_data.get("text", "")
        id = int(request_data.get("id", app.config.get("ID", 0)))
        format = request_data.get("format", app.config.get("FORMAT", "wav"))
        lang = request_data.get("lang", app.config.get("LANG", "auto")).lower()
        length = float(request_data.get("length", app.config.get("LENGTH", 1)))
        noise = float(request_data.get("noise", app.config.get("NOISE", 0.667)))
        noisew = float(request_data.get("noisew", app.config.get("NOISEW", 0.8)))
        max = int(request_data.get("max", app.config.get("MAX", 50)))
        emotion = int(request_data.get("emotion", app.config.get("EMOTION", 0)))
        use_streaming = request_data.get('streaming', False, type=bool)
    except Exception as e:
        logger.error(f"[{ModelType.W2V2_VITS.value}] {e}")
        return make_response(f"parameter error", 400)

    logger.info(f"[{ModelType.W2V2_VITS.value}] id:{id} format:{format} lang:{lang} "
                f"length:{length} noise:{noise} noisew:{noisew} emotion:{emotion}")
    logger.info(f"[{ModelType.W2V2_VITS.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.W2V2_VITS.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.W2V2_VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.w2v2_speakers_count:
        logger.info(f"[{ModelType.W2V2_VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = tts.voice_speakers[ModelType.W2V2_VITS.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.W2V2_VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}), 400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    task = {"text": text,
            "id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "max": max,
            "lang": lang,
            "emotion": emotion,
            "speaker_lang": speaker_lang}

    t1 = time.time()
    audio = tts.w2v2_vits_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.W2V2_VITS.value}] finish in {(t2 - t1):.2f}s")

    if app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.W2V2_VITS.value}] {fname}")
        path = os.path.join(app.config.get('CACHE_PATH'), fname)
        save_audio(audio.getvalue(), path)

    if use_streaming:
        audio = tts.generate_audio_chunks(audio)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/conversion', methods=["POST"])
@app.route('/voice/vits/conversion', methods=["POST"])
@require_api_key
def vits_voice_conversion_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            original_id = int(request.form["original_id"])
            target_id = int(request.form["target_id"])
            format = request.form.get("format", voice.filename.split(".")[1])
            use_streaming = request.form.get('streaming', False, type=bool)
        except Exception as e:
            logger.error(f"[vits_voice_convertsion] {e}")
            return make_response("parameter error", 400)

        logger.info(f"[vits_voice_convertsion] orginal_id:{original_id} target_id:{target_id}")
        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        voice.save(audio_path)
        file_type = f"audio/{format}"
        task = {"audio_path": audio_path,
                "original_id": original_id,
                "target_id": target_id,
                "format": format}

        t1 = time.time()
        audio = tts.vits_voice_conversion(task)
        t2 = time.time()
        logger.info(f"[Voice conversion] finish in {(t2 - t1):.2f}s")

        if app.config.get("SAVE_AUDIO", False):
            logger.debug(f"[Voice conversion] {fname}")
            path = os.path.join(app.config.get('CACHE_PATH'), fname)
            save_audio(audio.getvalue(), path)

        if use_streaming:
            audio = tts.generate_audio_chunks(audio)
            response = make_response(audio)
            response.headers['Content-Disposition'] = f'attachment; filename={fname}'
            response.headers['Content-Type'] = file_type
            return response
        else:
            return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/ssml', methods=["POST"])
@require_api_key
def ssml_api():
    try:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form

        ssml = request_data.get("ssml")
    except Exception as e:
        logger.info(f"[ssml] {e}")
        return make_response(jsonify({"status": "error", "message": f"parameter error"}), 400)

    logger.debug(ssml)
    voice_tasks, format = tts.parse_ssml(ssml)
    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    t1 = time.time()
    audio = tts.process_ssml_infer_task(voice_tasks, format)
    t2 = time.time()
    logger.info(f"[ssml] finish in {(t2 - t1):.2f}s")

    if app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[ssml] {fname}")
        path = os.path.join(app.config.get('CACHE_PATH'), fname)
        save_audio(audio.getvalue(), path)

    return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/dimension-emotion', methods=["POST"])
@require_api_key
def dimensional_emotion():
    if request.method == "POST":
        try:
            audio = request.files['upload']
            use_streaming = request.form.get('streaming', False, type=bool)
        except Exception as e:
            logger.error(f"[dimensional_emotion] {e}")
            return make_response("parameter error", 400)

    content = BytesIO(audio.read())

    file_type = "application/octet-stream; charset=ascii"
    fname = os.path.splitext(audio.filename)[0] + ".npy"
    emotion_npy = tts.get_dimensional_emotion_npy(content)
    if use_streaming:
        emotion_npy = tts.generate_audio_chunks(emotion_npy)
        response = make_response(emotion_npy)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        return send_file(path_or_file=emotion_npy, mimetype=file_type, download_name=fname)


@app.route('/voice/bert-vits2', methods=["GET", "POST"])
@require_api_key
def voice_bert_vits2_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = request_data.get("text", "")
        id = int(request_data.get("id", app.config.get("ID", 0)))
        format = request_data.get("format", app.config.get("FORMAT", "wav"))
        lang = request_data.get("lang", "auto").lower()
        length = float(request_data.get("length", app.config.get("LENGTH", 1)))
        noise = float(request_data.get("noise", app.config.get("NOISE", 0.667)))
        noisew = float(request_data.get("noisew", app.config.get("NOISEW", 0.8)))
        sdp_ratio = float(request_data.get("sdp_ratio", app.config.get("SDP_RATIO", 0.2)))
        max = int(request_data.get("max", app.config.get("MAX", 50)))
    except Exception as e:
        logger.error(f"[{ModelType.BERT_VITS2.value}] {e}")
        return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.BERT_VITS2.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew} sdp_ratio:{sdp_ratio}")
    logger.info(f"[{ModelType.BERT_VITS2.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.BERT_VITS2.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.BERT_VITS2.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.bert_vits2_speakers_count:
        logger.info(f"[{ModelType.BERT_VITS2.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = tts.voice_speakers[ModelType.BERT_VITS2.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.BERT_VITS2.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}), 400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    task = {"text": text,
            "id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "sdp_ratio": sdp_ratio,
            "max": max,
            "lang": lang,
            "speaker_lang": speaker_lang}

    t1 = time.time()
    audio = tts.bert_vits2_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.BERT_VITS2.value}] finish in {(t2 - t1):.2f}s")

    if app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.BERT_VITS2.value}] {fname}")
        path = os.path.join(app.config.get('CACHE_PATH'), fname)
        save_audio(audio.getvalue(), path)

    return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@app.route('/voice/check', methods=["GET", "POST"])
def check():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        model_type_str = request_data.get("model_type", request_data.get("model")).upper()
        id = int(request_data.get("id"))
    except Exception as e:
        logger.info(f"[check] {e}")
        return make_response(jsonify({"status": "error", "message": "parameter error"}), 400)

    if check_is_none(model_type_str):
        logger.info(f"[check] model {model_type_str} is empty")
        return make_response(jsonify({"status": "error", "message": "model is empty"}), 400)

    if model_type_str not in ModelType._value2member_map_:
        res = make_response(jsonify({"status": "error", "message": f"model {model_type_str} does not exist"}))
        res.status = 404
        logger.info(f"[check] speaker id {id} error")
        return res

    if check_is_none(id):
        logger.info(f"[check] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    model_type = ModelType(model_type_str)
    speaker_list = tts.voice_speakers[model_type.value]

    if len(speaker_list) == 0:
        logger.info(f"[check] {model_type_str} not loaded")
        return make_response(jsonify({"status": "error", "message": f"{model_type_str} not loaded"}), 400)

    if id < 0 or id >= len(speaker_list):
        logger.info(f"[check] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)
    name = str(speaker_list[id]["name"])
    lang = speaker_list[id]["lang"]
    logger.info(f"[check] check id:{id} name:{name} lang:{lang}")

    return make_response(jsonify({"status": "success", "id": id, "name": name, "lang": lang}), 200)


# regular cleaning
@scheduler.task('interval', id='clean_task', seconds=app.config.get("CLEAN_INTERVAL_SECONDS", 3600),
                misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))  # 对外开放
    # app.run(host='127.0.0.1', port=app.config.get("PORT",23456), debug=True)  # 本地运行、调试
