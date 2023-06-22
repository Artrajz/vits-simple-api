import os
import logging
import time
import logzero
import uuid
from flask import Flask, request, send_file, jsonify, make_response, render_template
from werkzeug.utils import secure_filename
from flask_apscheduler import APScheduler
from functools import wraps
from utils.utils import clean_folder, check_is_none
from utils.merge import merge_model
from io import BytesIO

app = Flask(__name__)
app.config.from_pyfile("config.py")

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

logzero.loglevel(logging.WARNING)
logger = logging.getLogger("vits-simple-api")
level = app.config.get("LOGGING_LEVEL", "DEBUG")
level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL}
logging.basicConfig(level=level_dict[level])
logging.getLogger('numba').setLevel(logging.WARNING)

tts = merge_model(app.config["MODEL_LIST"])

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if not os.path.exists(app.config['CACHE_PATH']):
    os.makedirs(app.config['CACHE_PATH'], exist_ok=True)


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
        "speakers_count": tts.speakers_count
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
            text = request.args.get("text", "")
            id = int(request.args.get("id", app.config.get("ID", 0)))
            format = request.args.get("format", app.config.get("FORMAT", "wav"))
            lang = request.args.get("lang", app.config.get("LANG", "auto"))
            length = float(request.args.get("length", app.config.get("LENGTH", 1)))
            noise = float(request.args.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(request.args.get("noisew", app.config.get("NOISEW", 0.8)))
            max = int(request.args.get("max", app.config.get("MAX", 50)))
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                data = request.get_json()
            else:
                data = request.form
            text = data.get("text", "")
            id = int(data.get("id", app.config.get("ID", 0)))
            format = data.get("format", app.config.get("FORMAT", "wav"))
            lang = data.get("lang", app.config.get("LANG", "auto"))
            length = float(data.get("length", app.config.get("LENGTH", 1)))
            noise = float(data.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(data.get("noisew", app.config.get("NOISEW", 0.8)))
            max = int(data.get("max", app.config.get("MAX", 50)))
    except Exception as e:
        logger.error(f"[VITS] {e}")
        return make_response("parameter error", 400)

    logger.info(f"[VITS] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew}")
    logger.info(f"[VITS] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[VITS] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[VITS] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.vits_speakers_count:
        logger.info(f"[VITS] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = tts.voice_speakers["VITS"][id].get('lang')
    if lang.upper() != "AUTO" and lang.upper() != "MIX" and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[VITS] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}), 400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    t1 = time.time()
    output = tts.vits_infer({"text": text,
                             "id": id,
                             "format": format,
                             "length": length,
                             "noise": noise,
                             "noisew": noisew,
                             "max": max,
                             "lang": lang,
                             "speaker_lang": speaker_lang})
    t2 = time.time()
    logger.info(f"[VITS] finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


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
        except Exception as e:
            logger.error(f"[hubert] {e}")
            return make_response("parameter error", 400)

    logger.info(f"[hubert] id:{id} format:{format} length:{length} noise:{noise} noisew:{noisew}")

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    if check_is_none(id):
        logger.info(f"[hubert] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.hubert_speakers_count:
        logger.info(f"[hubert] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    file_type = f"audio/{format}"

    t1 = time.time()
    output = tts.hubert_vits_infer({"id": id,
                                    "format": format,
                                    "length": length,
                                    "noise": noise,
                                    "noisew": noisew,
                                    "audio_path": os.path.join(app.config['UPLOAD_FOLDER'], fname)})
    t2 = time.time()
    logger.info(f"[hubert] finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/w2v2-vits', methods=["GET", "POST"])
@require_api_key
def voice_w2v2_api():
    try:
        if request.method == "GET":
            text = request.args.get("text", "")
            id = int(request.args.get("id", app.config.get("ID", 0)))
            format = request.args.get("format", app.config.get("FORMAT", "wav"))
            lang = request.args.get("lang", app.config.get("LANG", "auto"))
            length = float(request.args.get("length", app.config.get("LENGTH", 1)))
            noise = float(request.args.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(request.args.get("noisew", app.config.get("NOISEW", 0.8)))
            max = int(request.args.get("max", app.config.get("MAX", 50)))
            emotion = int(request.args.get("emotion", app.config.get("EMOTION", 0)))
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                data = request.get_json()
            else:
                data = request.form
            text = data.get("text", "")
            id = int(data.get("id", app.config.get("ID", 0)))
            format = data.get("format", app.config.get("FORMAT", "wav"))
            lang = data.get("lang", app.config.get("LANG", "auto"))
            length = float(data.get("length"))
            noise = float(data.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(data.get("noisew", app.config.get("NOISEW", 0.8)))
            max = int(data.get("max", app.config.get("MAX", 50)))
            emotion = int(data.get("emotion", app.config.get("EMOTION", 0)))
    except Exception as e:
        logger.error(f"[w2v2] {e}")
        return make_response(f"parameter error", 400)

    logger.info(f"[w2v2] id:{id} format:{format} lang:{lang} "
                f"length:{length} noise:{noise} noisew:{noisew} emotion:{emotion}")
    logger.info(f"[w2v2] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[w2v2] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[w2v2] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= tts.w2v2_speakers_count:
        logger.info(f"[w2v2] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = tts.voice_speakers["W2V2-VITS"][id].get('lang')
    if lang.upper() != "AUTO" and lang.upper() != "MIX" and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[w2v2] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}), 400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    t1 = time.time()
    output = tts.w2v2_vits_infer({"text": text,
                                  "id": id,
                                  "format": format,
                                  "length": length,
                                  "noise": noise,
                                  "noisew": noisew,
                                  "max": max,
                                  "lang": lang,
                                  "emotion": emotion,
                                  "speaker_lang": speaker_lang})
    t2 = time.time()
    logger.info(f"[w2v2] finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


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
        except Exception as e:
            logger.error(f"[vits_voice_convertsion] {e}")
            return make_response("parameter error", 400)

        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        voice.save(audio_path)
        file_type = f"audio/{format}"

        logger.info(f"[vits_voice_convertsion] orginal_id:{original_id} target_id:{target_id}")
        t1 = time.time()
        try:
            output = tts.vits_voice_conversion({"audio_path": audio_path,
                                                "original_id": original_id,
                                                "target_id": target_id,
                                                "format": format})
        except Exception as e:
            logger.info(f"[vits_voice_convertsion] {e}")
            return make_response(jsonify({"status": "error", "message": f"synthesis failure"}), 400)
        t2 = time.time()
        logger.info(f"finish in {(t2 - t1):.2f}s")

        return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/ssml', methods=["POST"])
@require_api_key
def ssml():
    try:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form
        ssml = data.get("ssml")
    except Exception as e:
        logger.info(f"[ssml] {e}")
        return make_response(jsonify({"status": "error", "message": f"parameter error"}), 400)

    logger.debug(ssml)

    t1 = time.time()
    try:
        output, format = tts.create_ssml_infer_task(ssml)
    except Exception as e:
        logger.info(f"[ssml] {e}")
        return make_response(jsonify({"status": "error", "message": f"synthesis failure"}), 400)
    t2 = time.time()

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    logger.info(f"[ssml] finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/dimension-emotion', methods=["POST"])
def dimensional_emotion():
    if request.method == "POST":
        try:
            audio = request.files['upload']
        except Exception as e:
            logger.error(f"[dimensional_emotion] {e}")
            return make_response("parameter error", 400)

    content = BytesIO(audio.read())

    file_type = "application/octet-stream; charset=ascii"
    fname = os.path.splitext(audio.filename)[0] + ".npy"
    output = tts.get_dimensional_emotion_npy(content)

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/check', methods=["GET", "POST"])
def check():
    try:
        if request.method == "GET":
            model = request.args.get("model")
            id = int(request.args.get("id"))
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                data = request.get_json()
            else:
                data = request.form
            model = data.get("model")
            id = int(data.get("id"))
    except Exception as e:
        logger.info(f"[check] {e}")
        return make_response(jsonify({"status": "error", "message": "parameter error"}), 400)

    if check_is_none(model):
        logger.info(f"[check] model {model} is empty")
        return make_response(jsonify({"status": "error", "message": "model is empty"}), 400)

    if model.upper() not in ("VITS", "HUBERT", "W2V2"):
        res = make_response(jsonify({"status": "error", "message": f"model {model} does not exist"}))
        res.status = 404
        logger.info(f"[check] speaker id {id} error")
        return res

    if check_is_none(id):
        logger.info(f"[check] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if model.upper() == "VITS":
        speaker_list = tts.voice_speakers["VITS"]
    elif model.upper() == "HUBERT":
        speaker_list = tts.voice_speakers["HUBERT-VITS"]
    elif model.upper() == "W2V2":
        speaker_list = tts.voice_speakers["W2V2-VITS"]

    if len(speaker_list) == 0:
        logger.info(f"[check] {model} not loaded")
        return make_response(jsonify({"status": "error", "message": f"{model} not loaded"}), 400)

    if id < 0 or id >= len(speaker_list):
        logger.info(f"[check] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)
    name = str(speaker_list[id]["name"])
    lang = speaker_list[id]["lang"]
    logger.info(f"[check] check id:{id} name:{name} lang:{lang}")

    return make_response(jsonify({"status": "success", "id": id, "name": name, "lang": lang}), 200)


# regular cleaning
@scheduler.task('interval', id='clean_task', seconds=3600, misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))  # 对外开放
    # app.run(host='127.0.0.1', port=app.config.get("PORT",23456), debug=True)  # 本地运行、调试
