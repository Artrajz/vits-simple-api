import os
import logging
import time
import logzero
import uuid
from flask import Flask, request, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_apscheduler import APScheduler
from functools import wraps
from utils.utils import clean_folder, check_is_none
from utils.nlp import clasify_lang
from utils.merge import merge_model

app = Flask(__name__)
app.config.from_pyfile("config.py")

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

logger = logging.getLogger('vits-simple-api')
logger.setLevel(logging.INFO)
logzero.loglevel(logging.WARNING)

voice_obj, voice_speakers = merge_model(app.config["MODEL_LIST"])

print(f"loaded {sum([len(voice_speakers[i]) for i in range(len(voice_speakers))])} speakers")

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.mkdir(app.config['UPLOAD_FOLDER'])
    except:
        pass


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
                res = make_response(jsonify({"status": "error", "message": "Invalid API Key"}))
                res.status = 401
                res.headers["message"] = "Invalid API Key"
                return res

    return check_api_key


@app.route('/', methods=["GET", "POST"])
def index():
    return ""


@app.route('/voice/speakers', methods=["GET", "POST"])
def voice_speakers_api():
    models_json = {
        "VITS": voice_speakers[0],
        "HuBert-VITS": voice_speakers[1],
        "W2V2-VITS": voice_speakers[2]
    }

    return jsonify(models_json)


@app.route('/voice', methods=["GET", "POST"])
@app.route('/voice/vits', methods=["GET", "POST"])
@require_api_key
def voice_api():
    try:
        if request.method == "GET":
            text = request.args.get("text")
            speaker_id = int(request.args.get("id", app.config.get("ID", 0)))
            format = request.args.get("format", app.config.get("FORMAT", "wav"))
            lang = request.args.get("lang", app.config.get("LANG", "auto"))
            length = float(request.args.get("length", app.config.get("LENGTH", 1)))
            noise = float(request.args.get("noise", app.config.get("NOISE", 0.667)))
            noisew = float(request.args.get("noisew", app.config.get("NOISEW", 0.8)))
            max = int(request.args.get("max", app.config.get("MAX", 50)))
        elif request.method == "POST":
            text = request.form["text"]
            speaker_id = int(request.form["id"])
            format = request.form["format"]
            lang = request.form["lang"]
            length = float(request.form["length"])
            noise = float(request.form["noise"])
            noisew = float(request.form["noisew"])
            max = int(request.form["max"])
    except Exception as e:
        res = make_response("param error")
        res.status = 400
        res.headers["message"] = "param error"
        logger.error(msg=f"{e} {e.args}")
        return res

    if check_is_none(text):
        res = make_response(jsonify({"status": "error", "message": "text is empty"}))
        res.status = 404
        logger.info(msg=f"text is empty")
        return res

    real_id = voice_obj[0][speaker_id][0]
    real_obj = voice_obj[0][speaker_id][1]

    logger.info(msg=f"VITS id:{speaker_id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew}")
    logger.info(msg=f"len:{len(text)} text：{text}")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    t1 = time.time()
    output = real_obj.create_infer_task(text=text,
                                        speaker_id=real_id,
                                        format=format,
                                        length=length,
                                        noise=noise,
                                        noisew=noisew,
                                        max=max,
                                        lang=lang)
    t2 = time.time()
    logger.info(msg=f"finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/hubert-vits', methods=["GET", "POST"])
@require_api_key
def voice_hubert_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            target_id = int(request.form["target_id"])
            format = request.form["format"]
            length = float(request.form["length"])
            noise = float(request.form["noise"])
            noisew = float(request.form["noisew"])
        except Exception as e:
            res = make_response("param error")
            res.status = 400
            res.headers["message"] = "param error"
            logger.error(msg=f"{e} {e.args}")
            return res

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    try:
        real_id = voice_obj[1][target_id][0]
        real_obj = voice_obj[1][target_id][1]
    except Exception:
        res = make_response("target id error")
        res.status = 400
        res.headers["message"] = "target id error"
        return res

    file_type = f"audio/{format}"
    logger.info(msg=f"HuBert-soft id:{target_id} format:{format} length:{length} noise:{noise} noisew:{noisew}")
    t1 = time.time()
    output = real_obj.create_infer_task(target_id=real_id,
                                        format=format,
                                        length=length,
                                        noise=noise,
                                        noisew=noisew,
                                        audio_path=os.path.join(app.config['UPLOAD_FOLDER'], fname))
    t2 = time.time()
    logger.info(msg=f"finish in {(t2 - t1):.2f}s")

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/conversion', methods=["GET", "POST"])
@require_api_key
def voice_conversion_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            original_id = int(request.form["original_id"])
            target_id = int(request.form["target_id"])
        except Exception as e:
            res = make_response("param error")
            res.status = 400
            res.headers["message"] = "param error"
            logger.error(msg=f"{e} {e.args}")
            return res

        format = voice.filename.split(".")[1]

        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
        file_type = f"audio/{format}"

        real_original_id = int(voice_obj[0][original_id][0])
        real_target_id = int(voice_obj[0][target_id][0])
        real_obj = voice_obj[0][original_id][1]
        real_target_obj = voice_obj[0][target_id][1]

        # if voice_obj[0][original_id][2] != voice_obj[0][target_id][2]:
        #     res = make_response("speaker IDs are in diffrent Model!")
        #     res.status = 400
        #     res.headers["message"] = "speaker IDs are in diffrent Model!"
        #     return res

        logger.info(msg=f"voice_convetsion orginal_id:{original_id} target_id:{target_id}")
        t1 = time.time()
        output = real_obj.voice_conversion(os.path.join(app.config['UPLOAD_FOLDER'], fname),
                                           real_original_id, real_target_id)
        t2 = time.time()
        logger.info(msg=f"finish in {(t2 - t1):.2f}s")

        return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/check', methods=["GET", "POST"])
def check():
    try:
        if request.method == "GET":
            model = request.args.get("model")
            speaker_id = int(request.args.get("id"))
        elif request.method == "POST":
            model = request.form["model"]
            speaker_id = int(request.form["id"])
    except Exception as e:
        res = make_response(jsonify({"status": "error", "message": "param error"}))
        res.status = 400
        logger.info(msg=f"{e}")
        return res

    if check_is_none(model):
        res = make_response(jsonify({"status": "error", "message": "model is empty"}))
        res.status = 404
        logger.info(msg=f"model is empty")
        return res

    if model.upper() not in ("VITS", "HUBERT", "W2V2"):
        res = make_response(jsonify({"status": "error", "message": f"model {model} does not exist"}))
        res.status = 404
        logger.info(msg=f"speaker id {speaker_id} error")
        return res

    if check_is_none(speaker_id):
        res = make_response(jsonify({"status": "error", "message": "id is empty"}))
        res.status = 404
        logger.info(msg=f"speaker id {speaker_id} error")
        return res

    if model.upper() == "VITS":
        speaker_list = voice_speakers[0]
    elif model.upper() == "HUBERT":
        speaker_list = voice_speakers[1]
    elif model.upper() == "W2V2":
        speaker_list = voice_speakers[2]

    if len(speaker_list) == 0:
        res = make_response(jsonify({"status": "error", "message": f"{model} not loaded"}))
        res.status = 404
        logger.info(msg=f"{model} not loaded")
        return res

    if speaker_id < 0 or speaker_id >= len(speaker_list):
        res = make_response(jsonify({"status": "error", "message": f"id {speaker_id} does not exist"}))
        res.status = 404
        logger.info(msg=f"speaker id {speaker_id} does not exist")
        logger.info(msg=f"speaker id {speaker_id} does not exist")
        return res
    name = str(speaker_list[speaker_id][speaker_id])
    logger.info(msg=f"check id:{speaker_id} name:{name}")

    res = make_response(jsonify({"status": "success", "id": speaker_id, "name": name}))
    res.status = 200
    return res


# regular cleaning
@scheduler.task('interval', id='clean_task', seconds=3600, misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["PORT"])  # 如果对外开放用这个,docker部署也用这个
    # app.run(host='127.0.0.1', port=app.config["PORT"], debug=True)  # 本地运行、调试
