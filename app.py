import os
import logging
import uuid
from flask import Flask, request, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_apscheduler import APScheduler
from utils import clean_folder, merge_model, check_is_none, clasify_lang

app = Flask(__name__)
app.config.from_pyfile("config.py")

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

logger = logging.getLogger('moegoe-simple-api')
logger.setLevel(logging.INFO)

voice_obj, voice_speakers = merge_model(app.config["MODEL_LIST"])

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.mkdir(app.config['UPLOAD_FOLDER'])
    except:
        pass


@app.route('/')
@app.route('/voice/')
def index():
    return "usage:https://github.com/Artrajz/MoeGoe-Simple-API#readme"


@app.route('/voice/speakers', methods=["GET", "POST"])
def voice_speakers_api():
    speakers_list = voice_speakers
    json = {
        "VITS": speakers_list[0],
        "HuBert-VITS": speakers_list[1],
        "W2V2-VITS": speakers_list[2]
    }

    return jsonify(json)


@app.route('/voice', methods=["GET", "POST"])
@app.route('/voice/vits', methods=["GET", "POST"])
def voice_api():
    if request.method == "GET":
        text = request.args.get("text")
        speaker_id = int(request.args.get("id", 0))
        format = request.args.get("format", "wav")
        lang = request.args.get("lang", "auto")
        length = float(request.args.get("length", 1.0))
        noise = float(request.args.get("noise", 0.667))
        noisew = float(request.args.get("noisew", 0.8))
    elif request.method == "POST":
        text = request.form["text"]
        speaker_id = int(request.form["id"])
        format = request.form["format"]
        lang = request.form["lang"]
        length = float(request.form["length"])
        noise = float(request.form["noise"])
        noisew = float(request.form["noisew"])

    if lang.upper() == "MIX":
        pass
    elif lang.upper() == "ZH":
        text = f"[ZH]{text}[ZH]"
    elif lang.upper() == "JA":
        text = f"[JA]{text}[JA]"
    elif lang.upper() == "AUTO":
        text = clasify_lang(text)

    real_id = voice_obj[0][speaker_id][0]
    real_obj = voice_obj[0][speaker_id][1]
    logger.info(msg=f"角色id：{speaker_id}")
    logger.info(msg=f"合成文本：{text}")

    output, file_type, fname = real_obj.generate(text=text,
                                                 speaker_id=real_id,
                                                 format=format,
                                                 length=length,
                                                 noise=noise,
                                                 noisew=noisew)

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/hubert-vits', methods=["GET", "POST"])
def voice_hubert_api():
    if request.method == "POST":
        voice = request.files['upload']
        target_id = int(request.form["target_id"])
        format = request.form["format"]
        length = float(request.form["length"])
        noise = float(request.form["noise"])
        noisew = float(request.form["noisew"])

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    real_id = voice_obj[1][target_id][0]
    real_obj = voice_obj[1][target_id][1]

    output, file_type, fname = real_obj.generate(target_id=real_id,
                                                 format=format,
                                                 length=length,
                                                 noise=noise,
                                                 noisew=noisew,
                                                 audio_path=os.path.join(app.config['UPLOAD_FOLDER'], fname))

    return send_file(path_or_file=output, mimetype=file_type, download_name=fname)


@app.route('/voice/conversion', methods=["GET", "POST"])
def voice_conversion_api():
    if request.method == "GET":
        return jsonify("method should be POST")
    if request.method == "POST":
        voice = request.files['upload']
        original_id = int(request.form["original_id"])
        target_id = int(request.form["target_id"])

        format = voice.filename.split(".")[1]

        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

        real_original_id = int(voice_obj[0][original_id][0])
        real_target_id = int(voice_obj[0][target_id][0])
        real_obj = voice_obj[0][original_id][1]
        real_target_obj = voice_obj[0][target_id][1]

        if voice_obj[0][original_id][2] != voice_obj[0][target_id][2]:
            res = make_response("speaker IDs are in diffrent Model!")
            res.status = 600
            res.headers["msg"] = "speaker IDs are in diffrent Model!"
            return res

        output = real_obj.voice_conversion(os.path.join(app.config['UPLOAD_FOLDER'], fname),
                                           real_original_id, real_target_id)
        file_type = f"audio/{format}"

        return send_file(path_or_file=output, mimetype=file_type, download_name=fname)
        # return output


@app.route('/voice/check', methods=["GET", "POST"])
def check_id():
    if request.method == "GET":
        model = request.args.get("model")
        speaker_id = int(request.args.get("id"))
    elif request.method == "POST":
        model = request.form["model"]
        speaker_id = int(request.form["id"])

    if check_is_none(model):
        res = make_response("model is empty")
        res.status = 600
        res.headers["msg"] = "model is empty"
        return res

    if model.upper() not in ("VITS", "HUBERT", "W2V2"):
        res = make_response("model is not exist")
        res.status = 600
        res.headers["msg"] = "model is not exist"
        return res

    if check_is_none(speaker_id):
        res = make_response("id is not exist")
        res.status = 600
        res.headers["msg"] = "id is not exist"
        return res

    if model.upper() == "VITS":
        speaker_list = voice_speakers[0]
    elif model.upper() == "HUBERT-VITS":
        speaker_list = voice_speakers[1]
    elif model.upper() == "W2V2-VITS":
        speaker_list = voice_speakers[2]
    if speaker_id < 0 or speaker_id >= len(speaker_list):
        res = make_response("speaker id error")
        res.status = 600
        res.headers["msg"] = "speaker id error"
        return res
    name = str(speaker_list[speaker_id][speaker_id])
    res = make_response(f"success check id:{speaker_id} name:{name}")
    res.status = 200
    res.headers["msg"] = "success"
    return res


# 定时清理临时文件，每小时清一次
@scheduler.task('interval', id='clean_task', seconds=3600, misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["PORT"])  # 如果对外开放用这个,docker部署也用这个
    # app.run(host='127.0.0.1', port=app.config["PORT"], debug=True)  # 本地运行、调试
