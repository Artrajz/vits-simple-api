import os

import logging
import uuid

from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_apscheduler import APScheduler

from utils import clean_folder, merge_model

app = Flask(__name__)
app.config.from_pyfile("config.py")

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

logging.getLogger('numba').setLevel(logging.WARNING)

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
        "VITS":speakers_list[0],
        "HuBert-VITS":speakers_list[1],
        "W2V2-VITS":speakers_list[2]
    }

    return jsonify(json)


@app.route('/voice', methods=["GET", "POST"])
@app.route('/voice/vits', methods=["GET", "POST"])
def voice_api():
    if request.method == "GET":
        text = request.args.get("text")
        speaker_id = int(request.args.get("id", 0))
        format = request.args.get("format", "wav")
        lang = request.args.get("lang", "mix")
        length = float(request.args.get("length", 1.0))
        noise = float(request.args.get("noise", 0.667))
        noisew = float(request.args.get("noisew", 0.8))
    elif request.method == "POST":
        json_data = request.json
        text = json_data["text"]
        speaker_id = int(json_data["id"])
        format = json_data["format"]
        lang = json_data["lang"]
        length = float(json_data["length"])
        noise = float(json_data["noise"])
        noisew = float(json_data["noisew"])

    if lang.upper() == "ZH":
        text = f"[ZH]{text}[ZH]"
    elif lang.upper() == "JA":
        text = f"[JA]{text}[JA]"

    real_id = voice_obj[0][speaker_id][0]
    real_obj = voice_obj[0][speaker_id][1]

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
        # json_data = request.json
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

        form = {}
        if voice_obj[0][original_id][2] != voice_obj[0][target_id][2]:
            form["status"] = "error"
            form["message"] = "speaker IDs are in diffrent Model!"
            return form

        output = real_obj.voice_conversion(os.path.join(app.config['UPLOAD_FOLDER'], fname),
                                           real_original_id, real_target_id)
        file_type = f"audio/{format}"

        return send_file(path_or_file=output, mimetype=file_type, download_name=fname)
        # return output


# 定时清理临时文件，每小时清一次
@scheduler.task('interval', id='随便写', seconds=3600, misfire_grace_time=900)
def clean_task():
    clean_folder(app.config["UPLOAD_FOLDER"])
    clean_folder(app.config["CACHE_PATH"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["PORT"])  # 如果对外开放用这个,docker部署也用这个
    # app.run(host='127.0.0.1', port=app.config["PORT"], debug=True)  # 本地运行、调试
