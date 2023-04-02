import os

import logging
import uuid

from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

from voice import merge_model


app = Flask(__name__)
app.config.from_pyfile("config.py")


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
    return jsonify(speakers_list)


@app.route('/voice', methods=["GET", "POST"])
def voice_api():
    if request.method == "GET":
        text = request.args.get("text")
        speaker_id = int(request.args.get("id", 0))
        format = request.args.get("format", "wav")
        lang = request.args.get("lang", "mix")
    elif request.method == "POST":
        json_data = request.json
        text = json_data["text"]
        speaker_id = int(json_data["id"])
        format = json_data["format"]
        lang = json_data["lang"]

    if lang.upper() == "ZH":
        text = f"[ZH]{text}[ZH]"
    elif lang.upper() == "JA":
        text = f"[JA]{text}[JA]"

    real_id = voice_obj[speaker_id][0]
    real_obj = voice_obj[speaker_id][1]

    output, file_type, fname = real_obj.generate(text, real_id, format)

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

        form = {}

        format = voice.filename.split(".")[1]

        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        voice.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

        real_original_id = int(voice_obj[original_id][0])
        real_target_id = int(voice_obj[target_id][0])
        real_obj = voice_obj[original_id][1]
        real_target_obj = voice_obj[target_id][1]

        if voice_obj[original_id][2] != voice_obj[target_id][2]:
            form["status"] = "error"
            form["message"] = "speaker IDs are in diffrent Model!"
            return form

        output = real_obj.voice_conversion(os.path.join(app.config['UPLOAD_FOLDER'], fname),
                                           real_original_id, real_target_id, format)
        file_type = f"audio/{format}"

        return send_file(path_or_file=output, mimetype=file_type, download_name=fname)
        # return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["PORT"])  # 如果对外开放用这个
    # app.run(host='127.0.0.1', port=app.config["PORT"], debug=True)  # 本地运行
