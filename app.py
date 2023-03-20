import os

import logging
from flask import Flask, request, send_file, jsonify

from voice import merge_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config["port"] = 23456

logging.getLogger('numba').setLevel(logging.WARNING)

'''vits models path'''
model_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth"
config_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/config.json"

model_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/1158_epochs.pth"
config_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/config.json"

model_g = os.path.dirname(__file__) + "/Model/g/G_953000.pth"
config_g = os.path.dirname(__file__) + "/Model/g/config.json"

'''add models here'''
merging_list = [
    [model_zh, config_zh],
    [model_ja, config_ja],
    [model_g, config_g],
]
voice_obj, voice_speakers = merge_model(merging_list)

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

    output, file_type, file_name = real_obj.generate(text, real_id, format)

    return send_file(path_or_file=output, mimetype=file_type, download_name=file_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["port"])  # 如果对外开放用这个
    # app.run(host='127.0.0.1', port=app.config["port"], debug=True)  # 本地运行
