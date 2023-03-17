import os

import logging
from flask import Flask, request, send_file

from voice import merge_model
from voice import Voice

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config["port"] = 23456

logging.getLogger('numba').setLevel(logging.WARNING)

"""
VITS Model example

model_zh = "model_path"
config_zh = "config.json_path"
voice = Voice(model, config)
"""

# 可能遇到获取不到绝对路径的情况，取消以下注释使用可以取到绝对路径的方法替换下面的路径即可
# print("os.path.dirname(__file__)",os.path.dirname(__file__))
# print("os.path.dirname(sys.argv[0])",os.path.dirname(sys.argv[0]))
# print("os.path.realpath(sys.argv[0])",os.path.realpath(sys.argv[0]))
# print("os.path.dirname(os.path.realpath(sys.argv[0]))",os.path.dirname(__file__))

# out_path = os.path.dirname(__file__) + "/output/"

model_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth"
config_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/config.json"
voice_zh = Voice(model_zh, config_zh)

model_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/1158_epochs.pth"
config_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/config.json"
voice_ja = Voice(model_ja, config_ja)

model_g = os.path.dirname(__file__) + "/Model/g/G_953000.pth"
config_g = os.path.dirname(__file__) + "/Model/g/config.json"

merging_list = [
    [model_zh, config_zh],
    [model_ja, config_ja],
    [model_g, config_g],
]
voice_obj, voice_speakers = merge_model(merging_list)


@app.route('/voice/')
def index():
    return "usage:https://github.com/Artrajz/MoeGoe-Simple-API#readme"


@app.route('/voice/ja/speakers')
def voice_ja_speakers_api():
    escape = False
    speakers_list = voice_ja.return_speakers(escape)
    return speakers_list


@app.route('/voice/ja', methods=["GET"])
def voice_ja_api():
    text = request.args.get("text")
    speaker_id = int(request.args.get("id", 2))
    format = request.args.get("format", "wav")
    lang = request.args.get("lang", "ja")

    if lang.upper() == "ZH":
        text = f"[ZH]{text}[ZH]"
    elif lang.upper() == "JA":
        text = f"[JA]{text}[JA]"

    output = voice_ja.generate(text, speaker_id, format)
    return send_file(output)


@app.route('/voice/zh/speakers')
def voice_zh_speakers_api():
    escape = False
    speakers_list = voice_zh.return_speakers(escape)
    return speakers_list


@app.route('/voice/zh', methods=["GET"])
def voice_zh_api():
    text = request.args.get("text")
    speaker_id = int(request.args.get("id", 3))
    format = request.args.get("format", "wav")
    lang = request.args.get("lang", "zh")

    if lang.upper() == "ZH":
        text = f"[ZH]{text}[ZH]"
    elif lang.upper() == "JA":
        text = f"[JA]{text}[JA]"

    output, file_type, file_name = voice_zh.generate(text, speaker_id, format)

    return send_file(path_or_file=output, mimetype=file_type, download_name=file_name)


@app.route('/voice/speakers')
def voice_speakers_api():
    speakers_list = voice_speakers
    return speakers_list


@app.route('/voice', methods=["GET"])
def voice_api():
    text = request.args.get("text")
    speaker_id = int(request.args.get("id", 0))
    format = request.args.get("format", "wav")
    lang = request.args.get("lang", "mix")

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
