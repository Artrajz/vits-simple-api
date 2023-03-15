import os

import logging
from flask import Flask, request, send_file

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


@app.route('/api/')
def index():
    return "usage:https://github.com/pang-juzhong/MoeGoe-Simple-API#readme"


@app.route('/api/ja/speakers')
def voice_speakers_ja():
    escape = False
    speakers_list = voice_ja.return_speakers(escape)
    return speakers_list


@app.route('/api/ja', methods=["GET"])
def api_voice_ja():
    text = "[JA]" + request.args.get("text") + "[JA]"
    speaker_id = int(request.args.get("id", 0))
    format = request.args.get("format", "wav")

    output = voice_ja.generate(text, speaker_id, format)
    return send_file(output)


@app.route('/api/zh/speakers')
def voice_speakers_zh():
    escape = False
    speakers_list = voice_zh.return_speakers(escape)
    return speakers_list


@app.route('/api/zh', methods=["GET"])
def api_voice_zh():
    text = "[ZH]" + request.args.get("text") + "[ZH]"
    speaker_id = int(request.args.get("id", 3))
    format = request.args.get("format", "wav")

    output, type, file_name = voice_zh.generate(text, speaker_id, format)

    return send_file(path_or_file=output, mimetype=type, download_name=file_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config["port"])  # 如果对外开放用这个
    # app.run(host='127.0.0.1', port=app.config["port"], debug=True)  # 本地运行
