import os

from scipy.io.wavfile import write
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging
from flask import Flask, request, send_file
import uuid
import subprocess
import ffmpeg
from io import BytesIO

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

logging.getLogger('numba').setLevel(logging.WARNING)


class Voice:
    def __init__(self, model, config, out_path=None):
        self.out_path = out_path
        self.hps_ms = utils.get_hparams_from_file(config)
        n_speakers = self.hps_ms.data.n_speakers if 'n_speakers' in self.hps_ms.data.keys() else 0
        n_symbols = len(self.hps_ms.symbols) if 'symbols' in self.hps_ms.keys() else 0
        self.speakers = self.hps_ms.speakers if 'speakers' in self.hps_ms.keys() else ['0']
        use_f0 = self.hps_ms.data.use_f0 if 'use_f0' in self.hps_ms.data.keys() else False
        self.emotion_embedding = self.hps_ms.data.emotion_embedding if 'emotion_embedding' in self.hps_ms.data.keys() else False

        self.net_g_ms = SynthesizerTrn(
            n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=n_speakers,
            emotion_embedding=self.emotion_embedding,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(model, self.net_g_ms)

    def generate(self, text, speaker_id, format):
        if not self.emotion_embedding:
            length_scale, text = self.get_label_value(
                text, 'LENGTH', 1, 'length scale')
            noise_scale, text = self.get_label_value(
                text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = self.get_label_value(
                text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = self.get_label(text, 'CLEANED')

            stn_tst = self.get_text(text, self.hps_ms, cleaned=cleaned)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)])
                sid = LongTensor([speaker_id])
                audio = \
                    self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                        noise_scale_w=noise_scale_w,
                                        length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                file_name = str(uuid.uuid1())

                with BytesIO() as f:
                    if format == 'ogg':
                        file_path = self.out_path+"/"+file_name+".wav"
                        out_path = self.out_path+"/"+file_name+".ogg"
                        write(file_path, self.hps_ms.data.sampling_rate, audio)
                        f.seek(0, 0)
                        #file=BytesIO(f.getvalue())

                        with BytesIO() as ofp:
                            ffmpeg.input(file_path).output(out_path).run()
                            return out_path, "audio/ogg", file_name + ".ogg",
                    else:
                        write(f, self.hps_ms.data.sampling_rate, audio)
                        f.seek(0, 0)
                        return BytesIO(f.getvalue()), "audio/wav", file_name + ".wav",

    def run_script(self, file_path):
        out_path = file_path.split('.')[0] + ".ogg"
        ffmpeg.input(file_path).output(out_path).run()
        subprocess.run(["rm " + file_path], shell=True, timeout=5)
        return out_path

    def get_text(self, text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def get_label_value(self, text, label, default, warning_name='value'):
        value = re.search(rf'\[{label}=(.+?)\]', text)
        if value:
            try:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            except:
                print(f'Invalid {warning_name}!')
                sys.exit(1)
        else:
            value = default
        return value, text

    def ex_return(self, text, escape=False):
        if escape:
            return text.encode('unicode_escape').decode()
        else:
            return text

    def return_speakers(self, escape=False):
        if len(self.speakers) > 100:
            return
        # print('ID\tSpeaker')
        speakers_list = []
        for id, name in enumerate(self.speakers):
            speakers_list.append(self.ex_return(str(id) + '\t' + name, escape))
        return speakers_list

    def get_label(self, text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text


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


out_path = os.path.dirname(__file__) + "/output/"

model_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth"
config_zh = os.path.dirname(__file__) + "/Model/Nene_Nanami_Rong_Tang/config.json"
voice_zh = Voice(model_zh, config_zh, out_path)

model_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/1158_epochs.pth"
config_ja = os.path.dirname(__file__) + "/Model/Zero_no_tsukaima/config.json"
voice_ja = Voice(model_ja, config_ja, out_path)


@app.route('/api/')
def index():
    return "usage:/api/zh?text=text&id=3&format=wav"


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
    app.run(host='0.0.0.0', port=23456, debug=True)  # 如果对外开放用这个
    # app.run(host='127.0.0.1', port=23456)  # 本地运行
