import logging
import os
import sys
from io import BytesIO
from json import loads
import av
import pilk
from flask import Flask
from torch import load, FloatTensor
from numpy import float32
import librosa
import regex as re
from fastlid import fastlid
from voice import Voice

app = Flask(__name__)
app.config.from_pyfile("config.py")


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def load_checkpoint(checkpoint_path, model):
    checkpoint_dict = load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logging.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logging.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return


def get_hparams_from_file(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = f.read()
    config = loads(data)

    hparams = HParams(**config)
    return hparams


def load_audio_to_torch(full_path, target_sampling_rate):
    audio, sampling_rate = librosa.load(full_path, sr=target_sampling_rate, mono=True)
    return FloatTensor(audio.astype(float32))


def wav2ogg(input, output):
    with av.open(input, 'rb') as i:
        with av.open(output, 'wb', format='ogg') as o:
            out_stream = o.add_stream('libvorbis')
            for frame in i.decode(audio=0):
                for p in out_stream.encode(frame):
                    o.mux(p)

            for p in out_stream.encode(None):
                o.mux(p)


# def wav2silk(input, output):
#     with av.open(input) as in_wav:
#         in_stream = in_wav.streams.audio[0]
#         sample_rate = in_stream.codec_context.sample_rate
#         with BytesIO() as pcm:
#             with av.open(pcm, 'w', 's16le') as out_pcm:
#                 out_stream = out_pcm.add_stream(
#                     'pcm_s16le',
#                     rate=sample_rate,
#                     layout='mono'
#                 )
#                 for frame in in_wav.decode(in_stream):
#                     frame.pts = None
#                     for packet in out_stream.encode(frame):
#                         out_pcm.mux(packet)
#
#             pilk.encode(out_pcm, output, pcm_rate=sample_rate, tencent=True)


def to_pcm(in_path: str) -> tuple[str, int]:
    out_path = os.path.splitext(in_path)[0] + '.pcm'
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        sample_rate = in_stream.codec_context.sample_rate
        with av.open(out_path, 'w', 's16le') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=sample_rate,
                layout='mono'
            )
            try:
                for frame in in_container.decode(in_stream):
                    frame.pts = None
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)
            except:
                pass
    return out_path, sample_rate


def convert_to_silk(media_path: str) -> str:
    pcm_path, sample_rate = to_pcm(media_path)
    silk_path = os.path.splitext(pcm_path)[0] + '.silk'
    pilk.encode(pcm_path, silk_path, pcm_rate=sample_rate, tencent=True)
    os.remove(pcm_path)
    return silk_path


def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 如果是文件，则删除文件
        if os.path.isfile(file_path):
            os.remove(file_path)


def merge_model(merging_model):
    vits_obj = []
    vits_speakers = []
    hubert_vits_obj = []
    hubert_vits_speakers = []
    w2v2_vits_obj = []
    w2v2_vits_speakers = []

    # model list
    vits_list = []
    hubert_vits_list = []
    w2v2_vits_list = []

    out_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "/cache"

    # classify
    for l in merging_model:
        if len(l) == 2:
            vits_list.append(l)
        elif len(l) == 3:
            hubert_vits_list.append(l)

    # merging vits
    new_id = 0
    for obj_id, i in enumerate(vits_list):
        obj = Voice(model=i[0], config=i[1], out_path=out_path)

        for id, name in enumerate(obj.return_speakers()):
            vits_obj.append([int(id), obj, obj_id])
            vits_speakers.append({new_id: name})

            new_id += 1

    # merging hubert-vits
    new_id = 0
    for obj_id, i in enumerate(hubert_vits_list):
        obj = Voice(model=i[0], config=i[1], hubert_soft_model=i[2], out_path=out_path)

        for id, name in enumerate(obj.return_speakers()):
            hubert_vits_obj.append([int(id), obj, obj_id])
            hubert_vits_speakers.append({new_id: name})

            new_id += 1

    voice_obj = [vits_obj, hubert_vits_obj, w2v2_vits_obj]
    voice_speakers = [vits_speakers, hubert_vits_speakers, w2v2_vits_speakers]

    return voice_obj, voice_speakers


def clasify_lang(text: str) -> str:
    pattern = r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`' \
              r'\！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」' \
              r'『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+'
    words = re.split(pattern, text)

    pre = ""
    for word in words:
        if check_is_none(word): continue

        lang = fastlid(word)[0]
        if pre == "":
            text = text.replace(word, f'[{lang.upper()}]' + word)
        elif pre != lang:
            text = text.replace(word, f'[{pre.upper()}][{lang.upper()}]' + word)
        pre = lang
    text += f"[{pre.upper()}]"

    return text


# is none -> True,is not none -> False
def check_is_none(s: str) -> bool:
    s = str(s)
    if s == None or s == "" or s.isspace():
        return True
    return False
