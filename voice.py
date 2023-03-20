import os

from scipy.io.wavfile import write
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
# import torch
# torch.set_num_threads(1) #设置torch线程为1，防止多任务推理时服务崩溃，但flask仍然会使用多线程
from torch import no_grad, LongTensor
import uuid
from io import BytesIO


class Voice:
    def __init__(self, model, config, out_path=None):
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            try:
                os.mkdir(self.out_path)
            except:
                pass

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
            length_scale, text = self.get_label_value(text, 'LENGTH', 1, 'length scale')
            noise_scale, text = self.get_label_value(text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = self.get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = self.get_label(text, 'CLEANED')

            stn_tst = self.get_text(text, self.hps_ms, cleaned=cleaned)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)])
                sid = LongTensor([speaker_id])
                audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid,
                                            noise_scale=noise_scale,
                                            noise_scale_w=noise_scale_w,
                                            length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                file_name = str(uuid.uuid1())

                with BytesIO() as f:
                    file_path = self.out_path + "/" + file_name + ".wav"
                    # out_path = self.out_path + "/" + file_name + ".ogg"

                    if format == 'ogg':
                        write(f, self.hps_ms.data.sampling_rate, audio)
                        with BytesIO() as o:
                            utils.wav2ogg(f, o)
                            return BytesIO(o.getvalue()), "audio/ogg", file_name + ".ogg"
                    elif format == 'silk':
                        write(file_path, 24000, audio)
                        silk_path = utils.convert_to_silk(file_path)
                        os.remove(file_path)
                        return silk_path, "audio/silk", file_name + ".silk"
                    else:
                        write(f, self.hps_ms.data.sampling_rate, audio)
                        return BytesIO(f.getvalue()), "audio/wav", file_name + ".wav"

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
        return self.speakers

    def get_label(self, text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text


def merge_model(merging_model):
    voice_obj = []
    voice_speakers = []
    new_id = 0
    out_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "/out_slik"
    for i in merging_model:
        obj = Voice(i[0], i[1], out_path)
        for id, name in enumerate(obj.return_speakers()):
            voice_obj.append([int(id), obj])
            voice_speakers.append({new_id: name})

            new_id += 1

    return voice_obj, voice_speakers
