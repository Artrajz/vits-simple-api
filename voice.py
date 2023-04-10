import os

import librosa
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
import numpy as np
# import torch
# torch.set_num_threads(1) #设置torch线程为1，防止多任务推理时服务崩溃，但flask仍然会使用多线程
from torch import no_grad, LongTensor, inference_mode, FloatTensor
import audonnx
import uuid
from io import BytesIO
from graiax import silkcoder


class Voice:
    def __init__(self, model, config, hubert_soft_model=None, out_path=None):
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            try:
                os.mkdir(self.out_path)
            except:
                pass

        self.hps_ms = utils.get_hparams_from_file(config)
        self.n_speakers = self.hps_ms.data.n_speakers if 'n_speakers' in self.hps_ms.data.keys() else 0
        self.n_symbols = len(self.hps_ms.symbols) if 'symbols' in self.hps_ms.keys() else 0
        self.speakers = self.hps_ms.speakers if 'speakers' in self.hps_ms.keys() else ['0']
        self.use_f0 = self.hps_ms.data.use_f0 if 'use_f0' in self.hps_ms.data.keys() else False
        self.emotion_embedding = self.hps_ms.data.emotion_embedding if 'emotion_embedding' in self.hps_ms.data.keys() else False

        self.net_g_ms = SynthesizerTrn(
            self.n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.n_speakers,
            emotion_embedding=self.emotion_embedding,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(model, self.net_g_ms)

        # load hubert-soft model
        if hubert_soft_model != None and self.n_symbols == 0:
            from hubert_model import hubert_soft
            self.hubert = hubert_soft(hubert_soft_model)

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

    # def ex_return(self, text, escape=False):
    #     if escape:
    #         return text.encode('unicode_escape').decode()
    #     else:
    #         return text

    def return_speakers(self, escape=False):
        return self.speakers

    def get_label(self, text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    def infer(self, text=None, speaker_id=None, length=1, noise=0.667, noisew=0.8, audio_path=None,
              target_id=None, escape=False,
              option=None, w2v2_folder=None):
        if self.n_symbols != 0:
            if not self.emotion_embedding:
                length_scale, text = self.get_label_value(text, 'LENGTH', length, 'length scale')
                noise_scale, text = self.get_label_value(text, 'NOISE', noise, 'noise scale')
                noise_scale_w, text = self.get_label_value(text, 'NOISEW', noisew, 'deviation of noise')
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

            # else:
            #     w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))
            #
            #     if option == 'clean':
            #         self.ex_print(_clean_text(
            #             text, self.hps_ms.data.text_cleaners), escape)
            #
            #     length_scale, text = self.get_label_value(
            #         text, 'LENGTH', length, 'length scale')
            #     noise_scale, text = self.get_label_value(
            #         text, 'NOISE', noise, 'noise scale')
            #     noise_scale_w, text = self.get_label_value(
            #         text, 'NOISEW', noisew, 'deviation of noise')
            #     cleaned, text = self.get_label(text, 'CLEANED')
            #
            #     stn_tst = self.get_text(text, self.hps_ms, cleaned=cleaned)
            #
            #     emotion_reference = input('Path of an emotion reference: ')
            #     if emotion_reference.endswith('.npy'):
            #         emotion = np.load(emotion_reference)
            #         emotion = FloatTensor(emotion).unsqueeze(0)
            #     else:
            #         audio16000, sampling_rate = librosa.load(
            #             emotion_reference, sr=16000, mono=True)
            #         emotion = w2v2_model(audio16000, sampling_rate)[
            #             'hidden_states']
            #         emotion_reference = re.sub(
            #             r'\..*$', '', emotion_reference)
            #         np.save(emotion_reference, emotion.squeeze(0))
            #         emotion = FloatTensor(emotion)
            #
            #     with no_grad():
            #         x_tst = stn_tst.unsqueeze(0)
            #         x_tst_lengths = LongTensor([stn_tst.size(0)])
            #         sid = LongTensor([speaker_id])
            #         audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
            #                                     noise_scale_w=noise_scale_w,
            #                                     length_scale=length_scale, emotion_embedding=emotion)[0][
            #             0, 0].data.cpu().float().numpy()

        else:
            if audio_path != '[VC]':
                if self.use_f0:
                    audio, sampling_rate = librosa.load(
                        audio_path, sr=self.hps_ms.data.sampling_rate, mono=True)
                    audio16000 = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio16000, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True)

                fname = str(uuid.uuid1())
                tmp_path = self.out_path + "/" + fname + ".wav"

                length_scale, tmp_path = self.get_label_value(
                    tmp_path, 'LENGTH', length, 'length scale')
                noise_scale, tmp_path = self.get_label_value(
                    tmp_path, 'NOISE', noise, 'noise scale')
                noise_scale_w, tmp_path = self.get_label_value(
                    tmp_path, 'NOISEW', noisew, 'deviation of noise')

                with inference_mode():
                    units = self.hubert.units(FloatTensor(audio16000).unsqueeze(
                        0).unsqueeze(0)).squeeze(0).numpy()
                    if self.use_f0:
                        f0_scale, tmp_path = self.get_label_value(
                            tmp_path, 'F0', 1, 'f0 scale')
                        f0 = librosa.pyin(audio, sr=sampling_rate,
                                          fmin=librosa.note_to_hz('C0'),
                                          fmax=librosa.note_to_hz('C7'),
                                          frame_length=1780)[0]
                        target_length = len(units[:, 0])
                        f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                     np.arange(0, len(f0)), f0)) * f0_scale
                        units[:, 0] = f0 / 10

                stn_tst = FloatTensor(units)
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([target_id])
                    audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                                noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                        0, 0].data.float().numpy()

        return audio

    def encode(self, sampling_rate, audio, format):
        with BytesIO() as f:
            write(f, sampling_rate, audio)
            if format == 'ogg':
                with BytesIO() as o:
                    utils.wav2ogg(f, o)
                    return BytesIO(o.getvalue())
            elif format == 'silk':
                return BytesIO(silkcoder.encode(f))
            elif format == 'wav':
                return BytesIO(f.getvalue())

    def generate(self, text=None, speaker_id=None, format=None, length=1, noise=0.667, noisew=0.8, audio_path=None,
                 target_id=None, escape=False,
                 option=None, w2v2_folder=None):
        audio = self.infer(text, speaker_id, length, noise, noisew, audio_path,
                           target_id, escape,
                           option, w2v2_folder)

        return self.encode(self.hps_ms.data.sampling_rate, audio, format)

    def voice_conversion(self, audio_path, original_id, target_id):

        audio = utils.load_audio_to_torch(
            audio_path, self.hps_ms.data.sampling_rate)

        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, self.hps_ms.data.filter_length,
                                 self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length,
                                 self.hps_ms.data.win_length,
                                 center=False)
        spec_lengths = LongTensor([spec.size(-1)])
        sid_src = LongTensor([original_id])

        with no_grad():
            sid_tgt = LongTensor([target_id])
            audio = self.net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[
                0][0, 0].data.cpu().float().numpy()

        with BytesIO() as f:
            write(f, self.hps_ms.data.sampling_rate, audio)
            return BytesIO(f.getvalue())
