import os

import librosa
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
from utils import utils
import commons
import sys
import re
import numpy as np
import torch
from torch import no_grad, LongTensor, inference_mode, FloatTensor
import uuid
from io import BytesIO
from graiax import silkcoder
from utils.nlp import cut, sentence_split

# torch.set_num_threads(1) # 设置torch线程为1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch:{torch.__version__}", f"GPU_available:{torch.cuda.is_available()}")
print(f'device:{device} device.type:{device.type}')


class vits:
    def __init__(self, model, config, model_=None):
        self.mode_type = None
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

        if self.n_symbols != 0:
            if not self.emotion_embedding:
                self.mode_type = "vits"
            else:
                self.mode_type = "w2v2"
        else:
            self.mode_type = "hubert-soft"

        # load model
        self.load_model(model, model_)

    def load_model(self, model, model_=None):
        utils.load_checkpoint(model, self.net_g_ms)
        self.net_g_ms.to(device)
        if self.mode_type == "hubert-soft":
            from hubert_model import hubert_soft
            self.hubert = hubert_soft(model_)
        if self.mode_type == "w2v2":
            # import audonnx
            # self.w2v2 = audonnx.load(model)
            self.emotion_reference = np.load(model_)

    def get_cleaned_text(self, text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def get_label_value(self, label, default, warning_name='value', text=""):
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
        if text == "":
            return value
        else:
            return value, text

    def get_label(self, text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    def get_cleaner(self):
        return getattr(self.hps_ms.data, 'text_cleaners', [None])[0]

    def return_speakers(self, escape=False):
        return self.speakers

    def encode(self, sampling_rate, audio, format):
        with BytesIO() as f:
            write(f, sampling_rate, audio)
            if format == 'ogg':
                with BytesIO() as o:
                    utils.wav2ogg(f, o)
                    return BytesIO(o.getvalue())
            elif format == 'silk':
                return BytesIO(silkcoder.encode(f))
            elif format == 'mp3':
                with BytesIO() as o:
                    utils.wav2mp3(f, o)
                    return BytesIO(o.getvalue())
            elif format == 'wav':
                return BytesIO(f.getvalue())

    def infer(self, params):
        emotion = params.get("emotion", None)

        with no_grad():
            x_tst = params.get("stn_tst").unsqueeze(0)
            x_tst_lengths = LongTensor([params.get("stn_tst").size(0)])

            audio = self.net_g_ms.infer(x_tst.to(device), x_tst_lengths.to(device), sid=params.get("sid").to(device),
                                        noise_scale=params.get("noise_scale"),
                                        noise_scale_w=params.get("noise_scale_w"),
                                        length_scale=params.get("length_scale"),
                                        emotion_embedding=emotion.to(device) if emotion != None else None)[0][
                0, 0].data.float().cpu().numpy()

        torch.cuda.empty_cache()
        return audio

    def get_infer_param(self, length, noise, noisew, text=None, speaker_id=None, audio_path=None,
                        emotion=None):
        emo = None
        if self.mode_type != "hubert-soft":
            length_scale, text = self.get_label_value('LENGTH', length, 'length scale', text)
            noise_scale, text = self.get_label_value('NOISE', noise, 'noise scale', text)
            noise_scale_w, text = self.get_label_value('NOISEW', noisew, 'deviation of noise', text)
            cleaned, text = self.get_label(text, 'CLEANED')

            stn_tst = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
            sid = LongTensor([speaker_id])

        if self.mode_type == "w2v2":
            # if emotion_reference.endswith('.npy'):
            #     emotion = np.load(emotion_reference)
            #     emotion = FloatTensor(emotion).unsqueeze(0)
            # else:
            #     audio16000, sampling_rate = librosa.load(
            #         emotion_reference, sr=16000, mono=True)
            #     emotion = self.w2v2(audio16000, sampling_rate)[
            #         'hidden_states']
            #     emotion_reference = re.sub(
            #         r'\..*$', '', emotion_reference)
            #     np.save(emotion_reference, emotion.squeeze(0))
            #     emotion = FloatTensor(emotion)
            emo = torch.FloatTensor(self.emotion_reference[emotion]).unsqueeze(0)


        elif self.mode_type == "hubert-soft":
            if self.use_f0:
                audio, sampling_rate = librosa.load(
                    audio_path, sr=self.hps_ms.data.sampling_rate, mono=True)
                audio16000 = librosa.resample(
                    audio, orig_sr=sampling_rate, target_sr=16000)
            else:
                audio16000, sampling_rate = librosa.load(
                    audio_path, sr=16000, mono=True)

            length_scale = self.get_label_value('LENGTH', length, 'length scale')
            noise_scale = self.get_label_value('NOISE', noise, 'noise scale')
            noise_scale_w = self.get_label_value('NOISEW', noisew, 'deviation of noise')

            with inference_mode():
                units = self.hubert.units(FloatTensor(audio16000).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
                if self.use_f0:
                    f0_scale = self.get_label_value('F0', 1, 'f0 scale')
                    f0 = librosa.pyin(audio,
                                      sr=sampling_rate,
                                      fmin=librosa.note_to_hz('C0'),
                                      fmax=librosa.note_to_hz('C7'),
                                      frame_length=1780)[0]
                    target_length = len(units[:, 0])
                    f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                 np.arange(0, len(f0)), f0)) * f0_scale
                    units[:, 0] = f0 / 10

            stn_tst = FloatTensor(units)
            sid = LongTensor([speaker_id])
        params = {"length_scale": length_scale, "noise_scale": noise_scale,
                  "noise_scale_w": noise_scale_w, "stn_tst": stn_tst,
                  "sid": sid, "emotion": emo}
        return params

    def create_infer_task(self, text=None, speaker_id=None, format=None, length=1, noise=0.667, noisew=0.8,
                          audio_path=None, max=50, lang="auto", emotion=0, speaker_lang=None):
        # params = self.get_infer_param(text=text, speaker_id=speaker_id, length=length, noise=noise, noisew=noisew,
        #                               target_id=target_id)
        tasks = []
        if self.mode_type == "vits":

            sentence_list = sentence_split(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                tasks.append(
                    self.get_infer_param(text=sentence, speaker_id=speaker_id, length=length, noise=noise,
                                         noisew=noisew))
            audios = []
            for task in tasks:
                audios.append(self.infer(task))

            audio = np.concatenate(audios, axis=0)
        elif self.mode_type == "hubert-soft":
            params = self.get_infer_param(speaker_id=speaker_id, length=length, noise=noise, noisew=noisew,
                                          audio_path=audio_path)
            audio = self.infer(params)
        elif self.mode_type == "w2v2":
            sentence_list = sentence_split(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                tasks.append(
                    self.get_infer_param(text=sentence, speaker_id=speaker_id, length=length, noise=noise,
                                         noisew=noisew, emotion=emotion))
            audios = []
            for task in tasks:
                audios.append(self.infer(task))

            audio = np.concatenate(audios, axis=0)

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
            audio = self.net_g_ms.voice_conversion(spec.to(device),
                                                   spec_lengths.to(device),
                                                   sid_src=sid_src.to(device),
                                                   sid_tgt=sid_tgt.to(device))[0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()
        with BytesIO() as f:
            write(f, self.hps_ms.data.sampling_rate, audio)
            return BytesIO(f.getvalue())
