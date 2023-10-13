import librosa
import re
import numpy as np
import torch
from torch import no_grad, LongTensor, inference_mode, FloatTensor
import utils
from contants import ModelType
from utils import get_hparams_from_file, lang_dict
from utils.sentence import sentence_split_and_markup
from vits import commons
from vits.mel_processing import spectrogram_torch
from vits.text import text_to_sequence
from vits.models import SynthesizerTrn


class VITS:
    def __init__(self, model, config, additional_model=None, model_type=None, device=torch.device("cpu"), **kwargs):
        self.model_type = model_type
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.n_symbols = len(getattr(self.hps_ms, 'symbols', []))
        self.speakers = getattr(self.hps_ms, 'speakers', ['0'])
        if not isinstance(self.speakers, list):
            self.speakers = [item[0] for item in sorted(list(self.speakers.items()), key=lambda x: x[1])]
        self.use_f0 = getattr(self.hps_ms.data, 'use_f0', False)
        self.emotion_embedding = getattr(self.hps_ms.data, 'emotion_embedding',
                                         getattr(self.hps_ms.model, 'emotion_embedding', False))
        self.bert_embedding = getattr(self.hps_ms.data, 'bert_embedding',
                                      getattr(self.hps_ms.model, 'bert_embedding', False))
        self.hps_ms.model.emotion_embedding = self.emotion_embedding
        self.hps_ms.model.bert_embedding = self.bert_embedding

        self.net_g_ms = SynthesizerTrn(
            self.n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.n_speakers,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        self.device = device

        key = getattr(self.hps_ms.data, "text_cleaners", ["none"])[0]
        self.lang = lang_dict.get(key, ["unknown"])

        # load model
        self.load_model(model, additional_model)

    def load_model(self, model, additional_model=None):
        utils.load_checkpoint(model, self.net_g_ms)
        self.net_g_ms.to(self.device)
        if self.model_type == ModelType.HUBERT_VITS:
            self.hubert = additional_model
        elif self.model_type == ModelType.W2V2_VITS:
            self.emotion_reference = additional_model

    def get_cleaned_text(self, text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            if self.bert_embedding:
                text_norm, char_embed = text_to_sequence(text, hps.symbols, hps.data.text_cleaners,
                                                         bert_embedding=self.bert_embedding)
                text_norm = LongTensor(text_norm)
                return text_norm, char_embed
            else:
                text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def get_cleaner(self):
        return getattr(self.hps_ms.data, 'text_cleaners', [None])[0]

    def get_speakers(self, escape=False):
        return self.speakers

    @property
    def sampling_rate(self):
        return self.hps_ms.data.sampling_rate

    def infer(self, params):
        with no_grad():
            x_tst = params.get("stn_tst").unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([params.get("stn_tst").size(0)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(params.get("char_embeds")).unsqueeze(0).to(
                self.device) if self.bert_embedding else None
            sid = params.get("sid").to(self.device)
            emotion = params.get("emotion").to(self.device) if self.emotion_embedding else None

            audio = self.net_g_ms.infer(x=x_tst,
                                        x_lengths=x_tst_lengths,
                                        sid=sid,
                                        noise_scale=params.get("noise_scale"),
                                        noise_scale_w=params.get("noise_scale_w"),
                                        length_scale=params.get("length_scale"),
                                        emotion_embedding=emotion,
                                        bert=x_tst_prosody)[0][0, 0].data.float().cpu().numpy()

        torch.cuda.empty_cache()

        return audio

    def get_infer_param(self, length_scale, noise_scale, noise_scale_w, text=None, speaker_id=None, audio_path=None,
                        emotion=None, cleaned=False, f0_scale=1):
        emo = None
        char_embeds = None
        if self.model_type != ModelType.HUBERT_VITS:
            if self.bert_embedding:
                stn_tst, char_embeds = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
            else:
                stn_tst = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
            sid = LongTensor([speaker_id])

        if self.model_type == ModelType.W2V2_VITS:
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


        elif self.model_type == ModelType.HUBERT_VITS:
            if self.use_f0:
                audio, sampling_rate = librosa.load(audio_path, sr=self.hps_ms.data.sampling_rate, mono=True)
                audio16000 = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            else:
                audio16000, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

            with inference_mode():
                units = self.hubert.units(FloatTensor(audio16000).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
                if self.use_f0:
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
                  "sid": sid, "emotion": emo, "char_embeds": char_embeds}

        return params

    def get_tasks(self, voice):
        text = voice.get("text", None)
        speaker_id = voice.get("id", 0)
        length = voice.get("length", 1)
        noise = voice.get("noise", 0.667)
        noisew = voice.get("noisew", 0.8)
        max = voice.get("max", 50)
        lang = voice.get("lang", "auto")
        speaker_lang = voice.get("speaker_lang", None)
        audio_path = voice.get("audio_path", None)
        emotion = voice.get("emotion", 0)

        # 去除所有多余的空白字符
        if text is not None: text = re.sub(r'\s+', ' ', text).strip()

        tasks = []
        if self.model_type == ModelType.VITS:
            sentence_list = sentence_split_and_markup(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                params = self.get_infer_param(text=sentence, speaker_id=speaker_id, length_scale=length,
                                              noise_scale=noise, noise_scale_w=noisew)
                tasks.append(params)

        elif self.model_type == ModelType.HUBERT_VITS:
            params = self.get_infer_param(speaker_id=speaker_id, length_scale=length, noise_scale=noise,
                                          noise_scale_w=noisew, audio_path=audio_path)
            tasks.append(params)

        elif self.model_type == ModelType.W2V2_VITS:
            sentence_list = sentence_split_and_markup(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                params = self.get_infer_param(text=sentence, speaker_id=speaker_id, length_scale=length,
                                              noise_scale=noise, noise_scale_w=noisew, emotion=emotion)
                tasks.append(params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return tasks

    def get_audio(self, voice, auto_break=False):
        tasks = self.get_tasks(voice)
        # 停顿0.75s，避免语音分段合成再拼接后的连接突兀
        brk = np.zeros(int(0.75 * self.sampling_rate), dtype=np.int16)

        audios = []
        num_tasks = len(tasks)

        for i, task in enumerate(tasks):
            if auto_break and i < num_tasks - 1:
                chunk = np.concatenate((self.infer(task), brk), axis=0)
            else:
                chunk = self.infer(task)
            audios.append(chunk)

        audio = np.concatenate(audios, axis=0)
        return audio

    def get_stream_audio(self, voice, auto_break=False):
        tasks = self.get_tasks(voice)

        brk = np.zeros(int(0.75 * 22050), dtype=np.int16)

        for task in tasks:
            if auto_break:
                chunk = np.concatenate((self.infer(task), brk), axis=0)
            else:
                chunk = self.infer(task)

            yield chunk

    def voice_conversion(self, voice):
        audio_path = voice.get("audio_path")
        original_id = voice.get("original_id")
        target_id = voice.get("target_id")

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
            audio = self.net_g_ms.voice_conversion(spec.to(self.device),
                                                   spec_lengths.to(self.device),
                                                   sid_src=sid_src.to(self.device),
                                                   sid_tgt=sid_tgt.to(self.device))[0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()

        return audio
