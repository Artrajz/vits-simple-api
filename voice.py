import os
import librosa
import commons
import re
import numpy as np
import torch
import xml.etree.ElementTree as ET
import config
import soundfile as sf
from torch import no_grad, LongTensor, inference_mode, FloatTensor
from io import BytesIO
from graiax import silkcoder
from utils.nlp import sentence_split
from mel_processing import spectrogram_torch
from text import text_to_sequence
from models import SynthesizerTrn
from utils import utils
from logger import logger

# torch.set_num_threads(1) # 设置torch线程为1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class vits:
    def __init__(self, model, config, model_=None, model_type=None):
        self.model_type = model_type
        self.hps_ms = utils.get_hparams_from_file(config)
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.n_symbols = len(getattr(self.hps_ms, 'symbols', []))
        self.speakers = getattr(self.hps_ms, 'speakers', ['0'])
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

        # load model
        self.load_model(model, model_)

    def load_model(self, model, model_=None):
        utils.load_checkpoint(model, self.net_g_ms)
        self.net_g_ms.to(device)
        if self.model_type == "hubert":
            self.hubert = model_
        elif self.model_type == "w2v2":
            self.emotion_reference = model_

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

    def infer(self, params):
        with no_grad():
            x_tst = params.get("stn_tst").unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([params.get("stn_tst").size(0)]).to(device)
            x_tst_prosody = torch.FloatTensor(params.get("char_embeds")).unsqueeze(0).to(
                device) if self.bert_embedding else None
            sid = params.get("sid").to(device) if not self.bert_embedding else None
            emotion = params.get("emotion").to(device) if self.emotion_embedding else None

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
        if self.model_type != "hubert":
            if self.bert_embedding:
                stn_tst, char_embeds = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
                sid = None
            else:
                stn_tst = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
                sid = LongTensor([speaker_id])

        if self.model_type == "w2v2":
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


        elif self.model_type == "hubert":
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
        if self.model_type == "vits":
            sentence_list = sentence_split(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                params = self.get_infer_param(text=sentence, speaker_id=speaker_id, length_scale=length,
                                              noise_scale=noise, noise_scale_w=noisew)
                tasks.append(params)

        elif self.model_type == "hubert":
            params = self.get_infer_param(speaker_id=speaker_id, length_scale=length, noise_scale=noise,
                                          noise_scale_w=noisew, audio_path=audio_path)
            tasks.append(params)

        elif self.model_type == "w2v2":
            sentence_list = sentence_split(text, max, lang, speaker_lang)
            for sentence in sentence_list:
                params = self.get_infer_param(text=sentence, speaker_id=speaker_id, length_scale=length,
                                              noise_scale=noise, noise_scale_w=noisew, emotion=emotion)
                tasks.append(params)

        return tasks

    def get_audio(self, voice, auto_break=False):
        tasks = self.get_tasks(voice)
        # 停顿0.75s，避免语音分段合成再拼接后的连接突兀
        brk = np.zeros(int(0.75 * 22050), dtype=np.int16)

        audios = []
        for task in tasks:
            if auto_break:
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
            audio = self.net_g_ms.voice_conversion(spec.to(device),
                                                   spec_lengths.to(device),
                                                   sid_src=sid_src.to(device),
                                                   sid_tgt=sid_tgt.to(device))[0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()

        return audio


class TTS:
    def __init__(self, voice_obj, voice_speakers):
        self._voice_obj = voice_obj
        self._voice_speakers = voice_speakers
        self._strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}
        self._speakers_count = sum([len(self._voice_speakers[i]) for i in self._voice_speakers])
        self._vits_speakers_count = len(self._voice_speakers["VITS"])
        self._hubert_speakers_count = len(self._voice_speakers["HUBERT-VITS"])
        self._w2v2_speakers_count = len(self._voice_speakers["W2V2-VITS"])
        self.dem = None

        # Initialization information
        self.logger = logger
        self.logger.info(f"torch:{torch.__version__} cuda_available:{torch.cuda.is_available()}")
        self.logger.info(f'device:{device} device.type:{device.type}')

        if getattr(config, "DIMENSIONAL_EMOTION_MODEL", None) != None:
            try:
                import audonnx
                root = os.path.dirname(config.DIMENSIONAL_EMOTION_MODEL)
                model_file = config.DIMENSIONAL_EMOTION_MODEL
                self.dem = audonnx.load(root=root, model_file=model_file)
            except Exception as e:
                self.logger.warning(f"Load DIMENSIONAL_EMOTION_MODEL failed {e}")

        if self._vits_speakers_count != 0: self.logger.info(f"[VITS] {self._vits_speakers_count} speakers")
        if self._hubert_speakers_count != 0: self.logger.info(f"[hubert] {self._hubert_speakers_count} speakers")
        if self._w2v2_speakers_count != 0: self.logger.info(f"[w2v2] {self._w2v2_speakers_count} speakers")
        self.logger.info(f"{self._speakers_count} speakers in total")
        if self._speakers_count == 0:
            self.logger.warning(f"No model was loaded")

    @property
    def voice_speakers(self):
        return self._voice_speakers

    @property
    def speakers_count(self):
        return self._speakers_count

    @property
    def vits_speakers_count(self):
        return self._vits_speakers_count

    @property
    def hubert_speakers_count(self):
        return self._hubert_speakers_count

    @property
    def w2v2_speakers_count(self):
        return self._w2v2_speakers_count

    def encode(self, sampling_rate, audio, format):
        with BytesIO() as f:
            if format.upper() == 'OGG':
                sf.write(f, audio, sampling_rate, format="ogg")
                return BytesIO(f.getvalue())
            elif format.upper() == 'SILK':
                sf.write(f, audio, sampling_rate, format="wav")
                return BytesIO(silkcoder.encode(f))
            elif format.upper() == 'MP3':
                sf.write(f, audio, sampling_rate, format="mp3")
                return BytesIO(f.getvalue())
            elif format.upper() == 'WAV':
                sf.write(f, audio, sampling_rate, format="wav")
                return BytesIO(f.getvalue())
            elif format.upper() == 'FLAC':
                sf.write(f, audio, sampling_rate, format="flac")
                return BytesIO(f.getvalue())
            else:
                raise ValueError(f"Unsupported format:{format}")

    def convert_time_string(self, time_string):
        time_value = float(re.findall(r'\d+\.?\d*', time_string)[0])
        time_unit = re.findall(r'[a-zA-Z]+', time_string)[0].lower()

        if time_unit.upper() == 'MS':
            return time_value / 1000
        elif time_unit.upper() == 'S':
            return time_value
        elif time_unit.upper() == 'MIN':
            return time_value * 60
        elif time_unit.upper() == 'H':
            return time_value * 3600
        elif time_unit.upper() == 'D':
            return time_value * 24 * 3600  # 不会有人真写D吧？
        else:
            raise ValueError("Unsupported time unit: {}".format(time_unit))

    def generate_audio_chunks(self, audio):
        chunk_size = 4096
        while True:
            chunk = audio.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def parse_ssml(self, ssml):
        root = ET.fromstring(ssml)
        format = root.attrib.get("format", "wav")
        voice_tasks = []
        brk_count = 0
        strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}

        for element in root.iter():
            if element.tag == "voice":
                id = int(element.attrib.get("id", root.attrib.get("id", config.ID)))
                lang = element.attrib.get("lang", root.attrib.get("lang", config.LANG))
                length = float(element.attrib.get("length", root.attrib.get("length", config.LENGTH)))
                noise = float(element.attrib.get("noise", root.attrib.get("noise", config.NOISE)))
                noisew = float(element.attrib.get("noisew", root.attrib.get("noisew", config.NOISEW)))
                max = int(element.attrib.get("max", root.attrib.get("max", "0")))
                # 不填写默认就是vits
                model = element.attrib.get("model", root.attrib.get("model", "vits"))
                # w2v2-vits/emotion-vits才有emotion
                emotion = int(element.attrib.get("emotion", root.attrib.get("emotion", 0)))

                voice_element = ET.tostring(element, encoding='unicode')

                pattern_voice = r'<voice.*?>(.*?)</voice>'
                pattern_break = r'<break\s*?(.*?)\s*?/>'

                matches_voice = re.findall(pattern_voice, voice_element)[0]
                matches_break = re.split(pattern_break, matches_voice)
                for match in matches_break:
                    strength = re.search(r'\s*strength\s*=\s*[\'\"](.*?)[\'\"]', match)
                    time = re.search(r'\s*time\s*=\s*[\'\"](.*?)[\'\"]', match)
                    # break标签 strength属性
                    if strength:
                        brk = strength_dict[strength.group(1)]
                        voice_tasks.append({"break": brk})
                        brk_count += 1
                    # break标签 time属性
                    elif time:
                        brk = self.convert_time_string(time.group(1))
                        voice_tasks.append({"break": brk})
                        brk_count += 1
                    # break标签 为空说明只写了break，默认停顿0.75s
                    elif match == "":
                        voice_tasks.append({"break": 0.75})
                        brk_count += 1
                    # voice标签中除了break剩下的就是文本
                    else:
                        voice_tasks.append({"id": id,
                                            "text": match,
                                            "lang": lang,
                                            "length": length,
                                            "noise": noise,
                                            "noisew": noisew,
                                            "max": max,
                                            "model": model,
                                            "emotion": emotion
                                            })

                # 分段末尾停顿0.75s
                voice_tasks.append({"break": 0.75})
            elif element.tag == "break":
                # brk_count大于0说明voice标签中有break
                if brk_count > 0:
                    brk_count -= 1
                    continue
                brk = strength_dict.get(element.attrib.get("strength"),
                                        self.convert_time_string(element.attrib.get("time", "750ms")))
                voice_tasks.append({"break": brk})

        for i in voice_tasks:
            self.logger.debug(i)

        return voice_tasks, format

    def create_ssml_infer_task(self, voice_tasks, format, fname):
        audios = []
        for voice in voice_tasks:
            if voice.get("break"):
                audios.append(np.zeros(int(voice.get("break") * 22050), dtype=np.int16))
            else:
                model = voice.get("model").upper()
                if model != "VITS" and model != "W2V2-VITS" and model != "EMOTION-VITS":
                    raise ValueError(f"Unsupported model: {voice.get('model')}")
                voice_obj = self._voice_obj[model][voice.get("id")][1]
                voice["id"] = self._voice_obj[model][voice.get("id")][0]
                audio = voice_obj.get_audio(voice)
                audios.append(audio)

        audio = np.concatenate(audios, axis=0)
        encoded_audio = self.encode(voice_obj.hps_ms.data.sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio

    def vits_infer(self, voice, fname):
        format = voice.get("format", "wav")
        voice_obj = self._voice_obj["VITS"][voice.get("id")][1]
        voice["id"] = self._voice_obj["VITS"][voice.get("id")][0]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate
        audio = voice_obj.get_audio(voice, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio

    def stream_vits_infer(self, voice, fname):
        format = voice.get("format", "wav")
        voice_obj = self._voice_obj["VITS"][voice.get("id")][1]
        voice["id"] = self._voice_obj["VITS"][voice.get("id")][0]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate
        genertator = voice_obj.get_stream_audio(voice, auto_break=True)
        audio = BytesIO()
        for chunk in genertator:
            encoded_audio = self.encode(sampling_rate, chunk, format)
            for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                yield encoded_audio_chunk
            if getattr(config, "SAVE_AUDIO", False):
                audio.write(encoded_audio.getvalue())
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(audio.getvalue(), path)

    def hubert_vits_infer(self, voice, fname):
        format = voice.get("format", "wav")
        voice_obj = self._voice_obj["HUBERT-VITS"][voice.get("id")][1]
        voice["id"] = self._voice_obj["HUBERT-VITS"][voice.get("id")][0]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate
        audio = voice_obj.get_audio(voice)
        encoded_audio = self.encode(sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio

    def w2v2_vits_infer(self, voice, fname):
        format = voice.get("format", "wav")
        voice_obj = self._voice_obj["W2V2-VITS"][voice.get("id")][1]
        voice["id"] = self._voice_obj["W2V2-VITS"][voice.get("id")][0]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate
        audio = voice_obj.get_audio(voice, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio

    def vits_voice_conversion(self, voice, fname):
        original_id = voice.get("original_id")
        target_id = voice.get("target_id")
        format = voice.get("format")

        original_id_obj = int(self._voice_obj["VITS"][original_id][2])
        target_id_obj = int(self._voice_obj["VITS"][target_id][2])

        if original_id_obj != target_id_obj:
            raise ValueError(f"speakers are in diffrent VITS Model")

        voice["original_id"] = int(self._voice_obj["VITS"][original_id][0])
        voice["target_id"] = int(self._voice_obj["VITS"][target_id][0])

        voice_obj = self._voice_obj["VITS"][original_id][1]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate

        audio = voice_obj.voice_conversion(voice)
        encoded_audio = self.encode(sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio

    def get_dimensional_emotion_npy(self, audio):
        if self.dem is None:
            raise ValueError(f"Please configure DIMENSIONAL_EMOTION_MODEL path in config.py")
        audio16000, sampling_rate = librosa.load(audio, sr=16000, mono=True)
        emotion = self.dem(audio16000, sampling_rate)['hidden_states']
        emotion_npy = BytesIO()
        np.save(emotion_npy, emotion.squeeze(0))
        emotion_npy.seek(0)

        return emotion_npy
