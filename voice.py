import os
import librosa
import re
import numpy as np
import torch
import xml.etree.ElementTree as ET
import config
import soundfile as sf
from io import BytesIO
from graiax import silkcoder
from utils import utils
from logger import logger


# torch.set_num_threads(1) # 设置torch线程为1


class TTS:
    def __init__(self, voice_obj, voice_speakers, w2v2_emotion_count=0, device=torch.device("cpu")):
        self._voice_obj = voice_obj
        self._voice_speakers = voice_speakers
        self._strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}
        self._speakers_count = sum([len(self._voice_speakers[i]) for i in self._voice_speakers])
        self._vits_speakers_count = len(self._voice_speakers["VITS"])
        self._hubert_speakers_count = len(self._voice_speakers["HUBERT-VITS"])
        self._w2v2_speakers_count = len(self._voice_speakers["W2V2-VITS"])
        self._w2v2_emotion_count = w2v2_emotion_count
        self._bert_vits2_speakers_count = len(self._voice_speakers["BERT-VITS2"])
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

    @property
    def bert_vits2_speakers_count(self):
        return self._bert_vits2_speakers_count

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

    def bert_vits2_infer(self, voice, fname):
        format = voice.get("format", "wav")
        voice_obj = self._voice_obj["BERT-VITS2"][voice.get("id")][1]
        voice["id"] = self._voice_obj["BERT-VITS2"][voice.get("id")][0]
        sampling_rate = voice_obj.hps_ms.data.sampling_rate
        audio = voice_obj.get_audio(voice, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        if getattr(config, "SAVE_AUDIO", False):
            path = f"{config.CACHE_PATH}/{fname}"
            utils.save_audio(encoded_audio.getvalue(), path)
        return encoded_audio
