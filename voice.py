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
from logger import logger
from contants import ModelType
from scipy.signal import resample_poly


# torch.set_num_threads(1) # 设置torch线程为1


class TTS:
    def __init__(self, voice_obj, voice_speakers, **kwargs):
        self._voice_obj = voice_obj
        self._voice_speakers = voice_speakers
        self._strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}
        self._speakers_count = sum([len(self._voice_speakers[i]) for i in self._voice_speakers])
        self._vits_speakers_count = len(self._voice_speakers[ModelType.VITS.value])
        self._hubert_speakers_count = len(self._voice_speakers[ModelType.HUBERT_VITS.value])
        self._w2v2_speakers_count = len(self._voice_speakers[ModelType.W2V2_VITS.value])
        self._w2v2_emotion_count = kwargs.get("w2v2_emotion_count", 0)
        self._bert_vits2_speakers_count = len(self._voice_speakers[ModelType.BERT_VITS2.value])
        self.dem = None

        # Initialization information
        self.logger = logger
        self.logger.info(f"torch:{torch.__version__} cuda_available:{torch.cuda.is_available()}")
        self.logger.info(f'device:{kwargs.get("device")} device.type:{kwargs.get("device").type}')

        if getattr(config, "DIMENSIONAL_EMOTION_MODEL", None) != None:
            try:
                import audonnx
                root = os.path.dirname(config.DIMENSIONAL_EMOTION_MODEL)
                model_file = config.DIMENSIONAL_EMOTION_MODEL
                self.dem = audonnx.load(root=root, model_file=model_file)
            except Exception as e:
                self.logger.warning(f"Load DIMENSIONAL_EMOTION_MODEL failed {e}")

        if self._vits_speakers_count != 0: self.logger.info(f"[{ModelType.VITS.value}] {self._vits_speakers_count} speakers")
        if self._hubert_speakers_count != 0: self.logger.info(f"[{ModelType.HUBERT_VITS.value}] {self._hubert_speakers_count} speakers")
        if self._w2v2_speakers_count != 0: self.logger.info(f"[{ModelType.W2V2_VITS.value}] {self._w2v2_speakers_count} speakers")
        if self._bert_vits2_speakers_count != 0: self.logger.info(
            f"[{ModelType.BERT_VITS2.value}] {self._bert_vits2_speakers_count} speakers")
        self.logger.info(f"{self._speakers_count} speakers in total.")
        if self._speakers_count == 0:
            self.logger.warning(f"No model was loaded.")

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
    def w2v2_emotion_count(self):
        return self._w2v2_emotion_count

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

    def resample_audio(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio

        gcd = np.gcd(orig_sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, orig_sr // gcd)

        return audio

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
                model_type = element.attrib.get("model_type", root.attrib.get("model_type", "vits"))
                # w2v2-vits/emotion-vits才有emotion
                emotion = int(element.attrib.get("emotion", root.attrib.get("emotion", 0)))
                # Bert-VITS2的参数
                sdp_ratio = int(element.attrib.get("sdp_ratio", root.attrib.get("sdp_ratio", config.SDP_RATIO)))

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
                                            "model_type": model_type,
                                            "emotion": emotion,
                                            "sdp_ratio": sdp_ratio
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

    def process_ssml_infer_task(self, tasks, format):
        audios = []
        sampling_rates = []
        last_sampling_rate = 22050
        for task in tasks:
            if task.get("break"):
                audios.append(np.zeros(int(task.get("break") * 22050), dtype=np.int16))
                sampling_rates.append(last_sampling_rate)
            else:
                model_type_str = task.get("model_type").upper()
                if model_type_str not in [ModelType.VITS.value, ModelType.W2V2_VITS.value, ModelType.BERT_VITS2.value]:
                    raise ValueError(f"Unsupported model type: {task.get('model_type')}")
                model_type = ModelType(model_type_str)
                voice_obj = self._voice_obj[model_type][task.get("id")][1]
                real_id = self._voice_obj[model_type][task.get("id")][0]
                task["id"] = real_id
                sampling_rates.append(voice_obj.sampling_rate)
                last_sampling_rate = voice_obj.sampling_rate
                audio = voice_obj.get_audio(task)
                audios.append(audio)
        # 得到最高的采样率
        target_sr = max(sampling_rates)
        # 所有音频要与最高采样率保持一致
        resampled_audios = [self.resample_audio(audio, sr, target_sr) for audio, sr in zip(audios, sampling_rates)]
        audio = np.concatenate(resampled_audios, axis=0)
        encoded_audio = self.encode(target_sr, audio, format)
        return encoded_audio

    def vits_infer(self, task):
        format = task.get("format", "wav")
        voice_obj = self._voice_obj[ModelType.VITS][task.get("id")][1]
        real_id = self._voice_obj[ModelType.VITS][task.get("id")][0]
        task["id"] = real_id  # Change to real id
        sampling_rate = voice_obj.sampling_rate
        audio = voice_obj.get_audio(task, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        return encoded_audio

    def stream_vits_infer(self, task, fname=None):
        format = task.get("format", "wav")
        voice_obj = self._voice_obj[ModelType.VITS][task.get("id")][1]
        task["id"] = self._voice_obj[ModelType.VITS][task.get("id")][0]
        sampling_rate = voice_obj.sampling_rate
        genertator = voice_obj.get_stream_audio(task, auto_break=True)
        # audio = BytesIO()
        for chunk in genertator:
            encoded_audio = self.encode(sampling_rate, chunk, format)
            for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                yield encoded_audio_chunk
        #     if getattr(config, "SAVE_AUDIO", False):
        #         audio.write(encoded_audio.getvalue())
        # if getattr(config, "SAVE_AUDIO", False):
        #     path = f"{config.CACHE_PATH}/{fname}"
        #     utils.save_audio(audio.getvalue(), path)

    def hubert_vits_infer(self, task):
        format = task.get("format", "wav")
        voice_obj = self._voice_obj[ModelType.HUBERT_VITS][task.get("id")][1]
        task["id"] = self._voice_obj[ModelType.HUBERT_VITS][task.get("id")][0]
        sampling_rate = voice_obj.sampling_rate
        audio = voice_obj.get_audio(task)
        encoded_audio = self.encode(sampling_rate, audio, format)
        return encoded_audio

    def w2v2_vits_infer(self, task):
        format = task.get("format", "wav")
        voice_obj = self._voice_obj[ModelType.W2V2_VITS][task.get("id")][1]
        task["id"] = self._voice_obj[ModelType.W2V2_VITS][task.get("id")][0]
        sampling_rate = voice_obj.sampling_rate
        audio = voice_obj.get_audio(task, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        return encoded_audio

    def vits_voice_conversion(self, task):
        original_id = task.get("original_id")
        target_id = task.get("target_id")
        format = task.get("format")

        original_id_obj = int(self._voice_obj[ModelType.VITS][original_id][2])
        target_id_obj = int(self._voice_obj[ModelType.VITS][target_id][2])

        if original_id_obj != target_id_obj:
            raise ValueError(f"speakers are in diffrent VITS Model")

        task["original_id"] = int(self._voice_obj[ModelType.VITS][original_id][0])
        task["target_id"] = int(self._voice_obj[ModelType.VITS][target_id][0])

        voice_obj = self._voice_obj[ModelType.VITS][original_id][1]
        sampling_rate = voice_obj.sampling_rate

        audio = voice_obj.voice_conversion(task)
        encoded_audio = self.encode(sampling_rate, audio, format)
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

    def bert_vits2_infer(self, task):
        format = task.get("format", "wav")
        voice_obj = self._voice_obj[ModelType.BERT_VITS2][task.get("id")][1]
        task["id"] = self._voice_obj[ModelType.BERT_VITS2][task.get("id")][0]
        sampling_rate = voice_obj.sampling_rate
        audio = voice_obj.get_audio(task, auto_break=True)
        encoded_audio = self.encode(sampling_rate, audio, format)
        return encoded_audio
