import librosa
import re
import numpy as np
import xml.etree.ElementTree as ET
from utils.config_manager import global_config as config
import soundfile as sf
from io import BytesIO
from graiax import silkcoder
from contants import ModelType
from scipy.signal import resample_poly

from logger import logger
from observer import Observer
from utils.sentence import sentence_split_and_markup, split_by_language, sentence_split


class TTSManager(Observer):
    def __init__(self, model_manager, **kwargs):
        self.model_manager = model_manager
        self.strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}
        self.logger = logger
        self.infer_map = {
            ModelType.VITS: self.vits_infer,
            ModelType.W2V2_VITS: self.w2v2_vits_infer,
            ModelType.HUBERT_VITS: self.hubert_vits_infer,
            ModelType.BERT_VITS2: self.bert_vits2_infer,
        }
        self.speaker_lang = None
        if getattr(config, "LANGUAGE_AUTOMATIC_DETECT", []) != []:
            self.speaker_lang = getattr(config, "LANGUAGE_AUTOMATIC_DETECT")

    @property
    def sid2model(self):
        return self.model_manager.sid2model

    @property
    def voice_speakers(self):
        return self.model_manager.voice_speakers

    @property
    def dimensional_emotion_model(self):
        return self.model_manager.dimensional_emotion_model

    def update(self, message, **kwargs):
        if message == "model_loaded":
            self._handle_model_loaded(kwargs['model_manager'])
        elif message == "model_unloaded":
            self._handle_model_unloaded(kwargs['model_manager'])

    def _handle_model_loaded(self, model_manager):
        self.model_manager = model_manager

    def _handle_model_unloaded(self, model_manager):
        self.model_manager = model_manager

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

    def get_model(self, model_type, id):
        return self.sid2model[model_type][id]["model"]

    def get_real_id(self, model_type, id):
        return self.sid2model[model_type][id]["real_id"]

    def get_model_id(self, model_type, id):
        return self.sid2model[model_type][id]["model_id"]

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
                segment_size = int(element.attrib.get("segment_size", root.attrib.get("segment_size", "0")))
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
                                            "segment_size": segment_size,
                                            "model_type": model_type,
                                            "emotion": emotion,
                                            "sdp_ratio": sdp_ratio,
                                            "speaker_lang": self.speaker_lang
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
                audios.append(np.zeros(int(task.get("break") * last_sampling_rate), dtype=np.int16))
                sampling_rates.append(last_sampling_rate)
            else:
                model_type_str = task.get("model_type").upper()
                if model_type_str not in [ModelType.VITS.value, ModelType.W2V2_VITS.value, ModelType.BERT_VITS2.value]:
                    raise ValueError(f"Unsupported model type: {task.get('model_type')}")
                model_type = ModelType(model_type_str)
                model = self.get_model(model_type, task.get("id"))
                task["id"] = self.get_real_id(model_type, task.get("id"))
                sampling_rates.append(model.sampling_rate)
                last_sampling_rate = model.sampling_rate
                audio = self.infer_map[model_type](task, encode=False)
                audios.append(audio)
        # 得到最高的采样率
        target_sr = max(sampling_rates)
        # 所有音频要与最高采样率保持一致
        resampled_audios = [self.resample_audio(audio, sr, target_sr) for audio, sr in zip(audios, sampling_rates)]
        audio = np.concatenate(resampled_audios, axis=0)
        encoded_audio = self.encode(target_sr, audio, format)
        return encoded_audio

    def vits_infer(self, state, encode=True):
        model = self.get_model(ModelType.VITS, state["id"])
        if config["DYNAMIC_LOADING"]:
            model.load_model()
        state["id"] = self.get_real_id(ModelType.VITS, state["id"])  # Change to real id
        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate

        sentences_list = sentence_split_and_markup(state["text"], state["segment_size"], state["lang"],
                                                   state["speaker_lang"])
        # 停顿0.5s，避免语音分段合成再拼接后的连接突兀
        brk = np.zeros(int(0.5 * sampling_rate), dtype=np.int16)

        audios = []
        sentences_num = len(sentences_list)

        for i, sentence in enumerate(sentences_list):
            sentence_audio = model.infer(sentence, state["id"], state["noise"], state["noisew"], state["length"])
            audios.append(sentence_audio)
            if i < sentences_num - 1:
                audios.append(brk)

        audio = np.concatenate(audios, axis=0)
        if config["DYNAMIC_LOADING"]:
            model.release_model()
        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def stream_vits_infer(self, state, fname=None):
        model = self.get_model(ModelType.VITS, state["id"])
        state["id"] = self.get_real_id(ModelType.VITS, state["id"])

        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate

        sentences_list = sentence_split_and_markup(state["text"], state["segment_size"], state["lang"],
                                                   state["speaker_lang"])
        # 停顿0.5s，避免语音分段合成再拼接后的连接突兀
        brk = np.zeros(int(0.5 * sampling_rate), dtype=np.int16)

        audios = []
        sentences_num = len(sentences_list)

        for i, sentence in enumerate(sentences_list):
            sentence_audio = model.infer(sentence, state["id"], state["noise"], state["noisew"], state["length"])
            audios.append(sentence_audio)
            if i < sentences_num - 1:
                audios.append(brk)

            audio = np.concatenate(audios, axis=0)
            encoded_audio = self.encode(sampling_rate, audio, state["format"])

            for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                yield encoded_audio_chunk
        #     if getattr(config, "SAVE_AUDIO", False):
        #         audio.write(encoded_audio.getvalue())
        # if getattr(config, "SAVE_AUDIO", False):
        #     path = f"{config.CACHE_PATH}/{fname}"
        #     utils.save_audio(audio.getvalue(), path)

    def hubert_vits_infer(self, state, encode=True):
        model = self.get_model(ModelType.HUBERT_VITS, state["id"])
        state["id"] = self.get_real_id(ModelType.HUBERT_VITS, state["id"])
        sampling_rate = model.sampling_rate
        audio = model.infer(state["audio_path"], state["id"], state["noise"], state["noisew"], state["length"],
                            f0_scale=1)
        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def w2v2_vits_infer(self, state, encode=True):
        model = self.get_model(ModelType.W2V2_VITS, state["id"])
        state["id"] = self.get_real_id(ModelType.W2V2_VITS, state["id"])
        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        emotion = state["emotion_reference"] if state["emotion_reference"] is not None else state["emotion"]

        sampling_rate = model.sampling_rate

        sentences_list = sentence_split_and_markup(state["text"], state["segment_size"], state["lang"],
                                                   state["speaker_lang"])
        # 停顿0.5s，避免语音分段合成再拼接后的连接突兀
        brk = np.zeros(int(0.5 * sampling_rate), dtype=np.int16)

        audios = []
        sentences_num = len(sentences_list)

        for i, sentence in enumerate(sentences_list):
            sentence_audio = model.infer(sentence, state["id"], state["noise"], state["noisew"], state["length"],
                                         emotion)
            audios.append(sentence_audio)
            if i < sentences_num - 1:
                audios.append(brk)

        audio = np.concatenate(audios, axis=0)
        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def vits_voice_conversion(self, state, encode=True):
        original_model_id = int(self.get_model_id(ModelType.VITS, state["original_id"]))
        target_model_id = int(self.get_model_id(ModelType.VITS, state["target_id"]))

        if original_model_id != target_model_id:
            raise ValueError(f"speakers are in diffrent VITS Model")

        model = self.get_model(ModelType.VITS, state["original_id"])
        state["original_id"] = int(self.get_real_id(ModelType.VITS, state["original_id"]))
        state["target_id"] = int(self.get_real_id(ModelType.VITS, state["target_id"]))

        sampling_rate = model.sampling_rate

        audio = model.voice_conversion(state["audio_path"], state["original_id"], state["target_id"])
        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def get_dimensional_emotion_npy(self, audio):
        audio16000, sampling_rate = librosa.load(audio, sr=16000, mono=True)
        emotion = self.dimensional_emotion_model(audio16000, sampling_rate)['hidden_states']
        emotion_npy = BytesIO()
        np.save(emotion_npy, emotion.squeeze(0))
        emotion_npy.seek(0)

        return emotion_npy

    def bert_vits2_infer(self, state, encode=True):
        model = self.get_model(model_type=ModelType.BERT_VITS2, id=state["id"])
        state["id"] = self.get_real_id(model_type=ModelType.BERT_VITS2, id=state["id"])

        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate

        # if state["lang"] == "auto":
        # state["lang"] = classify_language(state["text"], target_languages=model.lang)
        if state["lang"] == "auto":
            sentences_list = split_by_language(state["text"], state["speaker_lang"])
        else:
            sentences_list = [(state["text"], state["lang"])]
        audios = []

        for (text, lang) in sentences_list:
            sentences = sentence_split(text, state["segment_size"])
            if lang == 'zh' and state["length_zh"] > 0:
                length = state["length_zh"]
            elif lang == 'ja' and state["length_ja"] > 0:
                length = state["length_ja"]
            elif lang == 'en' and state["length_en"] > 0:
                length = state["length_en"]
            else:
                length = state["length"]
            for sentence in sentences:
                audio = model.infer(sentence, state["id"], lang, state["sdp_ratio"], state["noise"],
                                    state["noise"], length, emotion=state["emotion"],
                                    reference_audio=state["reference_audio"])
                audios.append(audio)
        audio = np.concatenate(audios)

        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def stream_bert_vits2_infer(self, state, fname=None):
        model = self.get_model(ModelType.BERT_VITS2, state["id"])
        state["id"] = self.get_real_id(ModelType.BERT_VITS2, state["id"])

        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate

        sentences_list = split_by_language(state["text"], state["speaker_lang"])

        # audios = []

        for (text, lang) in sentences_list:
            sentences = sentence_split(text, state["segment_size"])
            for sentence in sentences:
                audio = model.infer(sentence, state["id"], lang, state["sdp_ratio"], state["noise"],
                                    state["noise"], state["length"], emotion=state["emotion"],
                                    reference_audio=state["reference_audio"])
                # audios.append(audio)
                # audio = np.concatenate(audios, axis=0)
                encoded_audio = self.encode(sampling_rate, audio, state["format"])

                for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                    yield encoded_audio_chunk
