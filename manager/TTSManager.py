import logging
import os
import traceback

import librosa
import re
import numpy as np
import xml.etree.ElementTree as ET

from contants import config
import soundfile as sf
from io import BytesIO
from graiax import silkcoder
from contants import ModelType
from scipy.signal import resample_poly

from logger import logger
from manager.observer import Observer
from utils.data_utils import check_is_none
from utils.sentence import sentence_split_and_markup, split_languages, sentence_split, sentence_split_reading


class TTSManager(Observer):
    def __init__(self, model_manager, **kwargs):
        self.model_manager = model_manager
        self.strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}
        self.logger = logger
        self.infer_map = {
            ModelType.VITS: self.vits_infer,
            ModelType.W2V2_VITS: self.w2v2_vits_infer,
            ModelType.HUBERT_VITS: self.hubert_vits_infer,
            ModelType.BERT_VITS2: self.bert_vits2_infer_v2,
            ModelType.GPT_SOVITS: self.gpt_sovits_infer
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

    def normalize(self, state):
        int_keys = ["id", "segement_size", "emotion", "top_k"]
        float_keys = ["noise", "noisew", "length", "sdp_ratio", "style_weight", "top_p", "temperature"]

        for key in state:
            if key in int_keys and state[key] is not None:
                state[key] = int(state[key])
            elif key in float_keys and state[key] is not None:
                state[key] = float(state[key])

        return state

    def parse_ssml(self, ssml):
        root = ET.fromstring(ssml)
        format = root.attrib.get("format", "wav")
        voice_tasks = []
        brk_count = 0
        strength_dict = {"x-weak": 0.25, "weak": 0.5, "Medium": 0.75, "Strong": 1, "x-strong": 1.25}

        params = {ModelType.VITS.value: config.vits_config.asdict(),
                  ModelType.W2V2_VITS.value: config.w2v2_vits_config.asdict(),
                  ModelType.HUBERT_VITS.value: config.hubert_vits_config.asdict(),
                  ModelType.BERT_VITS2.value: config.bert_vits2_config.asdict(),
                  ModelType.GPT_SOVITS.value: config.gpt_sovits_config.asdict(),
                  }

        for element in root.iter():
            if element.tag == "voice":
                # 不填写则默认从已加载的模型中选择
                model_type = element.attrib.get("model_type", root.attrib.get("model_type", list(
                    self.model_manager.available_tts_model)[0]))
                if model_type is None:
                    raise ValueError(f"None model_type was specified")
                else:
                    model_type = model_type.upper()
                # logging.debug(f"Default model:{list(self.model_manager.available_tts_model)[0]}")
                # id = int(element.attrib.get("id", root.attrib.get("id", default_parameter.id)))
                # lang = element.attrib.get("lang", root.attrib.get("lang", default_parameter.lang))
                # length = float(
                #     element.attrib.get("length", root.attrib.get("length", default_parameter.length)))
                # noise = float(
                #     element.attrib.get("noise", root.attrib.get("noise", default_parameter.noise)))
                # noisew = float(
                #     element.attrib.get("noisew", root.attrib.get("noisew", default_parameter.noisew)))
                # segment_size = int(element.attrib.get("segment_size", root.attrib.get("segment_size",
                #                                                                       config["default_parameter"][
                #                                                                           "segment_size"])))

                # emotion = int(element.attrib.get("emotion", root.attrib.get("emotion", 0)))
                # Bert-VITS2的参数
                # sdp_ratio = int(element.attrib.get("sdp_ratio", root.attrib.get("sdp_ratio",
                #                                                                 config["default_parameter"][
                #                                                                     "sdp_ratio"])))

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
                        task = {
                            "model_type": model_type,
                            "speaker_lang": self.speaker_lang,
                            "text": match,
                        }
                        try:
                            task.update(params.get(model_type))  # 默认参数
                        except Exception as e:
                            raise ValueError(f"Invalid model_type:{model_type}")
                        task.update(root.attrib)  # 所有参数都放进去，推理函数会选出需要的参数
                        task = self.normalize(task)
                        voice_tasks.append(task)

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

        # for i in voice_tasks:
        #     self.logger.debug(i)

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
                if model_type_str not in [ModelType.VITS.value, ModelType.W2V2_VITS.value, ModelType.BERT_VITS2.value,
                                          ModelType.GPT_SOVITS.value]:
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
        if config.vits_config.dynamic_loading:
            model.load_model()
        state["id"] = self.get_real_id(ModelType.VITS, state["id"])  # Change to real id
        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate

        sentences_list = sentence_split_and_markup(state["text"], state["segment_size"], state["lang"],
                                                   model.lang)
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
        if config.vits_config.dynamic_loading:
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
                                                   model.lang)
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
                                                   model.lang)
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

    # def bert_vits2_infer(self, state, encode=True):
    #     model = self.get_model(model_type=ModelType.BERT_VITS2, id=state["id"])
    #     state["id"] = self.get_real_id(model_type=ModelType.BERT_VITS2, id=state["id"])
    #
    #     # 去除所有多余的空白字符
    #     if state["text"] is not None:
    #         state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
    #     sampling_rate = model.sampling_rate
    #
    #     # if state["lang"] == "auto":
    #     # state["lang"] = classify_language(state["text"], target_languages=model.lang)
    #     if state["lang"] == "auto":
    #         sentences_list = split_languages(state["text"], model.lang, expand_abbreviations=True,
    #                                          expand_hyphens=True)
    #     else:
    #         sentences_list = [(state["text"], state["lang"])]
    #     audios = []
    #
    #     for idx, (text, lang) in enumerate(sentences_list):
    #         skip_start = idx != 0
    #         skip_end = idx != len(sentences_list) - 1
    #         sentences = sentence_split(text, state["segment_size"])
    #         if lang == 'zh' and state["length_zh"] > 0:
    #             length = state["length_zh"]
    #         elif lang == 'ja' and state["length_ja"] > 0:
    #             length = state["length_ja"]
    #         elif lang == 'en' and state["length_en"] > 0:
    #             length = state["length_en"]
    #         else:
    #             length = state["length"]
    #         for _idx, sentence in enumerate(sentences):
    #             _skip_start = (_idx != 0) or (skip_start and _idx == 0)
    #             _skip_end = (_idx != len(sentences) - 1) or skip_end
    #             audio = model.infer(sentence, state["id"], lang, state["sdp_ratio"], state["noise"],
    #                                 state["noise"], length, emotion=state.get("emotion", None),
    #                                 reference_audio=state.get("reference_audio", None),
    #                                 text_prompt=state.get("text_prompt", None),
    #                                 style_text=state.get("style_text", None),
    #                                 style_weight=state.get("style_weight", 0.7))
    #             audios.extend(audio)
    #     audio = np.concatenate(audios)
    #
    #     return self.encode(sampling_rate, audio, state["format"]) if encode else audios

    # def stream_bert_vits2_infer(self, state, fname=None):
    #     model = self.get_model(ModelType.BERT_VITS2, state["id"])
    #     state["id"] = self.get_real_id(ModelType.BERT_VITS2, state["id"])
    #
    #     # 去除所有多余的空白字符
    #     if state["text"] is not None:
    #         state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
    #     sampling_rate = model.sampling_rate
    #
    #     sentences_list = split_languages(state["text"], model.lang, expand_abbreviations=True,
    #                                      expand_hyphens=True)
    #
    #     # audios = []
    #
    #     for (text, lang) in sentences_list:
    #         sentences = sentence_split(text, state["segment_size"])
    #         if lang == 'zh' and state["length_zh"] > 0:
    #             length = state["length_zh"]
    #         elif lang == 'ja' and state["length_ja"] > 0:
    #             length = state["length_ja"]
    #         elif lang == 'en' and state["length_en"] > 0:
    #             length = state["length_en"]
    #         else:
    #             length = state["length"]
    #         for sentence in sentences:
    #             audio = model.infer(sentence, state["id"], lang, state["sdp_ratio"], state["noise"],
    #                                 state["noise"], emotion=state.get("emotion", None),
    #                                 reference_audio=state.get("reference_audio", None),
    #                                 text_prompt=state.get("text_prompt", None),
    #                                 style_text=state.get("style_text", None),
    #                                 style_weight=state.get("style_weight", 0.7))
    #             # audios.append(audio)
    #             # audio = np.concatenate(audios, axis=0)
    #             encoded_audio = self.encode(sampling_rate, audio, state["format"])
    #
    #             for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
    #                 yield encoded_audio_chunk

    def bert_vits2_infer_v2(self, state, encode=True):
        model = self.get_model(model_type=ModelType.BERT_VITS2, id=state["id"])
        state["id"] = self.get_real_id(model_type=ModelType.BERT_VITS2, id=state["id"])

        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate
        sentences_list = sentence_split(state["text"], state["segment_size"])
        audios = []
        for sentences in sentences_list:
            if model.zh_bert_extra:
                infer_func = model.infer
                state["lang"] = "zh"
            elif state["lang"].lower() == "auto":
                infer_func = model.infer_multilang
            else:
                infer_func = model.infer
            audio = infer_func(sentences, state["id"], state["lang"], state["sdp_ratio"], state["noise"],
                               state["noise"], state["length"], emotion=state.get("emotion", None),
                               reference_audio=state.get("reference_audio", None),
                               text_prompt=state.get("text_prompt", None),
                               style_text=state.get("style_text", None),
                               style_weight=state.get("style_weight", 0.7))
            audios.append(audio)
        audio = np.concatenate(audios)

        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def stream_bert_vits2_infer_v2(self, state, fname=None):
        model = self.get_model(ModelType.BERT_VITS2, state["id"])
        state["id"] = self.get_real_id(ModelType.BERT_VITS2, state["id"])

        # 去除所有多余的空白字符
        if state["text"] is not None:
            state["text"] = re.sub(r'\s+', ' ', state["text"]).strip()
        sampling_rate = model.sampling_rate
        sentences_list = sentence_split(state["text"], state["segment_size"])
        # audios = []
        for sentences in sentences_list:
            if state["lang"].lower() == "auto":
                infer_func = model.infer_multilang
            else:
                infer_func = model.infer
            audio = infer_func(sentences, state["id"], state["lang"], state["sdp_ratio"], state["noise"],
                               state["noise"], state["length"], emotion=state.get("emotion", None),
                               reference_audio=state.get("reference_audio", None),
                               text_prompt=state.get("text_prompt", None),
                               style_text=state.get("style_text", None),
                               style_weight=state.get("style_weight", 0.7))
            # audios.append(audio)
            # audio = np.concatenate(audios, axis=0)
            encoded_audio = self.encode(sampling_rate, audio, state["format"])

            for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                yield encoded_audio_chunk

    def gpt_sovits_infer(self, state, encode=True):
        model = self.get_model(ModelType.GPT_SOVITS, state["id"])

        # 检查参考音频
        if check_is_none(state.get("reference_audio")):  # 无参考音频
            # 未选择预设
            if check_is_none(state.get("preset")):
                presets = config.gpt_sovits_config.presets
                refer_preset = presets.get(next(iter(presets)))
            else:  # 已选择预设
                refer_preset = config.gpt_sovits_config.presets.get(state.get("preset"))
            refer_wav_path = refer_preset.refer_wav_path
            if check_is_none(refer_wav_path):
                raise ValueError(f"The refer_wav_path:{refer_wav_path} in preset:{state.get('preset')} is None!")
            refer_wav_path = os.path.join(config.abs_path, config.system.data_path, refer_wav_path)
            state["prompt_text"], state["prompt_lang"] = refer_preset.prompt_text, refer_preset.prompt_lang

            # 将reference_audio换成指定预设里的参考音频
            state["reference_audio"] = refer_wav_path

        # if check_is_none(state.get("prompt_text")):
        #     raise ValueError(f"Error prompt_text:{state.get('prompt_text')}")

        if check_is_none(state.get("prompt_lang")):
            presets = config.gpt_sovits_config.presets
            state["prompt_lang"] = presets.get(next(iter(presets)), "auto")

        state["reference_audio"], state["reference_audio_sr"] = librosa.load(state["reference_audio"], sr=None,
                                                                             dtype=np.float32)
        state["reference_audio"] = state["reference_audio"].flatten()

        if state.get("lang").lower() == "auto":
            infer_func = model.infer_multilang
        else:
            infer_func = model.infer
        sentences_list = sentence_split(state["text"], state["segment_size"])
        audios = []
        for sentence in sentences_list:
            audio = infer_func(text=sentence,
                               lang=state.get("lang"),
                               reference_audio=state.get("reference_audio"),
                               reference_audio_sr=state.get("reference_audio_sr"),
                               prompt_text=state.get("prompt_text"),
                               prompt_lang=state.get("prompt_lang"),
                               top_k=state.get("top_k"),
                               top_p=state.get("top_p"),
                               temperature=state.get("temperature"),
                               )
            audios.append(audio)
        sampling_rate = model.sampling_rate
        audio = np.concatenate(audios)

        return self.encode(sampling_rate, audio, state["format"]) if encode else audio

    def stream_gpt_sovits_infer(self, state, encode=True):
        model = self.get_model(ModelType.GPT_SOVITS, state["id"])

        if state.get("lang").lower() == "auto":
            infer_func = model.infer_multilang
        else:
            infer_func = model.infer
        sentences_list = sentence_split(state["text"], state["segment_size"])
        audios = []
        for sentence in sentences_list:
            audio = infer_func(text=sentence,
                               lang=state.get("lang"),
                               reference_audio=state.get("reference_audio"),
                               reference_audio_sr=state.get("reference_audio_sr"),
                               prompt_text=state.get("prompt_text"),
                               prompt_lang=state.get("prompt_lang"),
                               top_k=state.get("top_k"),
                               top_p=state.get("top_p"),
                               temperature=state.get("temperature"),
                               )
            sampling_rate = model.sampling_rate
            encoded_audio = self.encode(sampling_rate, audio, state["format"])

            for encoded_audio_chunk in self.generate_audio_chunks(encoded_audio):
                yield encoded_audio_chunk

    def reading(self, in_state, nr_state):
        in_model = self.get_model(in_state["model_type"], in_state["id"])
        nr_model = self.get_model(nr_state["model_type"], nr_state["id"])

        infer_func = {ModelType.VITS: self.vits_infer,
                      ModelType.W2V2_VITS: self.w2v2_vits_infer,
                      ModelType.BERT_VITS2: self.bert_vits2_infer_v2,
                      ModelType.GPT_SOVITS: self.gpt_sovits_infer
                      }

        sentences_list = sentence_split_reading(in_state["text"])
        audios = []
        sampling_rates = []
        for sentence, is_quote in sentences_list:
            try:
                if is_quote:
                    in_state["text"] = sentence
                    audio = infer_func[in_state["model_type"]](in_state, encode=False)
                    sampling_rates.append(in_model.sampling_rate)
                else:
                    nr_state["text"] = sentence
                    audio = infer_func[nr_state["model_type"]](nr_state, encode=False)
                    sampling_rates.append(nr_model.sampling_rate)
                audios.append(audio)
            except Exception as e:
                logging.error(traceback.print_exc())
                logging.error(e)

        # 得到最高的采样率
        target_sr = max(sampling_rates)
        # 所有音频要与最高采样率保持一致
        resampled_audios = [self.resample_audio(audio, sr, target_sr) for audio, sr in
                            zip(audios, sampling_rates)]
        audio = np.concatenate(resampled_audios, axis=0)
        encoded_audio = self.encode(target_sr, audio, in_state["format"])

        return encoded_audio
