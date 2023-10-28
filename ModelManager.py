import logging
import os

import numpy as np
import torch

import config
import utils
from bert_vits2 import Bert_VITS2
from bert_vits2.text import BertHandler
from contants import ModelType
from logger import logger
from observer import Subject
from utils.data_utils import HParams
from config import DEVICE as device
from vits import VITS
from vits.hubert_vits import HuBert_VITS
from vits.text.vits_pinyin import VITS_PinYin
from vits.w2v2_vits import W2V2_VITS

CHINESE_ROBERTA_WWM_EXT_LARGE = os.path.join(config.ABS_PATH, "bert_vits2/bert/chinese-roberta-wwm-ext-large")
BERT_BASE_JAPANESE_V3 = os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-base-japanese-v3")
BERT_LARGE_JAPANESE_V2 = os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-large-japanese-v2")
DEBERTA_V2_LARGE_JAPANESE = os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese")
DEBERTA_V3_LARGE = os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v3-large")


class ModelManager(Subject):
    def __init__(self):
        self.device = device
        self.logger = logger

        self.model = []
        self.voice_objs = {
            ModelType.VITS: [],
            ModelType.HUBERT_VITS: [],
            ModelType.W2V2_VITS: [],
            ModelType.BERT_VITS2: []
        }
        self.voice_speakers = {
            ModelType.VITS.value: [],
            ModelType.HUBERT_VITS.value: [],
            ModelType.W2V2_VITS.value: [],
            ModelType.BERT_VITS2.value: []
        }

        self.emotion_reference = None
        self.hubert = None
        self.dimensional_emotion_model = None
        self.tts_front = None
        self.bert_models = {}
        self.bert_handler = BertHandler(self.device)

        self.id_mapping_obj = []
        self.name_mapping_id = []

        self.voice_objs_count = 0

        self._observers = []

        self.model_class_map = {
            ModelType.VITS: VITS,
            ModelType.HUBERT_VITS: HuBert_VITS,
            ModelType.W2V2_VITS: W2V2_VITS,
            ModelType.BERT_VITS2: Bert_VITS2
        }

    def model_init(self, model_list):
        for model_path, model_config in model_list:
            self.load_model(model_path, model_config)

        if getattr(config, "DIMENSIONAL_EMOTION_MODEL", None) is not None:
            if self.dimensional_emotion_model is None:
                self.dimensional_emotion_model = self.load_dimensional_emotion_model(config.DIMENSIONAL_EMOTION_MODEL)

        # Initialization information
        self.logger.info(f"torch:{torch.__version__} cuda_available:{torch.cuda.is_available()}")
        self.logger.info(f'device:{self.device} device.type:{self.device.type}')

        if self.vits_speakers_count != 0: self.logger.info(
            f"[{ModelType.VITS.value}] {self.vits_speakers_count} speakers")
        if self.hubert_speakers_count != 0: self.logger.info(
            f"[{ModelType.HUBERT_VITS.value}] {self.hubert_speakers_count} speakers")
        if self.w2v2_speakers_count != 0: self.logger.info(
            f"[{ModelType.W2V2_VITS.value}] {self.w2v2_speakers_count} speakers")
        if self.bert_vits2_speakers_count != 0: self.logger.info(
            f"[{ModelType.BERT_VITS2.value}] {self.bert_vits2_speakers_count} speakers")
        self.logger.info(f"{self.speakers_count} speakers in total.")
        if self.speakers_count == 0:
            self.logger.warning(f"No model was loaded.")

    @property
    def vits_speakers(self):
        return self.voice_speakers[ModelType.VITS]

    @property
    def speakers_count(self):
        return self.vits_speakers_count + self.hubert_speakers_count + self.w2v2_speakers_count + self.bert_vits2_speakers_count

    @property
    def vits_speakers_count(self):
        return len(self.voice_speakers[ModelType.VITS.value])

    @property
    def hubert_speakers_count(self):
        return len(self.voice_speakers[ModelType.HUBERT_VITS.value])

    @property
    def w2v2_speakers_count(self):
        return len(self.voice_speakers[ModelType.W2V2_VITS.value])

    @property
    def w2v2_emotion_count(self):
        return len(self.emotion_reference) if self.emotion_reference is not None else 0

    @property
    def bert_vits2_speakers_count(self):
        return len(self.voice_speakers[ModelType.BERT_VITS2.value])

    # 添加观察者
    def attach(self, observer):
        self._observers.append(observer)

    # 移除观察者
    def detach(self, observer):
        self._observers.remove(observer)

    # 通知所有观察者
    def notify(self, event_type, **kwargs):
        for observer in self._observers:
            observer.update(event_type, **kwargs)

    def _load_model_from_path(self, model_path, model_config):
        hps = utils.get_hparams_from_file(model_config)
        model_type = self.recognition_model_type(hps)

        obj_args = {
            "model_path": model_path,
            "config_path": model_config,
            "config": hps,
            "device": self.device
        }

        model_class = self.model_class_map[model_type]

        if model_type == ModelType.VITS:
            bert_embedding = getattr(hps.data, 'bert_embedding', getattr(hps.model, 'bert_embedding', False))
            if bert_embedding and self.tts_front is None:
                self.load_VITS_PinYin_model(os.path.join(config.ABS_PATH, "vits/bert"))

        if model_type == ModelType.W2V2_VITS:
            if self.emotion_reference is None:
                self.emotion_reference = self.load_npy(config.DIMENSIONAL_EMOTION_NPY)
            obj_args.update({"emotion_reference": self.emotion_reference,
                             "dimensional_emotion_model": self.dimensional_emotion_model})

        if model_type == ModelType.HUBERT_VITS:
            if self.hubert is None:
                self.hubert = self.load_hubert_model(config.HUBERT_SOFT_MODEL)
            obj_args.update({"hubert": self.hubert})

        obj = model_class(**obj_args)

        if model_type == ModelType.BERT_VITS2:
            bert_model_names = obj.bert_model_names
            for bert_model_name in bert_model_names.values():
                self.bert_handler.load_bert(bert_model_name)
            obj.load_model(self.bert_handler)

        id_mapping_obj = []
        speakers = []
        new_id = len(self.voice_speakers[model_type.value])
        obj_id = self.voice_objs_count
        for real_id, name in enumerate(obj.speakers):
            id_mapping_obj.append({"real_id": real_id, "obj": obj, "obj_id": obj_id})
            speakers.append({"id": new_id, "name": name, "lang": obj.lang})
            new_id += 1

        model_data = {
            "model": obj,
            "type": model_type,
            "config": hps,
            "id_mapping_obj": id_mapping_obj,
            "speakers": speakers
        }

        return model_data

    def load_model(self, model_path, model_config):
        model_data = self._load_model_from_path(model_path, model_config)
        id_mapping_obj = model_data["id_mapping_obj"]
        model_type = model_data["type"]

        self.voice_objs[model_type].extend(id_mapping_obj)
        self.voice_speakers[model_type.value].extend(model_data["speakers"])

        self.notify("model_loaded", model_manager=self)

    def unload_model(self, index):
        if 0 <= index < len(self.models):
            model = self.models[index][1]  # Assuming (path, model) tuple structure
            model_type = model.type

            self.voice_objs[model_type].remove(model)
            self.voice_speakers[model_type.value].remove(model.get_speaker_data())

            del self.models[index]
            self.notify("model_unloaded", model_manager=self)

    def load_dimensional_emotion_model(self, model_path):
        try:
            import audonnx
            root = os.path.dirname(model_path)
            model_file = model_path
            dimensional_emotion_model = audonnx.load(root=root, model_file=model_file)

            self.notify("model_loaded", model_manager=self)
        except Exception as e:
            self.logger.warning(f"Load DIMENSIONAL_EMOTION_MODEL failed {e}")

        return dimensional_emotion_model

    def unload_dimensional_emotion_model(self):
        self.dimensional_emotion_model = None
        self.notify("model_unloaded", model_manager=self)

    def load_hubert_model(self, model_path):
        """"HuBERT-VITS"""
        try:
            from vits.hubert_model import hubert_soft
            hubert = hubert_soft(model_path)
        except Exception as e:
            self.logger.warning(f"Load HUBERT_SOFT_MODEL failed {e}")
        return hubert

    def unload_hubert_model(self):
        self.hubert = None
        self.notify("model_unloaded", model_manager=self)

    # def load_bert_model(self, bert_model_name):
    #     """"Bert-VITS2"""
    #     if bert_model_name not in self.BERT_MODELS:
    #         raise ValueError(f"Unknown BERT model name: {bert_model_name}")
    #     model_path = self.BERT_MODELS[bert_model_name]
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
    #     return tokenizer, model

    def load_VITS_PinYin_model(self, bert_path):
        """"vits_chinese"""
        if self.tts_front is None:
            self.tts_front = VITS_PinYin(bert_path, self.device)

    def reorder_model(self, old_index, new_index):
        """重新排序模型，将old_index位置的模型移动到new_index位置"""
        if 0 <= old_index < len(self.models) and 0 <= new_index < len(self.models):
            model = self.models[old_index]
            del self.models[old_index]
            self.models.insert(new_index, model)

    def get_models(self):
        """返回所有模型的路径列表"""
        return [path for path, _ in self.models]

    def get_model_by_index(self, index):
        """根据给定的索引返回模型"""
        if 0 <= index < len(self.models):
            _, model = self.models[index]
            return model
        return None

    # def get_bert_model(self, bert_model_name):
    #     if bert_model_name not in self.bert_models:
    #         raise ValueError(f"Model {bert_model_name} not loaded!")
    #     return self.bert_models[bert_model_name]

    def clear_all(self):
        """清除所有模型"""
        self.models.clear()

    def recognition_model_type(self, hps: HParams) -> str:
        # model_config = json.load(model_config_json)
        symbols = getattr(hps, "symbols", None)
        # symbols = model_config.get("symbols", None)
        emotion_embedding = getattr(hps.data, "emotion_embedding", False)

        if "use_spk_conditioned_encoder" in hps.model:
            model_type = ModelType.BERT_VITS2
            return model_type

        if symbols != None:
            if not emotion_embedding:
                mode_type = ModelType.VITS
            else:
                mode_type = ModelType.W2V2_VITS
        else:
            mode_type = ModelType.HUBERT_VITS

        return mode_type

    def _load_npy_from_path(self, path):
        model_extention = os.path.splitext(path)[1]
        if model_extention != ".npy":
            raise ValueError(f"Unsupported model type: {model_extention}")
        return np.load(path).reshape(-1, 1024)

    def load_npy(self, emotion_reference_npy):
        emotion_reference = np.empty((0, 1024))

        if isinstance(emotion_reference_npy, list):
            for i in emotion_reference_npy:
                emotion_reference = np.append(emotion_reference, self._load_npy_from_path(i), axis=0)

        elif os.path.isdir(emotion_reference_npy):
            for root, dirs, files in os.walk(emotion_reference_npy):
                for file_name in files:
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(root, file_name)
                        emotion_reference = np.append(emotion_reference, self._load_npy_from_path(file_path), axis=0)

        elif os.path.isfile(emotion_reference_npy):
            emotion_reference = self._load_npy_from_path(emotion_reference_npy)

        logging.info(f"Loaded emotional dimention npy range: {len(emotion_reference)}")
        return emotion_reference
