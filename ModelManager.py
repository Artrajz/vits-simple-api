import gc
import glob
import logging
import os
import traceback

import cpuinfo
import numpy as np
import psutil
import torch

from utils.config_manager import global_config as config
import utils
from bert_vits2 import Bert_VITS2
from contants import ModelType
from logger import logger
from observer import Subject
from utils.data_utils import HParams
from vits import VITS
from vits.hubert_vits import HuBert_VITS
from vits.w2v2_vits import W2V2_VITS

CHINESE_ROBERTA_WWM_EXT_LARGE = os.path.join(config.ABS_PATH, "bert_vits2/bert/chinese-roberta-wwm-ext-large")
BERT_BASE_JAPANESE_V3 = os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-base-japanese-v3")
BERT_LARGE_JAPANESE_V2 = os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-large-japanese-v2")
DEBERTA_V2_LARGE_JAPANESE = os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese")
DEBERTA_V3_LARGE = os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v3-large")


class ModelManager(Subject):
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.logger = logger

        self.models = {  # "model_id":([model_path, config_path], model, n_speakers)
            ModelType.VITS: {},
            ModelType.HUBERT_VITS: {},
            ModelType.W2V2_VITS: {},
            ModelType.BERT_VITS2: {}
        }
        self.sid2model = {  # [real_id, model, model_id]
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
        self.bert_handler = None

        # self.sid2model = []
        # self.name_mapping_id = []

        self.voice_objs_count = 0

        self._observers = []

        self.model_class_map = {
            ModelType.VITS: VITS,
            ModelType.HUBERT_VITS: HuBert_VITS,
            ModelType.W2V2_VITS: W2V2_VITS,
            ModelType.BERT_VITS2: Bert_VITS2
        }

    def model_init(self, model_list):
        if model_list is None: model_list = []
        for model_path, config_path in model_list:
            self.load_model(model_path, config_path)

        if config.model_config.get("dimensional_emotion_model", None) is not None:
            if self.dimensional_emotion_model is None:
                self.dimensional_emotion_model = self.load_dimensional_emotion_model(
                    config.model_list["dimensional_emotion_model"])

        self.log_device_info()

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

    def log_device_info(self):
        cuda_available = torch.cuda.is_available()
        self.logger.info(
            f"PyTorch Version: {torch.__version__} Cuda available:{cuda_available} Device type:{self.device.type}")
        if self.device.type == 'cuda':
            if cuda_available:
                device_name = torch.cuda.get_device_name(self.device.index)
                self.logger.info(f"Using GPU on {device_name}, GPU Device Index: {self.device.index}")
            else:
                self.logger.warning("GPU device specified, but CUDA is not available.")
        else:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info['brand_raw']
            cpu_count = psutil.cpu_count(logical=False)
            thread_count = psutil.cpu_count(logical=True)
            self.logger.info(f"Using CPU on {cpu_name} with {cpu_count} cores and {thread_count} threads.")

    def _load_model_from_path(self, model_path, config_path):
        hps = utils.get_hparams_from_file(config_path)
        model_type = self.recognition_model_type(hps)

        model_args = {
            "model_path": model_path,
            "config_path": config_path,
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
                self.emotion_reference = self.load_npy(config["model_config"]["dimensional_emotion_model"])
            model_args.update({"emotion_reference": self.emotion_reference,
                               "dimensional_emotion_model": self.dimensional_emotion_model})

        if model_type == ModelType.HUBERT_VITS:
            if self.hubert is None:
                self.hubert = self.load_hubert_model(config["model_config"]["hubert_soft_model"])
            model_args.update({"hubert": self.hubert})

        model = model_class(**model_args)

        if model_type == ModelType.BERT_VITS2:
            bert_model_names = model.bert_model_names
            for bert_model_name in bert_model_names.values():
                if self.bert_handler is None:
                    from bert_vits2.text.bert_handler import BertHandler
                    self.bert_handler = BertHandler(self.device)
                self.bert_handler.load_bert(bert_model_name)
            model.load_model(self.bert_handler)

        sid2model = []
        speakers = []
        new_id = len(self.voice_speakers[model_type.value])
        model_id = max([-1] + list(self.models[model_type].keys())) + 1

        for real_id, name in enumerate(model.speakers):
            sid2model.append({"real_id": real_id, "model": model, "model_id": model_id})
            speakers.append({"id": new_id, "name": name, "lang": model.lang})
            new_id += 1

        model_data = {
            "model": model,
            "model_type": model_type,
            "model_id": model_id,
            "model_path": model_path,
            "config": hps,
            "sid2model": sid2model,
            "speakers": speakers
        }

        logging.info(
            f"model_type:{model_type.value} model_id:{model_id} n_speakers:{len(speakers)} model_path:{model_path}")

        return model_data

    def load_model(self, model_path: str, config_path: str):
        try:
            folder_path = os.path.join(config.ABS_PATH, 'Model')
            model_path = model_path if os.path.isabs(model_path) else os.path.join(folder_path, model_path)
            config_path = config_path if os.path.isabs(config_path) else os.path.join(folder_path, config_path)

            model_data = self._load_model_from_path(model_path, config_path)
            model_id = model_data["model_id"]
            sid2model = model_data["sid2model"]
            model_type = model_data["model_type"]

            self.models[model_type][model_id] = (
                [model_path, config_path], model_data["model"], len(model_data["speakers"]))
            self.sid2model[model_type].extend(sid2model)
            self.voice_speakers[model_type.value].extend(model_data["speakers"])

            self.notify("model_loaded", model_manager=self)
            state = True
        except Exception as e:
            self.logger.info(f"Loading failed. {e}")
            self.logger.error(traceback.format_exc())
            state = False
        return state

    def unload_model(self, model_type_value: str, model_id: str):
        state = False
        model_type = ModelType(model_type_value)
        model_id = int(model_id)
        try:
            if model_id in self.models[model_type].keys():
                model_path, model, n_speakers = self.models[model_type][model_id]
                start = 0
                for key, (_, _, ns) in self.models[model_type].items():
                    if key == model_id:
                        break
                    start += ns

                if model_type == ModelType.BERT_VITS2:
                    for bert_model_name in self.models[model_type][model_id][1].bert_model_names.values():
                        self.bert_handler.release_bert(bert_model_name)

                del self.sid2model[model_type][start:start + n_speakers]
                del self.voice_speakers[model_type.value][start:start + n_speakers]
                del self.models[model_type][model_id]

                for new_id, speaker in enumerate(self.voice_speakers[model_type.value]):
                    speaker["id"] = new_id

                gc.collect()
                torch.cuda.empty_cache()

                state = True
                self.notify("model_unloaded", model_manager=self)
                self.logger.info(f"Unloading success.")
        except Exception as e:
            self.logger.info(f"Unloading failed. {e}")
            state = False

        return state

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
        from vits.text.vits_pinyin import VITS_PinYin
        if self.tts_front is None:
            self.tts_front = VITS_PinYin(bert_path, self.device)

    def reorder_model(self, old_index, new_index):
        """重新排序模型，将old_index位置的模型移动到new_index位置"""
        if 0 <= old_index < len(self.models) and 0 <= new_index < len(self.models):
            model = self.models[old_index]
            del self.models[old_index]
            self.models.insert(new_index, model)

    def get_models_path(self):
        """按返回模型路径列表，列表每一项为[model_path, config_path]"""
        info = []

        for models in self.models.values():
            for values in models.values():
                info.append(values[0])

        return info

    def get_models_path_by_type(self):
        """按模型类型返回模型路径"""
        info = {
            ModelType.VITS.value: [],
            ModelType.HUBERT_VITS.value: [],
            ModelType.W2V2_VITS.value: [],
            ModelType.BERT_VITS2.value: []
        }
        for model_type, models in self.models.items():
            for values in models.values():
                info[model_type].append(values[0])

        return info

    def get_models_info(self):
        """按模型类型返回模型文件夹名以及模型文件名，speakers数量"""
        info = {
            ModelType.VITS.value: [],
            ModelType.HUBERT_VITS.value: [],
            ModelType.W2V2_VITS.value: [],
            ModelType.BERT_VITS2.value: []
        }
        for model_type, model_data in self.models.items():
            for model_id, (path, _, n_speakers) in model_data.items():
                info[model_type.value].append(
                    {"model_id": model_id,
                     "model_path": os.path.basename(os.path.dirname(path[0])) + "/" + os.path.basename(path[0]),
                     "config_path": os.path.basename(os.path.dirname(path[1])) + "/" + os.path.basename(path[1]),
                     "n_speakers": n_speakers})

        return info

    def get_model_by_index(self, model_type, model_id):
        """根据给定的索引返回模型"""
        if 0 <= model_id < len(self.models):
            _, model, _ = self.models[model_type][model_id]
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

    def scan_path(self):
        folder_path = os.path.join(config.ABS_PATH, 'Model')
        pth_files = glob.glob(folder_path + "/**/*.pth", recursive=True)
        all_paths = []
        unload_paths = []

        loaded_paths = []
        for path in self.get_models_path():
            # 只取已加载的模型路径
            loaded_paths.append(path[0])

        for id, pth_file in enumerate(pth_files):
            dir_name = os.path.dirname(pth_file)
            json_file = glob.glob(dir_name + "/*.json", recursive=True)[0]
            relative_pth_path = os.path.relpath(pth_file, folder_path)
            relative_pth_path = f"{os.path.dirname(relative_pth_path)}/{os.path.basename(relative_pth_path)}"
            relative_json_path = os.path.relpath(json_file, folder_path)
            relative_json_path = f"{os.path.dirname(relative_json_path)}/{os.path.basename(relative_json_path)}"
            info = {
                'model_id': id,
                'model_path': relative_pth_path,
                'config_path': relative_json_path
            }
            all_paths.append(info)

            if not self.is_path_loaded(pth_file, loaded_paths):
                unload_paths.append(info)

        return unload_paths

    def is_path_loaded(self, path, loaded_paths):
        normalized_path = os.path.normpath(path)

        for loaded_path in loaded_paths:
            normalized_loaded_path = os.path.normpath(loaded_path)
            if normalized_path == normalized_loaded_path:
                return True

        return False
