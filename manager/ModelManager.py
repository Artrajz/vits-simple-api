import gc
import glob
import logging
import os
import traceback

import cpuinfo
import numpy as np
import psutil
import torch

# from contants import config
from contants import config
import utils
from bert_vits2 import Bert_VITS2
from contants import ModelType
from gpt_sovits.gpt_sovits import GPT_SoVITS
from logger import logger
from manager.observer import Subject
from utils.data_utils import HParams, check_is_none
from vits import VITS
from vits.hubert_vits import HuBert_VITS
from vits.w2v2_vits import W2V2_VITS


class ModelManager(Subject):
    def __init__(self, device=config.system.device):
        self.device = device
        self.logger = logger

        self.models = {
            # "model_id": {"model_path": model_path, "config_path": config_path, "model": model, "n_speakers": n_speakers}
            ModelType.VITS: {},
            ModelType.HUBERT_VITS: {},
            ModelType.W2V2_VITS: {},
            ModelType.BERT_VITS2: {},
            ModelType.GPT_SOVITS: {},
        }
        self.sid2model = {  # [real_id, model, model_id]
            ModelType.VITS: [],
            ModelType.HUBERT_VITS: [],
            ModelType.W2V2_VITS: [],
            ModelType.BERT_VITS2: [],
            ModelType.GPT_SOVITS: [],
        }
        self.voice_speakers = {
            ModelType.VITS.value: [],
            ModelType.HUBERT_VITS.value: [],
            ModelType.W2V2_VITS.value: [],
            ModelType.BERT_VITS2.value: [],
            ModelType.GPT_SOVITS.value: [],
        }

        self.emotion_reference = None
        self.hubert = None
        self.dimensional_emotion_model = None
        self.tts_front = None
        self.bert_models = {}
        self.model_handler = None
        self.emotion_model = None
        self.processor = None

        # self.sid2model = []
        # self.name_mapping_id = []

        self.voice_objs_count = 0

        self._observers = []

        self.model_class_map = {
            ModelType.VITS: VITS,
            ModelType.HUBERT_VITS: HuBert_VITS,
            ModelType.W2V2_VITS: W2V2_VITS,
            ModelType.BERT_VITS2: Bert_VITS2,
            ModelType.GPT_SOVITS: GPT_SoVITS,
        }

        self.available_tts_model = set()

    def model_init(self):
        if config.tts_config.auto_load:
            models = self.scan_path()
        else:
            models = config.tts_config.asdict().get("models")

        for model in models:
            self.load_model(model_path=model.get("model_path"),
                            config_path=model.get("config_path"),
                            sovits_path=model.get("sovits_path"),
                            gpt_path=model.get("gpt_path"))

        dimensional_emotion_model_path = os.path.join(config.abs_path, config.system.data_path,
                                                      config.model_config.dimensional_emotion_model)
        if os.path.isfile(dimensional_emotion_model_path):
            if self.dimensional_emotion_model is None:
                self.dimensional_emotion_model = self.load_dimensional_emotion_model(dimensional_emotion_model_path)

        self.log_device_info()

        if self.vits_speakers_count != 0:
            self.logger.info(f"[{ModelType.VITS.value}] {self.vits_speakers_count} speakers")
        if self.hubert_speakers_count != 0:
            self.logger.info(f"[{ModelType.HUBERT_VITS.value}] {self.hubert_speakers_count} speakers")
        if self.w2v2_speakers_count != 0:
            self.logger.info(f"[{ModelType.W2V2_VITS.value}] {self.w2v2_speakers_count} speakers")
        if self.bert_vits2_speakers_count != 0:
            self.logger.info(f"[{ModelType.BERT_VITS2.value}] {self.bert_vits2_speakers_count} speakers")
        if self.gpt_sovits_speakers_count != 0:
            self.logger.info(f"[{ModelType.GPT_SOVITS.value}] {self.gpt_sovits_speakers_count} speakers")
        self.logger.info(f"{self.speakers_count} speakers in total.")
        if self.speakers_count == 0:
            self.logger.warning(f"No model was loaded.")

    @property
    def vits_speakers(self):
        return self.voice_speakers[ModelType.VITS]

    @property
    def speakers_count(self):
        return self.vits_speakers_count + self.hubert_speakers_count + self.w2v2_speakers_count + self.bert_vits2_speakers_count + self.gpt_sovits_speakers_count

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

    @property
    def gpt_sovits_speakers_count(self):
        return len(self.voice_speakers[ModelType.GPT_SOVITS.value])

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
                gpu_memory_info = round(torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3)  # GB
                self.logger.info(
                    f"Using GPU on {device_name} {gpu_memory_info}GB, GPU Device Index: {self.device.index}")
            else:
                self.logger.warning("GPU device specified, but CUDA is not available.")
        else:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info.get("brand_raw")
            cpu_count = psutil.cpu_count(logical=False)
            thread_count = psutil.cpu_count(logical=True)
            memory_info = psutil.virtual_memory()
            total_memory = round(memory_info.total / (1024 ** 3))
            self.logger.info(
                f"Using CPU on {cpu_name} with {cpu_count} cores and {thread_count} threads. Total memory: {total_memory}GB")

    def relative_to_absolute_path(self, *paths):
        absolute_paths = []

        for path in paths:
            if path is None:
                return None
            path = os.path.normpath(path)
            if path.startswith('models'):
                path = os.path.join(config.abs_path, config.system.data_path, path)
            else:
                path = os.path.join(config.abs_path, config.system.data_path, config.tts_config.models_path,
                                    path)
            absolute_paths.append(path)

        return absolute_paths

    def absolute_to_relative_path(self, *paths):
        relative_paths = []
        for path in paths:
            if path is None:
                relative_paths.append(None)
                continue

            # 获取models目录下的相对路径
            relative_path = os.path.relpath(path, os.path.join(config.abs_path, config.system.data_path,
                                                               config.tts_config.models_path))

            relative_paths.append(relative_path)

        return relative_paths

    def _load_model_from_path(self, model_path, config_path, sovits_path, gpt_path):
        if check_is_none(sovits_path, gpt_path):
            hps = utils.get_hparams_from_file(config_path)
            model_type = self.recognition_model_type(hps)
        else:
            hps = None
            model_type = ModelType.GPT_SOVITS

        model_args = {
            "model_type": model_type,
            "model_path": model_path,
            "config_path": config_path,
            "sovits_path": sovits_path,
            "gpt_path": gpt_path,
            "config": hps,
            "device": self.device
        }

        model_class = self.model_class_map[model_type]
        model = model_class(**model_args)

        if model_type == ModelType.VITS:
            bert_embedding = getattr(hps.data, 'bert_embedding', getattr(hps.model, 'bert_embedding', False))
            if bert_embedding and self.tts_front is None:
                self.load_VITS_PinYin_model(
                    os.path.join(config.abs_path, config.system.data_path, config.model_config.vits_chinese_bert))
            if not config.vits_config.dynamic_loading:
                model.load_model()
            self.available_tts_model.add(ModelType.VITS.value)

        elif model_type == ModelType.W2V2_VITS:
            if self.emotion_reference is None:
                self.emotion_reference = self.load_npy(
                    os.path.join(config.abs_path, config.system.data_path, config.model_config.dimensional_emotion_npy))
            model.load_model(emotion_reference=self.emotion_reference,
                             dimensional_emotion_model=self.dimensional_emotion_model)
            self.available_tts_model.add(ModelType.W2V2_VITS.value)

        elif model_type == ModelType.HUBERT_VITS:
            if self.hubert is None:
                self.hubert = self.load_hubert_model(
                    os.path.join(config.abs_path, config.system.data_path, config.model_config.hubert_soft_0d54a1f4))
            model.load_model(hubert=self.hubert)

        elif model_type == ModelType.BERT_VITS2:
            bert_model_names = model.bert_model_names
            for bert_model_name in bert_model_names.values():
                if self.model_handler is None:
                    from manager.model_handler import ModelHandler
                    self.model_handler = ModelHandler(self.device)
                self.model_handler.load_bert(bert_model_name)
            if model.hps_ms.model.emotion_embedding == 1:
                self.model_handler.load_emotion()
            elif model.hps_ms.model.emotion_embedding == 2:
                self.model_handler.load_clap()

            model.load_model(self.model_handler)

            self.available_tts_model.add(ModelType.BERT_VITS2.value)

        elif model_type == ModelType.GPT_SOVITS:
            if self.model_handler is None:
                from manager.model_handler import ModelHandler
                self.model_handler = ModelHandler(self.device)
            self.model_handler.load_ssl()
            self.model_handler.load_bert("CHINESE_ROBERTA_WWM_EXT_LARGE")
            model.load_model(self.model_handler)

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

    def load_model(self, model_path: str, config_path: str, sovits_path: str, gpt_path: str):
        try:
            if not check_is_none(model_path, config_path):
                model_path, config_path = self.relative_to_absolute_path(model_path, config_path)
            else:
                sovits_path, gpt_path = self.relative_to_absolute_path(sovits_path, gpt_path)

            model_data = self._load_model_from_path(model_path, config_path, sovits_path, gpt_path)
            model_id = model_data["model_id"]
            sid2model = model_data["sid2model"]
            model_type = model_data["model_type"]

            self.models[model_type][model_id] = {
                "model_type": model_data.get("model_type"),
                "model_path": model_path,
                "config_path": config_path,
                "sovits_path": sovits_path,
                "gpt_path": gpt_path,
                "model": model_data.get("model"),
                "n_speakers": len(model_data["speakers"])}
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

                model_path, config_path, model, n_speakers = self.models[model_type][model_id].values()
                start = 0

                for key, value in self.models[model_type].items():
                    if key == model_id:
                        break
                    start += value.get("n_speakers")

                if model_type == ModelType.BERT_VITS2:
                    for bert_model_name in self.models[model_type][model_id]["model"].bert_model_names.values():
                        self.model_handler.release_bert(bert_model_name)

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
            logging.error(traceback.print_exc())
            logging.error(f"Unloading failed. {e}")
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
        """按返回模型路径列表，列表每一项为{"model_path": model_path, "config_path": config_path}"""
        info = []
        for models in self.models.values():
            for model in models.values():
                info.append({
                    "model_type": model.get("model_type"),
                    "model_path": model.get("model_path"),
                    "config_path": model.get("config_path"),
                    "sovits_path": model.get("sovits_path"),
                    "gpt_path": model.get("gpt_path"),
                })

        return info

    def get_models_path_by_type(self):
        """按模型类型返回模型路径"""
        info = {
            ModelType.VITS.value: [],
            ModelType.HUBERT_VITS.value: [],
            ModelType.W2V2_VITS.value: [],
            ModelType.BERT_VITS2.value: [],
            ModelType.GPT_SOVITS.value: [],
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
            ModelType.BERT_VITS2.value: [],
            ModelType.GPT_SOVITS.value: [],
        }
        for model_type, model_data in self.models.items():
            if model_type != ModelType.GPT_SOVITS:
                for model_id, model in model_data.items():
                    model_path = model.get("model_path")
                    config_path = model.get("config_path")

                    model_path = self.absolute_to_relative_path(model_path)[0].replace("\\", "/")
                    config_path = self.absolute_to_relative_path(config_path)[0].replace("\\", "/")

                    info[model_type.value].append(
                        {"model_id": model_id,
                         "model_path": model_path,
                         "config_path": config_path,
                         "n_speakers": model.get("n_speakers")})
            else:
                for model_id, model in model_data.items():
                    sovits_path = model.get("sovits_path")
                    gpt_path = model.get("gpt_path")

                    sovits_path = self.absolute_to_relative_path(sovits_path)[0].replace("\\", "/")
                    gpt_path = self.absolute_to_relative_path(gpt_path)[0].replace("\\", "/")

                    info[model_type.value].append(
                        {"model_id": model_id,
                         "sovits_path": sovits_path,
                         "gpt_path": gpt_path,
                         "n_speakers": model.get("n_speakers")})

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
                        emotion_reference = np.append(emotion_reference, self._load_npy_from_path(file_path),
                                                      axis=0)

        elif os.path.isfile(emotion_reference_npy):
            emotion_reference = self._load_npy_from_path(emotion_reference_npy)

        logging.info(f"Loaded emotional dimention npy range: {len(emotion_reference)}")
        return emotion_reference

    def scan_path(self):
        folder_path = os.path.join(config.abs_path, config.system.data_path, config.tts_config.models_path)
        model_paths = glob.glob(folder_path + "/**/*.pth", recursive=True)
        gpt_paths = glob.glob(folder_path + "/**/*.ckpt", recursive=True)
        all_paths = []

        for id, pth_path in enumerate(model_paths):
            pth_name = os.path.basename(pth_path)
            if pth_name.startswith(("D_", "DUR_")):
                continue
            dir_name = os.path.dirname(pth_path)
            config_paths = glob.glob(dir_name + "/*.json", recursive=True)
            sovits_paths = glob.glob(dir_name + "/*.ckpt", recursive=True)
            model_path, config_path, sovits_path, gpt_path, model_type = (None,) * 5
            if len(config_paths) > 0:
                model_path = pth_path
                config_path = config_paths[0]
            elif len(sovits_paths) > 0:
                gpt_path = gpt_paths[0]
                sovits_path = pth_path
                model_type = ModelType.GPT_SOVITS
            else:
                continue

            info = {
                "model_id": id,
                "model_type": model_type,
                "model_path": model_path,
                "config_path": config_path,
                "sovits_path": sovits_path,
                "gpt_path": gpt_path,
            }
            all_paths.append(info)

        return all_paths

    def scan_unload_path(self):
        all_paths = self.scan_path()
        unload_paths = []
        loaded_paths = []
        loaded_paths_2 = []

        for model in self.get_models_path():
            # 只取已加载的模型路径
            if model.get("model_type") == ModelType.GPT_SOVITS:
                sovits_path, gpt_path = self.absolute_to_relative_path(model.get("sovits_path"),
                                                                       model.get("gpt_path"))
                sovits_path, gpt_path = sovits_path.replace("\\", "/"), gpt_path.replace("\\", "/")
                loaded_paths_2.append((sovits_path, gpt_path))
            else:
                model_path = self.absolute_to_relative_path(model.get("model_path"))[0].replace("\\", "/")
                loaded_paths.append(model_path)

        for info in all_paths:
            # 将绝对路径修改为相对路径，并将分隔符格式化为'/'
            if info.get("model_type") == ModelType.GPT_SOVITS:
                sovits_path, gpt_path = self.absolute_to_relative_path(info.get("sovits_path"),
                                                                       info.get("gpt_path"))
                sovits_path, gpt_path = sovits_path.replace("\\", "/"), gpt_path.replace("\\", "/")
                if not self.is_path_loaded((sovits_path, gpt_path), loaded_paths_2):
                    info.update(
                        {"model_type": info.get("model_type").value, "sovits_path": sovits_path, "gpt_path": gpt_path})
                    unload_paths.append(info)
            else:
                model_path, config_path = self.absolute_to_relative_path(info.get("model_path"),
                                                                         info.get("config_path"))
                model_path, config_path = model_path.replace("\\", "/"), config_path.replace("\\", "/")
                if not self.is_path_loaded(model_path, loaded_paths):
                    info.update({"model_path": model_path, "config_path": config_path})
                    unload_paths.append(info)

        return unload_paths

    def is_path_loaded(self, paths, loaded_paths):
        if len(paths) == 2:
            sovits_path, gpt_path = paths
            for loaded_path in loaded_paths:
                if sovits_path == loaded_path[0] and gpt_path == loaded_path[1]:
                    return True
        else:
            path = paths

            for loaded_path in loaded_paths:
                if path == loaded_path:
                    return True

        return False
