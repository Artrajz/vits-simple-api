import os
import json
import logging
import config
import numpy as np

import utils
from utils.data_utils import check_is_none, HParams
from vits import VITS
from voice import TTS
from config import DEVICE as device
from utils.lang_dict import lang_dict
from contants import ModelType


def recognition_model_type(hps: HParams) -> str:
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


def load_npy(emotion_reference_npy):
    if isinstance(emotion_reference_npy, list):
        # check if emotion_reference_npy is endwith .npy
        for i in emotion_reference_npy:
            model_extention = os.path.splitext(i)[1]
            if model_extention != ".npy":
                raise ValueError(f"Unsupported model type: {model_extention}")

        # merge npy files
        emotion_reference = np.empty((0, 1024))
        for i in emotion_reference_npy:
            tmp = np.load(i).reshape(-1, 1024)
            emotion_reference = np.append(emotion_reference, tmp, axis=0)

    elif os.path.isdir(emotion_reference_npy):
        emotion_reference = np.empty((0, 1024))
        for root, dirs, files in os.walk(emotion_reference_npy):
            for file_name in files:
                # check if emotion_reference_npy is endwith .npy
                model_extention = os.path.splitext(file_name)[1]
                if model_extention != ".npy":
                    continue
                file_path = os.path.join(root, file_name)

                # merge npy files
                tmp = np.load(file_path).reshape(-1, 1024)
                emotion_reference = np.append(emotion_reference, tmp, axis=0)

    elif os.path.isfile(emotion_reference_npy):
        # check if emotion_reference_npy is endwith .npy
        model_extention = os.path.splitext(emotion_reference_npy)[1]
        if model_extention != ".npy":
            raise ValueError(f"Unsupported model type: {model_extention}")

        emotion_reference = np.load(emotion_reference_npy)
    logging.info(f"Loaded emotional dimention npy range:{len(emotion_reference)}")
    return emotion_reference


def parse_models(model_list):
    categorized_models = {
        ModelType.VITS: [],
        ModelType.HUBERT_VITS: [],
        ModelType.W2V2_VITS: [],
        ModelType.BERT_VITS2: []
    }

    for model_info in model_list:
        config_path = model_info[1]
        hps = utils.get_hparams_from_file(config_path)
        model_info.append(hps)
        model_type = recognition_model_type(hps)
        # with open(config_path, 'r', encoding='utf-8') as model_config:
        #     model_type = recognition_model_type(model_config)
        if model_type in categorized_models:
            categorized_models[model_type].append(model_info)

    return categorized_models


def merge_models(model_list, model_class, model_type, additional_arg=None):
    id_mapping_objs = []
    speakers = []
    new_id = 0

    for obj_id, (model_path, config_path, hps) in enumerate(model_list):
        obj_args = {
            "model": model_path,
            "config": hps,
            "model_type": model_type,
            "device": device
        }

        if model_type == ModelType.BERT_VITS2:
            from bert_vits2.utils import process_legacy_versions
            legacy_versions = process_legacy_versions(hps)
            key = f"{model_type.value}_v{legacy_versions}" if legacy_versions else model_type.value         
        else:
            key = getattr(hps.data, "text_cleaners", ["none"])[0]

        if additional_arg:
            obj_args.update(additional_arg)

        obj = model_class(**obj_args)

        lang = lang_dict.get(key, ["unknown"])

        for real_id, name in enumerate(obj.get_speakers()):
            id_mapping_objs.append([real_id, obj, obj_id])
            speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    return id_mapping_objs, speakers


def load_model(model_list) -> TTS:
    categorized_models = parse_models(model_list)

    # Handle VITS
    vits_objs, vits_speakers = merge_models(categorized_models[ModelType.VITS], VITS, ModelType.VITS)

    # Handle HUBERT-VITS
    hubert_vits_objs, hubert_vits_speakers = [], []
    if len(categorized_models[ModelType.HUBERT_VITS]) != 0:
        if getattr(config, "HUBERT_SOFT_MODEL", None) is None or check_is_none(config.HUBERT_SOFT_MODEL):
            raise ValueError(f"Please configure HUBERT_SOFT_MODEL path in config.py")
        try:
            from vits.hubert_model import hubert_soft
            hubert = hubert_soft(config.HUBERT_SOFT_MODEL)
        except Exception as e:
            raise ValueError(f"Load HUBERT_SOFT_MODEL failed {e}")

        hubert_vits_objs, hubert_vits_speakers = merge_models(categorized_models[ModelType.HUBERT_VITS], VITS, ModelType.HUBERT_VITS,
                                                              additional_arg={"additional_model": hubert})

    # Handle W2V2-VITS
    w2v2_vits_objs, w2v2_vits_speakers = [], []
    w2v2_emotion_count = 0
    if len(categorized_models[ModelType.W2V2_VITS]) != 0:
        if getattr(config, "DIMENSIONAL_EMOTION_NPY", None) is None or check_is_none(
                config.DIMENSIONAL_EMOTION_NPY):
            raise ValueError(f"Please configure DIMENSIONAL_EMOTION_NPY path in config.py")
        try:
            emotion_reference = load_npy(config.DIMENSIONAL_EMOTION_NPY)
        except Exception as e:
            emotion_reference = None
            raise ValueError(f"Load DIMENSIONAL_EMOTION_NPY failed {e}")

        w2v2_vits_objs, w2v2_vits_speakers = merge_models(categorized_models[ModelType.W2V2_VITS], VITS, ModelType.W2V2_VITS,
                                                          additional_arg={"additional_model": emotion_reference})
        w2v2_emotion_count = len(emotion_reference) if emotion_reference is not None else 0

    # Handle BERT-VITS2
    bert_vits2_objs, bert_vits2_speakers = [], []
    if len(categorized_models[ModelType.BERT_VITS2]) != 0:
        from bert_vits2 import Bert_VITS2
        bert_vits2_objs, bert_vits2_speakers = merge_models(categorized_models[ModelType.BERT_VITS2], Bert_VITS2, ModelType.BERT_VITS2)

    voice_obj = {ModelType.VITS: vits_objs,
                 ModelType.HUBERT_VITS: hubert_vits_objs,
                 ModelType.W2V2_VITS: w2v2_vits_objs,
                 ModelType.BERT_VITS2: bert_vits2_objs}
    voice_speakers = {ModelType.VITS.value: vits_speakers,
                      ModelType.HUBERT_VITS.value: hubert_vits_speakers,
                      ModelType.W2V2_VITS.value: w2v2_vits_speakers,
                      ModelType.BERT_VITS2.value: bert_vits2_speakers}

    tts = TTS(voice_obj, voice_speakers, device=device, w2v2_emotion_count=w2v2_emotion_count)
    return tts
