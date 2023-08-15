import os
import json
import logging
import config
import numpy as np
from utils.utils import check_is_none
from voice import vits, TTS

lang_dict = {
    "english_cleaners": ["en"],
    "english_cleaners2": ["en"],
    "japanese_cleaners": ["ja"],
    "japanese_cleaners2": ["ja"],
    "korean_cleaners": ["ko"],
    "chinese_cleaners": ["zh"],
    "zh_ja_mixture_cleaners": ["zh", "ja"],
    "sanskrit_cleaners": ["sa"],
    "cjks_cleaners": ["zh", "ja", "ko", "sa"],
    "cjke_cleaners": ["zh", "ja", "ko", "en"],
    "cjke_cleaners2": ["zh", "ja", "ko", "en"],
    "cje_cleaners": ["zh", "ja", "en"],
    "cje_cleaners2": ["zh", "ja", "en"],
    "thai_cleaners": ["th"],
    "shanghainese_cleaners": ["sh"],
    "chinese_dialect_cleaners": ["zh", "ja", "sh", "gd", "en", "SZ", "WX", "CZ", "HZ", "SX", "NB", "JJ", "YX", "JD",
                                 "ZR", "PH", "TX", "JS", "HN", "LP", "XS", "FY", "RA", "CX", "SM", "TT", "WZ", "SC",
                                 "YB"],
    "bert_chinese_cleaners": ["zh"],
}


def analysis(model_config_json):
    model_config = json.load(model_config_json)
    symbols = model_config.get("symbols", None)
    emotion_embedding = model_config.get("data").get("emotion_embedding", False)
    if symbols != None:
        if not emotion_embedding:
            mode_type = "vits"
        else:
            mode_type = "w2v2"
    else:
        mode_type = "hubert"
    return mode_type


def load_npy(model_):
    if isinstance(model_, list):
        # check if is .npy
        for i in model_:
            _model_extention = os.path.splitext(i)[1]
            if _model_extention != ".npy":
                raise ValueError(f"Unsupported model type: {_model_extention}")

        # merge npy files
        emotion_reference = np.empty((0, 1024))
        for i in model_:
            tmp = np.load(i).reshape(-1, 1024)
            emotion_reference = np.append(emotion_reference, tmp, axis=0)

    elif os.path.isdir(model_):
        emotion_reference = np.empty((0, 1024))
        for root, dirs, files in os.walk(model_):
            for file_name in files:
                # check if is .npy
                _model_extention = os.path.splitext(file_name)[1]
                if _model_extention != ".npy":
                    continue
                file_path = os.path.join(root, file_name)

                # merge npy files
                tmp = np.load(file_path).reshape(-1, 1024)
                emotion_reference = np.append(emotion_reference, tmp, axis=0)

    elif os.path.isfile(model_):
        # check if is .npy
        _model_extention = os.path.splitext(model_)[1]
        if _model_extention != ".npy":
            raise ValueError(f"Unsupported model type: {_model_extention}")

        emotion_reference = np.load(model_)
    logging.info(f"Loaded emotional dimention npy range:{len(emotion_reference)}")
    return emotion_reference


def merge_model(merging_model):
    vits_obj = []
    vits_speakers = []
    hubert_vits_obj = []
    hubert_vits_speakers = []
    w2v2_vits_obj = []
    w2v2_vits_speakers = []

    # model list
    vits_list = []
    hubert_vits_list = []
    w2v2_vits_list = []

    for l in merging_model:
        with open(l[1], 'r', encoding='utf-8') as model_config:
            model_type = analysis(model_config)
        if model_type == "vits":
            vits_list.append(l)
        elif model_type == "hubert":
            hubert_vits_list.append(l)
        elif model_type == "w2v2":
            w2v2_vits_list.append(l)

    # merge vits
    new_id = 0
    for obj_id, i in enumerate(vits_list):
        obj = vits(model=i[0], config=i[1], model_type="vits")
        lang = lang_dict.get(obj.get_cleaner(), ["unknown"])
        if isinstance(obj.get_speakers(), list):
            for id, name in enumerate(obj.get_speakers()):
                vits_obj.append([int(id), obj, obj_id])
                vits_speakers.append({"id": new_id, "name": name, "lang": lang})
                new_id += 1
        else:
            for id, (name, _) in enumerate(obj.get_speakers().items()):
                vits_obj.append([int(id), obj, obj_id])
                vits_speakers.append({"id": new_id, "name": name, "lang": lang})
                new_id += 1

    # merge hubert-vits
    if len(hubert_vits_list) != 0:
        if getattr(config, "HUBERT_SOFT_MODEL", None) == None or check_is_none(config.HUBERT_SOFT_MODEL):
            raise ValueError(f"Please configure HUBERT_SOFT_MODEL path in config.py")
        try:
            from hubert_model import hubert_soft
            hubert = hubert_soft(config.HUBERT_SOFT_MODEL)
        except Exception as e:
            raise ValueError(f"Load HUBERT_SOFT_MODEL failed {e}")

    new_id = 0
    for obj_id, i in enumerate(hubert_vits_list):
        obj = vits(model=i[0], config=i[1], model_=hubert, model_type="hubert")
        lang = lang_dict.get(obj.get_cleaner(), ["unknown"])

        for id, name in enumerate(obj.get_speakers()):
            hubert_vits_obj.append([int(id), obj, obj_id])
            hubert_vits_speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    # merge w2v2-vits
    if len(w2v2_vits_list) != 0:
        if getattr(config, "DIMENSIONAL_EMOTION_NPY", None) == None or check_is_none(config.DIMENSIONAL_EMOTION_NPY):
            raise ValueError(f"Please configure DIMENSIONAL_EMOTION_NPY path in config.py")
        try:
            emotion_reference = load_npy(config.DIMENSIONAL_EMOTION_NPY)
        except Exception as e:
            raise ValueError(f"Load DIMENSIONAL_EMOTION_NPY failed {e}")

    new_id = 0
    for obj_id, i in enumerate(w2v2_vits_list):
        obj = vits(model=i[0], config=i[1], model_=emotion_reference, model_type="w2v2")
        lang = lang_dict.get(obj.get_cleaner(), ["unknown"])

        for id, name in enumerate(obj.get_speakers()):
            w2v2_vits_obj.append([int(id), obj, obj_id])
            w2v2_vits_speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    voice_obj = {"VITS": vits_obj, "HUBERT-VITS": hubert_vits_obj, "W2V2-VITS": w2v2_vits_obj}
    voice_speakers = {"VITS": vits_speakers, "HUBERT-VITS": hubert_vits_speakers, "W2V2-VITS": w2v2_vits_speakers}

    tts = TTS(voice_obj, voice_speakers)

    return tts
