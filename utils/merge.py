import os
import sys

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
    "thai_cleaners": ["th"],
    "shanghainese_cleaners": ["sh"],
    "chinese_dialect_cleaners": ["zh", "ja", "sh", "gd", "en", "SZ", "WX", "CZ", "HZ", "SX", "NB", "JJ", "YX", "JD",
                                 "ZR", "PH", "TX", "JS", "HN", "LP", "XS", "FY", "RA", "CX", "SM", "TT", "WZ", "SC",
                                 "YB"],
}


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

    cache_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "/cache"

    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # Analysis model type
    for l in merging_model:
        if len(l) == 2:
            vits_list.append(l)
        elif len(l) == 3:
            _model = l[2]

            if isinstance(_model, list):

                for i in _model:
                    _model_extention = os.path.splitext(i)[1]

                    if _model_extention != ".npy":
                        raise ValueError(f"Unsupported model type: {_model_extention}")

                w2v2_vits_list.append(l)
            else:
                _model_extention = os.path.splitext(_model)[1]

                if _model_extention == ".pt":
                    hubert_vits_list.append(l)
                elif _model_extention == ".npy":
                    w2v2_vits_list.append(l)
                else:
                    raise ValueError(f"Unsupported model type: {_model_extention}")

    # merge vits
    new_id = 0
    for obj_id, i in enumerate(vits_list):
        obj = vits(model=i[0], config=i[1])
        lang = lang_dict.get(obj.get_cleaner(), obj.get_cleaner())

        for id, name in enumerate(obj.return_speakers()):
            vits_obj.append([int(id), obj, obj_id])
            vits_speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    # merge hubert-vits
    new_id = 0
    for obj_id, i in enumerate(hubert_vits_list):
        obj = vits(model=i[0], config=i[1], model_=i[2])
        lang = lang_dict.get(obj.get_cleaner(), obj.get_cleaner())

        for id, name in enumerate(obj.return_speakers()):
            hubert_vits_obj.append([int(id), obj, obj_id])
            hubert_vits_speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    # merge w2v2-vits
    new_id = 0
    for obj_id, i in enumerate(w2v2_vits_list):
        obj = vits(model=i[0], config=i[1], model_=i[2])
        lang = lang_dict.get(obj.get_cleaner(), obj.get_cleaner())

        for id, name in enumerate(obj.return_speakers()):
            w2v2_vits_obj.append([int(id), obj, obj_id])
            w2v2_vits_speakers.append({"id": new_id, "name": name, "lang": lang})
            new_id += 1

    voice_obj = {"VITS": vits_obj, "HUBERT-VITS": hubert_vits_obj, "W2V2-VITS": w2v2_vits_obj}
    voice_speakers = {"VITS": vits_speakers, "HUBERT-VITS": hubert_vits_speakers, "W2V2-VITS": w2v2_vits_speakers}

    tts = TTS(voice_obj, voice_speakers)

    return tts
