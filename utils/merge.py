import os
import sys

from voice import vits


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
        try:
            os.mkdir(cache_path)
        except:
            pass

    # classify
    for l in merging_model:
        if len(l) == 2:
            vits_list.append(l)
        elif len(l) == 3:
            hubert_vits_list.append(l)

    # merging vits
    new_id = 0
    for obj_id, i in enumerate(vits_list):
        obj = vits(model=i[0], config=i[1])

        for id, name in enumerate(obj.return_speakers()):
            vits_obj.append([int(id), obj, obj_id])
            # vits_speakers.append({"id": new_id, "name": name})
            vits_speakers.append({new_id: name})
            new_id += 1

    # merging hubert-vits
    new_id = 0
    for obj_id, i in enumerate(hubert_vits_list):
        obj = vits(model=i[0], config=i[1], model_=i[2])

        for id, name in enumerate(obj.return_speakers()):
            hubert_vits_obj.append([int(id), obj, obj_id])
            # hubert_vits_speakers.append({"id": new_id, "name": name})
            hubert_vits_speakers.append({new_id: name})

            new_id += 1

    voice_obj = [vits_obj, hubert_vits_obj, w2v2_vits_obj]
    voice_speakers = [vits_speakers, hubert_vits_speakers, w2v2_vits_speakers]

    return voice_obj, voice_speakers
