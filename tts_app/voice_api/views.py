import copy
import logging
import os
import time
import traceback
import uuid
from io import BytesIO

import librosa
import numpy as np
from flask import request, jsonify, make_response, send_file, Blueprint
from werkzeug.utils import secure_filename

from contants import config
# from gpt_sovits.utils import load_audio
from logger import logger
from contants import ModelType
from tts_app.voice_api.auth import require_api_key
from tts_app.model_manager import model_manager, tts_manager
from tts_app.voice_api.utils import *
from utils.data_utils import check_is_none

voice_api = Blueprint("voice_api", __name__)


def get_param(request_data, key, default, data_type=None):
    if key == "segment_size" and "max" in request_data:
        logger.warning(
            "The 'max' parameter is deprecated and will be phased out in the future. Please use 'segment_size' instead.")
        return get_param(request_data, "max", default, data_type)

    value = request_data.get(key, "")

    if data_type:
        try:
            value = data_type(value)
        except:
            value = default

    if value == "":
        value = default

    return value


def extract_filename_and_directory(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    directory_name = os.path.basename(directory)
    if not directory:  # 如果文件所在文件夹为空（即在根目录）
        return filename
    else:
        return directory_name + "/" + filename


@voice_api.route('/default_parameter', methods=["GET", "POST"])
def default_parameter():
    gpt_sovits_config = copy.deepcopy(config.gpt_sovits_config.asdict())
    for preset_name, preset in gpt_sovits_config["presets"].items():
        if not check_is_none(preset["refer_wav_path"]):
            preset["refer_wav_path"] = extract_filename_and_directory(preset["refer_wav_path"])

    data = {"vits_config": config.vits_config.asdict(),
            "w2v2_vits_config": config.w2v2_vits_config.asdict(),
            "hubert_vits_config": config.hubert_vits_config.asdict(),
            "bert_vits2_config": config.bert_vits2_config.asdict(),
            "gpt_sovits_config": gpt_sovits_config
            }
    return jsonify(data)


@voice_api.route('/speakers', methods=["GET", "POST"])
def voice_speakers_api():
    return jsonify(model_manager.voice_speakers)


@voice_api.route('/', methods=["GET", "POST"])
@voice_api.route('/vits', methods=["GET", "POST"])
@require_api_key
def voice_vits_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = get_param(request_data, "text", "", str)
        id = get_param(request_data, "id", config.vits_config.id, int)
        format = get_param(request_data, "format", config.vits_config.format, str)
        lang = get_param(request_data, "lang", config.vits_config.lang, str).lower()
        length = get_param(request_data, "length", config.vits_config.length, float)
        noise = get_param(request_data, "noise", config.vits_config.noise, float)
        noisew = get_param(request_data, "noisew", config.vits_config.noisew, float)
        segment_size = get_param(request_data, "segment_size", config.vits_config.segment_size, int)
        use_streaming = get_param(request_data, 'streaming', config.vits_config.use_streaming, bool)
    except Exception as e:
        logger.error(f"[{ModelType.VITS.value}] {e}")
        return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.VITS.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew} segment_size:{segment_size}")
    logger.info(f"[{ModelType.VITS.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.VITS.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= model_manager.vits_speakers_count:
        logger.info(f"[{ModelType.VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = model_manager.voice_speakers[ModelType.VITS.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) > 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if (lang_detect := config.language_identification.language_automatic_detect) and isinstance(lang_detect, list):
        speaker_lang = lang_detect

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    state = {"text": text,
             "id": id,
             "format": format,
             "length": length,
             "noise": noise,
             "noisew": noisew,
             "segment_size": segment_size,
             "lang": lang,
             "speaker_lang": speaker_lang}

    if use_streaming:
        audio = tts_manager.stream_vits_infer(state)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        t1 = time.time()
        audio = tts_manager.vits_infer(state)
        t2 = time.time()
        logger.info(f"[{ModelType.VITS.value}] finish in {(t2 - t1):.2f}s")

        if config.system.cache_audio:
            logger.debug(f"[{ModelType.VITS.value}] {fname}")
            path = os.path.join(config.system.cache_path, fname)
            save_audio(audio.getvalue(), path)

        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/hubert-vits', methods=["POST"])
@require_api_key
def voice_hubert_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            id = get_param(request.form, "id", config.hubert_vits_config.id, int)
            format = get_param(request.form, "format", config.hubert_vits_config.format)
            length = get_param(request.form, "length", config.hubert_vits_config.length, float)
            noise = get_param(request.form, "noise", config.hubert_vits_config.noise, float)
            noisew = get_param(request.form, "noisew", config.hubert_vits_config.noisew, float)
            use_streaming = get_param(request.form, 'streaming', False, bool)
        except Exception as e:
            logger.error(f"[{ModelType.HUBERT_VITS.value}] {e}")
            return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.HUBERT_VITS.value}] id:{id} format:{format} length:{length} noise:{noise} noisew:{noisew}")

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(config.system.upload_folder, fname))

    if check_is_none(id):
        logger.info(f"[{ModelType.HUBERT_VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= model_manager.hubert_speakers_count:
        logger.info(f"[{ModelType.HUBERT_VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    file_type = f"audio/{format}"
    task = {"id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "audio_path": os.path.join(config.system.upload_folder, fname)}

    t1 = time.time()
    audio = tts_manager.hubert_vits_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.HUBERT_VITS.value}] finish in {(t2 - t1):.2f}s")

    if config.system.cache_audio:
        logger.debug(f"[{ModelType.HUBERT_VITS.value}] {fname}")
        path = os.path.join(config.system.cache_path, fname)
        save_audio(audio.getvalue(), path)

    if use_streaming:
        audio = tts_manager.generate_audio_chunks(audio)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/w2v2-vits', methods=["GET", "POST"])
@require_api_key
def voice_w2v2_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = get_param(request_data, "text", "", str)
        id = get_param(request_data, "id", config.w2v2_vits_config.id, int)
        format = get_param(request_data, "format", config.w2v2_vits_config.format, str)
        lang = get_param(request_data, "lang", config.w2v2_vits_config.lang, str).lower()
        length = get_param(request_data, "length", config.w2v2_vits_config.length, float)
        noise = get_param(request_data, "noise", config.w2v2_vits_config.noise, float)
        noisew = get_param(request_data, "noisew", config.w2v2_vits_config.noisew, float)
        segment_size = get_param(request_data, "segment_size", config.w2v2_vits_config.segment_size, int)
        emotion = get_param(request_data, "emotion", config.w2v2_vits_config.emotion, int)
        emotion_reference = get_param(request_data, "emotion_reference", None, str)
        use_streaming = get_param(request_data, 'streaming', False, bool)
    except Exception as e:
        logger.error(f"[{ModelType.W2V2_VITS.value}] {e}")
        return make_response(f"parameter error", 400)

    logger.info(f"[{ModelType.W2V2_VITS.value}] id:{id} format:{format} lang:{lang} "
                f"length:{length} noise:{noise} noisew:{noisew} emotion:{emotion} segment_size:{segment_size}")
    logger.info(f"[{ModelType.W2V2_VITS.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.W2V2_VITS.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.W2V2_VITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= model_manager.w2v2_speakers_count:
        logger.info(f"[{ModelType.W2V2_VITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = model_manager.voice_speakers[ModelType.W2V2_VITS.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) > 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.W2V2_VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if (lang_detect := config.language_identification.language_automatic_detect) and isinstance(lang_detect, list):
        speaker_lang = lang_detect

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    task = {"text": text,
            "id": id,
            "format": format,
            "length": length,
            "noise": noise,
            "noisew": noisew,
            "segment_size": segment_size,
            "lang": lang,
            "emotion": emotion,
            "emotion_reference": emotion_reference,
            "speaker_lang": speaker_lang}

    t1 = time.time()
    audio = tts_manager.w2v2_vits_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.W2V2_VITS.value}] finish in {(t2 - t1):.2f}s")

    if config.system.cache_audio:
        logger.debug(f"[{ModelType.W2V2_VITS.value}] {fname}")
        path = os.path.join(config.system.cache_path, fname)
        save_audio(audio.getvalue(), path)

    if use_streaming:
        audio = tts_manager.generate_audio_chunks(audio)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/conversion', methods=["POST"])
@voice_api.route('/vits/conversion', methods=["POST"])
@require_api_key
def vits_voice_conversion_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            original_id = get_param(request.form, "original_id", 0, int)
            target_id = get_param(request.form, "target_id", 0, int)
            format = get_param(request.form, "format", voice.filename.split(".")[1], str)
            use_streaming = get_param(request.form, 'streaming', False, bool)
        except Exception as e:
            logger.error(f"[vits_voice_convertsion] {e}")
            return make_response("parameter error", 400)

        logger.info(f"[vits_voice_convertsion] orginal_id:{original_id} target_id:{target_id}")
        fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
        audio_path = os.path.join(config.system.upload_folder, fname)
        voice.save(audio_path)
        file_type = f"audio/{format}"
        state = {"audio_path": audio_path,
                 "original_id": original_id,
                 "target_id": target_id,
                 "format": format}

        t1 = time.time()
        audio = tts_manager.vits_voice_conversion(state)
        t2 = time.time()
        logger.info(f"[Voice conversion] finish in {(t2 - t1):.2f}s")

        if config.system.cache_audio:
            logger.debug(f"[Voice conversion] {fname}")
            path = os.path.join(config.system.cache_path, fname)
            save_audio(audio.getvalue(), path)

        if use_streaming:
            audio = tts_manager.generate_audio_chunks(audio)
            response = make_response(audio)
            response.headers['Content-Disposition'] = f'attachment; filename={fname}'
            response.headers['Content-Type'] = file_type
            return response
        else:
            return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/ssml', methods=["POST"])
@require_api_key
def ssml_api():
    try:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            request_data = request.get_json()
        else:
            request_data = request.form

        ssml = request_data.get("ssml")
    except Exception as e:
        logger.info(f"[ssml] {e}")
        return make_response(jsonify({"status": "error", "message": f"parameter error"}), 400)

    logger.debug(ssml)
    voice_tasks, format = tts_manager.parse_ssml(ssml)
    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"

    t1 = time.time()
    audio = tts_manager.process_ssml_infer_task(voice_tasks, format)
    t2 = time.time()
    logger.info(f"[ssml] finish in {(t2 - t1):.2f}s")

    if config.system.cache_audio:
        logger.debug(f"[ssml] {fname}")
        path = os.path.join(config.system.cache_path, fname)
        save_audio(audio.getvalue(), path)

    return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/dimension-emotion', methods=["POST"])
@require_api_key
def dimensional_emotion_api():
    if request.method == "POST":
        try:
            audio = request.files['upload']
        except Exception as e:
            logger.error(f"[dimensional_emotion] {e}")
            return make_response("parameter error", 400)

    content = BytesIO(audio.read())

    file_type = "application/octet-stream; charset=ascii"
    fname = os.path.splitext(audio.filename)[0] + ".npy"
    emotion_npy = tts_manager.get_dimensional_emotion_npy(content)
    return send_file(path_or_file=emotion_npy, mimetype=file_type, download_name=fname)


@voice_api.route('/bert-vits2', methods=["GET", "POST"])
@require_api_key
def voice_bert_vits2_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = get_param(request_data, "text", "", str)
        id = get_param(request_data, "id", config.bert_vits2_config.id, int)
        format = get_param(request_data, "format", config.bert_vits2_config.format, str)
        lang = get_param(request_data, "lang", config.bert_vits2_config.lang, str).lower()
        length = get_param(request_data, "length", config.bert_vits2_config.length, float)
        # length_zh = get_param(request_data, "length_zh", config.bert_vits2_config.length_zh, float)
        # length_ja = get_param(request_data, "length_ja", config.bert_vits2_config.length_ja, float)
        # length_en = get_param(request_data, "length_en", config.bert_vits2_config.length_en, float)
        noise = get_param(request_data, "noise", config.bert_vits2_config.noise, float)
        noisew = get_param(request_data, "noisew", config.bert_vits2_config.noisew, float)
        sdp_ratio = get_param(request_data, "sdp_ratio", config.bert_vits2_config.sdp_ratio, float)
        segment_size = get_param(request_data, "segment_size", config.bert_vits2_config.segment_size, int)
        use_streaming = get_param(request_data, 'streaming', config.bert_vits2_config.use_streaming, bool)
        emotion = get_param(request_data, 'emotion', config.bert_vits2_config.emotion, int)
        reference_audio = request.files.get("reference_audio", None)
        text_prompt = get_param(request_data, 'text_prompt', config.bert_vits2_config.text_prompt, str)
        style_text = get_param(request_data, 'style_text', config.bert_vits2_config.style_text, str)
        style_weight = get_param(request_data, 'style_weight', config.bert_vits2_config.style_weight, float)
    except Exception as e:
        logger.error(f"[{ModelType.BERT_VITS2.value}] {e}")
        return make_response("parameter error", 400)

    # logger.info(
    #     f"[{ModelType.BERT_VITS2.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew} sdp_ratio:{sdp_ratio} segment_size:{segment_size}"
    #     f" length_zh:{length_zh} length_ja:{length_ja} length_en:{length_en}")

    logger.info(
        f"[{ModelType.BERT_VITS2.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew} sdp_ratio:{sdp_ratio} segment_size:{segment_size}")
    logger.info(f"[{ModelType.BERT_VITS2.value}] len:{len(text)} text：{text}")
    if reference_audio:
        logger.info(f"[{ModelType.BERT_VITS2.value}] reference_audio:{reference_audio.filename}")
    elif emotion:
        logger.info(f"[{ModelType.BERT_VITS2.value}] emotion:{emotion}")
    elif text_prompt:
        logger.info(f"[{ModelType.BERT_VITS2.value}] text_prompt:{text_prompt}")
    elif style_text:
        logger.info(f"[{ModelType.BERT_VITS2.value}] style_text:{style_text} style_weight:{style_weight}")
    logger.info(f"[{ModelType.BERT_VITS2.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.BERT_VITS2.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.BERT_VITS2.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= model_manager.bert_vits2_speakers_count:
        logger.info(f"[{ModelType.BERT_VITS2.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    if emotion and (emotion < 0 or emotion > 9):
        logger.info(f"[{ModelType.BERT_VITS2.value}] emotion {emotion} out of the range 0-9")
        return make_response(jsonify({"status": "error", "message": f"emotion {emotion} out of the range 0-9"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = model_manager.voice_speakers[ModelType.BERT_VITS2.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) > 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.BERT_VITS2.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if (lang_detect := config.language_identification.language_automatic_detect) and isinstance(lang_detect, list):
        speaker_lang = lang_detect

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    state = {"text": text,
             "id": id,
             "format": format,
             "length": length,
             # "length_zh": length_zh,
             # "length_ja": length_ja,
             # "length_en": length_en,
             "noise": noise,
             "noisew": noisew,
             "sdp_ratio": sdp_ratio,
             "segment_size": segment_size,
             "lang": lang,
             "speaker_lang": speaker_lang,
             "emotion": emotion,
             "reference_audio": reference_audio,
             "text_prompt": text_prompt,
             "style_text": style_text,
             "style_weight": style_weight,
             }

    if use_streaming:
        # audio = tts_manager.stream_bert_vits2_infer(state)
        audio = tts_manager.stream_bert_vits2_infer_v2(state)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        t1 = time.time()
        # audio = tts_manager.bert_vits2_infer(state)
        audio = tts_manager.bert_vits2_infer_v2(state)
        t2 = time.time()
        logger.info(f"[{ModelType.BERT_VITS2.value}] finish in {(t2 - t1):.2f}s")

    if config.system.cache_audio:
        logger.debug(f"[{ModelType.BERT_VITS2.value}] {fname}")
        path = os.path.join(config.system.cache_path, fname)
        save_audio(audio.getvalue(), path)

    return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/gpt-sovits', methods=["GET", "POST"])
@require_api_key
def voice_gpt_sovits_api():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        text = get_param(request_data, "text", "", str)
        id = get_param(request_data, "id", config.gpt_sovits_config.id, int)
        lang = get_param(request_data, "lang", config.gpt_sovits_config.lang, str)
        format = get_param(request_data, "format", config.gpt_sovits_config.format, str)
        segment_size = get_param(request_data, "segment_size", config.gpt_sovits_config.segment_size, int)
        reference_audio = request.files.get("reference_audio", None)
        preset = get_param(request_data, "preset", "default", str)
        refer_wav_path = get_param(request_data, "refer_wav_path",
                                   config.gpt_sovits_config.presets.get("default").refer_wav_path, str)
        prompt_text = get_param(request_data, "prompt_text",
                                config.gpt_sovits_config.presets.get("default").prompt_text, str)
        prompt_lang = get_param(request_data, "prompt_lang",
                                config.gpt_sovits_config.presets.get("default").prompt_lang, str)
        top_k = get_param(request_data, "top_k", config.gpt_sovits_config.top_k, int)
        top_p = get_param(request_data, "top_p", config.gpt_sovits_config.top_p, float)
        temperature = get_param(request_data, "temperature", config.gpt_sovits_config.temperature, float)
        # use_streaming = get_param(request_data, 'streaming', config.gpt_sovits_config.use_streaming, bool)
    except Exception as e:
        logger.error(f"[{ModelType.GPT_SOVITS.value}] {e}")
        return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.GPT_SOVITS.value}] id:{id} format:{format} lang:{lang} segment_size:{segment_size} top_k:{top_k} top_p:{top_p} temperature:{temperature}")
    logger.info(f"[{ModelType.GPT_SOVITS.value}] len:{len(text)} text：{text}")

    if check_is_none(text):
        logger.info(f"[{ModelType.GPT_SOVITS.value}] text is empty")
        return make_response(jsonify({"status": "error", "message": "text is empty"}), 400)

    if check_is_none(id):
        logger.info(f"[{ModelType.GPT_SOVITS.value}] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    if id < 0 or id >= model_manager.gpt_sovits_speakers_count:
        logger.info(f"[{ModelType.GPT_SOVITS.value}] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)

    # 校验模型是否支持输入的语言
    speaker_lang = model_manager.voice_speakers[ModelType.GPT_SOVITS.value][id].get('lang')
    if lang not in ["auto", "mix"] and len(speaker_lang) > 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.GPT_SOVITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if (lang_detect := config.language_identification.language_automatic_detect) and isinstance(lang_detect, list):
        speaker_lang = lang_detect

    # 检查参考音频
    if check_is_none(reference_audio):  # 无参考音频
        # 已选择预设
        if preset != "default":
            refer_preset = config.gpt_sovits_config.presets.get(preset)
            if check_is_none(refer_wav_path):
                refer_wav_path = refer_preset.refer_wav_path

            prompt_text, prompt_lang = refer_preset.prompt_text, refer_preset.prompt_lang

        # 未选择预设，使用预设default
        reference_audio = refer_wav_path

    reference_audio, reference_audio_sr = librosa.load(reference_audio, sr=None, dtype=np.float32)
    reference_audio = reference_audio.flatten()

    # 检查修改后的参考音频，如果既没有上传参考音频且没设置预设，就会报错
    # if check_is_none(reference_audio, prompt_text):
    #     # 未指定参考音频且配置文件无预设
    #     message = "No reference audio specified, and no default setting in the config. 未指定参考音频且配置文件无预设"
    #     logging.error(message)
    #     return make_response(jsonify(
    #         {"status": "error", "message": message}),
    #         400)

    logger.info(
        f"[{ModelType.GPT_SOVITS.value}] prompt_text:{prompt_text} prompt_lang:{prompt_lang} ")

    # if use_streaming and format.upper() != "MP3":
    #     format = "mp3"
    #     logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    state = {"text": text,
             "id": id,
             "format": format,
             "segment_size": segment_size,
             "lang": lang,
             "speaker_lang": speaker_lang,
             "reference_audio": reference_audio,
             "reference_audio_sr": reference_audio_sr,
             "prompt_text": prompt_text,
             "prompt_lang": prompt_lang,
             "top_k": top_k,
             "top_p": top_p,
             "temperature": temperature,
             }

    t1 = time.time()
    audio = tts_manager.gpt_sovits_infer(state)
    t2 = time.time()
    logger.info(f"[{ModelType.GPT_SOVITS.value}] finish in {(t2 - t1):.2f}s")

    if config.system.cache_audio:
        logger.debug(f"[{ModelType.GPT_SOVITS.value}] {fname}")
        path = os.path.join(config.system.cache_path, fname)
        save_audio(audio.getvalue(), path)

    return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/check', methods=["GET", "POST"])
def check():
    try:
        if request.method == "GET":
            request_data = request.args
        elif request.method == "POST":
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                request_data = request.get_json()
            else:
                request_data = request.form

        model_type_str = request_data.get("model_type", request_data.get("model")).upper()
        id = int(request_data.get("id"))
    except Exception as e:
        logger.info(f"[check] {e}")
        return make_response(jsonify({"status": "error", "message": "parameter error"}), 400)

    if check_is_none(model_type_str):
        logger.info(f"[check] model {model_type_str} is empty")
        return make_response(jsonify({"status": "error", "message": "model is empty"}), 400)

    if model_type_str not in ModelType._value2member_map_:
        res = make_response(jsonify({"status": "error", "message": f"model {model_type_str} does not exist"}))
        res.status = 404
        logger.info(f"[check] speaker id {id} error")
        return res

    if check_is_none(id):
        logger.info(f"[check] speaker id is empty")
        return make_response(jsonify({"status": "error", "message": "speaker id is empty"}), 400)

    model_type = ModelType(model_type_str)
    speaker_list = model_manager.voice_speakers[model_type.value]

    if len(speaker_list) == 0:
        logger.info(f"[check] {model_type_str} not loaded")
        return make_response(jsonify({"status": "error", "message": f"{model_type_str} not loaded"}), 400)

    if id < 0 or id >= len(speaker_list):
        logger.info(f"[check] speaker id {id} does not exist")
        return make_response(jsonify({"status": "error", "message": f"id {id} does not exist"}), 400)
    name = str(speaker_list[id]["name"])
    lang = speaker_list[id]["lang"]
    logger.info(f"[check] check id:{id} name:{name} lang:{lang}")

    return make_response(jsonify({"status": "success", "id": id, "name": name, "lang": lang}), 200)
