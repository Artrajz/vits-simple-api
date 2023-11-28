import os
import time
import uuid
from io import BytesIO

from flask import request, jsonify, make_response, send_file, Blueprint, current_app
from werkzeug.utils import secure_filename

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


@voice_api.route('/default_parameter', methods=["GET", "POST"])
def default_parameter():
    return jsonify(current_app.config.get("default_parameter"))


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
        id = get_param(request_data, "id", current_app.config.get("ID", 0), int)
        format = get_param(request_data, "format", current_app.config.get("FORMAT", "wav"), str)
        lang = get_param(request_data, "lang", current_app.config.get("LANG", "auto"), str).lower()
        length = get_param(request_data, "length", current_app.config.get("LENGTH", 1), float)
        noise = get_param(request_data, "noise", current_app.config.get("NOISE", 0.667), float)
        noisew = get_param(request_data, "noisew", current_app.config.get("NOISEW", 0.8), float)
        segment_size = get_param(request_data, "segment_size", current_app.config.get("SEGMENT_SIZE", 50), int)
        use_streaming = get_param(request_data, 'streaming', False, bool)
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
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if current_app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = current_app.config.get("LANGUAGE_AUTOMATIC_DETECT")

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

        if current_app.config.get("SAVE_AUDIO", False):
            logger.debug(f"[{ModelType.VITS.value}] {fname}")
            path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
            save_audio(audio.getvalue(), path)

        return send_file(path_or_file=audio, mimetype=file_type, download_name=fname)


@voice_api.route('/hubert-vits', methods=["POST"])
@require_api_key
def voice_hubert_api():
    if request.method == "POST":
        try:
            voice = request.files['upload']
            id = get_param(request.form, "id", 0, int)
            format = get_param(request.form, "format", current_app.config.get("LANG", "auto"))
            length = get_param(request.form, "length", current_app.config.get("LENGTH", 1), float)
            noise = get_param(request.form, "noise", current_app.config.get("NOISE", 0.667), float)
            noisew = get_param(request.form, "noisew", current_app.config.get("NOISEW", 0.8), float)
            use_streaming = get_param(request.form, 'streaming', False, bool)
        except Exception as e:
            logger.error(f"[{ModelType.HUBERT_VITS.value}] {e}")
            return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.HUBERT_VITS.value}] id:{id} format:{format} length:{length} noise:{noise} noisew:{noisew}")

    fname = secure_filename(str(uuid.uuid1()) + "." + voice.filename.split(".")[1])
    voice.save(os.path.join(current_app.config['UPLOAD_FOLDER'], fname))

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
            "audio_path": os.path.join(current_app.config['UPLOAD_FOLDER'], fname)}

    t1 = time.time()
    audio = tts_manager.hubert_vits_infer(task)
    t2 = time.time()
    logger.info(f"[{ModelType.HUBERT_VITS.value}] finish in {(t2 - t1):.2f}s")

    if current_app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.HUBERT_VITS.value}] {fname}")
        path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
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
        id = get_param(request_data, "id", current_app.config.get("ID", 0), int)
        format = get_param(request_data, "format", current_app.config.get("FORMAT", "wav"), str)
        lang = get_param(request_data, "lang", current_app.config.get("LANG", "auto"), str).lower()
        length = get_param(request_data, "length", current_app.config.get("LENGTH", 1), float)
        noise = get_param(request_data, "noise", current_app.config.get("NOISE", 0.667), float)
        noisew = get_param(request_data, "noisew", current_app.config.get("NOISEW", 0.8), float)
        segment_size = get_param(request_data, "segment_size", current_app.config.get("SEGMENT_SIZE", 50), int)
        emotion = get_param(request_data, "emotion", current_app.config.get("EMOTION", 0), int)
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
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.W2V2_VITS.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if current_app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = current_app.config.get("LANGUAGE_AUTOMATIC_DETECT")

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

    if current_app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.W2V2_VITS.value}] {fname}")
        path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
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
        audio_path = os.path.join(current_app.config['UPLOAD_FOLDER'], fname)
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

        if current_app.config.get("SAVE_AUDIO", False):
            logger.debug(f"[Voice conversion] {fname}")
            path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
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

    if current_app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[ssml] {fname}")
        path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
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
        id = get_param(request_data, "id", current_app.config.get("ID", 0), int)
        format = get_param(request_data, "format", current_app.config.get("FORMAT", "wav"), str)
        lang = get_param(request_data, "lang", current_app.config.get("LANG", "auto"), str).lower()
        length = get_param(request_data, "length", current_app.config.get("LENGTH", 1), float)
        length_zh = get_param(request_data, "length_zh", current_app.config.get("LENGTH_ZH", 0), float)
        length_ja = get_param(request_data, "length_ja", current_app.config.get("LENGTH_JA", 0), float)
        length_en = get_param(request_data, "length_en", current_app.config.get("LENGTH_EN", 0), float)
        noise = get_param(request_data, "noise", current_app.config.get("NOISE", 0.667), float)
        noisew = get_param(request_data, "noisew", current_app.config.get("NOISEW", 0.8), float)
        sdp_ratio = get_param(request_data, "sdp_ratio", current_app.config.get("SDP_RATIO", 0.2), float)
        segment_size = get_param(request_data, "segment_size", current_app.config.get("SEGMENT_SIZE", 50), int)
        use_streaming = get_param(request_data, 'streaming', False, bool)
        emotion = get_param(request_data, 'emotion', None, int)
        reference_audio = request.files.get("reference_audio", None)
    except Exception as e:
        logger.error(f"[{ModelType.BERT_VITS2.value}] {e}")
        return make_response("parameter error", 400)

    logger.info(
        f"[{ModelType.BERT_VITS2.value}] id:{id} format:{format} lang:{lang} length:{length} noise:{noise} noisew:{noisew} sdp_ratio:{sdp_ratio} segment_size:{segment_size}"
        f" length_zh:{length_zh} length_ja:{length_ja} length_en:{length_en}")
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
    if lang not in ["auto", "mix"] and len(speaker_lang) != 1 and lang not in speaker_lang:
        logger.info(f"[{ModelType.BERT_VITS2.value}] lang \"{lang}\" is not in {speaker_lang}")
        return make_response(jsonify({"status": "error", "message": f"lang '{lang}' is not in {speaker_lang}"}),
                             400)

    # 如果配置文件中设置了LANGUAGE_AUTOMATIC_DETECT则强制将speaker_lang设置为LANGUAGE_AUTOMATIC_DETECT
    if current_app.config.get("LANGUAGE_AUTOMATIC_DETECT", []) != []:
        speaker_lang = current_app.config.get("LANGUAGE_AUTOMATIC_DETECT")

    if use_streaming and format.upper() != "MP3":
        format = "mp3"
        logger.warning("Streaming response only supports MP3 format.")

    fname = f"{str(uuid.uuid1())}.{format}"
    file_type = f"audio/{format}"
    state = {"text": text,
             "id": id,
             "format": format,
             "length": length,
             "length_zh": length_zh,
             "length_ja": length_ja,
             "length_en": length_en,
             "noise": noise,
             "noisew": noisew,
             "sdp_ratio": sdp_ratio,
             "segment_size": segment_size,
             "lang": lang,
             "speaker_lang": speaker_lang,
             "emotion": emotion,
             "reference_audio": reference_audio}

    if use_streaming:
        audio = tts_manager.stream_bert_vits2_infer(state)
        response = make_response(audio)
        response.headers['Content-Disposition'] = f'attachment; filename={fname}'
        response.headers['Content-Type'] = file_type
        return response
    else:
        t1 = time.time()
        audio = tts_manager.bert_vits2_infer(state)
        t2 = time.time()
        logger.info(f"[{ModelType.BERT_VITS2.value}] finish in {(t2 - t1):.2f}s")

    if current_app.config.get("SAVE_AUDIO", False):
        logger.debug(f"[{ModelType.BERT_VITS2.value}] {fname}")
        path = os.path.join(current_app.config.get('CACHE_PATH'), fname)
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
