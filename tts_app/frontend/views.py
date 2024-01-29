from flask import Blueprint, render_template

from contants import ModelType
from tts_app.model_manager import model_manager

frontend = Blueprint('frontend', __name__)


@frontend.route('/webui/<model_id>', methods=["GET", "POST"])
def index(model_id):
    kwargs = {
        "speakers": {ModelType.VITS.value: [],
                     ModelType.W2V2_VITS.value: [],
                     ModelType.BERT_VITS2.value: model_manager.voice_speakers[ModelType.BERT_VITS2.value][model_id]},
        "speakers_count": model_manager.speakers_count,
        "vits_speakers_count": model_manager.vits_speakers_count,
        "w2v2_speakers_count": model_manager.w2v2_speakers_count,
        "w2v2_emotion_count": model_manager.w2v2_emotion_count,
        "bert_vits2_speakers_count": len(model_manager.voice_speakers[ModelType.BERT_VITS2.value][model_id])
    }
    return render_template("pages/index.html", **kwargs)
