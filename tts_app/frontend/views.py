from flask import Blueprint, render_template

from tts_app.voice_api.views import model_manager

frontend_blueprint = Blueprint('frontend', __name__)


@frontend_blueprint.route('/', methods=["GET", "POST"])
def index():
    kwargs = {
        "speakers": model_manager.voice_speakers,
        "speakers_count": model_manager.speakers_count,
        "vits_speakers_count": model_manager.vits_speakers_count,
        "w2v2_speakers_count": model_manager.w2v2_speakers_count,
        "w2v2_emotion_count": model_manager.w2v2_emotion_count,
        "bert_vits2_speakers_count": model_manager.bert_vits2_speakers_count
    }
    return render_template("index.html", **kwargs)
