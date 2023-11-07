from flask import Blueprint, render_template

from tts_app.model_manager import model_manager

frontend = Blueprint('frontend', __name__)


@frontend.route('/', methods=["GET", "POST"])
def index():
    kwargs = {
        "speakers": model_manager.voice_speakers,
        "speakers_count": model_manager.speakers_count,
        "vits_speakers_count": model_manager.vits_speakers_count,
        "w2v2_speakers_count": model_manager.w2v2_speakers_count,
        "w2v2_emotion_count": model_manager.w2v2_emotion_count,
        "bert_vits2_speakers_count": model_manager.bert_vits2_speakers_count
    }
    return render_template("pages/index.html", **kwargs)
