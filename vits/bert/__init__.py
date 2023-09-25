""" from https://github.com/PlayVoice/vits_chinese """
import os

import config
from utils.download import download_and_verify
from .ProsodyModel import TTSProsody

URLS = [
    "https://huggingface.co/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt",
]
TARGET_PATH = os.path.join(config.ABS_PATH, "vits/bert/prosody_model.pt")
EXPECTED_MD5 = None

if not os.path.exists(TARGET_PATH):
    success, message = download_and_verify(URLS, TARGET_PATH, EXPECTED_MD5)