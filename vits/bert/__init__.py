""" from https://github.com/PlayVoice/vits_chinese """
import os

from contants import config
from utils.download import download_file
from .ProsodyModel import TTSProsody

URLS = [
    "https://huggingface.co/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt",
    "https://hf-mirror.com/spaces/maxmax20160403/vits_chinese/resolve/main/bert/prosody_model.pt"
]
TARGET_PATH = os.path.join(config.abs_path, "vits/bert/prosody_model.pt")
EXPECTED_SHA256 = "3accec7a0d5cbfccaa8a42b96374a91d442a69801c6a01402baae3bf06b8c015"

if not os.path.exists(TARGET_PATH):
    success, message = download_file(URLS, TARGET_PATH, EXPECTED_SHA256)