import gc
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import config
from utils.download import download_file
from .chinese_bert import get_bert_feature as zh_bert
from .english_bert_mock import get_bert_feature as en_bert
from .japanese_bert import get_bert_feature as ja_bert
from .japanese_bert_v111 import get_bert_feature as ja_bert_v111
from .japanese_bert_v200 import get_bert_feature as ja_bert_v200
from .english_bert_mock_v200 import get_bert_feature as en_bert_v200


class BertHandler:
    def __init__(self, device):
        self.bert_model_path = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": os.path.join(config.ABS_PATH,
                                                          "bert_vits2/bert/chinese-roberta-wwm-ext-large"),
            "BERT_BASE_JAPANESE_V3": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-base-japanese-v3"),
            "BERT_LARGE_JAPANESE_V2": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-large-japanese-v2"),
            "DEBERTA_V2_LARGE_JAPANESE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese"),
            "DEBERTA_V3_LARGE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v3-large"),
            "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM":os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese-char-wwm")
        }
        self.lang_bert_func_map = {"zh": zh_bert, "en": en_bert, "ja": ja_bert, "ja_v111": ja_bert_v111,
                                   "ja_v200": ja_bert_v200, "en_v200": en_bert_v200}

        self.bert_models = {}  # Value: (tokenizer, model, reference_count)
        self.device = device

    def _download_model(self, bert_model_name, target_path=None):
        DOWNLOAD_PATHS = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": [
                "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
            ],
            "BERT_BASE_JAPANESE_V3": [
                "https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin",
            ],
            "BERT_LARGE_JAPANESE_V2": [
                "https://huggingface.co/cl-tohoku/bert-large-japanese-v2/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/cl-tohoku/bert-large-japanese-v2/resolve/main/pytorch_model.bin",
            ],
            "DEBERTA_V2_LARGE_JAPANESE": [
                "https://huggingface.co/ku-nlp/deberta-v2-large-japanese/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/ku-nlp/deberta-v2-large-japanese/resolve/main/pytorch_model.bin",
            ],
            "DEBERTA_V3_LARGE": [
                "https://huggingface.co/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin",
            ],
            "SPM": [
                "https://huggingface.co/microsoft/deberta-v3-large/resolve/main/spm.model",
                "https://hf-mirror.com/microsoft/deberta-v3-large/resolve/main/spm.model",
            ],
            "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM": [
                "https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/ku-nlp/deberta-v2-large-japanese-char-wwm/resolve/main/pytorch_model.bin",
            ]
        }

        SHA256 = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": "4ac62d49144d770c5ca9a5d1d3039c4995665a080febe63198189857c6bd11cd",
            "BERT_BASE_JAPANESE_V3": "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4",
            "BERT_LARGE_JAPANESE_V2": "50212d714f79af45d3e47205faa356d0e5030e1c9a37138eadda544180f9e7c9",
            "DEBERTA_V2_LARGE_JAPANESE": "a6c15feac0dea77ab8835c70e1befa4cf4c2137862c6fb2443b1553f70840047",
            "DEBERTA_V3_LARGE": "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5",
            "SPM": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
            "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM": "bf0dab8ad87bd7c22e85ec71e04f2240804fda6d33196157d6b5923af6ea1201"
        }
        urls = DOWNLOAD_PATHS[bert_model_name]

        if target_path is None:
            target_path = os.path.join(self.bert_model_path[bert_model_name], "pytorch_model.bin")

        expected_sha256 = SHA256[bert_model_name]
        success, message = download_file(urls, target_path, expected_sha256=expected_sha256)
        if not success:
            logging.error(f"Failed to download {bert_model_name}: {message}")
        else:
            logging.info(f"{message}")

    def load_bert(self, bert_model_name, max_retries=3):
        if bert_model_name not in self.bert_models:
            retries = 0
            while retries < max_retries:
                model_path = self.bert_model_path[bert_model_name]
                logging.info(f"Loading BERT model: {model_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
                    self.bert_models[bert_model_name] = (tokenizer, model, 1)  # 初始化引用计数为1\
                    logging.info(f"Success loading: {model_path}")
                    break
                except Exception as e:
                    logging.error(f"Failed loading {model_path}. {e}")
                    logging.info(f"Trying to download.")
                    if bert_model_name == "DEBERTA_V3_LARGE" and not os.path.exists(
                            os.path.join(model_path, "spm.model")):
                        self._download_model("SPM", os.path.join(model_path, "spm.model"))
                    self._download_model(bert_model_name)
                    retries += 1
            if retries == max_retries:
                logging.error(f"Failed to load {model_path} after {max_retries} retries.")
        else:
            tokenizer, model, count = self.bert_models[bert_model_name]
            self.bert_models[bert_model_name] = (tokenizer, model, count + 1)

    def get_bert_model(self, bert_model_name):
        if bert_model_name not in self.bert_models:
            self.load_bert(bert_model_name)

        tokenizer, model, _ = self.bert_models[bert_model_name]
        return tokenizer, model

    def get_bert_feature(self, norm_text, word2ph, language, bert_model_name):
        tokenizer, model = self.get_bert_model(bert_model_name)
        bert_feature = self.lang_bert_func_map[language](norm_text, word2ph, tokenizer, model, self.device)
        return bert_feature

    def release_bert(self, bert_model_name):
        if bert_model_name in self.bert_models:
            _, _, count = self.bert_models[bert_model_name]
            count -= 1
            if count == 0:
                # 当引用计数为0时，删除模型并释放其资源
                del self.bert_models[bert_model_name]
                gc.collect()
                torch.cuda.empty_cache()
                logging.info(f"BERT model {bert_model_name} has been released.")
            else:
                tokenizer, model = self.bert_models[bert_model_name][:2]
                self.bert_models[bert_model_name] = (tokenizer, model, count)

    def is_model_loaded(self, bert_model_name):
        return bert_model_name in self.bert_models

    def reference_count(self, bert_model_name):
        return self.bert_models[bert_model_name][2] if bert_model_name in self.bert_models else 0
