import gc
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, MegatronBertModel

import config
from utils.download import download_file
from bert_vits2.text.chinese_bert import get_bert_feature as zh_bert
from bert_vits2.text.english_bert_mock import get_bert_feature as en_bert
from bert_vits2.text.japanese_bert import get_bert_feature as ja_bert
from bert_vits2.text.japanese_bert_v111 import get_bert_feature as ja_bert_v111
from bert_vits2.text.japanese_bert_v200 import get_bert_feature as ja_bert_v200
from bert_vits2.text.english_bert_mock_v200 import get_bert_feature as en_bert_v200
from bert_vits2.text.chinese_bert_extra import get_bert_feature as zh_bert_extra


class ModelHandler:
    def __init__(self, device):
        self.DOWNLOAD_PATHS = {
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
            ],
            "WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM": [
                "https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim/resolve/main/pytorch_model.bin",
            ],
            "CLAP_HTSAT_FUSED": [
                "https://huggingface.co/laion/clap-htsat-fused/resolve/main/pytorch_model.bin?download=true",
                "https://hf-mirror.com/laion/clap-htsat-fused/resolve/main/pytorch_model.bin?download=true",
            ],
            "Erlangshen-MegatronBert-1.3B-Chinese": [
                "https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-MegatronBERT-1.3B-Chinese/resolve/main/pytorch_model.bin",
                "https://hf-mirror.com/IDEA-CCNL/Erlangshen-UniMC-MegatronBERT-1.3B-Chinese/resolve/main/pytorch_model.bin",
            ]
        }

        self.SHA256 = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": "4ac62d49144d770c5ca9a5d1d3039c4995665a080febe63198189857c6bd11cd",
            "BERT_BASE_JAPANESE_V3": "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4",
            "BERT_LARGE_JAPANESE_V2": "50212d714f79af45d3e47205faa356d0e5030e1c9a37138eadda544180f9e7c9",
            "DEBERTA_V2_LARGE_JAPANESE": "a6c15feac0dea77ab8835c70e1befa4cf4c2137862c6fb2443b1553f70840047",
            "DEBERTA_V3_LARGE": "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5",
            "SPM": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
            "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM": "bf0dab8ad87bd7c22e85ec71e04f2240804fda6d33196157d6b5923af6ea1201",
            "WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM": "176d9d1ce29a8bddbab44068b9c1c194c51624c7f1812905e01355da58b18816",
            "CLAP_HTSAT_FUSED": "1ed5d0215d887551ddd0a49ce7311b21429ebdf1e6a129d4e68f743357225253",
            "Erlangshen-MegatronBert-1.3B-Chinese": "3456bb8f2c7157985688a4cb5cecdb9e229cb1dcf785b01545c611462ffe3579",
        }
        self.model_path = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": os.path.join(config.ABS_PATH,
                                                          "bert_vits2/bert/chinese-roberta-wwm-ext-large"),
            "BERT_BASE_JAPANESE_V3": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-base-japanese-v3"),
            "BERT_LARGE_JAPANESE_V2": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-large-japanese-v2"),
            "DEBERTA_V2_LARGE_JAPANESE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese"),
            "DEBERTA_V3_LARGE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v3-large"),
            "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM": os.path.join(config.ABS_PATH,
                                                               "bert_vits2/bert/deberta-v2-large-japanese-char-wwm"),
            "WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM": os.path.join(config.ABS_PATH,
                                                                        "bert_vits2/emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
            "CLAP_HTSAT_FUSED": os.path.join(config.ABS_PATH, "bert_vits2/emotional/clap-htsat-fused"),
            "Erlangshen-MegatronBert-1.3B-Chinese": os.path.join(config.ABS_PATH,
                                                                 "bert_vits2/bert/Erlangshen-MegatronBert-1.3B-Chinese"),
        }

        self.lang_bert_func_map = {"zh": zh_bert, "en": en_bert, "ja": ja_bert, "ja_v111": ja_bert_v111,
                                   "ja_v200": ja_bert_v200, "en_v200": en_bert_v200, "zh_extra": zh_bert_extra}

        self.bert_models = {}  # Value: (tokenizer, model, reference_count)
        self.emotion = None
        self.clap = None
        self.device = device

    @property
    def emotion_model(self):
        return self.emotion["model"]

    @property
    def emotion_processor(self):
        return self.emotion["processor"]

    @property
    def clap_model(self):
        return self.clap["model"]

    @property
    def clap_processor(self):
        return self.clap["processor"]

    def _download_model(self, model_name, target_path=None):
        urls = self.DOWNLOAD_PATHS[model_name]

        if target_path is None:
            target_path = os.path.join(self.model_path[model_name], "pytorch_model.bin")

        expected_sha256 = self.SHA256[model_name]
        success, message = download_file(urls, target_path, expected_sha256=expected_sha256)
        if not success:
            logging.error(f"Failed to download {model_name}: {message}")
        else:
            logging.info(f"{message}")

    def load_bert(self, bert_model_name, max_retries=3):
        if bert_model_name not in self.bert_models:
            retries = 0
            while retries < max_retries:
                model_path = self.model_path[bert_model_name]
                logging.info(f"Loading BERT model: {model_path}")
                try:
                    if bert_model_name == "bert_model_name":
                        tokenizer = BertTokenizer.from_pretrained(model_path)
                        model = MegatronBertModel.from_pretrained(model_path).to(self.device)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
                    self.bert_models[bert_model_name] = (tokenizer, model, 1)  # 初始化引用计数为1
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

    def load_emotion(self, max_retries=3):
        """Bert-VITS2 v2.1 EmotionModel"""
        if self.emotion is None:
            from transformers import Wav2Vec2Processor
            from bert_vits2.get_emo import EmotionModel
            retries = 0
            model_path = self.model_path["WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM"]
            while retries < max_retries:
                logging.info(f"Loading WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM: {model_path}")
                try:
                    self.emotion = {}
                    self.emotion["model"] = EmotionModel.from_pretrained(model_path).to(self.device)
                    self.emotion["processor"] = Wav2Vec2Processor.from_pretrained(model_path)
                    self.emotion["reference_count"] = 1
                    logging.info(f"Success loading: {model_path}")
                    break
                except Exception as e:
                    logging.error(f"Failed loading {model_path}. {e}")
                    self._download_model("WAV2VEC2_LARGE_ROBUST_12_FT_EMOTION_MSP_DIM")
                    retries += 1
            if retries == max_retries:
                logging.error(f"Failed to load {model_path} after {max_retries} retries.")
        else:
            self.emotion["reference_count"] += 1

    def load_clap(self, max_retries=3):
        """Bert-VITS2 v2.2 ClapModel"""
        if self.clap is None:
            from transformers import ClapModel, ClapProcessor
            retries = 0
            model_path = self.model_path["CLAP_HTSAT_FUSED"]
            while retries < max_retries:
                logging.info(f"Loading CLAP_HTSAT_FUSED: {model_path}")
                try:
                    self.clap = {}
                    self.clap["model"] = ClapModel.from_pretrained(model_path).to(self.device)
                    self.clap["processor"] = ClapProcessor.from_pretrained(model_path)
                    self.clap["reference_count"] = 1
                    logging.info(f"Success loading: {model_path}")
                    break
                except Exception as e:
                    logging.error(f"Failed loading {model_path}. {e}")
                    self._download_model("CLAP_HTSAT_FUSED")
                    retries += 1
            if retries == max_retries:
                logging.error(f"Failed to load {model_path} after {max_retries} retries.")
        else:
            self.clap["reference_count"] += 1

    def get_bert_model(self, bert_model_name):
        if bert_model_name not in self.bert_models:
            self.load_bert(bert_model_name)

        tokenizer, model, _ = self.bert_models[bert_model_name]
        return tokenizer, model

    def get_bert_feature(self, norm_text, word2ph, language, bert_model_name, style_text=None, style_weight=0.7):
        tokenizer, model = self.get_bert_model(bert_model_name)
        bert_feature = self.lang_bert_func_map[language](norm_text, word2ph, tokenizer, model, self.device,
                                                         style_text=style_text, style_weight=style_weight)
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
