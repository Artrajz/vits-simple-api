import logging
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.config_manager import global_config as config
from logger import logger
from utils.download import download_and_verify
from .chinese_bert import get_bert_feature as zh_bert
from .english_bert_mock import get_bert_feature as en_bert
from .japanese_bert import get_bert_feature as ja_bert


class BertHandler:
    def __init__(self, device):
        self.bert_model_path = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": os.path.join(config.ABS_PATH,
                                                          "bert_vits2/bert/chinese-roberta-wwm-ext-large"),
            "BERT_BASE_JAPANESE_V3": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-base-japanese-v3"),
            "BERT_LARGE_JAPANESE_V2": os.path.join(config.ABS_PATH, "bert_vits2/bert/bert-large-japanese-v2"),
            "DEBERTA_V2_LARGE_JAPANESE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v2-large-japanese"),
            "DEBERTA_V3_LARGE": os.path.join(config.ABS_PATH, "bert_vits2/bert/deberta-v3-large")
        }
        self.lang_bert_func_map = {"zh": zh_bert, "en": en_bert, "ja": ja_bert}

        self.bert_models = {}  # Value: (tokenizer, model, reference_count)
        self.device = device

    def _download_model(self, bert_model_name):
        DOWNLOAD_PATHS = {
            "CHINESE_ROBERTA_WWM_EXT_LARGE": [
                "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
                "https://openi.pcl.ac.cn/Stardust_minus/Bert-VITS2/modelmanage/61b6e7f7-65f7-413c-8c3b-9cbd6996c935/downloadsingle?parentDir=&fileName=pytorch_model.bin", ],
            "BERT_BASE_JAPANESE_V3": [
                "https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin",
                "https://openi.pcl.ac.cn/Stardust_minus/Bert-VITS2/modelmanage/d77420a6-3438-412f-9199-69c6342ffb06/downloadsingle?parentDir=&fileName=pytorch_model.bin", ],
            "BERT_LARGE_JAPANESE_V2": [
                "https://huggingface.co/cl-tohoku/bert-large-japanese-v2/blob/main/pytorch_model.bin", ],
            "DEBERTA_V2_LARGE_JAPANESE": [
                "https://huggingface.co/ku-nlp/deberta-v2-large-japanese/blob/main/pytorch_model.bin", ],
            "DEBERTA_V3_LARGE": ["https://huggingface.co/microsoft/deberta-v3-large/blob/main/pytorch_model.bin", ],

        }
        urls = DOWNLOAD_PATHS[bert_model_name]
        target_path = os.path.join(self.bert_model_path[bert_model_name], "pytorch_model.bin"),

        if not os.path.exists(target_path):
            success, message = download_and_verify(urls, target_path, None)
            if not success:
                logger.error(f"Failed to download {bert_model_name}: {message}")

    def load_bert(self, bert_model_name):
        if bert_model_name not in self.bert_models:
            model_path = self.bert_model_path[bert_model_name]
            if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                self._download_model(bert_model_name)
            logging.info(f"Loading BERT model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
            self.bert_models[bert_model_name] = (tokenizer, model, 1)  # 初始化引用计数为1
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
                logger(f"Model {bert_model_name} has been released.")
            else:
                tokenizer, model = self.bert_models[bert_model_name][:2]
                self.bert_models[bert_model_name] = (tokenizer, model, count)

    def is_model_loaded(self, bert_model_name):
        return bert_model_name in self.bert_models

    def reference_count(self, bert_model_name):
        return self.bert_models[bert_model_name][2] if bert_model_name in self.bert_models else 0
