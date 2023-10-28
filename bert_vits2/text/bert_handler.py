# import importlib
# 
# 
# class BertHandler:
#     _bert_functions = {}
# 
#     BERT_IMPORT_MAP = {
#         "zh": "bert_vits2.text.chinese_bert.get_bert_feature",
#         "en": "bert_vits2.text.english_bert_mock.get_bert_feature",
#         "ja": "bert_vits2.text.japanese_bert.get_bert_feature",
#     }
# 
#     def __init__(self, languages):
#         for lang in languages:
#             if lang not in BertHandler._bert_functions:
#                 self.load_bert_function(lang)
# 
#     def load_bert_function(self, language):
#         if language not in BertHandler.BERT_IMPORT_MAP:
#             raise ValueError(f"Unsupported language: {language}")
# 
#         module_path, function_name = BertHandler.BERT_IMPORT_MAP[language].rsplit('.', 1)
#         module = importlib.import_module(module_path, package=__package__)
#         bert_function = getattr(module, function_name)
# 
#         BertHandler._bert_functions[language] = bert_function
# 
#     def get_bert(self, norm_text, word2ph, language):
#         if language not in BertHandler._bert_functions:
#             raise ValueError(f"BERT for {language} has not been initialized. Please initialize first.")
# 
#         bert_func = BertHandler._bert_functions[language]
#         return bert_func(norm_text, word2ph)
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

import config
from logger import logger
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

    def load_bert(self, bert_model_name):
        if bert_model_name not in self.bert_models:
            model_path = self.bert_model_path[bert_model_name]
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
