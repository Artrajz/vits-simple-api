import importlib


class BertHandler:
    _bert_functions = {}

    BERT_IMPORT_MAP = {
        "zh": "bert_vits2.text.chinese_bert.get_bert_feature",
        "en": "bert_vits2.text.english_bert_mock.get_bert_feature",
        "ja": "bert_vits2.text.japanese_bert.get_bert_feature",
    }

    def __init__(self, languages):
        for lang in languages:
            if lang not in BertHandler._bert_functions:
                self.load_bert_function(lang)

    def load_bert_function(self, language):
        if language not in BertHandler.BERT_IMPORT_MAP:
            raise ValueError(f"Unsupported language: {language}")

        module_path, function_name = BertHandler.BERT_IMPORT_MAP[language].rsplit('.', 1)
        module = importlib.import_module(module_path, package=__package__)
        bert_function = getattr(module, function_name)

        BertHandler._bert_functions[language] = bert_function

    def get_bert(self, norm_text, word2ph, language):
        if language not in BertHandler._bert_functions:
            raise ValueError(f"BERT for {language} has not been initialized. Please initialize first.")

        bert_func = BertHandler._bert_functions[language]
        return bert_func(norm_text, word2ph)
