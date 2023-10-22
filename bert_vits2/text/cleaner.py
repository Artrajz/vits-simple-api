import importlib
from bert_vits2.text import cleaned_text_to_sequence

language_module_map = {
    'zh': "bert_vits2.text.chinese",
    'ja': "bert_vits2.text.japanese"
}

_loaded_modules = {}


def get_language_module(language):
    if language not in _loaded_modules:
        module_path = language_module_map.get(language)
        if not module_path:
            raise ValueError(f"Unsupported language: {language}")

        _loaded_modules[language] = importlib.import_module(module_path)

    return _loaded_modules[language]


def clean_text(text, language):
    language_module = get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == '__main__':
    pass
