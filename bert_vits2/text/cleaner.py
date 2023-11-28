from bert_vits2.text import chinese, japanese, english, cleaned_text_to_sequence, japanese_v111, chinese_v100, \
    japanese_v200, english_v200

language_module_map = {
    'zh': chinese,
    'ja': japanese,
    'en': english,
    'ja_v111': japanese_v111,
    'zh_v100': chinese_v100,
    'ja_v200': japanese_v200,
    'en_v200': english_v200
}


# _loaded_modules = {}
# 
# 
# def get_language_module(language):
#     if language not in _loaded_modules:
#         module_path = language_module_map.get(language)
#         if not module_path:
#             raise ValueError(f"Unsupported language: {language}")
# 
#         _loaded_modules[language] = importlib.import_module(module_path)
# 
#     return _loaded_modules[language]


def clean_text(text, language, tokenizer):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text, tokenizer=tokenizer)
    return norm_text, phones, tones, word2ph


# def clean_text_bert(text, language, tokenizer):
#     language_module = language_module_map[language]
#     norm_text = language_module.text_normalize(text)
#     phones, tones, word2ph = language_module.g2p(norm_text, tokenizer)
#     bert = language_module.get_bert_feature(norm_text, word2ph)
#     return phones, tones, bert


def text_to_sequence(text, language, tokenizer):
    norm_text, phones, tones, word2ph = clean_text(text, language, tokenizer)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == '__main__':
    pass
