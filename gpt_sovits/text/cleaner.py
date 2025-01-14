from . import chinese as chinese_v1
from . import chinese2 as chinese_v2
from . import japanese
from . import english
from . import cantonese
from . import korean

from . import symbols as symbols_v1
from . import symbols2 as symbols_v2

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

language_module_map = {
    'v1': {
        'zh': chinese_v1,
        'ja': japanese,
        'en': english,
    },
    'v2': {
        'zh': chinese_v2,
        'ja': japanese,
        'en': english,
        'yue': cantonese,
        'ko': korean,
    },
}


def get_language_module(language, version='v2'):
    return language_module_map[version][language]


def get_symbols(version='v1'):
    if version == "v1":
        return symbols_v1.symbols
    else:
        return symbols_v2.symbols


def clean_text(text, language, version='v2', pinyin_g2pw=None):
    language_module = get_language_module(language, version)
    symbols = get_symbols(version)

    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)

    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "zh" or language == "yue":
        phones, word2ph = language_module.g2p(norm_text, pinyin_g2pw=pinyin_g2pw)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ['UNK' if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol, version='v2'):
    language_module = get_language_module(language, version)
    symbols = get_symbols(version)

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


# def text_to_sequence(text, language, version='v2'):
#     phones = clean_text(text, version)
#     return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
