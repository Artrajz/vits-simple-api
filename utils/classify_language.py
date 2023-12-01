import regex as re

try:
    from utils.config_manager import global_config
except:
    pass

langid_languages = ["af", "am", "an", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de",
                    "dz", "el",
                    "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "ga", "gl", "gu", "he", "hi", "hr", "ht",
                    "hu", "hy",
                    "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt",
                    "lv", "mg",
                    "mk", "ml", "mn", "mr", "ms", "mt", "nb", "ne", "nl", "nn", "no", "oc", "or", "pa", "pl", "ps",
                    "pt", "qu",
                    "ro", "ru", "rw", "se", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr",
                    "ug", "uk",
                    "ur", "vi", "vo", "wa", "xh", "zh", "zu"]


def classify_language(text: str, target_languages: list = None) -> str:
    try:
        module = global_config["LANGUAGE_IDENTIFICATION_LIBRARY"].lower()
    except:
        module = "langid"
    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid, supported_langs
        classifier = fastlid
        if target_languages != None:
            target_languages = [lang for lang in target_languages if lang in supported_langs]
            fastlid.set_languages = target_languages
    elif module == "langid":
        import langid
        classifier = langid.classify
        if target_languages != None:
            target_languages = [lang for lang in target_languages if lang in langid_languages]
            langid.set_languages(target_languages)
    else:
        raise ValueError(f"Wrong LANGUAGE_IDENTIFICATION_LIBRARY in config.py")

    lang = classifier(text)[0]

    return lang


def classify_zh_ja(text: str) -> str:
    for idx, char in enumerate(text):
        unicode_val = ord(char)

        # 检测日语字符
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
            return "ja"

        # 检测汉字字符
        if 0x4E00 <= unicode_val <= 0x9FFF:
            # 检查周围的字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None

            if next_char and (0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF):
                return "ja"

    return "zh"


def split_alpha_nonalpha(text, mode=1):
    """
    Splits the input text based on the specified mode.

    Parameters:
    - text (str): The input text to be split.
    - mode (int): The mode for splitting (1 or 2).
        - Mode 1: Splits based on the pattern - Chinese/Japanese followed by English or vice versa.
        - Mode 2: Splits based on the pattern - Chinese/Japanese followed by English/digit or vice versa.

    Returns:
    - list: A list of substrings after the split.
    """
    if mode == 1:
        pattern = r'(?<=[\u4e00-\u9fff\u3040-\u30FF\d])(?=[\p{Latin}])|(?<=[\p{Latin}])(?=[\u4e00-\u9fff\u3040-\u30FF\d])'
    elif mode == 2:
        pattern = r'(?<=[\u4e00-\u9fff\u3040-\u30FF])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d])(?=[\u4e00-\u9fff\u3040-\u30FF])'
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    return re.split(pattern, text)


if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))
    print(classify_zh_ja(text))  # "zh"

    text = "これはテストテキストです"
    print(classify_language(text))
    print(classify_zh_ja(text))  # "ja"
