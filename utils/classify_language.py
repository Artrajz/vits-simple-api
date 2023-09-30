from config import LANGUAGE_IDENTIFICATION_LIBRARY

module = LANGUAGE_IDENTIFICATION_LIBRARY.lower()


def classify_language(text: str, target_languages: list = None) -> str:
    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid
        classifier = fastlid
        if target_languages != None: fastlid.set_languages = target_languages
    elif module == "langid":
        import langid
        classifier = langid.classify
        if target_languages != None: langid.set_languages(target_languages)
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


if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))
    print(classify_zh_ja(text))  # "zh"

    text = "これはテストテキストです"
    print(classify_language(text))
    print(classify_zh_ja(text))  # "ja"
