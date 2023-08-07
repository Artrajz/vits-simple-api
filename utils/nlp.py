import regex as re
import config
from .utils import check_is_none
from logger import logger


def clasify_lang(text, speaker_lang):
    pattern = r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`' \
              r'\！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」' \
              r'『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+'
    words = re.split(pattern, text)

    pre = ""
    p = 0
    for word in words:

        if check_is_none(word): continue

        # 读取配置选择语种识别库
        clf = getattr(config, "LANGUAGE_IDENTIFICATION_LIBRARY", "fastlid")
        if clf.upper() == "FASTLID" or clf.upper() == "FASTTEXT":
            from fastlid import fastlid
            lang = fastlid(word)[0]
            if speaker_lang != None: fastlid.set_languages = speaker_lang
        elif clf.upper() == "LANGID":
            import langid
            lang = langid.classify(word)[0]
            if speaker_lang != None: langid.set_languages(speaker_lang)
        else:
            raise ValueError(f"Wrong LANGUAGE_IDENTIFICATION_LIBRARY in config.py")

        if pre == "":
            text = text[:p] + text[p:].replace(word, f'[{lang.upper()}]' + word, 1)
            p += len(f'[{lang.upper()}]')
        elif pre != lang:
            text = text[:p] + text[p:].replace(word, f'[{pre.upper()}][{lang.upper()}]' + word, 1)
            p += len(f'[{pre.upper()}][{lang.upper()}]')
        pre = lang
        p += text[p:].index(word) + len(word)
    text += f"[{pre.upper()}]"

    return text


def cut(text, max):
    pattern = r'[!(),—+\-.:;?？。，、；：]+'
    sentences = re.split(pattern, text)
    discarded_chars = re.findall(pattern, text)

    sentence_list, count, p = [], 0, 0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):
        count += len(sentences[i]) + len(discarded_chars)
        if count >= max:
            sentence_list.append(text[p:p + count].strip())
            p += count
            count = 0

    # 加入最后剩余的文本
    if p < len(text):
        sentence_list.append(text[p:])

    return sentence_list


def sentence_split(text, max=50, lang="auto", speaker_lang=None):
    # 如果该speaker只支持一种语言
    if speaker_lang is not None and len(speaker_lang) == 1:
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            logger.debug(
                f"lang \"{lang}\" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}")
        lang = speaker_lang[0]

    sentence_list = []
    if lang.upper() != "MIX":
        if max <= 0:
            sentence_list.append(
                clasify_lang(text,
                             speaker_lang) if lang.upper() == "AUTO" else f"[{lang.upper()}]{text}[{lang.upper()}]")
        else:
            for i in cut(text, max):
                if check_is_none(i): continue
                sentence_list.append(
                    clasify_lang(i,
                                 speaker_lang) if lang.upper() == "AUTO" else f"[{lang.upper()}]{i}[{lang.upper()}]")
    else:
        sentence_list.append(text)

    for i in sentence_list:
        logger.debug(i)

    return sentence_list
