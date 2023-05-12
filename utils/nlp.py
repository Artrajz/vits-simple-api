import regex as re
import logging
import config
from fastlid import fastlid
from .utils import check_is_none

logger = logging.getLogger("vits-simple-api")
level = getattr(config, "LOGGING_LEVEL", "DEBUG")
level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL}
logger.setLevel(level_dict[level])


def clasify_lang(text):
    pattern = r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`' \
              r'\！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」' \
              r'『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+'
    words = re.split(pattern, text)

    pre = ""
    p = 0
    for word in words:

        if check_is_none(word): continue
        lang = fastlid(word)[0]
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
    pattern = r'[\!\(\)\,\-\.\/\:\;\?\？\。\，\、\；\：]+'
    sentences = re.split(pattern, text)
    sentence_list = []
    count = 0
    p = 0
    for sentence in sentences:
        count += len(sentence) + 1
        if count >= max:
            sentence_list.append(text[p:p + count])
            p += count
            count = 0
    if p < len(text):
        sentence_list.append(text[p:])
    return sentence_list


def sentence_split(text, max=50, lang="auto", speaker_lang=None):
    fastlid.set_languages = speaker_lang

    sentence_list = []
    if lang.upper() != "MIX":
        if max <= 0:
            sentence_list.append(
                clasify_lang(text) if lang.upper() == "AUTO" else f"[{lang.upper()}]{text}[{lang.upper()}]")
        else:
            for i in cut(text, max):
                if check_is_none(i): continue
                sentence_list.append(
                    clasify_lang(i) if lang.upper() == "AUTO" else f"[{lang.upper()}]{i}[{lang.upper()}]")
    else:
        sentence_list.append(text)

    for i in sentence_list:
        logger.debug(i)

    return sentence_list
