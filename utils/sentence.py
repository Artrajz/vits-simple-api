import regex as re

from logger import logger
from utils.data_utils import check_is_none
from utils.classify_language import classify_language


def markup_language_type(text: str, target_languages: list = None) -> str:
    pattern = r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`' \
              r'\！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」' \
              r'『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+'
    sentences = re.split(pattern, text)

    pre_lang = ""
    p = 0

    for sentence in sentences:

        if check_is_none(sentence): continue

        lang = classify_language(sentence, target_languages)

        if pre_lang == "":
            text = text[:p] + text[p:].replace(sentence, f"[{lang.upper()}]{sentence}", 1)
            p += len(f"[{lang.upper()}]")
        elif pre_lang != lang:
            text = text[:p] + text[p:].replace(sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1)
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")
        pre_lang = lang
        p += text[p:].index(sentence) + len(sentence)
    text += f"[{pre_lang.upper()}]"

    return text


def cut(text: str, max: int) -> list:
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


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
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
                markup_language_type(text,
                                     speaker_lang) if lang.upper() == "AUTO" else f"[{lang.upper()}]{text}[{lang.upper()}]")
        else:
            for i in cut(text, max):
                if check_is_none(i): continue
                sentence_list.append(
                    markup_language_type(i,
                                         speaker_lang) if lang.upper() == "AUTO" else f"[{lang.upper()}]{i}[{lang.upper()}]")
    else:
        sentence_list.append(text)

    for i in sentence_list:
        logger.debug(i)

    return sentence_list


if __name__ == '__main__':
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language_type(text, languages=None))
    print(cut(text, max=50))
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))
