import regex as re
from fastlid import fastlid

fastlid.set_languages = ["zh", "ja"]


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


def cut(text,max=50):
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
    if p < len(sentence):
        sentence_list.append(text[p:])
    return sentence_list


def sentence_split(text):
    sentence_list = []
    for i in cut(text):
        if check_is_none(i): continue
        sentence_list.append(clasify_lang(i))
    return sentence_list


# is none -> True,is not none -> False
def check_is_none(s):
    s = str(s)
    if s == None or s == "" or s.isspace():
        return True
    return False
