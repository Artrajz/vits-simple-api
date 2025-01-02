import os

import pypinyin
import yaml
import jieba
from pypinyin.contrib.tone_convert import to_tone, to_initials, to_finals_tone3

from config import config, BASE_DIR


# 定义自定义的表示器来确保列表显示在同一行
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


# 自定义表示器来强制使用单引号和流格式
def represent_list(self, data):
    value = self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    for item in value.value:
        item.style = "'"
    return value


def represent_str(self, data):
    if isinstance(data, str):
        return self.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    return self.represent_scalar('tag:yaml.org,2002:str', data)


MyDumper.add_representer(list, represent_list)
MyDumper.add_representer(str, represent_str)


def load_polyphonic_dict(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as polyphonic_file:
        # 解析yaml
        polyphonic_dict = yaml.load(polyphonic_file, Loader=yaml.FullLoader)

    return polyphonic_dict


def save_polyphonic_dict(path: str, polyphonic_dict: dict):
    with open(path, 'w', encoding='utf-8') as polyphonic_file:
        yaml.dump(polyphonic_dict, polyphonic_file, Dumper=MyDumper, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)


class Polyphonic:
    def __init__(self):
        self.path: str = os.path.join(BASE_DIR, config.system.data_path, config.polyphonic.dict_path)
        self.polyphonic_dict = load_polyphonic_dict(self.path)
        self.polyphonic_words = self.polyphonic_dict["polyphonic"]

        for word in self.polyphonic_words.keys():
            jieba.add_word(word)

        pypinyin.load_phrases_dict(self.polyphonic_dict)

    def correct_pronunciation(self, word: str, pinyin: list[list[str]] | tuple[list[str], list[str]], style=1):
        if word in self.polyphonic_words.keys():
            _pinyin = self.polyphonic_words[word]
            if style == 1:
                pinyin: list[list[str]] = [[to_tone(x)] for x in _pinyin]
            elif style == 2:
                initials = [to_initials(x) for x in _pinyin]
                finals = [to_finals_tone3(x, neutral_tone_with_five=True) for x in _pinyin]
                pinyin: tuple[list[str], list[str]] = (initials, finals)
            elif style == 3:  # GPT-SoVITS v2
                pinyin: list[str] = [to_tone(x) for x in _pinyin]

        return pinyin

    def update_polyphonic(self):
        self.polyphonic_dict = load_polyphonic_dict(self.path)
        self.polyphonic_words = self.polyphonic_dict["polyphonic"]

        for word in self.polyphonic_words.keys():
            jieba.add_word(word)

        pypinyin.load_phrases_dict(self.polyphonic_dict)

    def save_polyphonic(self):
        save_polyphonic_dict(self.path, self.polyphonic_dict)


corrector: Polyphonic | None = None


def load_polyphonic():
    global corrector
    if corrector is None:
        corrector = Polyphonic()


def get_polyphonic_dict():
    if corrector is None:
        load_polyphonic()
    return corrector.polyphonic_dict


def correct_pronunciation(word, pinyin, style=1):
    if corrector is None:
        load_polyphonic()
    return corrector.correct_pronunciation(word, pinyin, style)


def add_polyphonic(word, pinyin):
    if corrector is None:
        load_polyphonic()

    if len(word) != len(pinyin):
        return False

    corrector.polyphonic_words[word] = pinyin
    corrector.save_polyphonic()
    return True


def delete_polyphonic(word):
    if corrector is None:
        load_polyphonic()

    if word in corrector.polyphonic_words.keys():
        del corrector.polyphonic_words[word]
        corrector.save_polyphonic()
        return True

    return False


def update_polyphonic():
    if corrector is None:
        load_polyphonic()
    corrector.update_polyphonic()
