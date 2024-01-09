import ast
import logging

import jieba
import pypinyin
from pypinyin_dict.phrase_pinyin_data import large_pinyin
from pypinyin_dict.pinyin_data import cc_cedict

import config

phrases_dict = {
    "一骑当千": [["yí"], ["jì"], ["dāng"], ["qiān"]],
    "桔子": [["jú"], ["zi"]],
    "重生": [["chóng"], ["shēng"]],
    "重重地": [["zhòng"], ["zhòng"], ["de"]],
    "自少时": [["zì"], ["shào"], ["shí"]],
}


def load_phrases_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            additional_phrases = ast.literal_eval(file.read())
            phrases_dict.update(additional_phrases)
            logging.info(f"Additional phrases loaded from {file_path}")
    except FileNotFoundError:
        logging.debug(f"File {file_path} not found. You can create {file_path} and write your phrases_dict.")
    except Exception as e:
        logging.error(f"Error loading additional phrases from {file_path}: {str(e)}")


def phrases_dict_init():
    logging.info("Loading phrases_dict")
    large_pinyin.load()
    cc_cedict.load()
    additional_phrases_file = config.ABS_PATH + "/phrases_dict.txt"
    load_phrases_from_file(additional_phrases_file)

    for word in phrases_dict.keys():
        jieba.add_word(word)
    pypinyin.load_phrases_dict(phrases_dict)
