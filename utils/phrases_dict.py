import ast
import logging

import jieba
import pypinyin
from pypinyin_dict.pinyin_data import kmandarin_8105

import config

phrases_dict = {u"一骑当千": [[u"yí"], [u"jì"], [u"dāng"], [u"qiān"]], }


def load_phrases_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            additional_phrases = ast.literal_eval(file.read())
            phrases_dict.update(additional_phrases)
            logging.info(f"Additional phrases loaded from {file_path}")
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found. You can create {file_path} and write your phrases_dict.")
    except Exception as e:
        logging.error(f"Error loading additional phrases from {file_path}: {str(e)}")


def phrases_dict_init():
    logging.info("Loading phrases_dict")
    kmandarin_8105.load()
    additional_phrases_file = config.ABS_PATH + "/phrases_dict.txt"
    load_phrases_from_file(additional_phrases_file)
    for word in phrases_dict.keys():
        jieba.add_word(word)
    pypinyin.load_phrases_dict(phrases_dict)
