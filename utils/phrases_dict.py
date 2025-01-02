import logging

from pypinyin_dict.phrase_pinyin_data import large_pinyin


def phrases_dict_init():
    logging.info("Loading large_pinyin")
    large_pinyin.load()
