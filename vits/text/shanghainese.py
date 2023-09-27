import os
import re
import cn2an
import opencc
import config
from utils.download import download_and_verify

URLS = [
    "https://github.com/CjangCjengh/chinese-dialect-lexicons/releases/download/v1.0.3/chinese_dialects.7z",
    "https://ghproxy.com/https://github.com/CjangCjengh/chinese-dialect-lexicons/releases/download/v1.0.3/chinese_dialects.7z",
]
TARGET_PATH = os.path.join(config.ABS_PATH, "vits/text/chinese_dialects.7z")
EXTRACT_DESTINATION = os.path.join(config.ABS_PATH, "vits/text/chinese_dialect_lexicons/")
EXPECTED_MD5 = None
OPENCC_FILE_PATH = os.path.join(config.ABS_PATH, "vits/text/chinese_dialect_lexicons/zaonhe.json")

if not os.path.exists(OPENCC_FILE_PATH):
    success, message = download_and_verify(URLS, TARGET_PATH, EXPECTED_MD5, EXTRACT_DESTINATION)

converter = opencc.OpenCC(OPENCC_FILE_PATH)

# List of (Latin alphabet, ipa) pairs:
_latin_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('A', 'ᴇ'),
    ('B', 'bi'),
    ('C', 'si'),
    ('D', 'di'),
    ('E', 'i'),
    ('F', 'ᴇf'),
    ('G', 'dʑi'),
    ('H', 'ᴇtɕʰ'),
    ('I', 'ᴀi'),
    ('J', 'dʑᴇ'),
    ('K', 'kʰᴇ'),
    ('L', 'ᴇl'),
    ('M', 'ᴇm'),
    ('N', 'ᴇn'),
    ('O', 'o'),
    ('P', 'pʰi'),
    ('Q', 'kʰiu'),
    ('R', 'ᴀl'),
    ('S', 'ᴇs'),
    ('T', 'tʰi'),
    ('U', 'ɦiu'),
    ('V', 'vi'),
    ('W', 'dᴀbɤliu'),
    ('X', 'ᴇks'),
    ('Y', 'uᴀi'),
    ('Z', 'zᴇ')
]]


def _number_to_shanghainese(num):
    num = cn2an.an2cn(num).replace('一十', '十').replace('二十', '廿').replace('二', '两')
    return re.sub(r'((?:^|[^三四五六七八九])十|廿)两', r'\1二', num)


def number_to_shanghainese(text):
    return re.sub(r'\d+(?:\.?\d+)?', lambda x: _number_to_shanghainese(x.group()), text)


def latin_to_ipa(text):
    for regex, replacement in _latin_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def shanghainese_to_ipa(text):
    from vits.text.mandarin import symbols_to_chinese
    text = symbols_to_chinese(text)
    text = number_to_shanghainese(text.upper())
    text = converter.convert(text).replace('-', '').replace('$', ' ')
    text = re.sub(r'[A-Z]', lambda x: latin_to_ipa(x.group()) + ' ', text)
    text = re.sub(r'[、；：]', '，', text)
    text = re.sub(r'\s*，\s*', ', ', text)
    text = re.sub(r'\s*。\s*', '. ', text)
    text = re.sub(r'\s*？\s*', '? ', text)
    text = re.sub(r'\s*！\s*', '! ', text)
    text = re.sub(r'\s*$', '', text)
    return text
