import re
import cn2an
import opencc
import config

converter = opencc.OpenCC(config.ABS_PATH + '/chinese_dialect_lexicons/zaonhe')

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

_symbols_to_chinese = [(re.compile(f'{x[0]}'), x[1]) for x in [
    ('([0-9]+(?:\.?[0-9]+)?)%', r'百分之\1'),
    ('([0-9]+)/([0-9]+)', r'\2分之\1'),
    ('\+', r'加'),
    ('([0-9]+)-([0-9]+)', r'\1减\2'),
    ('×', r'乘以'),
    ('([0-9]+)x([0-9]+)', r'\1乘以\2'),
    ('([0-9]+)\*([0-9]+)', r'\1乘以\2'),
    ('÷', r'除以'),
    ('=', r'等于'),
    ('≠', r'不等于'),
]]


def symbols_to_chinese(text):
    for regex, replacement in _symbols_to_chinese:
        text = re.sub(regex, replacement, text)
    return text


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
