import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper

try:
    from utils.config_manager import global_config as config
    ESPEAK_LIBRARY = getattr(config, "ESPEAK_LIBRARY", "")
except:
    import config
    ESPEAK_LIBRARY = getattr(config, "ESPEAK_LIBRARY", "")
if ESPEAK_LIBRARY != "":
    EspeakWrapper.set_library(ESPEAK_LIBRARY)

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = expand_abbreviations(text)
    return text


# for English text
def english_cleaners(text):
    '''Pipeline for English text, including abbreviation expansion.'''
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: transliteration_cleaners(x.group(1)) + ' ', text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    return phonemes


# for non-English text that can be transliterated to ASCII
def english_cleaners2(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: transliteration_cleaners(x.group(1)) + ' ', text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True,
                         with_stress=True)
    return phonemes


def japanese_cleaners(text):
    from vits.text.japanese import japanese_to_romaji_with_accent

    def clean(text):
        text = japanese_to_romaji_with_accent(text)
        text = re.sub(r'([A-Za-z])$', r'\1.', text)
        return text

    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: clean(x.group(1)) + ' ', text)
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace('ts', 'ʦ').replace('...', '…')


def korean_cleaners(text):
    '''Pipeline for Korean text'''
    from vits.text.korean import latin_to_hangul, number_to_hangul, divide_hangul

    def clean(text):
        text = latin_to_hangul(text)
        text = number_to_hangul(text)
        text = divide_hangul(text)
        text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
        return text

    text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: clean(x.group(1)) + ' ', text)
    return text


def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    from vits.text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo, symbols_to_chinese

    def clean(text):
        text = symbols_to_chinese(text)
        text = number_to_chinese(text)
        text = chinese_to_bopomofo(text)
        text = latin_to_bopomofo(text)
        text = re.sub(r'([ˉˊˇˋ˙])$', r'\1。', text)
        return text

    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: clean(x.group(1)) + ' ', text)
    return text


def zh_ja_mixture_cleaners(text):
    from vits.text.mandarin import chinese_to_romaji
    from vits.text.japanese import japanese_to_romaji_with_accent
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_romaji(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_romaji_with_accent(
        x.group(1)).replace('ts', 'ʦ').replace('u', 'ɯ').replace('...', '…') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def sanskrit_cleaners(text):
    text = text.replace('॥', '।').replace('ॐ', 'ओम्')
    text = re.sub(r'([^।])$', r'\1।', text)
    return text


def cjks_cleaners(text):
    from vits.text.mandarin import chinese_to_lazy_ipa
    from vits.text.japanese import japanese_to_ipa
    from vits.text.korean import korean_to_lazy_ipa
    from vits.text.sanskrit import devanagari_to_ipa
    from vits.text.english import english_to_lazy_ipa
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[SA\](.*?)\[SA\]',
                  lambda x: devanagari_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners(text):
    from vits.text.mandarin import chinese_to_lazy_ipa
    from vits.text.japanese import japanese_to_ipa
    from vits.text.korean import korean_to_ipa
    from vits.text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: chinese_to_lazy_ipa(x.group(1)).replace(
        'ʧ', 'tʃ').replace('ʦ', 'ts').replace('ɥan', 'ɥæn') + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa(x.group(1)).replace('ʧ', 'tʃ').replace(
        'ʦ', 'ts').replace('ɥan', 'ɥæn').replace('ʥ', 'dz') + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english_to_ipa2(x.group(1)).replace('ɑ', 'a').replace(
        'ɔ', 'o').replace('ɛ', 'e').replace('ɪ', 'i').replace('ʊ', 'u') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners2(text):
    from vits.text.mandarin import chinese_to_ipa
    from vits.text.japanese import japanese_to_ipa2
    from vits.text.korean import korean_to_ipa
    from vits.text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cje_cleaners(text):
    from vits.text.mandarin import chinese_to_lazy_ipa
    from vits.text.japanese import japanese_to_ipa
    from vits.text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: chinese_to_lazy_ipa(x.group(1)).replace(
        'ʧ', 'tʃ').replace('ʦ', 'ts').replace('ɥan', 'ɥæn') + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa(x.group(1)).replace('ʧ', 'tʃ').replace(
        'ʦ', 'ts').replace('ɥan', 'ɥæn').replace('ʥ', 'dz') + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english_to_ipa2(x.group(1)).replace('ɑ', 'a').replace(
        'ɔ', 'o').replace('ɛ', 'e').replace('ɪ', 'i').replace('ʊ', 'u') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cje_cleaners2(text):
    from vits.text.mandarin import chinese_to_ipa
    from vits.text.japanese import japanese_to_ipa2
    from vits.text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def thai_cleaners(text):
    from vits.text.thai import num_to_thai, latin_to_thai

    def clean(text):
        text = num_to_thai(text)
        text = latin_to_thai(text)
        return text

    text = re.sub(r'\[TH\](.*?)\[TH\]', lambda x: clean(x.group(1)) + ' ', text)
    return text


def shanghainese_cleaners(text):
    from vits.text.shanghainese import shanghainese_to_ipa

    def clean(text):
        text = shanghainese_to_ipa(text)
        text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
        return text

    text = re.sub(r'\[SH\](.*?)\[SH\]', lambda x: clean(x.group(1)) + ' ', text)
    return text


def chinese_dialect_cleaners(text):
    from vits.text.mandarin import chinese_to_ipa2
    from vits.text.japanese import japanese_to_ipa3
    from vits.text.shanghainese import shanghainese_to_ipa
    from vits.text.cantonese import cantonese_to_ipa
    from vits.text.english import english_to_lazy_ipa2
    from vits.text.ngu_dialect import ngu_dialect_to_ipa
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa3(x.group(1)).replace('Q', 'ʔ') + ' ', text)
    text = re.sub(r'\[SH\](.*?)\[SH\]', lambda x: shanghainese_to_ipa(x.group(1)).replace('1', '˥˧').replace('5',
                                                                                                             '˧˧˦').replace(
        '6', '˩˩˧').replace('7', '˥').replace('8', '˩˨').replace('ᴀ', 'ɐ').replace('ᴇ', 'e') + ' ', text)
    text = re.sub(r'\[GD\](.*?)\[GD\]',
                  lambda x: cantonese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_lazy_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', lambda x: ngu_dialect_to_ipa(x.group(2), x.group(
        1)).replace('ʣ', 'dz').replace('ʥ', 'dʑ').replace('ʦ', 'ts').replace('ʨ', 'tɕ') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def bert_chinese_cleaners(text):
    from vits.text import mandarin
    matches = re.findall(r"\[ZH\](.*?)\[ZH\]", text)
    text = "".join(matches)
    if text[-1] not in [".", "。", ",", "，"]: text += "."
    text = mandarin.symbols_to_chinese(text)
    text = mandarin.number_transform_to_chinese(text)
    from tts_app.model_manager import model_manager
    cleaned_text, char_embeds = model_manager.tts_front.chinese_to_phonemes(text)
    return cleaned_text, char_embeds
