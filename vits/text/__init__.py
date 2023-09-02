""" from https://github.com/keithito/tacotron """
from vits.text import cleaners


def text_to_sequence(text, symbols, cleaner_names, bert_embedding=False):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text
    '''

    _symbol_to_id = {s: i for i, s in enumerate(symbols)}

    if bert_embedding:
        cleaned_text, char_embeds = _clean_text(text, cleaner_names)
        sequence = [_symbol_to_id[symbol] for symbol in cleaned_text.split()]
        return sequence, char_embeds
    else:
        cleaned_text = _clean_text(text, cleaner_names)
        sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id.keys()]
        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
