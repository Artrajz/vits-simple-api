import re

import numpy as np
import torch

from bert_vits2 import utils, commons
from bert_vits2.models import SynthesizerTrn
from bert_vits2.text import symbols, cleaned_text_to_sequence, get_bert
from bert_vits2.text.cleaner import clean_text
from bert_vits2.text.symbols import get_symbols
from utils.sentence import sentence_split, cut


class Bert_VITS2:
    def __init__(self, model, config, device=torch.device("cpu")):
        self.hps_ms = utils.get_hparams_from_file(config)
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.speakers = [item[0] for item in
                         sorted(list(getattr(self.hps_ms.data, 'spk2id', {'0': 0}).items()), key=lambda x: x[1])]
        
        self.legacy = getattr(self.hps_ms.data, 'legacy', False)
        symbols, num_tones, self.language_id_map, num_languages, self.language_tone_start_map = get_symbols(
            legacy=self.legacy)
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            symbols=symbols,
            num_tones=num_tones,
            num_languages=num_languages,
            **self.hps_ms.model).to(device)
        _ = self.net_g.eval()
        self.device = device
        self.load_model(model)

    def load_model(self, model):
        utils.load_checkpoint(model, self.net_g, None, skip_optimizer=True)

    def get_speakers(self):
        return self.speakers

    def get_text(self, text, language_str, hps):
        norm_text, phone, tone, word2ph = clean_text(text, language_str)
        # print([f"{p}{t}" for p, t in zip(phone, tone)])
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, self._symbol_to_id,
                                                         self.language_tone_start_map, self.language_id_map)

        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert = get_bert(norm_text, word2ph, language_str)

        assert bert.shape[-1] == len(phone)

        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)

        return bert, phone, tone, language

    def infer(self, text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
        bert, phones, tones, lang_ids = self.get_text(text, "ZH", self.hps_ms)
        with torch.no_grad():
            x_tst = phones.to(self.device).unsqueeze(0)
            tones = tones.to(self.device).unsqueeze(0)
            lang_ids = lang_ids.to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            speakers = torch.LongTensor([int(sid)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, sdp_ratio=sdp_ratio
                                     , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[
                0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()
        return audio

    def get_audio(self, voice, auto_break=False):
        text = voice.get("text", None)
        sdp_ratio = voice.get("sdp_ratio", 0.2)
        noise_scale = voice.get("noise", 0.5)
        noise_scale_w = voice.get("noisew", 0.6)
        length_scale = voice.get("length", 1)
        sid = voice.get("id", 0)
        max = voice.get("max", 50)
        # sentence_list = sentence_split(text, max, "ZH", ["zh"])
        sentence_list = cut(text, max)
        audios = []
        for sentence in sentence_list:
            audio = self.infer(sentence, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid)
            audios.append(audio)
        audio = np.concatenate(audios)
        return audio
