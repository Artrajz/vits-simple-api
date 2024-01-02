import logging

import torch

from bert_vits2 import commons
from bert_vits2 import utils as bert_vits2_utils
from bert_vits2.clap_wrapper import get_clap_audio_feature, get_clap_text_feature
from bert_vits2.get_emo import get_emo
from bert_vits2.models import SynthesizerTrn
from bert_vits2.models_v230 import SynthesizerTrn as SynthesizerTrn_v230
from bert_vits2.text import *
from bert_vits2.text.cleaner import clean_text
from bert_vits2.utils import process_legacy_versions
from utils import get_hparams_from_file


class Bert_VITS2:
    def __init__(self, model_path, config, device=torch.device("cpu"), **kwargs):
        self.model_path = model_path
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.speakers = [item[0] for item in
                         sorted(list(getattr(self.hps_ms.data, 'spk2id', {'0': 0}).items()), key=lambda x: x[1])]
        self.symbols = symbols
        self.sampling_rate = self.hps_ms.data.sampling_rate

        self.bert_model_names = {}
        self.ja_bert_dim = 1024
        self.num_tones = num_tones

        # Compatible with legacy versions
        self.version = process_legacy_versions(self.hps_ms)
        self.text_extra_str_map = {"zh": "", "ja": "", "en": ""}
        self.bert_extra_str_map = {"zh": "", "ja": "", "en": ""}
        self.hps_ms.model.emotion_embedding = None
        if self.version in ["1.0", "1.0.0", "1.0.1"]:
            self.symbols = symbols_legacy
            self.hps_ms.model.n_layers_trans_flow = 3
            self.lang = getattr(self.hps_ms.data, "lang", ["zh"])
            self.ja_bert_dim = 768
            self.num_tones = num_tones_v111
            self.text_extra_str_map.update({"zh": "_v100"})

        elif self.version in ["1.1.0-transition"]:
            self.hps_ms.model.n_layers_trans_flow = 3
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja"])
            self.ja_bert_dim = 768
            self.num_tones = num_tones_v111
            if "ja" in self.lang: self.bert_model_names.update({"ja": "BERT_BASE_JAPANESE_V3"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v111"})
            self.bert_extra_str_map.update({"ja": "_v111"})

        elif self.version in ["1.1", "1.1.0", "1.1.1"]:
            self.hps_ms.model.n_layers_trans_flow = 6
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja"])
            self.ja_bert_dim = 768
            self.num_tones = num_tones_v111
            if "ja" in self.lang: self.bert_model_names.update({"ja": "BERT_BASE_JAPANESE_V3"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v111"})
            self.bert_extra_str_map.update({"ja": "_v111"})

        elif self.version in ["2.0", "2.0.0", "2.0.1", "2.0.2"]:
            self.hps_ms.model.n_layers_trans_flow = 4
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v200", "en": "_v200"})
            self.bert_extra_str_map.update({"ja": "_v200", "en": "_v200"})

        elif self.version in ["2.1", "2.1.0"]:
            self.hps_ms.model.n_layers_trans_flow = 4
            self.hps_ms.model.emotion_embedding = 1
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})

        elif self.version in ["2.2", "2.2.0"]:
            self.hps_ms.model.n_layers_trans_flow = 4
            self.hps_ms.model.emotion_embedding = 2
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
        elif self.version in ["2.3", "2.3.0"]:
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            self.text_extra_str_map.update({"en": "_v230"})
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
        else:
            logging.debug("Version information not found. Loaded as the newest version: v2.3.")
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            self.text_extra_str_map.update({"en": "_v230"})
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})

        if "zh" in self.lang:
            self.bert_model_names.update({"zh": "CHINESE_ROBERTA_WWM_EXT_LARGE"})

        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

        self.device = device

    def load_model(self, model_handler):
        self.model_handler = model_handler

        if self.version in ["2.3", "2.3.0"]:
            self.net_g = SynthesizerTrn_v230(
                len(symbols),
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                n_speakers=self.hps_ms.data.n_speakers,
                **self.hps_ms.model,
            ).to(self.device)
        else:
            self.net_g = SynthesizerTrn(
                len(self.symbols),
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                n_speakers=self.hps_ms.data.n_speakers,
                symbols=self.symbols,
                ja_bert_dim=self.ja_bert_dim,
                num_tones=self.num_tones,
                **self.hps_ms.model).to(self.device)
        _ = self.net_g.eval()
        bert_vits2_utils.load_checkpoint(self.model_path, self.net_g, None, skip_optimizer=True, version=self.version)

    def get_speakers(self):
        return self.speakers

    def get_text(self, text, language_str, hps, style_text=None, style_weight=0.7):
        clean_text_lang_str = language_str + self.text_extra_str_map.get(language_str, "")
        bert_feature_lang_str = language_str + self.bert_extra_str_map.get(language_str, "")

        tokenizer, _ = self.model_handler.get_bert_model(self.bert_model_names[language_str])

        norm_text, phone, tone, word2ph = clean_text(text, clean_text_lang_str, tokenizer)

        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, self._symbol_to_id)

        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        style_text = None if style_text == "" else style_text
        bert = self.model_handler.get_bert_feature(norm_text, word2ph, bert_feature_lang_str,
                                                   self.bert_model_names[language_str], style_text, style_weight)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "zh":
            zh_bert = bert
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "ja":
            zh_bert = torch.zeros(1024, len(phone))
            ja_bert = bert
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "en":
            zh_bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone))
            en_bert = bert
        else:
            zh_bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return zh_bert, ja_bert, en_bert, phone, tone, language

    def get_emo_(self, reference_audio, emotion):
        if reference_audio:
            emo = torch.from_numpy(
                get_emo(reference_audio, self.model_handler.emotion_model,
                        self.model_handler.emotion_processor))
        else:
            if emotion is None: emotion = 0
            emo = torch.Tensor([emotion])

        return emo

    def infer(self, text, id, lang, sdp_ratio, noise, noisew, length, reference_audio=None, emotion=None,
              skip_start=False, skip_end=False, text_prompt=None, style_text=None, style_weigth=0.7, **kwargs):
        zh_bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text(text, lang, self.hps_ms, style_text,
                                                                           style_weigth)

        if self.hps_ms.model.emotion_embedding == 1:
            emo = self.get_emo_(reference_audio, emotion).to(self.device).unsqueeze(0)
        elif self.hps_ms.model.emotion_embedding == 2:
            if isinstance(text_prompt, str):
                emo = get_clap_text_feature(text_prompt, self.model_handler.clap_model,
                                            self.model_handler.clap_processor, self.device)
            else:
                emo = get_clap_audio_feature(reference_audio, self.model_handler.clap_model,
                                             self.model_handler.clap_processor, self.device)
            emo = torch.squeeze(emo, dim=1).unsqueeze(0)
        else:
            emo = None

        if skip_start:
            phones = phones[3:]
            tones = tones[3:]
            lang_ids = lang_ids[3:]
            zh_bert = zh_bert[:, 3:]
            ja_bert = ja_bert[:, 3:]
            en_bert = en_bert[:, 3:]
        if skip_end:
            phones = phones[:-2]
            tones = tones[:-2]
            lang_ids = lang_ids[:-2]
            zh_bert = zh_bert[:, :-2]
            ja_bert = ja_bert[:, :-2]
            en_bert = en_bert[:, :-2]

        with torch.no_grad():
            x_tst = phones.to(self.device).unsqueeze(0)
            tones = tones.to(self.device).unsqueeze(0)
            lang_ids = lang_ids.to(self.device).unsqueeze(0)
            zh_bert = zh_bert.to(self.device).unsqueeze(0)
            ja_bert = ja_bert.to(self.device).unsqueeze(0)
            en_bert = en_bert.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            speakers = torch.LongTensor([int(id)]).to(self.device)
            audio = self.net_g.infer(x_tst,
                                     x_tst_lengths,
                                     speakers,
                                     tones,
                                     lang_ids,
                                     zh_bert,
                                     ja_bert,
                                     en_bert,
                                     sdp_ratio=sdp_ratio,
                                     noise_scale=noise,
                                     noise_scale_w=noisew,
                                     length_scale=length,
                                     emo=emo
                                     )[0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()
        return audio
