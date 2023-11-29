import torch

from bert_vits2 import commons
from bert_vits2 import utils as bert_vits2_utils
from bert_vits2.get_emo import get_emo
from bert_vits2.models import SynthesizerTrn
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
        self.emotion_model = None
        self.processor = None

        # Compatible with legacy versions
        self.version = process_legacy_versions(self.hps_ms)
        self.text_extra_str_map = {"zh": "", "ja": "", "en": ""}
        self.bert_extra_str_map = {"zh": "", "ja": "", "en": ""}
        self.hps_ms.model.emotion_embedding = False
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
            self.hps_ms.model.emotion_embedding = True
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
        else:
            self.hps_ms.model.n_layers_trans_flow = 4
            self.hps_ms.model.emotion_embedding = True
            self.lang = getattr(self.hps_ms.data, "lang", ["zh", "ja", "en"])
            self.num_tones = num_tones
            if "ja" in self.lang: self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            if "en" in self.lang: self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})

        if "zh" in self.lang:
            self.bert_model_names.update({"zh": "CHINESE_ROBERTA_WWM_EXT_LARGE"})

        # self.bert_handler = BertHandler(self.lang)

        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

        self.device = device

    def load_model(self, bert_handler, emotion_model=None, processor=None):
        self.bert_handler = bert_handler
        self.emotion_model = emotion_model
        self.processor = processor

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

    def get_text(self, text, language_str, hps):
        clean_text_lang_str = language_str + self.text_extra_str_map.get(language_str, "")
        bert_feature_lang_str = language_str + self.bert_extra_str_map.get(language_str, "")

        tokenizer, _ = self.bert_handler.get_bert_model(self.bert_model_names[language_str])

        norm_text, phone, tone, word2ph = clean_text(text, clean_text_lang_str, tokenizer)

        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, self._symbol_to_id)

        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        bert = self.bert_handler.get_bert_feature(norm_text, word2ph, bert_feature_lang_str,
                                                  self.bert_model_names[language_str])
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
        emo = torch.from_numpy(
            get_emo(reference_audio, self.emotion_model, self.processor)) if reference_audio else torch.Tensor(
            [emotion])

        return emo

    def infer(self, text, id, lang, sdp_ratio, noise, noisew, length, reference_audio=None, emotion=None,
              skip_start=False, skip_end=False, **kwargs):
        zh_bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text(text, lang, self.hps_ms)
        
        if self.hps_ms.model.emotion_embedding:
            emo = self.get_emo_(reference_audio, emotion).to(self.device).unsqueeze(0)
        else:
            emo = None
            
        if skip_start:
            phones = phones[1:]
            tones = tones[1:]
            lang_ids = lang_ids[1:]
            zh_bert = zh_bert[:, 1:]
            ja_bert = ja_bert[:, 1:]
            en_bert = en_bert[:, 1:]
        if skip_end:
            phones = phones[:-1]
            tones = tones[:-1]
            lang_ids = lang_ids[:-1]
            zh_bert = zh_bert[:, :-1]
            ja_bert = ja_bert[:, :-1]
            en_bert = en_bert[:, :-1]

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
