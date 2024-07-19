import logging
import math
import os.path
import random
import re
from typing import List

import librosa
import numpy as np
import torch

from contants import config
from gpt_sovits.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from gpt_sovits.module.mel_processing import spectrogram_torch
from gpt_sovits.module.models import SynthesizerTrn
from gpt_sovits.text import cleaned_text_to_sequence
from gpt_sovits.text.cleaner import clean_text
from gpt_sovits.utils import DictToAttrRecursive
from utils.classify_language import classify_language
from utils.data_utils import check_is_none
from utils.sentence import split_languages, sentence_split

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


class GPT_SoVITS:
    def __init__(self, sovits_path, gpt_path, device, **kwargs):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.hz = config.gpt_sovits_config.hz
        self.sampling_rate = None
        self.device = device
        self.model_handler = None
        self.is_half = config.gpt_sovits_config.is_half
        self.np_dtype = np.float16 if self.is_half else np.float32
        self.torch_dtype = torch.float16 if self.is_half else torch.float32
        self.speakers = None
        self.lang = ["zh", "ja", "en"]
        self.flash_attn_enabled = True
        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spepc": None,
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
        }

    def load_model(self, model_handler):
        self.model_handler = model_handler

        self.load_sovits(self.sovits_path)
        self.load_gpt(self.gpt_path)

        self.tokenizer, self.bert_model = self.model_handler.get_bert_model("CHINESE_ROBERTA_WWM_EXT_LARGE")

        self.ssl_model = self.model_handler.get_ssl_model()

    def load_weight(self, saved_state_dict, model):
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                # logging.info(f"{k} is not in the checkpoint")
                new_state_dict[k] = v
        if hasattr(model, 'module'):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)

    def load_sovits(self, sovits_path):
        # self.n_semantic = 1024
        logging.info(f"Loaded checkpoint '{sovits_path}'")
        dict_s2 = torch.load(sovits_path, map_location=self.device)
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        # self.speakers = [self.hps.get("name")] # 从模型配置中获取名字
        self.speakers = [os.path.basename(os.path.dirname(self.sovits_path))]  # 用模型文件夹作为名字

        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device)

        if config.gpt_sovits_config.is_half:
            self.vq_model = self.vq_model.half()

        self.vq_model.eval()
        self.sampling_rate = self.hps.data.sampling_rate

        self.load_weight(dict_s2['weight'], self.vq_model)

    def load_gpt(self, gpt_path):
        logging.info(f"Loaded checkpoint '{gpt_path}'")
        dict_s1 = torch.load(gpt_path, map_location=self.device)

        self.gpt_config = dict_s1["config"]
        self.max_sec = self.gpt_config.get("data").get("max_sec")

        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False,
                                                      flash_attn_enabled=self.flash_attn_enabled).to(
            self.device)

        self.load_weight(dict_s1['weight'], self.t2s_model)

        if config.gpt_sovits_config.is_half:
            self.t2s_model = self.t2s_model.half()

        self.t2s_model.eval()

        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        logging.info(f"Number of parameter: {total / 1e6:.2f}M")

    def set_seed(self, seed: int):
        seed = int(seed)
        seed = seed if seed != -1 else random.randrange(1 << 32)
        logging.debug(f"Set seed to {seed}")
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
                # torch.backends.cudnn.enabled = True
        except:
            pass
        return seed

    def get_speakers(self):
        return self.speakers

    def get_cleaned_text(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language.replace("all_", ""))
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_cleaned_text_multilang(self, text):
        sentences = split_languages(text, expand_abbreviations=True, expand_hyphens=True)
        phones, word2ph, norm_text = [], [], []
        for sentence, lang in sentences:
            lang = classify_language(sentence)
            _phones, _word2ph, _norm_text = self.get_cleaned_text(sentence, lang)
            phones.extend(_phones)
            word2ph.extend(_word2ph)
            norm_text.extend(_norm_text)

        return phones, word2ph, norm_text

    def get_bert_feature(self, text, phones, word2ph, language):
        if language == "zh":
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(self.device)  #####输入是long不用管精度问题，精度随bert_model
                res = self.bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            assert len(word2ph) == len(text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
            phone_level_feature = torch.cat(phone_level_feature, dim=0)
            # if(config.gpt_sovits_config.is_half==True):phone_level_feature=phone_level_feature.half()
            bert = phone_level_feature.T
            torch.cuda.empty_cache()
        else:
            bert = torch.zeros((1024, len(phones)), dtype=self.torch_dtype)

        return bert

    def get_bert_and_cleaned_text_multilang(self, text: str, lang_list: list):
        target_languages = lang_list
        if len(lang_list) == 1 and lang_list[0] == "auto":
            target_languages = self.lang

        sentences = split_languages(text, expand_abbreviations=True, expand_hyphens=True,
                                    target_languages=target_languages)

        phones_list, word2ph_list, norm_text_list, bert_list = [], [], [], []

        for sentence, lang in sentences:
            phones, word2ph, _norm_text = self.get_cleaned_text(sentence, lang)
            bert = self.get_bert_feature(sentence, phones, word2ph, _norm_text)
            phones_list.extend(phones)
            if word2ph is not None:
                word2ph_list.extend(word2ph)
            norm_text_list.extend(_norm_text)
            bert_list.append(bert)

        norm_text = ''.join(norm_text_list)
        bert = torch.cat(bert_list, dim=1).to(self.device, dtype=self.torch_dtype)

        return phones_list, word2ph_list, norm_text, bert

    def get_spepc(self, audio, orig_sr):
        """audio的sampling_rate与模型相同"""
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec

    def _set_prompt_semantic(self, reference_audio, reference_audio_sr):
        zero_wav = np.zeros(
            int(self.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half else np.float32,
        )
        wav16k = librosa.resample(reference_audio, orig_sr=reference_audio_sr, target_sr=16000)
        with torch.no_grad():
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)

            if self.is_half == True:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)

            wav16k = torch.cat([wav16k, zero_wav_torch]).unsqueeze(0)

            ssl_content = self.ssl_model.model(wav16k)[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0].to(self.device)
            # prompt_semantic = prompt_semantic.unsqueeze(0).to(self.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic
        torch.cuda.empty_cache()

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def preprocess_text(self, text: str, lang_list: list, segment_size: int):
        texts = sentence_split(text, segment_size)
        lang = lang_list[0]  # main language

        result = []
        for text in texts:
            text = text.strip("\n")
            if (text[0] not in splits and len(self.get_first(text)) < 4):
                text = "。" + text if lang != "en" else "." + text
            if (text[-1] not in splits):
                text += "。" if lang != "en" else "."

            phones, word2ph, norm_text, bert_features = self.get_bert_and_cleaned_text_multilang(text, lang_list)

            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def preprocess_prompt(self, reference_audio, reference_audio_sr, prompt_text: str, prompt_lang: str):
        if self.prompt_cache.get("prompt_text") != prompt_text:
            if prompt_lang.lower() == "auto":
                prompt_lang = classify_language(prompt_text)
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits):
                prompt_text += "。" if prompt_lang != "en" else "."
            phones, word2ph, norm_text = self.get_cleaned_text(prompt_text, prompt_lang)
            bert_features = self.get_bert_feature(norm_text, phones, word2ph, prompt_lang).to(self.device,
                                                                                              dtype=self.torch_dtype)
            self.prompt_cache["prompt_text"] = prompt_text
            self.prompt_cache["prompt_lang"] = prompt_lang
            self.prompt_cache["phones"] = phones
            self.prompt_cache["bert_features"] = bert_features
            self.prompt_cache["norm_text"] = norm_text
            self.prompt_cache["refer_spepc"] = self.get_spepc(reference_audio, orig_sr=reference_audio_sr)

            self._set_prompt_semantic(reference_audio, reference_audio_sr)

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length: int = None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype: torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(self, data: list, prompt_data: dict = None, batch_size: int = 5, threshold: float = 0.75,
                 split_bucket: bool = True):

        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos + batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end - pos) // 2] / batch.mean()
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end - 1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            # bert_features_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            bert_max_len = 0
            phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1)
                    all_phones = torch.LongTensor(prompt_data["phones"] + item["phones"])
                    phones = torch.LongTensor(item["phones"])
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"]
                    phones = torch.LongTensor(item["phones"])
                    all_phones = phones
                    # norm_text = item["norm_text"]

                bert_max_len = max(bert_max_len, all_bert_features.shape[-1])
                phones_max_len = max(phones_max_len, phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            max_len = max(bert_max_len, phones_max_len)
            # phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            all_bert_features_batch = torch.FloatTensor(len(item_list), 1024, max_len)
            all_bert_features_batch.zero_()

            for idx, item in enumerate(all_bert_features_list):
                if item != None:
                    all_bert_features_batch[idx, :, : item.shape[-1]] = item

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        '''
        Recovery the order of the audio according to the batch_index_list.

        Args:
            data (List[list(np.ndarray)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.

        Returns:
            list (List[np.ndarray]): the data in the original order.
        '''
        lenght = len(sum(batch_index_list, []))
        _data = [None] * lenght
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def audio_postprocess(self, audio: List[torch.Tensor], sr: int, batch_index_list: list = None,
                          speed_factor: float = 1.0, split_bucket: bool = True) -> tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
            int(self.sampling_rate * 0.3),
            dtype=torch.float16 if self.is_half else torch.float32,
            device=self.device
        )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio = torch.abs(audio_fragment).max()  # 简单防止16bit爆音
                if max_audio > 1: audio_fragment /= max_audio
                audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
                audio[i][j] = audio_fragment.cpu().numpy()

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            # audio = [item for batch in audio for item in batch]
            audio = sum(audio, [])

        audio = np.concatenate(audio, 0)

        try:
            if speed_factor != 1.0:
                from .utils import speed_change
                audio = speed_change(audio, speed_factor=speed_factor, sr=int(sr))
        except Exception as e:
            logging.error(f"Failed to change speed of audio: \n{e}")

        return audio

    def infer(self, text, lang, reference_audio, reference_audio_sr, prompt_text, prompt_lang, top_k, top_p,
              temperature, batch_size: int = 5, batch_threshold: float = 0.75, split_bucket: bool = True,
              return_fragment: bool = False, speed_factor: float = 1.0, seed: int = -1,
              segment_size: int = config.gpt_sovits_config.segment_size, **kwargs):

        self.set_seed(seed)

        if return_fragment:
            split_bucket = False

        data = self.preprocess_text(text, lang, segment_size)

        no_prompt_text = False
        if check_is_none(prompt_text):
            no_prompt_text = True
        else:
            self.preprocess_prompt(reference_audio, reference_audio_sr, prompt_text, prompt_lang)

        data, batch_index_list = self.to_batch(data,
                                               prompt_data=self.prompt_cache if not no_prompt_text else None,
                                               batch_size=batch_size,
                                               threshold=batch_threshold,
                                               split_bucket=split_bucket
                                               )

        audio = []
        for item in data:
            batch_phones = item["phones"]
            batch_phones_len = item["phones_len"]
            all_phoneme_ids = item["all_phones"]
            all_phoneme_lens = item["all_phones_len"]
            all_bert_features = item["all_bert_features"]
            norm_text = item["norm_text"]

            # batch_phones = batch_phones.to(self.device)
            batch_phones_len = batch_phones_len.to(self.device)
            all_phoneme_ids = all_phoneme_ids.to(self.device)
            all_phoneme_lens = all_phoneme_lens.to(self.device)
            all_bert_features = all_bert_features.to(self.device)
            if self.is_half:
                all_bert_features = all_bert_features.half()

            logging.debug(f"Infer text:{norm_text}")
            if no_prompt_text:
                prompt = None
            else:
                prompt = self.prompt_cache["prompt_semantic"].expand(all_phoneme_ids.shape[0], -1).to(
                    self.device)

            with torch.no_grad():
                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )

            refer_audio_spepc: torch.Tensor = self.prompt_cache["refer_spepc"].to(self.device)
            if self.is_half:
                refer_audio_spepc = refer_audio_spepc.half()

            pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
            upsample_rate = math.prod(self.vq_model.upsample_rates)
            audio_frag_idx = [pred_semantic_list[i].shape[0] * 2 * upsample_rate for i in
                              range(0, len(pred_semantic_list))]
            audio_frag_end_idx = [sum(audio_frag_idx[:i + 1]) for i in range(0, len(audio_frag_idx))]
            all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.device)
            _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.device)
            _batch_audio_fragment = (self.vq_model.decode(
                all_pred_semantic, _batch_phones, refer_audio_spepc
            ).detach()[0, 0, :])
            audio_frag_end_idx.insert(0, 0)
            batch_audio_fragment = [_batch_audio_fragment[audio_frag_end_idx[i - 1]:audio_frag_end_idx[i]] for i in
                                    range(1, len(audio_frag_end_idx))]

            torch.cuda.empty_cache()

            if return_fragment:
                yield self.audio_postprocess([batch_audio_fragment],
                                             reference_audio_sr,
                                             batch_index_list,
                                             speed_factor,
                                             split_bucket)
            else:
                audio.append(batch_audio_fragment)

        if not return_fragment:
            yield self.audio_postprocess(audio,
                                         reference_audio_sr,
                                         batch_index_list,
                                         speed_factor,
                                         split_bucket)
