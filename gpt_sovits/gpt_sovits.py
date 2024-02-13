import logging
import os.path
import re

import librosa
import numpy as np
import torch
from time import time as ttime

from contants import config
from gpt_sovits.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from gpt_sovits.module.mel_processing import spectrogram_torch
from gpt_sovits.module.models import SynthesizerTrn
from gpt_sovits.utils import DictToAttrRecursive
from gpt_sovits.text import cleaned_text_to_sequence
from gpt_sovits.text.cleaner import clean_text
from utils.classify_language import classify_language

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
        self.lang = []

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

        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False).to(self.device)

        self.load_weight(dict_s1['weight'], self.t2s_model)

        if config.gpt_sovits_config.is_half:
            self.t2s_model = self.t2s_model.half()

        self.t2s_model.eval()

        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        logging.info(f"Number of parameter: {total / 1e6:.2f}M")

    def get_speakers(self):
        return self.speakers

    def get_cleaned_text(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language.replace("all_", ""))
        phones = cleaned_text_to_sequence(phones)
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
        else:
            bert = torch.zeros((1024, len(phones)), dtype=self.torch_dtype)

        return bert.to(self.device)

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

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def infer(self, text, lang, reference_audio, reference_audio_sr, prompt_text, prompt_lang, top_k, top_p,
              temperature, **kwargs):
        # t0 = ttime()

        if lang.lower() == "auto":
            lang = classify_language(text, target_languages=self.lang)

        if prompt_lang.lower() == "auto":
            prompt_lang = classify_language(prompt_text)

        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_lang != "en" else "."

        # 应该是文本太短需要加符号补短
        if (text[0] not in splits and len(self.get_first(text)) < 4):
            text = "。" + text if lang == "zh" else "." + text

        if (text[-1] not in splits):
            text += "。" if lang != "en" else "."

        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3), dtype=self.np_dtype)
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
            prompt_semantic = codes[0, 0]
        # t1 = ttime()

        phones1, word2ph1, norm_text1 = self.get_cleaned_text(prompt_text, prompt_lang)

        bert1 = self.get_bert_feature(norm_text1, phones1, word2ph1, prompt_lang).to(dtype=self.torch_dtype)

        phones2, word2ph2, norm_text2 = self.get_cleaned_text(text, lang)
        bert2 = self.get_bert_feature(norm_text2, phones2, word2ph2, lang).to(dtype=self.torch_dtype)

        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        bert = bert.to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
        prompt = prompt_semantic.unsqueeze(0).to(self.device)
        # t2 = ttime()

        audios = []
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=self.hz * self.max_sec,
            )
            # t3 = ttime()
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self.get_spepc(reference_audio, orig_sr=reference_audio_sr)  # .to(device)
            if self.is_half:
                refer = refer.half()
            refer = refer.to(self.device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]

            audio = (
                self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                                     refer).detach().cpu().numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1: audio /= max_audio
            audios.append(audio)
            audios.append(zero_wav)
            # t4 = ttime()

            # logging.debug(f"{t1 - t0:.3f}\t{t2 - t1:.3f}\t{t3 - t2:.3f}\t{t4 - t3:.3f}")
            audio = (np.concatenate(audios, 0) * 32768).astype(np.int16)
            return audio
