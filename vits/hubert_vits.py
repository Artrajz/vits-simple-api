import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor, inference_mode, FloatTensor
import utils
from utils import get_hparams_from_file, lang_dict
from vits import commons
from vits.text import text_to_sequence
from vits.models import SynthesizerTrn


class HuBert_VITS:
    def __init__(self, model_path, config, device=torch.device("cpu"), **kwargs):
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.n_symbols = len(getattr(self.hps_ms, 'symbols', []))
        self.speakers = getattr(self.hps_ms, 'speakers', ['0'])
        if not isinstance(self.speakers, list):
            self.speakers = [item[0] for item in sorted(list(self.speakers.items()), key=lambda x: x[1])]
        self.use_f0 = getattr(self.hps_ms.data, 'use_f0', False)
        self.model_path = model_path
        self.device = device

        key = getattr(self.hps_ms.data, "text_cleaners", ["none"])[0]
        self.lang = lang_dict.get(key, ["unknown"])

    def load_model(self, hubert):
        self.hubert = hubert

        self.net_g_ms = SynthesizerTrn(
            self.n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.n_speakers,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(self.model_path, self.net_g_ms)
        self.net_g_ms.to(self.device)
        
    def get_cleaned_text(self, text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            if self.bert_embedding:
                text_norm, char_embed = text_to_sequence(text, hps.symbols, hps.data.text_cleaners,
                                                         bert_embedding=self.bert_embedding)
                text_norm = LongTensor(text_norm)
                return text_norm, char_embed
            else:
                text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def get_cleaner(self):
        return getattr(self.hps_ms.data, 'text_cleaners', [None])[0]

    def get_speakers(self, escape=False):
        return self.speakers

    @property
    def sampling_rate(self):
        return self.hps_ms.data.sampling_rate

    def infer(self, audio_path, id, noise, noisew, length, f0_scale=1, **kwargs):
        if self.use_f0:
            audio, sampling_rate = librosa.load(audio_path, sr=self.hps_ms.data.sampling_rate, mono=True)
            audio16000 = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        else:
            audio16000, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

        with inference_mode():
            units = self.hubert.units(FloatTensor(audio16000).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
            if self.use_f0:
                f0 = librosa.pyin(audio,
                                  sr=sampling_rate,
                                  fmin=librosa.note_to_hz('C0'),
                                  fmax=librosa.note_to_hz('C7'),
                                  frame_length=1780)[0]
                target_length = len(units[:, 0])
                f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                             np.arange(0, len(f0)), f0)) * f0_scale
                units[:, 0] = f0 / 10

        stn_tst = FloatTensor(units)
        id = LongTensor([id])

        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            id = id.to(self.device)

            audio = self.net_g_ms.infer(x=x_tst,
                                        x_lengths=x_tst_lengths,
                                        sid=id,
                                        noise_scale=noise,
                                        noise_scale_w=noisew,
                                        length_scale=length)[0][0, 0].data.float().cpu().numpy()

        torch.cuda.empty_cache()

        return audio
