import librosa
import re
import numpy as np
import torch
from torch import no_grad, LongTensor
import utils
from utils import get_hparams_from_file, lang_dict
from vits import commons
from vits.text import text_to_sequence
from vits.models import SynthesizerTrn


class W2V2_VITS:
    def __init__(self, model_path, config, device=torch.device("cpu"), **kwargs):
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.n_symbols = len(getattr(self.hps_ms, 'symbols', []))
        self.speakers = getattr(self.hps_ms, 'speakers', ['0'])
        if not isinstance(self.speakers, list):
            self.speakers = [item[0] for item in sorted(list(self.speakers.items()), key=lambda x: x[1])]
        self.use_f0 = getattr(self.hps_ms.data, 'use_f0', False)
        self.emotion_embedding = getattr(self.hps_ms.data, 'emotion_embedding',
                                         getattr(self.hps_ms.model, 'emotion_embedding', False))
        self.hps_ms.model.emotion_embedding = self.emotion_embedding

        self.text_cleaners = getattr(self.hps_ms.data, 'text_cleaners', [None])[0]
        self.sampling_rate = self.hps_ms.data.sampling_rate
        self.device = device
        self.model_path = model_path

        self.lang = lang_dict.get(self.text_cleaners, ["unknown"])

    def load_model(self, emotion_reference, dimensional_emotion_model):
        self.emotion_reference = emotion_reference
        self.dimensional_emotion_model = dimensional_emotion_model

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
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def infer(self, text, id, noise, noisew, length, emotion, cleaned=False, **kwargs):
        stn_tst = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
        id = LongTensor([id])

        if isinstance(emotion, int):
            emotion_emb = self.emotion_reference[emotion]
        elif isinstance(emotion, str) and emotion.endswith('.npy'):
            emotion_emb = np.load(emotion).reshape(-1, 1024)[0]
        else:
            audio16000, sampling_rate = librosa.load(emotion, sr=16000, mono=True)
            emotion_emb = self.dimensional_emotion_model(audio16000, sampling_rate)['hidden_states']
            emotion_emb = re.sub(r'\..*$', '', emotion_emb)

        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            id = id.to(self.device)
            emotion_emb = torch.FloatTensor(emotion_emb).unsqueeze(0).to(self.device)

            audio = self.net_g_ms.infer(x=x_tst,
                                        x_lengths=x_tst_lengths,
                                        sid=id,
                                        noise_scale=noise,
                                        noise_scale_w=noisew,
                                        length_scale=length,
                                        emotion_embedding=emotion_emb)[0][0, 0].data.float().cpu().numpy()

        torch.cuda.empty_cache()

        return audio
