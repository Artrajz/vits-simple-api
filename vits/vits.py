import torch
from torch import no_grad, LongTensor
import utils
from utils import get_hparams_from_file, lang_dict
from vits import commons
from vits.mel_processing import spectrogram_torch
from vits.text import text_to_sequence
from vits.models import SynthesizerTrn


class VITS:
    def __init__(self, model_path, config, device="cpu", **kwargs):
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.n_speakers = getattr(self.hps_ms.data, 'n_speakers', 0)
        self.n_symbols = len(getattr(self.hps_ms, 'symbols', []))
        self.speakers = getattr(self.hps_ms, 'speakers', ['0'])
        if not isinstance(self.speakers, list):
            self.speakers = [item[0] for item in sorted(list(self.speakers.items()), key=lambda x: x[1])]
        self.bert_embedding = getattr(self.hps_ms.data, 'bert_embedding',
                                      getattr(self.hps_ms.model, 'bert_embedding', False))
        self.hps_ms.model.bert_embedding = self.bert_embedding
        self.text_cleaners = getattr(self.hps_ms.data, 'text_cleaners', [None])[0]
        self.sampling_rate = self.hps_ms.data.sampling_rate
        self.device = device
        self.model_path = model_path

        # load checkpoint
        # self.load_model()

        self.lang = lang_dict.get(self.text_cleaners, ["unknown"])

    def load_model(self):
        self.net_g_ms = SynthesizerTrn(
            self.n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.n_speakers,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(self.model_path, self.net_g_ms)
        self.net_g_ms.to(self.device)
    
    def release_model(self):
        del self.net_g_ms
        

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

    def infer(self, text, id, noise, noisew, length, cleaned=False, **kwargs):
        char_embeds = None
        if self.bert_embedding:
            stn_tst, char_embeds = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
        else:
            stn_tst = self.get_cleaned_text(text, self.hps_ms, cleaned=cleaned)
        id = LongTensor([id])

        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(
                self.device) if self.bert_embedding else None
            id = id.to(self.device)

            audio = self.net_g_ms.infer(x=x_tst,
                                        x_lengths=x_tst_lengths,
                                        sid=id,
                                        noise_scale=noise,
                                        noise_scale_w=noisew,
                                        length_scale=length,
                                        bert=x_tst_prosody)[0][0, 0].data.float().cpu().numpy()

        torch.cuda.empty_cache()

        return audio

    def voice_conversion(self, audio_path, original_id, target_id):

        audio = utils.load_audio_to_torch(
            audio_path, self.sampling_rate)

        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, self.hps_ms.data.filter_length,
                                 self.sampling_rate, self.hps_ms.data.hop_length,
                                 self.hps_ms.data.win_length,
                                 center=False)
        spec_lengths = LongTensor([spec.size(-1)])
        sid_src = LongTensor([original_id])

        with no_grad():
            sid_tgt = LongTensor([target_id])
            audio = self.net_g_ms.voice_conversion(spec.to(self.device),
                                                   spec_lengths.to(self.device),
                                                   sid_src=sid_src.to(self.device),
                                                   sid_tgt=sid_tgt.to(self.device))[0][0, 0].data.cpu().float().numpy()

        torch.cuda.empty_cache()

        return audio
