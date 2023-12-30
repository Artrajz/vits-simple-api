import torch


def get_clap_audio_feature(audio_data, clap_model, processor, device):
    with torch.no_grad():
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        emb = clap_model.get_audio_features(**inputs)
    return emb.T


def get_clap_text_feature(text, clap_model, processor, device):
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        emb = clap_model.get_text_features(**inputs)
    return emb.T
