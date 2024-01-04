import torch

from utils.config_manager import global_config
from bert_vits2.text.japanese import text2sep_kata

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"


def get_bert_feature(text, word2ph, tokenizer, model, device=global_config.DEVICE, style_text=None, style_weight=0.7, **kwargs):
    text = "".join(text2sep_kata(text)[0])
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if style_text:
            repeat_feature = (
                    res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                    + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
