import torch

from utils.config_manager import global_config
from bert_vits2.text.japanese_v200 import text2sep_kata


def get_bert_feature(text, word2ph, tokenizer, model, device=global_config.DEVICE):
    sep_text, _, _ = text2sep_kata(text)
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    return get_bert_feature_with_token(sep_ids, word2ph, tokenizer, model, device)


def get_bert_feature_with_token(tokens, word2ph, tokenizer, model, device=global_config.DEVICE):
    with torch.no_grad():
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        # for i in inputs:
        #     inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
