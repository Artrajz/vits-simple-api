import torch

from config import config


def get_bert_feature(text, word2ph, tokenizer, model, device=config.system.device, style_text=None, style_weight=0.7,
                     **kwargs):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.nn.functional.normalize(torch.cat(res["hidden_states"][-3:-2], -1)[0], dim=0).float().cpu()
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.nn.functional.normalize(
                torch.cat(style_res["hidden_states"][-3:-2], -1)[0], dim=0
            ).float().cpu()
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


if __name__ == '__main__':

    word_level_feature = torch.rand(38, 2048)  # 12个词,每个词2048维特征
    word2phone = [1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2,
                  2, 2, 2, 1]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 2048])
