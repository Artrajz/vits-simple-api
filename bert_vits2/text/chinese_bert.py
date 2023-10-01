import os

import config
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from logger import logger
from utils.download import download_and_verify
from config import DEVICE as device

URLS = [
    "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
]
TARGET_PATH = os.path.join(config.ABS_PATH, "bert_vits2/bert/chinese-roberta-wwm-ext-large/pytorch_model.bin")
EXPECTED_MD5 = None

if not os.path.exists(TARGET_PATH):
    success, message = download_and_verify(URLS, TARGET_PATH, EXPECTED_MD5)

try:
    logger.info("Loading chinese-roberta-wwm-ext-large...")
    tokenizer = AutoTokenizer.from_pretrained(config.ABS_PATH + "/bert_vits2/bert/chinese-roberta-wwm-ext-large")
    model = AutoModelForMaskedLM.from_pretrained(config.ABS_PATH + "/bert_vits2/bert/chinese-roberta-wwm-ext-large").to(
        device)
    logger.info("Loading finished.")
except Exception as e:
    logger.error(e)
    logger.error(f"Please download pytorch_model.bin from hfl/chinese-roberta-wwm-ext-large.")


def get_bert_feature(text, word2ph, device=config.DEVICE):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res['hidden_states'][-3:-2], -1)[0].cpu()

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


if __name__ == '__main__':
    import torch

    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
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
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
