from enum import Enum


class ModelType(str, Enum):
    VITS = "VITS"
    HUBERT_VITS = "HUBERT-VITS"
    W2V2_VITS = "W2V2-VITS"
    BERT_VITS2 = "BERT-VITS2"
    GPT_SOVITS = "GPT-SOVITS"
