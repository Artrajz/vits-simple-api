from enum import Enum
from config import Config

config = Config.load_config()

class ModelType(Enum):
    VITS = "VITS"
    HUBERT_VITS = "HUBERT-VITS"
    W2V2_VITS = "W2V2-VITS"
    BERT_VITS2 = "BERT-VITS2"
    GPT_SOVITS = "GPT-SOVITS"
