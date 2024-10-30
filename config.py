import io
import logging
import os
import secrets
import string
import sys

import torch
import yaml
from typing import List, Union, Optional, Dict, Type
from pydantic import BaseModel, Field, ValidationError

from contants import ModelType

JSON_AS_ASCII = False
MAX_CONTENT_LENGTH = 5242880

# Absolute path of vits-simple-api (current program root path)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Configuration file path
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# WTForms CSRF
SECRET_KEY = secrets.token_hex(16)


def update_nested_dict(original, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            update_nested_dict(original[key], value)
        else:
            original[key] = value


class VitsConfig(BaseModel):
    id: int = 0
    format: str = "wav"
    lang: str = "auto"
    length: float = 1.0
    noise: float = 0.33
    noisew: float = 0.4
    segment_size: int = 50
    use_streaming: bool = False


class W2V2VitsConfig(BaseModel):
    id: int = 0
    format: str = "wav"
    lang: str = "auto"
    length: float = 1.0
    noise: float = 0.33
    noisew: float = 0.4
    segment_size: int = 50
    emotion: int = 0


class HuBertVitsConfig(BaseModel):
    id: int = 0
    format: str = "wav"
    length: float = 1.0
    noise: float = 0.33
    noisew: float = 0.4


class BertVits2Config(BaseModel):
    id: int = 0
    speaker: Optional[str] = None
    format: str = "wav"
    lang: str = "auto"
    length: float = 1.0
    noise: float = 0.33
    noisew: float = 0.4
    segment_size: int = 50
    sdp_ratio: float = 0.2
    emotion: int = 0
    text_prompt: str = "Happy"
    style_text: Optional[str] = None
    style_weight: float = 0.7
    use_streaming: bool = False
    torch_data_type: Optional[str] = None


class GPTSoVitsPreset(BaseModel):
    refer_wav_path: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_lang: str = "auto"


class GPTSoVitsConfig(BaseModel):
    hz: int = 50
    is_half: bool = False
    id: int = 0
    lang: str = "auto"
    format: str = "wav"
    segment_size: int = 30
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    use_streaming: bool = False
    batch_size: int = 5
    speed: float = 1.0
    seed: int = -1
    presets: Dict[str, GPTSoVitsPreset] = Field(default_factory=lambda: {"default": GPTSoVitsPreset(),
                                                                         "default2": GPTSoVitsPreset()})


class Reader(BaseModel):
    model_type: str = "VITS"
    id: int = 0
    preset: str = "default"


class ReadingConfig(BaseModel):
    interlocutor: Reader = Reader()
    narrator: Reader = Reader()


class ResourcePathsConfig(BaseModel):
    chinese_roberta_wwm_ext_large: str = "bert/chinese-roberta-wwm-ext-large"
    bert_base_japanese_v3: str = "bert/bert-base-japanese-v3"
    bert_large_japanese_v2: str = "bert/bert-large-japanese-v2"
    deberta_v2_large_japanese: str = "bert/deberta-v2-large-japanese"
    deberta_v3_large: str = "bert/deberta-v3-large"
    deberta_v2_large_japanese_char_wwm: str = "bert/deberta-v2-large-japanese-char-wwm"
    wav2vec2_large_robust_12_ft_emotion_msp_dim: str = "emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    clap_htsat_fused: str = "emotional/clap-htsat-fused"
    erlangshen_MegatronBert_1_3B_Chinese: str = "bert/Erlangshen-MegatronBert-1.3B-Chinese"
    vits_chinese_bert: str = "bert/vits_chinese_bert"
    hubert_soft_0d54a1f4: str = "hubert/hubert_soft/hubert-soft-0d54a1f4.pt"
    dimensional_emotion_npy: Union[str, List[str]] = "emotional/dimensional_emotion_npy"
    dimensional_emotion_model: str = "emotional/dimensional_emotion_model/models.yaml"
    g2pw_model: str = "G2PWModel"
    chinese_hubert_base: str = "hubert/chinese_hubert_base"


class BaseModelConfig(BaseModel):
    model_type: str

    class Config:
        protected_namespaces = ()


class VITSModelConfig(BaseModelConfig):
    model_type: str = ModelType.VITS
    vits_path: str = None
    config_path: str = None


class W2V2VITSModelConfig(BaseModelConfig):
    model_type: str = ModelType.W2V2_VITS
    vits_path: str = None
    config_path: str = None


class HuBertVITSModelConfig(BaseModelConfig):
    model_type: str = ModelType.HUBERT_VITS
    vits_path: str = None
    config_path: str = None


class BertVITS2ModelConfig(BaseModelConfig):
    model_type: str = ModelType.BERT_VITS2
    vits_path: str = None
    config_path: str = None


class GPTSoVITSModelConfig(BaseModelConfig):
    model_type: str = ModelType.GPT_SOVITS
    gpt_path: str = None
    sovits_path: str = None


MODEL_TYPE_MAP: Dict[str, Type[BaseModelConfig]] = {
    ModelType.VITS: VITSModelConfig,
    ModelType.W2V2_VITS: W2V2VITSModelConfig,
    ModelType.HUBERT_VITS: HuBertVITSModelConfig,
    ModelType.BERT_VITS2: BertVITS2ModelConfig,
    ModelType.GPT_SOVITS: GPTSoVITSModelConfig
}


class TTSModelConfig(BaseModel):
    models_dir: str = "models"
    auto_load: bool = True
    tts_models: List[Union[
        VITSModelConfig,
        W2V2VITSModelConfig,
        HuBertVITSModelConfig,
        BertVITS2ModelConfig,
        GPTSoVITSModelConfig,
    ]] = Field(default_factory=list)

    def add_model(self, model_config: BaseModelConfig):
        if not isinstance(model_config, BaseModelConfig):
            raise TypeError("model_config must be an instance of BaseModelConfig")

        model_class = MODEL_TYPE_MAP.get(model_config.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model_type: {model_config.model_type}")

        self.tts_models.append(model_class(**model_config.model_dump()))

    def update_tts_models(self, tts_models: list):
        self.tts_models = []
        for item in tts_models:
            tts_model = item["tts_model"]
            model_type = tts_model.get("model_type")
            if model_type:
                model_type = model_type.upper().replace("_", "-")
            model_class = MODEL_TYPE_MAP.get(ModelType(model_type))
            if model_class is not None:
                try:
                    model_instance = model_class.model_validate(tts_model)
                    self.tts_models.append(model_instance)
                except ValidationError as e:
                    logging.error(f"Validation error for item {tts_model}: {e}")
            else:
                logging.error(f"Unknown model_type in data: {model_type}")


class HttpService(BaseModel):
    host: str = "0.0.0.0"
    port: int = 23456
    debug: bool = False
    origins: str = "*"


class LogConfig(BaseModel):
    logs_path: str = "logs"
    logs_backup_count: int = 30
    logging_level: str = "DEBUG"


class APIKey(BaseModel):
    key: str = Field(
        default_factory=lambda: ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24)))
    enabled: bool = True


class System(BaseModel):
    device: str = Field(default_factory=lambda: str(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    upload_folder: str = "upload"
    cache_path: str = "cache"
    clean_interval_seconds: int = 3600
    cache_audio: bool = False
    api_key_enabled: bool = False
    api_keys: List[APIKey] = Field(default_factory=lambda: [APIKey() for _ in range(5)])
    is_admin_enabled: bool = True
    admin_route: str = '/admin'
    data_path: str = "data"


class LanguageIdentification(BaseModel):
    language_identification_library: str = "langid"
    espeak_library: str = r"C:/Program Files/eSpeak NG/libespeak-ng.dll" if "win" in sys.platform else ""
    language_automatic_detect: List[str] = Field(default_factory=list)
    split_pattern: str = r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`' \
                         r'\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」' \
                         r'『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+'


class User(BaseModel):
    id: int = 0
    username: str = Field(
        default_factory=lambda: ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8)))
    password: str = Field(
        default_factory=lambda: ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16)))

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


class NgrokConfig(BaseModel):
    auth_token: Optional[str] = None


class Config(BaseModel):
    http_service: HttpService = HttpService()
    ngrok_config: NgrokConfig = NgrokConfig()
    resource_paths_config: ResourcePathsConfig = ResourcePathsConfig()
    tts_model_config: TTSModelConfig = TTSModelConfig()
    admin: User = User()
    system: System = System()
    log_config: LogConfig = LogConfig()
    language_identification: LanguageIdentification = LanguageIdentification()
    reading_config: ReadingConfig = ReadingConfig()
    vits_config: VitsConfig = VitsConfig()
    w2v2_vits_config: W2V2VitsConfig = W2V2VitsConfig()
    hubert_vits_config: HuBertVitsConfig = HuBertVitsConfig()
    bert_vits2_config: BertVits2Config = BertVits2Config()
    gpt_sovits_config: GPTSoVitsConfig = GPTSoVitsConfig()

    @staticmethod
    def load_config(file_path: str):
        if not os.path.exists(file_path):
            config = Config()
            save_config_to_yaml(config)
            return config

        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        if config_data:
            try:
                config = Config(**config_data)
            except ValidationError as e:
                logging.error(f"Config validation error: {e}")
                config = Config()  # Load defaults
                for error in e.errors():
                    field = error['loc'][0]
                    if field in config.__annotations__:
                        default_value = getattr(Config, field, None)
                        if default_value is not None:
                            # Apply default value
                            setattr(config, field, default_value)
        else:
            config = Config()

        save_config_to_yaml(config)

        return config

    def update_config(self, update_data: Dict):
        try:
            new_config_data = self.model_dump()

            update_nested_dict(new_config_data, update_data)

            updated_config = Config(**new_config_data)

            save_config_to_yaml(updated_config)

            self.__dict__.update(updated_config.__dict__)

            return self
        except ValidationError as e:
            logging.error(f"Config validation error: {e}")
            return self


def save_config_to_yaml(config: Config):
    temp_file = io.StringIO()

    yaml.safe_dump(config.model_dump(), temp_file, allow_unicode=True, sort_keys=False)

    data = temp_file.getvalue()
    temp_file.close()

    with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
        file.write(data)


config = Config.load_config(CONFIG_PATH)
