import logging
import os
import secrets
import shutil
import string
import sys
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import List, Union

import torch
import yaml

JSON_AS_ASCII = False

MAX_CONTENT_LENGTH = 5242880

# Absolute path of vits-simple-api
ABS_PATH = os.path.dirname(os.path.realpath(__file__))

# WTForms CSRF
SECRET_KEY = secrets.token_hex(16)

# Fill in the models path here
model_list = [
    # VITS
    # {"model_path": "g/G_953000.pth", "config_path": "g/config.json"},
    # HuBert-VITS (Need to configure HUBERT_SOFT_MODEL)
    # {"model_path": "louise/360_epochs.pth", "config_path": "louise/config.json"},
    # W2V2-VITS (Need to configure DIMENSIONAL_EMOTION_NPY)
    # {"model_path": "w2v2-vits/1026_epochs.pth", "config_path": "w2v2-vits/1026_epochs.pth"},
    # Bert-VITS2
    # {"model_path": "bert_vits2/G_9000.pth", "config_path": "bert_vits2/config.json"},
]


# # torch.device
# def represent_torch_device(dumper, device_obj):
#     return dumper.represent_scalar('!torch.device', str(device_obj))
# 
# 
# def construct_torch_device(loader, node):
#     device_str = loader.construct_scalar(node)
#     return torch.device(device_str)
# 
# 
# yaml.add_representer(torch.device, represent_torch_device, Dumper=yaml.SafeDumper)
# yaml.add_constructor('!torch.device', construct_torch_device, Loader=yaml.SafeLoader)


@dataclass
class AsDictMixin:
    def asdict(self):
        return asdict(self)

    def __iter__(self):
        for key, value in self.asdict().items():
            yield key, value

    def update_config(self, new_config_dict):
        for field in fields(self):
            field_name = field.name
            field_type = field.type

            if field_name in new_config_dict:
                new_value = new_config_dict[field_name]

                if is_dataclass(field_type):
                    nested_config = getattr(self, field_name)
                    nested_config.update_config(new_value)
                elif field_type == bool:
                    new_value = bool(new_value)
                elif field_type == int:
                    new_value = int(new_value)
                elif field_type == float:
                    new_value = float(new_value)
                elif field_type == str:
                    new_value = str(new_value)
                elif field_type == torch.device:
                    new_value = torch.device(new_value)

                setattr(self, field_name, new_value)


@dataclass
class VitsConfig(AsDictMixin):
    # For VITS: Load models during inference, dynamically release models after inference.
    dynamic_loading: bool = False
    id: int = 0
    format: str = "wav"
    lang: str = "auto"
    length: float = 1
    noise: float = 0.33
    noisew: float = 0.4
    # Batch processing threshold. Text will not be processed in batches if segment_size<=0
    segment_size: int = 50


@dataclass
class W2V2VitsConfig(AsDictMixin):
    id: int = 0
    format: str = "wav"
    lang: str = "auto"
    length: float = 1
    noise: float = 0.33
    noisew: float = 0.4
    # Batch processing threshold. Text will not be processed in batches if segment_size<=0
    segment_size: int = 50
    emotion: int = 0


@dataclass
class HuBertVitsConfig(AsDictMixin):
    id: int = 0
    format: str = "wav"
    length: float = 1
    noise: float = 0.33
    noisew: float = 0.4


@dataclass
class BertVits2Config(AsDictMixin):
    id: int = 0
    format: str = "wav"
    lang: str = "auto"
    length: float = 1
    noise: float = 0.33
    noisew: float = 0.4
    # Batch processing threshold. Text will not be processed in batches if segment_size<=0
    segment_size: int = 50
    sdp_ratio: float = 0.2
    emotion: int = 0
    text_prompy: str = "Happy"
    style_weight: float = 0.7


@dataclass
class ModelConfig(AsDictMixin):
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
    # hubert-vits
    hubert_soft_0d54a1f4: str = "hubert_soft/hubert-soft-0d54a1f4.pt"
    # w2v2-vits: .npy file or folder are alvailable
    dimensional_emotion_npy: Union[str, List[str]] = "dimensional_emotion_npy"
    # w2v2-vits: Need to have both `models.onnx` and `models.yaml` files in the same path.
    dimensional_emotion_model: str = "dimensional_emotion_model/models.yaml"


@dataclass
class TTSModelConfig(AsDictMixin):
    model_path: str
    config_path: str


@dataclass
class TTSConfig(AsDictMixin):
    models: List[TTSModelConfig] = field(default_factory=list)


@dataclass
class HttpService(AsDictMixin):
    host: str = "0.0.0.0"
    port: int = 23456
    debug: bool = False


@dataclass
class LogConfig(AsDictMixin):
    # Logs path
    logs_path: str = "logs"
    # Set the number of backup log files to keep. 
    logs_backupcount: int = 30
    # logging_level:DEBUG/INFO/WARNING/ERROR/CRITICAL
    logging_level: str = "DEBUG"


@dataclass
class System(AsDictMixin):
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Upload path
    upload_folder: str = "upload"
    # Cahce path
    cache_path: str = "cache"
    # If CLEAN_INTERVAL_SECONDS <= 0, the cleaning task will not be executed.
    clean_interval_seconds: int = 3600
    # save audio to CACHE_PATH
    cache_audio: bool = False
    # Set to True to enable API Key authentication
    api_key_enabled: bool = False
    # API_KEY is required for authentication
    api_key: str = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))
    # Control whether to enable the admin backend functionality. Set to False to disable the admin backend.
    is_admin_enabled: bool = True
    # Define the route for the admin backend. You can change this to your desired route
    admin_route: str = '/admin'
    # Path to the 'data' folder, where various models are stored
    data_path: str = "data"


@dataclass
class LanguageIdentification(AsDictMixin):
    # Language identification library. Optional fastlid, langid
    language_identification_library: str = "langid"
    # To use the english_cleaner, you need to install espeak and provide the path of libespeak-ng.dll as input here.
    # If ESPEAK_LIBRARY is set to empty, it will be read from the environment variable.
    # For windows : "C:/Program Files/eSpeak NG/libespeak-ng.dll"
    espeak_library: str = r"C:/Program Files/eSpeak NG/libespeak-ng.dll" if "win" in sys.platform else ""
    # zh ja ko en... If it is empty, it will be read based on the text_cleaners specified in the config.json.
    language_automatic_detect: list = field(default_factory=list)


@dataclass
class User(AsDictMixin):
    id: int = 0
    username: str = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    password: str = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


@dataclass
class Config(AsDictMixin):
    abs_path: str = ABS_PATH
    http_service: HttpService = HttpService()
    log_config: LogConfig = LogConfig()
    system: System = System()
    language_identification: LanguageIdentification = LanguageIdentification()
    vits_config: VitsConfig = VitsConfig()
    w2v2_vits_config: W2V2VitsConfig = W2V2VitsConfig()
    hubert_vits_config: HuBertVitsConfig = HuBertVitsConfig()
    bert_vits2_config: BertVits2Config = BertVits2Config()
    model_config: ModelConfig = ModelConfig()
    tts_config: TTSConfig = TTSConfig()
    admin: User = User()

    def asdict(self):
        self.system.device = str(self.system.device)
        data = asdict(self)
        self.system.device = torch.device(self.system.device)
        return data

    @staticmethod
    def load_config():
        logging.getLogger().setLevel(logging.INFO)
        config_path = os.path.join(Config.abs_path, "config.yaml")
        if not os.path.exists(config_path):
            logging.info("config.yaml not found. Generating a new config.yaml based on config.py.")
            config = Config()
            # 初始化管理员账号密码
            logging.info(
                f"New admin user created:\n"
                f"{'-' * 40}\n"
                f"| Username: {config.admin.username:<26} |\n"
                f"| Password: {config.admin.password:<26} |\n"
                f"{'-' * 40}\n"
                f"Please do not share this information.")
            Config.save_config(config)
            return config
        else:
            try:
                logging.info("Loading config...")
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    config = Config()
                    config.update_config(loaded_config)
                    logging.info("Loading config success!")
                    return config
            except Exception as e:
                ValueError(e)

    @staticmethod
    def save_config(config):
        temp_filename = os.path.join(Config.abs_path, "config.yaml.tmp")
        with open(temp_filename, 'w') as f:
            yaml.safe_dump(config.asdict(), f, default_style=None)
        shutil.move(temp_filename, os.path.join(Config.abs_path, "config.yaml"))
        logging.info(f"Config is saved.")

    def update_config(self, new_config_dict):
        for field in fields(self):
            field_name = field.name
            field_type = field.type

            if field_name in new_config_dict:
                new_value = new_config_dict[field_name]

                if is_dataclass(field_type):
                    nested_config = getattr(self, field_name)
                    nested_config.update_config(new_value)
                else:
                    if field_type == bool:
                        new_value = bool(new_value)
                    elif field_type == int:
                        new_value = int(new_value)
                    elif field_type == float:
                        new_value = float(new_value)
                    elif field_type == str:
                        new_value = str(new_value)
                    elif field_type == torch.device:
                        new_value = torch.device(new_value)

                    setattr(self, field_name, new_value)

