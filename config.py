"""
在首次启动自动生成config.yaml文件后，对配置进行修改时，应该直接在config.yaml文件中进行，而不是在config.py文件中修改。

初回の起動後にconfig.yamlが自動生成された場合、設定の変更はconfig.pyではなくconfig.yamlで行うべきです。

After the initial launch that automatically generates the config.yaml file, any modifications to the configuration should be made directly in the config.yaml file, not in the config.py file.
"""

import copy
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

"""
模型存放在data/models文件夹下，每个文件夹包含一个模型文件和一个配置文件。请按照以下格式填写路径信息：
{"model_path": "文件夹名/模型文件.pth", "config_path": "文件夹名/config.json"},
注意：只有当auto_load（自动加载模型）为False时才有效。当auto_load为True（默认值）时，
此处填写的模型路径具有最高优先级，将在每次启动时加载。如非必要，请尽量在config.yaml填写模型路径。

Models are stored in the data/models folder, with each folder containing a model file and a configuration file. 
Please fill in the paths following the format: {"model_path": "folder_name/model_file.pth", "config_path": "folder_name/config.json"},
Note: This is effective only when auto_load (automatic model loading) is set to False. When auto_load is True (default),
the model paths specified here have the highest priority and will be loaded each time the program starts. If not necessary, 
it's recommended to specify model paths in config.yaml.
"""

model_list = [
    # {"model_path": "model_name/G_9000.pth", "config_path": "model_name/config.json"},
]


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
                    if isinstance(new_value, list):
                        # If the field type is a dataclass and the new value is a list
                        # Convert each element of the list to the corresponding class object
                        new_value = [field_type(**item) for item in new_value]
                        setattr(self, field_name, new_value)
                    else:
                        # If the field type is a dataclass but not a list, recursively update the dataclass
                        nested_config = getattr(self, field_name)
                        nested_config.update_config(new_value)
                else:
                    if field_type == bool:
                        new_value = str(new_value).lower() == "true"
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
    use_streaming: bool = False


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
    text_prompt: str = "Happy"
    style_text: str = "Happy"
    style_weight: float = 0.7
    use_streaming: bool = False
    # Can be set to "float16"/"fp16" or "int8".
    torch_data_type:str = ""


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
    # Directory name for models under the data folder
    models_path: str = "models"
    # List to store configurations of Text-to-Speech models
    models: List[TTSModelConfig] = field(default_factory=list)
    # If set to True (default), models under the specified models_path will be automatically loaded.
    # When set to False, you can manually specify the models to load.
    auto_load: bool = True

    def asdict(self):
        config_copy = copy.deepcopy(self)
        config_copy.models = [[tts_model.model_path, tts_model.config_path] for tts_model in self.models]
        data = asdict(config_copy)
        return data

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
                        new_value = str(new_value).lower() == "true"
                    elif field_type == int:
                        new_value = int(new_value)
                    elif field_type == float:
                        new_value = float(new_value)
                    elif field_type == str:
                        new_value = str(new_value)
                    elif field_type == torch.device:
                        new_value = torch.device(new_value)
                    elif field_type == List[TTSModelConfig]:
                        new_value = [TTSModelConfig(model.get("model_path"), model.get("config_path")) for model in
                                     new_value]

                    setattr(self, field_name, new_value)


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

    def asdict(self):
        self.models = [[tts_model.model_path, tts_model.config_path] for tts_model in self.models]
        return asdict(self)


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
        if not os.path.exists(config_path) or not os.path.isfile(config_path):
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

                if loaded_config is not None:
                    config.update_config(loaded_config)
                    logging.info("Loading config success!")
                else:
                    logging.info("config.yaml is empty, initializing config.yaml...")

                # Load default models from config.py.
                config.update_config(model_list)

                # If parameters are incomplete, they will be automatically filled in upon saving.
                Config.save_config(config)
                
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
