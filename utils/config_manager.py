import os
import secrets
import shutil
import logging

import torch
import yaml

import config as default_config
from utils.data_utils import check_is_none

YAML_CONFIG_FILE = os.path.join(default_config.ABS_PATH, 'config.yml')

logging.getLogger().setLevel(logging.DEBUG)

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


global_config = Config()


# 修改表示和构造器的定义
def represent_torch_device(dumper, device_obj):
    return dumper.represent_scalar('!torch.device', str(device_obj))


def construct_torch_device(loader, node):
    device_str = loader.construct_scalar(node)
    return torch.device(device_str)


yaml.add_representer(torch.device, represent_torch_device, Dumper=yaml.SafeDumper)
yaml.add_constructor('!torch.device', construct_torch_device, Loader=yaml.SafeLoader)


def load_yaml_config(filename):
    with open(filename, 'r') as f:
        yaml_config = yaml.safe_load(f)
    global global_config
    global_config.update(yaml_config)
    logging.info(f"Loading yaml from {YAML_CONFIG_FILE}")
    return Config(yaml_config)


def save_yaml_config(filename, data):
    temp_filename = filename + '.tmp'
    try:
        dict_data = dict(data)
        with open(temp_filename, 'w') as f:
            yaml.safe_dump(dict_data, f, default_style="'")
        global global_config
        global_config.update(dict_data)
        shutil.move(temp_filename, filename)
        logging.info(f"Saving yaml to {YAML_CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error while saving yaml: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def generate_secret_key(length=32):
    return secrets.token_hex(length)


model_path = ["MODEL_LIST", "HUBERT_SOFT_MODEL", "DIMENSIONAL_EMOTION_NPY", "DIMENSIONAL_EMOTION_MODEL"]
default_parameter = ["ID", "FORMAT", "LANG", "LENGTH", "NOISE", "NOISEW", "MAX", "SDP_RATIO"]

if os.path.exists(YAML_CONFIG_FILE):
    global_config.update(load_yaml_config(YAML_CONFIG_FILE))
else:
    global_config.setdefault("model_path", {})
    global_config.setdefault("default_parameter", {})

    for key, value in vars(default_config).items():
        if key.islower():
            continue
        if key in model_path:
            global_config["model_path"][key.lower()] = value
        elif key in default_parameter:
            global_config["default_parameter"][key.lower()] = value
        else:
            global_config[key] = value
    logging.info("config.yml not found. Generating a new config.yml based on config.py.")
    save_yaml_config(YAML_CONFIG_FILE, global_config)

if check_is_none(global_config.SECRET_KEY):
    secret_key = generate_secret_key()
    global_config["SECRET_KEY"] = secret_key
    logging.info(f"SECRET_KEY is not found or is None. Generating a new SECRET_KEY:{secret_key}")
    save_yaml_config(YAML_CONFIG_FILE, global_config)

if check_is_none(global_config.API_KEY):
    secret_key = generate_secret_key()
    global_config["API_KEY"] = secret_key
    logging.info(f"API_KEY is not found or is None. Generating a new API_KEY:{secret_key}")
    save_yaml_config(YAML_CONFIG_FILE, global_config)
