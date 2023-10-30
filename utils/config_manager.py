import os
import shutil

import torch
import yaml

import config as default_config
from logger import logger

YAML_CONFIG_FILE = os.path.join(default_config.ABS_PATH, 'config.yml')

global_config = {}


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
    logger.info(f"Loading yaml from {YAML_CONFIG_FILE}")
    return yaml_config


def save_yaml_config(filename, data):
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w') as f:
            yaml.safe_dump(data, f)

        shutil.move(temp_filename, filename)
        logger.info(f"Saving yaml to {YAML_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error while saving yaml: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)



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
    save_yaml_config(YAML_CONFIG_FILE, global_config)
