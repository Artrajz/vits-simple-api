import os
import shutil
import logging

import torch
import yaml

import config as default_config

YAML_CONFIG_FILE = os.path.join(default_config.ABS_PATH, 'config.yml')


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def update(self, data: dict):
        self.__dict__.update(data)


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
    return yaml_config


def save_yaml_config(filename, data):
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w') as f:
            yaml.safe_dump(data, f)
        global global_config
        global_config.update(data)
        shutil.move(temp_filename, filename)
        logging.info(f"Saving yaml to {YAML_CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error while saving yaml: {e}")
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
