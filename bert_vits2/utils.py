import os
import sys
import logging
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False, version=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    elif optimizer is None and not skip_optimizer:
        # else: #Disable this line if Infer ,and enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict['param_groups'][0]['params']
        new_opt_dict['param_groups'] = checkpoint_dict['optimizer']['param_groups']
        new_opt_dict['param_groups'][0]['params'] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except:
            # Handle legacy model versions and provide appropriate warnings
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                if version is None:
                    logger.error(f"{k} is not in the checkpoint")
                    logger.warning(
                        f"If you're using an older version of the model, consider adding the \"version\" parameter to the model's config.json under the \"data\" section. For instance: \"legacy_version\": \"1.0.1\"")
            elif "flow.flows.0.enc.attn_layers.3" in k:
                logger.error(f"{k} is not in the checkpoint")
                logger.warning(
                    f"If you're using a transitional version, please add the \"version\": \"1.1.0-transition\" parameter within the \"data\" section of the model's config.json.")
            else:
                logger.error(f"{k} is not in the checkpoint")

            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    # print("load ")
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def process_legacy_versions(hps):
    version = getattr(hps, "version", getattr(hps.data, "version", None))
    if version:
        prefix = version[0].lower()
        if prefix == "v":
            version = version[1:]
    return version
