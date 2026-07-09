import os
import importlib

import torch
import wandb

import train  # run_fold() 재사용

PC_CFG = 'pad_cfg'
MODEL_CFG = '2019.pad_bio_baseline'


def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_mdl_name(fusion_method, use_meta_after, lr):
    fusion_tag = 'co' if fusion_method == 'concat' else 'ew'
    after_tag = '_after1024' if use_meta_after else ''
    lr_tag = f"{lr:.0e}"  # 0.001 -> "1e-03", 0.0001 -> "1e-04", 0.00001 -> "1e-05"
    return f"bio_{fusion_tag}{after_tag}_{lr_tag}"


def main():
    wandb.init(project="mediskin")
    cfg = wandb.config

    pc_cfg = importlib.import_module('pc_cfgs.' + PC_CFG)
    mdlParams = {}
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.' + MODEL_CFG)
    mdlParams.update(model_cfg.init(mdlParams))

    mdlParams['trainSetState'] = 'train'
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + MODEL_CFG

    mdlParams['learning_rate_meta'] = cfg.learning_rate_meta
    mdlParams['fc_layers_after'] = [1024] if cfg.use_meta_after else []
    mdlParams['fusion_method'] = cfg.fusion_method
    mdlParams['epoch'] = 50

    mdlParams['mdlName'] = build_mdl_name(cfg.fusion_method, cfg.use_meta_after, cfg.learning_rate_meta)
    wandb.run.name = mdlParams['mdlName']

    mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/' + mdlParams['mdlName']
    os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
    os.makedirs(mdlParams['saveDir'], exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")

    train.run_fold(mdlParams, device, cv=0)

    wandb.finish()


if __name__ == '__main__':
    main()
