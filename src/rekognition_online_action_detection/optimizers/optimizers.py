# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_optimizer']

import torch.optim as optim
import timm.optim.optim_factory as optim_factory
# from .lr_decay_spikformer import param_groups_lrd
import json
def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = 3

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:  # 仅针对需要利用梯度进行更新的参数
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name in ['final_query', 'pos_encoding']:
        return 0
    elif name.startswith("block"):
        # return int(name.split('.')[1]) + 1
        return num_layers
    else:
        return num_layers

def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM,
        )
    elif cfg.SOLVER.OPTIMIZER == 'RMSprop':
        optimizer = optim.RMSprop(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, alpha=0.9, eps=1e-6
        )
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == 'Lamb':
        param_groups = param_groups_lrd(
            model,
            cfg.SOLVER.WEIGHT_DECAY,
            # no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=1.0,
        )
        # param_groups[0]['initial_lr'] = cfg.SOLVER.BASE_LR
        optimizer = optim_factory.Lamb(model.parameters(), trust_clip=True, lr=cfg.SOLVER.BASE_LR)
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(cfg.SOLVER.OPTIMIZER))
    return optimizer
