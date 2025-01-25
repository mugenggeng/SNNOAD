# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_optimizer']

import torch.optim as optim
import timm.optim.optim_factory as optim_factory
from lr_decay_spikformer import param_groups_lrd

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
        optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=cfg.SOLVER.BASE_LR)
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(cfg.SOLVER.OPTIMIZER))
    return optimizer
