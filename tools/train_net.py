# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(os.getcwd()+'src/')
sys.path.append(os.getcwd()+'/src/')
# sys.path.append('/home/dx/data/houlin/RGB_Only/src/')
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.datasets import build_data_loader
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.criterions import build_criterion,BRDLoss,LaSNNLoss
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.optimizers import build_ema
from rekognition_online_action_detection.engines import do_train
from rekognition_online_action_detection.optimizers import NativeScalerWithGradNormCount as NativeScaler


from thop import profile
from thop import clever_format
import torch
def main(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    # print(device,'device')
    checkpointer = setup_checkpointer(cfg, phase='train')
    # print('22222222')
    logger = setup_logger(cfg, phase='train')

    # Build data loaders
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # Build model
    # print('333333333')
    model = build_model(cfg,device=device)

    cfg.MODEL.CHECKPOINT = './MatCheckpoints/epoch-CCI_2.pth'
    checkpointer1 = setup_checkpointer(cfg, phase='test')
    teach_model = build_model(cfg,name='MAT',device=device)
    checkpointer1.load(teach_model)
    # teach_model.eval()

    # bdr_loss = LaSNNLoss()
    # bdr_loss.to(device)
    # print(teach_model)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model1 = build_model(cfg, device)
    # model1.eval()
    # input1 = torch.randn(1, 1024, 256)
    # input2 = torch.randn(1, 1024, 256)
    # flops, params = profile(model1, inputs=(input,input2),verbose=False)
    # print(flops, params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

    # print(model)
    # print(model.device,'model.device')
    # print(next(model.parameters()).device)
    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    # param_groups = lrd.param_groups_lrd(
    #     model_without_ddp,
    #     args.weight_decay,
    #     # no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay,
    # )
    # optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
    loss_scaler = NativeScaler()
    # Build ema
    ema = build_ema(model, 0.999)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    do_train(
        cfg,
        data_loaders,
        model,
        teach_model,
        criterion,
        loss_scaler,
        optimizer,
        scheduler,
        ema,
        device,
        checkpointer,
        logger,
    )


if __name__ == '__main__':
    # print('11111111111111111')
    main(load_cfg())
