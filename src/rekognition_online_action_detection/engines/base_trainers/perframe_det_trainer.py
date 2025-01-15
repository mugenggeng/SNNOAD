# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0



import time
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from thop import profile
from rekognition_online_action_detection.evaluation import compute_result
from torch.cuda.amp import GradScaler, autocast

import sys
# sys.path.append('../../../../external/')
# sys.path.append("/home/wanghongtao/project/Memory-and-Anticipation-Transformer/")
# sys.path.append("/home/dx/data/houlin/Memory-and-Anticipation-Transformer/")
sys.path.append(os.getcwd())
try:
    from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                                        topk_accuracy_multiple_timesteps,
                                                        topk_recall_multiple_timesteps,
                                                        tta)
except:
    raise ModuleNotFoundError
def get_logits_loss(fc_t, fc_s, label, temp, num_classes=1000):
    # print(fc_t.shape,fc_s.shape,label.shape)
    fc_s = fc_s.reshape(-1,num_classes)
    fc_t = fc_t.reshape(-1,num_classes)
    s_input_for_softmax = fc_s / temp
    t_input_for_softmax = fc_t / temp

    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    t_soft_label = softmax(t_input_for_softmax)

    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    fc_s_auto = fc_s.detach()
    fc_t_auto = fc_t.detach()
    log_softmax_s = logsoftmax(fc_s_auto)
    log_softmax_t = logsoftmax(fc_t_auto)
    # print(label)
    # one_hot_label = F.one_hot(label, num_classes=num_classes).float()
    one_hot_label = label
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    soft_loss = (temp ** 2) * torch.mean(softmax_loss)

    return soft_loss

def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          teach_model,
                          criterion,
                          brdloss,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    if len(cfg.GPU.split(','))>1:
        model = nn.DataParallel(model)
    # print(torch.cuda.device_count(),'torch.cuda.device_count() > 1:')
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))

    # scaler = GradScaler()

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_log = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_log = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    # print(batch_size,'batch_size')
                    # print(cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES)
                    if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES <= 0:
                        det_target = data[-1].to(device)

                        det_score = model(*[x.to(device) for x in data[:-1]])
                        # print(det_target.shape)
                        # print(det_score.shape)
                        # print(det_target.shape)
                        # print('====')
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        # print(det_target.shape)
                        # print(det_score.shape)
                        det_loss = criterion['MCE'](det_score, det_target)
                        det_losses[phase] += det_loss.item() * batch_size
                        # print(det_loss)
                        # print(det_loss.shape)
                    else:
                        det_target, fut_target = data[-1][0].to(device), data[-1][1].to(device)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        # print(fut_target.shape)
                        # # for x in data[:-1]:
                        # #     print(x.shape)
                        # # print(model.device_ids,'model.device_ids')

                        det_scores, fut_scores,feature_SW,feature_SF = model(*[x.to(device) for x in data[:-1]],epoch)
                        if training:
                            _,feature_TW,feature_TF = teach_model(*[x.to(device) for x in data[:-1]],epoch=epoch)
                            # print(feature_SW[0].shape,feature_TW[0].shape)
                            brd_loss_w = brdloss(feature_SW[0], feature_TW[0].permute(1,2,0))
                            brd_loss_f = brdloss(feature_SF[0], feature_TF[0].permute(1,2,0))
                            # print(brd_loss_w,brd_loss_f)
                            # brd_loss_l = brdloss(feature_S[0], feature_T[0].permute(0,2,1))
                            # brd_loss = brd_loss_w + brd_loss_f
                            distill_weight = 1.
                            temp = 2.
                            loss_dist_logits_W = distill_weight * get_logits_loss(feature_TW[-1], feature_SW[-1], det_target,
                                                                                temp, cfg.DATA.NUM_CLASSES)

                            loss_dist_logits_F = distill_weight * get_logits_loss(feature_TF[-1], feature_SF[-1],
                                                                                  fut_target,
                                                                                  temp, cfg.DATA.NUM_CLASSES)
                            loss_dist_W = brd_loss_w + loss_dist_logits_W * 0.5
                            loss_dist_F = brd_loss_f + loss_dist_logits_F * 0.5
                            loss_TW = loss_dist_F+loss_dist_W
                            # print(loss_TW,loss_dist_F,loss_dist_W,'loss_TW,loss_dist_T,loss_dist_W')
                            # print(brd_loss_w,loss_dist_logits_W,'brd_loss_w,loss_dist_logits_')
                            # print(brd_loss_f,loss_dist_logits_F,'brd_loss_f,loss_dist_logits_F')
                        # print(feature_T)
                        # print(det_scores.shape,fut_scores.shape)
                        # det_scores = det_scores.permute(1,0,2,3)
                        # fut_scores = fut_scores.permute(1,0,2,3)
                        # loss_dist_feat = losses(feature[0], feature_T[0])
                        # loss_dist_logits = distill_weight * get_logits_loss(feature_T[-1], feature[-1], target, temp,
                        #                                                     num_classes)
                        # loss_dist = loss_dist_logits + loss_dist_feat

                        # print(brd_loss)
                        for i, det_score in enumerate(det_scores):
                            # print(det_score.shape,'det_score.shape')
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            # print(det_score.shape)
                            # print(det_target.shape)
                            # det_loss = 0.7* criterion['MCE'](det_score, det_target)
                            # print(det_target.shape,det_score.shape)
                            # print(i)
                            if i == 0:
                                det_loss = 0.8 * criterion['MCE'](det_score, det_target)
                            else:
                                det_loss += (0.8*i - 0.6) * criterion['MCE'](det_score, det_target)
                        # det_loss = det_loss/det_scores.shape[0]
                        det_losses[phase] += det_loss.item() * batch_size
                        det_log[phase] += det_loss
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                fut_loss = 0.2 * criterion['MCE'](fut_score, fut_target)
                            else:
                                fut_loss += 0.1 * criterion['MCE'](fut_score, fut_target)
                            # fut_loss = 0.1 * criterion['MCE'](fut_score, fut_target)
                        # fut_loss = fut_loss / fut_scores.shape[0]
                        fut_losses[phase] += fut_loss.item() * batch_size
                        fut_log[phase] += fut_loss
                        det_loss += fut_loss
                    if training:
                        # print(brd_loss)
                        det_loss += loss_TW
                        # print(det_loss)
                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })
                    # print(det_loss,batch_idx,'det_loss')
                    if training:
                        optimizer.zero_grad()

                        det_loss.backward()
                        # for name, param in model.named_parameters():
                        #     if param.grad is None:
                        #         print(name)

                        # print(N,batch_idx,'batch_idx')
                        # scaler.scale(det_loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()
                        optimizer.step()
                        ema.update()
                        scheduler.step()

                    else:
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](
                cfg,
                det_gt_targets,
                det_pred_scores,
            )
            log.append('test det_loss: {:.5f}, det_log: {:.5f},fut_log: {:.5f} det_mAP: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset,),
                det_log['test'] / len(data_loaders['test'].dataset,),
                fut_log['test'] / len(data_loaders['test'].dataset,),
                det_result['mean_AP'],
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        # logger.info(' | '.join(log))
        print(' | '.join(log))
        # Save checkpoint for model and optimizer
        checkpointer.save(epoch, model, optimizer)
        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()


def do_ek100_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES <= 0:
                        det_target = data[-1].to(device)
                        det_score = model(*[x.to(device) for x in data[:-1]])
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion['MCE'](det_score, det_target)
                        det_losses[phase] += det_loss.item()
                    else:
                        det_target, noun_target, verb_target, \
                        fut_target, fut_noun_target, fut_verb_target = data[-1][0].to(device), data[-1][1].to(device), data[-1][2].to(device), \
                                                                                data[-1][3].to(device), data[-1][4].to(device), data[-1][5].to(device)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        noun_target = noun_target.reshape(-1, noun_target.shape[-1])
                        verb_target = verb_target.reshape(-1, verb_target.shape[-1])
                        fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        fut_noun_target = fut_noun_target.reshape(-1, fut_noun_target.shape[-1])
                        fut_verb_target = fut_verb_target.reshape(-1, fut_verb_target.shape[-1])

                        det_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score = model(*[x.to(device) for x in data[:-1]])
                        for i, det_score in enumerate(det_scores):
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                det_loss = 0.2 * criterion['MCE_EQL'](det_score, det_target)
                            else:
                                det_loss += (0.8 * i - 0.6) * criterion['MCE_EQL'](det_score, det_target)
                        det_loss = det_loss / det_scores.shape[0]
                        verb_score, noun_score = verb_score.reshape(-1, verb_score.shape[-1]), noun_score.reshape(-1, noun_score.shape[-1])
                        verb_loss = criterion['MCE'](verb_score, verb_target)
                        noun_loss = criterion['MCE'](noun_score, noun_target)
                        det_loss += (verb_loss + noun_loss)
                        det_losses[phase] += det_loss.item()
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                fut_loss = 0.1 * criterion['MCE_EQL'](fut_score, fut_target)
                            else:
                                fut_loss += 0.1 * criterion['MCE_EQL'](fut_score, fut_target)
                        fut_verb_score, fut_noun_score = fut_verb_score.reshape(-1, fut_verb_score.shape[-1]), fut_noun_score.reshape(-1, fut_noun_score.shape[-1])
                        fut_verb_loss = criterion['MCE'](fut_verb_score, fut_verb_target)
                        fut_noun_loss = criterion['MCE'](fut_noun_score, fut_noun_target)
                        fut_loss += 0.1 * (fut_verb_loss + fut_noun_loss)
                        fut_losses[phase] += fut_loss.item()

                        fut_loss = fut_loss/fut_score.shape[0]
                        det_loss += fut_loss

                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        ema.update()
                        scheduler.step()
                    else:
                        # Prepare for evaluation
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)

        end = time.time()

        # Output log for current epoch
        # print(det_losses,'det_losses')
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            gt, pred = np.array(det_gt_targets), np.array(det_pred_scores)[:, 1:]
            action_labels = np.argmax(gt, axis=-1)
            action_pred = pred.reshape(-1, 1, pred.shape[-1])
            valid_index = list(np.where(action_labels != 0))[0]
            det_result = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0]
            log.append('test det_loss: {:.5f}, det_Recall: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                float(det_result) * 100,
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        # logger.info(' | '.join(log))
        print(' | '.join(log))
        # Save checkpoint for model and optimizer
        checkpointer.save(epoch, model, optimizer)
        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

