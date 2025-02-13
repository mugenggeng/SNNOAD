# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
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
import geomloss
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
# 兼容geomloss提供的'euclidean'
# 注意：geomloss要求cost func计算两个batch的距离，也即接受(B, N, D)
def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M


class MACACCalculator:
    def __init__(self, model):
        self.mac_count = 0
        self.ac_count = 0
        self.hooks = []

        # 注册所有卷积、全连接和LIF层的钩子
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                self.hooks.append(layer.register_forward_hook(self._compute_ops))
            elif isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                self.hooks.append(layer.register_forward_hook(self._compute_bn_ops))
            elif isinstance(layer, (MultiStepLIFNode)):  # 明确的LIF类型检查
                self.hooks.append(layer.register_forward_hook(self._compute_lif_ops))

    def _compute_ops(self, module, input, output):
        """处理卷积/全连接层的MAC计算"""
        if isinstance(module, nn.Conv1d):
            # [T, B, C_in, L_in] -> [T, B, C_out, L_out]
            # print(type(input))
            # print(input[0].shape)
            T_B, C_in, L_in = input[0].shape
            T ,B = 4, T_B//4
            C_out = module.out_channels
            L_out = output.shape[-1]
            self.mac_count += C_in * C_out * module.kernel_size[0] * L_out * T * B
        elif isinstance(module, nn.Linear):
            T, B, C_in = input[0].shape[0], input[0].shape[1], input[0].shape[2]
            C_out = module.out_features
            self.mac_count += C_in * C_out * T * B

    def _compute_bn_ops(self, module, input, output):
        """处理BatchNorm的AC计算"""
        # [T, B, C, L] -> 每个元素有2次加法（平移和缩放）
        self.ac_count += 2 * input[0].numel()

    def _compute_lif_ops(self, module, input, output):
        """修复后的LIF神经元AC计算"""
        # output形状: [T, B, C, L]
        spike_count = (output > 0).float().sum().item()
        self.ac_count += 3 * spike_count  # 3次加法/脉冲

    def reset(self):
        self.mac_count = 0
        self.ac_count = 0

    def get_counts(self):
        return self.mac_count, self.ac_count
def Geomloss(pred,target):
    metric = 'cosine'
    entreg = .1
    p = 2
    OTLoss = geomloss.SamplesLoss(
        loss='sinkhorn', p=p,
        cost=lambda pred, target: cost_func(pred, target, p=p, metric=metric),
        blur=entreg**(1/p), backend='tensorized')
    # pW = OTLoss(a, b)
    return OTLoss(pred, target).sum(0)
def track_weight_update(model, prev_params):
    updates = []
    for (name, param), prev in zip(model.named_parameters(), prev_params):
        if param.requires_grad and param.grad is not None:
            update_ratio = (param.data - prev).abs().mean() / (param.abs().mean() + 1e-7)
            updates.append(update_ratio.item())
    return np.mean(updates)
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
    # macac_calculator = MACACCalculator(model)
    # macac_calculator1 = MACACCalculator(teach_model)
    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_log = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_log = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []
        # print(model)
        # print(hasattr(model, "set_T"))
        # if hasattr(model, "set_T"):
        #     if epoch < 10:
        #         setattr(model, "T",4)
        #             # model.set_T(4)
        #     elif epoch < 20:
        #         setattr(model, "T",8)
        #     else:
        #         setattr(model, "T",16)

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
                    # macac_calculator.reset()
                    # macac_calculator1.reset()
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
                        # batch_mac, batch_ac = macac_calculator.get_counts()

                        # print(batch_mac,batch_ac)

                        if training:
                            _,feature_TW,feature_TF = teach_model(*[x.to(device) for x in data[:-1]],epoch=epoch)
                            # batch_mac1, batch_ac1 = macac_calculator1.get_counts()
                            # print(batch_mac1, batch_ac1)
                            # print(feature_SW[0].shape,feature_TW[0].shape)
                            # brd_loss_w = brdloss(feature_SW[0], feature_TW[0].permute(1,2,0))
                            geomloss_w = Geomloss(feature_SW[0], feature_TW[0].permute(1,2,0))

                            # brd_loss_f = brdloss(feature_SF[0], feature_TF[0].permute(1,2,0))
                            geomloss_f = Geomloss(feature_SF[0], feature_TF[0].permute(1,2,0))

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

                            loss_TW = loss_dist_logits_W + loss_dist_logits_F + geomloss_w + geomloss_f

                        for i, det_score in enumerate(det_scores):

                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)

                            if i == 0:
                                det_loss = 0.2 * criterion['MCE'](det_score, det_target)
                            else:

                                det_loss += (0.8*i - 0.6) * criterion['MCE'](det_score, det_target)
                        # det_loss = det_loss/det_scores.shape[0]
                        det_losses[phase] += det_loss.item() * batch_size
                        det_log[phase] += det_loss
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:

                                fut_loss = 0.1 * criterion['MCE'](fut_score, fut_target)
                            else:

                                fut_loss += 0.1 * criterion['MCE'](fut_score, fut_target)
                            # fut_loss = 0.1 * criterion['MCE'](fut_score, fut_target)
                        # fut_loss = fut_loss / fut_scores.shape[0]
                        fut_losses[phase] += fut_loss.item() * batch_size
                        fut_log[phase] += fut_loss
                        det_loss += fut_loss
                    det_loss += loss_TW

                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })
                    # print(det_loss,batch_idx,'det_loss')
                    if training:
                        optimizer.zero_grad()

                        det_loss.backward()

                        total_grad = 0
                        valid_layers = 0
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'weight' in name:
                                layer_grad = param.grad.abs().mean().item()
                                total_grad += layer_grad
                                valid_layers += 1
                                if layer_grad < 1e-5:
                                    print(f"警告：{name} 层出现梯度消失 ({layer_grad:.2e})")

                        avg_grad = total_grad / valid_layers
                        logger.info(f"平均梯度量级：{avg_grad:.2e}")

                        # 梯度裁剪策略优化
                        max_norm = 10.0 if epoch < 10 else 1.0
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                        # clip_value = 10.0  # 设置裁剪阈值
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                        # for name, param in model.named_parameters():
                        #     if 'classifier' in name:
                        #         # print('111')
                        #         torch.nn.utils.clip_grad_norm_(param, 10.0)
                        prev_params = [p.clone().detach() for p in model.parameters()]
                        optimizer.step()
                        update_ratio = track_weight_update(model, prev_params)
                        logger.info(f"权重更新比率：{update_ratio:.2e}")
                        ema.update()
                        scheduler.step()

                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         print(f"Layer: {name}, Max Gradient: {param.grad.abs().max().item()}")

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

