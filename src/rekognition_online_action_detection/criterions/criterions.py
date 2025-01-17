# # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# # SPDX-License-Identifier: Apache-2.0
#
# __all__ = ['build_criterion']
#
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from rekognition_online_action_detection.utils.registry import Registry

CRITERIONS = Registry()

# 兼容geomloss提供的'euclidean'
# 注意：geomloss要求cost func计算两个batch的距离，也即接受(B, N, D)
# def cost_func(a, b, p=2, metric='cosine'):
#     """ a, b in shape: (B, N, D) or (N, D)
#     """
#     assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
#     if metric=='euclidean' and p==1:
#         return geomloss.utils.distances(a, b)
#     elif metric=='euclidean' and p==2:
#         return geomloss.utils.squared_distances(a, b)
#     else:
#         if a.dim() == 3:
#             x_norm = a / a.norm(dim=2)[:, :, None]
#             y_norm = b / b.norm(dim=2)[:, :, None]
#             M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
#         elif a.dim() == 2:
#             x_norm = a / a.norm(dim=1)[:, None]
#             y_norm = b / b.norm(dim=1)[:, None]
#             M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
#         M = pow(M, p)
#         return M
#
# metric = 'cosine'
# OTLoss = geomloss.SamplesLoss(
#     loss='sinkhorn', p=p,
#     cost=lambda a, b: cost_func(a, b, p=p, metric=metric),
#     blur=entreg**(1/p), backend='tensorized')
# pW = OTLoss(a, b)

class LaSNNLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """

    def __init__(self,
                 student_emb=1024,
                 teacher_emb=1024,
                 alpha_mgd=0.00007,
                 fnum=4,
                 ):
        super(LaSNNLoss, self).__init__()

        self.alpha_mgd = alpha_mgd
        self.fnum = fnum

        if student_emb != teacher_emb:
            self.align = [nn.Conv1d(student_emb, teacher_emb, kernel_size=1, stride=1, padding=0) for _ in range(self.fnum)]
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*D*H*W, student's feature map
            preds_T(Tensor): Bs*D, teacher's feature map
        """
        # assert preds_S.shape[-1:] == preds_T.shape[-1:]
        # print(len(preds_S),len(preds_T))
        assert len(preds_S) == len(preds_T)
        loss = 0.
        if self.align is not None:
            assert len(preds_S) == len(self.align)
            for ps, pt, a in zip(preds_S, preds_T, self.align):
                loss += self.get_dis_loss(a(ps), pt) * self.alpha_mgd
        else:
            # for ps, pt in zip(preds_S, preds_T):
            #     print(ps.shape,pt.shape)
            #     loss += self.get_dis_loss(ps, pt) * self.alpha_mgd
            loss += self.get_dis_loss(preds_S,preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        B,C,N = preds_S.shape

        # new_fea = preds_S.flatten(2).mean(2)
        # print(new_fea.shape, preds_T.shape)

        dis_loss = loss_mse(preds_S, preds_T) / B

        return dis_loss

@CRITERIONS.register('BRD')
class BRDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """

    def __init__(self,
                 student_emb=1024,
                 teacher_emb=1024,
                 alpha_mgd=0.0007,
                 lambda_mgd=0.15,
                 use_clip=True,
                 ):
        super(BRDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        if student_emb != teacher_emb:
            self.align = nn.Conv2d(student_emb, teacher_emb, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.use_clip = use_clip

        self.generation = nn.Sequential(
            nn.Conv1d(teacher_emb, teacher_emb, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(teacher_emb, teacher_emb, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*D*H*W, student's feature map
            preds_T(Tensor): Bs*D, teacher's feature map
        """
        # assert preds_S.shape[-1:] == preds_T.shape[-1:]
        # preds_S = torch.stack(preds_S)
        # preds_T = torch.stack(preds_T)
        if self.align is not None:
            preds_S = self.align(preds_S)

        if self.use_clip:
            # preds_T = torch.clip(preds_T, preds_T.min(), preds_S.max())
            preds_T = preds_T / (preds_T.max()) * preds_S.max()

        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        # loss_mse = nn.MSELoss(reduction='sum')
        loss_mse = nn.KLDivLoss(reduction='sum')
        log_soft = nn.LogSoftmax(dim=1)
        soft = nn.Softmax(dim=1)
        B, C, N = preds_S.shape

        device = preds_S.device
        mat = torch.rand((B, C, 1)).to(device)
        # mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        # print(masked_fea.shape)
        new_fea = self.generation(masked_fea)
        # new_fea = new_fea.flatten(2).mean(2)
        # print(new_fea.shape, preds_T.shape)
        new_fea = log_soft(new_fea)
        preds_T = soft(preds_T)
        dis_loss = loss_mse(new_fea, preds_T)
        # print(dis_loss)
        return dis_loss
@CRITERIONS.register('BCE')
class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('SCE')
class SingleCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(SingleCrossEntropyLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('MCE')
class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        # print(input.shape,target.shape,'input.shape,target.shape,')
        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-F.normalize(target[:, notice_index]) * logsoftmax(input[:, notice_index]), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-F.normalize(target) * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output

@CRITERIONS.register('MCE_EQL')
class MultipCrossEntropyEqualizedLoss(nn.Module):

    def __init__(self, gamma=0.95, lambda_=1.76e-3, reduction='mean', ignore_index=-100,
                 anno_path='external/rulstm/RULSTM/data/ek55/'):
        super(MultipCrossEntropyEqualizedLoss, self).__init__()

        # get label distribution
        segment_list = pd.read_csv(osp.join(anno_path, 'training.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        freq_info = np.zeros((max(segment_list['action']) + 1,))
        assert ignore_index == 0
        for segment in segment_list.iterrows():
            freq_info[segment[1]['action']] += 1.
        freq_info = freq_info / freq_info.sum()
        self.freq_info = torch.FloatTensor(freq_info)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # print(input.shape,target.shape,'nput.shape,target.shape')
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        weight = input.new_zeros(len(notice_index))
        weight[self.freq_info < self.lambda_] = 1.
        weight = weight.view(1, -1).repeat(input.shape[0], 1)

        eql_w = 1 - (torch.rand_like(target) < self.gamma) * weight * (1 - target)
        input = torch.log(eql_w + 1e-8) + input

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]

def build_criterion(cfg, device=None):
    criterion = {}
    for name, params in cfg.MODEL.CRITERIONS:
        if name in CRITERIONS:
            if 'ignore_index' not in params:
                params['ignore_index'] = cfg.DATA.IGNORE_INDEX
            criterion[name] = CRITERIONS[name](**params).to(device)
        else:
            raise RuntimeError('Unknown criterion: {}'.format(name))
    return criterion




