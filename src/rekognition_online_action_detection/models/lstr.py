# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul,mem_update

from  . import transformer as tr
from scipy.fft import fft, fftshift, ifft
from .models import META_ARCHITECTURES as registry
from .transformer.snn_module import MS_DownSampling,MS_ConvBlock,MS_Block, MS_ConvBlock_T,CGAFusion_SNN,MS_Star_Block,MS_Block_Cross_Weight,MS_Block_Cross_No_Weight
from spikingjelly.activation_based import layer
from typing import Any, List, Mapping
from .feature_head import build_feature_head
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# class LSTR(nn.Module):
#
#     def __init__(self, cfg):
#         super(LSTR, self).__init__()
#
#         self.cfg = cfg
#         # Build long feature heads
#         self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
#         self.long_enabled = self.long_memory_num_samples > 0
#         if self.long_enabled:
#             self.feature_head_long = build_feature_head(cfg)
#
#         # Build work feature head
#         self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
#         self.work_enabled = self.work_memory_num_samples > 0
#         if self.work_enabled:
#             self.feature_head_work = build_feature_head(cfg)
#
#         self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
#         self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
#         self.future_enabled = self.future_num_samples > 0
#
#         self.d_model = self.feature_head_work.d_model
#         self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
#         self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
#         self.dropout = cfg.MODEL.LSTR.DROPOUT
#         self.activation = cfg.MODEL.LSTR.ACTIVATION
#         self.num_classes = cfg.DATA.NUM_CLASSES
#
#         # Build position encoding
#         self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
#         self.group = self.cfg.MODEL.LSTR.GROUPS
#         # Build LSTR encoder
#         if self.long_enabled:
#             self.enc_queries = nn.ModuleList()
#             self.enc_modules = nn.ModuleList()
#             # self.enc_mode = nn.ModuleList()
#             # self.cross_attention = tr.MultiheadAttention(self.d_model,1)
#             index = 0
#             for param in cfg.MODEL.LSTR.ENC_MODULE:
#                 if param[0] != -1:
#                     self.enc_queries.append(nn.Embedding(param[0], self.d_model))
#                     enc_layer = tr.LongMemLayer(
#                         self.d_model, self.num_heads, self.dim_feedforward,
#                         self.dropout, self.activation,short=(index==0),gru=True,atten=True)
#                     self.enc_modules.append(tr.TransformerDecoder(
#                             enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
#                     # self.enc_modules.append(self.enc_mode)
#                 else:
#                     self.enc_queries.append(None)
#                     enc_layer = tr.TransformerEncoderLayer(
#                         self.d_model, self.num_heads, self.dim_feedforward,
#                         self.dropout, self.activation)
#                     self.enc_modules.append(tr.TransformerEncoder(
#                         enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
#                 index = index+1
#             self.average_pooling = nn.AdaptiveAvgPool1d(1)
#             self.max_polling = nn.AdaptiveMaxPool1d(1)
#             # self.norm = nn.LayerNorm(self.d_model)
#         else:
#             self.register_parameter('enc_queries', None)
#             self.register_parameter('enc_modules', None)
#
#         # Build LSTR decoder
#         if self.long_enabled:
#             param = cfg.MODEL.LSTR.DEC_MODULE
#             dec_layer = tr.TransformerDecoderLayer(
#                 self.d_model, self.num_heads, self.dim_feedforward,
#                 self.dropout, self.activation,gru=True,atten=True)
#             self.dec_modules = tr.TransformerDecoder(
#                 dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
#         else:
#             param = cfg.MODEL.LSTR.DEC_MODULE
#             dec_layer = tr.TransformerEncoderLayer(
#                 self.d_model, self.num_heads, self.dim_feedforward,
#                 self.dropout, self.activation)
#             self.dec_modules = tr.TransformerEncoder(
#                 dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
#         # self.norm = nn.LayerNorm(self.d_model)
#         # short_layer = tr.TransformerEncoderLayer(
#         #     self.d_model, self.num_heads, self.dim_feedforward,
#         #     self.dropout, self.activation)
#         # self.short_modules = tr.TransformerEncoder(
#         #     short_layer, param[1], tr.layer_norm(self.d_model, param[2]))
#
#         # self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
#         # Build Anticipation Generation
#         if self.future_enabled:
#             param = cfg.MODEL.LSTR.GEN_MODULE
#             self.gen_query = nn.Embedding(param[0], self.d_model)
#             gen_layer = tr.TransformerDecoderLayer(
#                 self.d_model, self.num_heads, self.dim_feedforward,
#                 self.dropout, self.activation,gru=True,atten=True)
#             self.gen_layer = tr.TransformerDecoder(
#                 gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
#             )
#             self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
#         #     # CCI
#             self.work_fusions = nn.ModuleList()
#             self.fut_fusions = nn.ModuleList()
#             for i in range(cfg.MODEL.LSTR.CCI_TIMES):
#                 work_enc_layer = tr.TransformerDecoderLayer(
#                     self.d_model, self.num_heads, self.dim_feedforward,
#                     self.dropout, self.activation,gru=True,atten=True)
#                 self.work_fusions.append(tr.TransformerDecoder(
#                     work_enc_layer, 1, tr.layer_norm(self.d_model, True)))
#                 if i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
#                     fut_enc_layer = tr.TransformerDecoderLayer(
#                         self.d_model, self.num_heads, self.dim_feedforward,
#                         self.dropout, self.activation,gru=True,atten=True)
#                     self.fut_fusions.append(tr.TransformerDecoder(
#                         fut_enc_layer, 1, tr.layer_norm(self.d_model, True)))
#
#         # Build classifier
#         self.classifier = nn.Linear(self.d_model, self.num_classes)
#         if self.cfg.DATA.DATA_NAME == 'EK100':
#             self.classifier_verb = nn.Linear(self.d_model, 98)
#             self.classifier_noun = nn.Linear(self.d_model, 301)
#             self.dropout_cls = nn.Dropout(0.8)
#
#         embed_dim = [512, 256, 512, 1024]
#         self.T = 10
#         mlp_ratios=4
#         self.downsample1_1 = MS_DownSampling(
#                     in_channels=self.dim_feedforward,
#                     embed_dims=embed_dim[0] // 2,
#                     kernel_size=7,
#                     stride=2,
#                     padding=3,
#                     first_layer=True,
#                 )
#
#         self.ConvBlock1_1 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
#                 )
#
#         self.downsample1_2 = MS_DownSampling(
#                     in_channels=embed_dim[0] // 2,
#                     embed_dims=embed_dim[0],
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     first_layer=False,
#                 )
#
#         self.ConvBlock1_2 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
#                 )
#
#         self.downsample2 = MS_DownSampling(
#                     in_channels=embed_dim[0],
#                     embed_dims=embed_dim[1],
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     first_layer=False,
#                 )
#
#         self.ConvBlock2_1 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#                 )
#
#         self.ConvBlock2_2 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#                 )
#
#         self.downsample3 = MS_DownSampling(
#                     in_channels=embed_dim[1],
#                     embed_dims=embed_dim[2],
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     first_layer=False,
#                 )
#
#         self.ConvBlock3_1 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[2], mlp_ratio=mlp_ratios)]
#                 )
#
#         self.downsample4 = MS_DownSampling(
#                     in_channels=embed_dim[2],
#                     embed_dims=embed_dim[3],
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     first_layer=False,
#                 )
#
#
#
#         self.ConvBlock4_1 = nn.ModuleList(
#                     [MS_ConvBlock(dim=embed_dim[3], mlp_ratio=mlp_ratios)]
#                 )
#
#         # self.downsample5 = MS_DownSampling(
#         #     in_channels=embed_dim[3],
#         #     embed_dims=embed_dim[3],
#         #     kernel_size=3,
#         #     stride=2,
#         #     padding=1,
#         #     first_layer=False,
#         # )
#         #
#         # self.ConvBlock5_1 = nn.ModuleList(
#         #     [MS_ConvBlock(dim=embed_dim[3], mlp_ratio=mlp_ratios)]
#         # )
#
#     def cci(self, memory, output, mask, short_mem):
#         # print(memory.shape,output.shape,'memory.shape,output.shape')
#         his_memory = torch.cat([memory, output])
#         # print(his_memory.shape,'his_memory.shape')
#         enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
#         future = self.gen_layer(enc_query, his_memory, knn=True)
#
#         dec_query = self.final_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
#         future_rep = [future]
#         short_rep = [output]
#         for i in range(self.cfg.MODEL.LSTR.CCI_TIMES):
#             mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
#             mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
#             the_mask = torch.cat((mask1, mask, mask2), dim=-1)
#             total_memory = torch.cat([memory, output, future])
#             output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask, knn=True)
#             # print(output.shape,'output.shape')
#             short_rep.append(output)
#             # print(memory.shape, output.shape, future.shape, 'memory.shape,output.shape,future.shape22')
#             total_memory = torch.cat([memory, output, future])
#             # print(total_memory.shape, 'total_memory.shape22')
#             if i == 0:
#                 future = self.fut_fusions[i](dec_query, total_memory, knn=True)
#                 future_rep.append(future)
#             elif i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
#                 # else:
#                 mask1 = torch.zeros((future.shape[0], memory.shape[0] + output.shape[0])).to(output.device)
#                 mask2 = tr.generate_square_subsequent_mask(future.shape[0]).to(output.device)
#                 future = self.fut_fusions[i](future, total_memory, tgt_mask=mask2,
#                                              memory_mask=torch.cat((mask1, mask2), dim=-1), knn=True)
#                 # print(future.shape,'future.shape')
#                 future_rep.append(future)
#         return short_rep, future_rep
#     def forward_features(self, x):
#         if x.dim()!=4:
#             x = x.unsqueeze(0).repeat(self.T,1,1,1)
#         x = self.downsample1_1(x)
#         for blk in self.ConvBlock1_1:
#             x = blk(x)
#         x = self.downsample1_2(x)
#         for blk in self.ConvBlock1_2:
#             x = blk(x)
#         res_x = x
#
#
#         x = self.downsample2(x)
#         for blk in self.ConvBlock2_1:
#             x = blk(x)
#         for blk in self.ConvBlock2_2:
#             x = blk(x)
#
#
#         x = self.downsample3(x)
#         for blk in self.ConvBlock3_1:
#             x = blk(x)
#         # for blk in self.block3:
#         #     x = blk(x)
#
#
#         x = self.downsample4(x)
#         for blk in self.ConvBlock4_1:
#             x = blk(x)
#
#         # x = self.downsample5(x)
#         # for blk in self.ConvBlock5_1:
#         #     x = blk(x)
#         # for blk in self.block4:
#         #     x = blk(x)
#
#         return x  # T,B,C,N
#
#     def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=1):
#         # print(visual_inputs.shape)
#         # print(self.long_memory_num_samples)
#         if self.long_enabled:
#             # Compute long memories
#             the_long_memories = self.feature_head_long(
#                 visual_inputs[:, :self.long_memory_num_samples],
#                 motion_inputs[:, :self.long_memory_num_samples]).transpose(0, 1)
#
#             work_memories = self.feature_head_work(
#                 visual_inputs[:, self.long_memory_num_samples:],
#                 motion_inputs[:, self.long_memory_num_samples:],
#             ).transpose(0, 1)
#             work_memories = self.pos_encoding(work_memories, padding=0)
#             # short_mem = self.short_modules(short_mem)
#             if len(self.enc_modules) > 0:
#                 enc_queries = [
#                     enc_query.weight.unsqueeze(1).repeat(1, the_long_memories.shape[1], 1)
#                     if enc_query is not None else None
#                     for enc_query in self.enc_queries
#                 ]
#
#                 # Encode long memories
#                 if enc_queries[0] is not None:
#                     if self.cfg.MODEL.LSTR.GROUPS > 0 and (
#                             memory_key_padding_mask == float('-inf')).sum() < self.cfg.MODEL.LSTR.GROUPS:
#
#                         T = the_long_memories.shape[0] // self.cfg.MODEL.LSTR.GROUPS
#                         enc_query = enc_queries[0]
#                         long_memories = []
#                         max_mem = []
#                         avg_mem = []
#                         for i in range(self.cfg.MODEL.LSTR.GROUPS):
#                             out = self.enc_modules[0](enc_query,the_long_memories[i * T:(i + 1) * T],
#                                                       memory_key_padding_mask=memory_key_padding_mask[:,
#                                                                               i * T:(i + 1) * T], knn=True,short_mem = work_memories)
#                             weight = self.max_polling(out.permute(1, 2, 0)).permute(2, 0, 1)
#                             out = self.average_pooling(out.permute(1, 2, 0)).permute(2, 0, 1)
#                             out = out+weight
#                             # max_mem.append(weight)
#                             # avg_mem.append(out)
#                             long_memories.append(out)
#                         long_memories = torch.cat(long_memories)
#
#                     else:
#                         # print(type(the_long_memories))
#                         # print(self.enc_modules[0])
#                         long_memories = self.enc_modules[0](enc_queries[0], the_long_memories,
#                                                             memory_key_padding_mask=memory_key_padding_mask, knn=True,short_mem = work_memories)
#                         # print(type(long_memories))
#                 else:
#                     long_memories = self.enc_modules[0](long_memories)
#
#                 for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
#                     if enc_query is not None:
#                         long_memories = enc_module(enc_query, long_memories, knn=True)
#                     else:
#                         # print(long_memories.shape)
#                         long_memories = enc_module(long_memories, knn=True)
#
#
#         if self.long_enabled:
#             memory = long_memories
#         snn_mem = self.forward_features(the_long_memories.permute(1, 2, 0)).mean(0).permute(2, 0, 1)
#         # print(memory.shape,snn_mem.shape)
#         memory = memory+snn_mem
#         if self.work_enabled:
#             if self.anticipation_num_samples > 0 and self.future_enabled:
#                 anticipation_queries = self.pos_encoding(
#                     self.final_query.weight[:self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH
#                                             :self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...].unsqueeze(1).repeat(1,
#                                                                                                                     work_memories.shape[
#                                                                                                                         1],
#                                                                                                                     1),
#                     padding=self.work_memory_num_samples)
#                 work_memories = torch.cat((work_memories, anticipation_queries), dim=0)
#
#             # Build mask
#             mask = tr.generate_square_subsequent_mask(
#                 work_memories.shape[0])
#             mask = mask.to(work_memories.device)
#
#
#             # print(work_memories.shape,memory.shape,'11')
#             if self.long_enabled:
#                 output = self.dec_modules(
#                     work_memories,
#                     memory=memory,
#                     tgt_mask=mask,
#                     knn=True
#                 )
#             else:
#                 output = self.dec_modules(
#                     work_memories,
#                     src_mask=mask,
#                     knn=True
#                 )
#
#         if self.future_enabled:
#             works, futs = self.cci(memory, output, mask, work_memories)
#             work_scores = []
#             fut_scores = []
#             for i, work in enumerate(works):
#                 # print(work.shape,'work.shape')
#                 if i == len(works) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
#                     noun_score = self.classifier_noun(work).transpose(0, 1)
#                     verb_score = self.classifier_verb(work).transpose(0, 1)
#                     work_scores.append(self.classifier(self.dropout_cls(work)).transpose(0, 1))
#                 else:
#                     work_scores.append(self.classifier(work).transpose(0, 1))
#             for i, fut in enumerate(futs):
#                 # print(fut.shape,'fut.shape')
#                 if i == 0:
#                     fut_scores.append(self.classifier(
#                         F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1)).transpose(0,
#                                                                                                                       1))
#                 else:
#                     fut_scores.append(self.classifier(fut).transpose(0, 1))
#                     if i == len(futs) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
#                         fut_noun_score = self.classifier_noun(fut).transpose(0, 1)
#                         fut_verb_score = self.classifier_verb(fut).transpose(0, 1)
#             # print(work_scores[0].shape, fut_scores[0].shape)
#             return (work_scores, fut_scores) if self.cfg.DATA.DATA_NAME != 'EK100' else (
#             work_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score)
#
#         # Compute classification score
#         score = self.classifier(output)
#         # print(score.shape)
#         return score.transpose(0, 1)
class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x
# class LSTR(nn.Module):
#
#     def __init__(self, cfg,
#                  layers=[
#             ['DSSA', ] * 1,
#             ['DSSA', ] * 2,
#             ['DSSA',] *3,
#             # ['DSSA', ] * 1,
#             # ['DSSA'] * 1,
#                  ],
#                  planes= [512, 512, 512],
#                  num_heads= [4, 4, 4],
#                  img_size=288,
#                  T=10,
#                  in_channels =1024,
#                  prologue=None,
#                  group_size=64,
#                  activation=mem_update,
#                  **kwargs,
#                  ):
#         super(LSTR, self).__init__()
#
#         self.cfg = cfg
#         # Build long feature heads
#         self.long_enabled=True
#         self.work_enabled=True
#         self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
#         if self.long_enabled:
#             self.feature_head_long = build_feature_head(cfg)
#         if self.work_enabled:
#             self.feature_head_work = build_feature_head(cfg)
#         self.d_model = self.feature_head_work.d_model
#         self.dropout = cfg.MODEL.LSTR.DROPOUT
#         self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
#         self.num_classes = cfg.DATA.NUM_CLASSES
#         self.future = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0
#         # Build position encoding
#         self.T = T
#         self.skip = ['prologue.0', 'classifier']
#         assert len(planes) == len(layers) == len(num_heads)
#
#         if prologue is None:
#             self.prologue = nn.Sequential(
#                 layer.Conv1d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode='m'),
#                 BN(planes[0]),
#                 layer.MaxPool1d(kernel_size=3, stride=2, padding=1, step_mode='m'),
#             )
#             img_size = img_size // 2
#         else:
#             self.prologue = prologue
#         self.prologue_work = layer.Conv1d(in_channels,planes[-1],kernel_size=3,stride=1,padding=1,step_mode='m')
#         self.layers = nn.Sequential()
#         for idx in range(len(planes)):
#             sub_layers = nn.Sequential()
#             # if idx != 0:
#             #     sub_layers.append(
#             #         DownsampleLayer(planes[idx-1], planes[idx], stride=2, activation=activation))
#             #     img_size = img_size // 2
#             for name in layers[idx]:
#                 if name == 'DSSA':
#                     sub_layers.append(
#                         tr.DSSA(planes[idx], num_heads[idx], img_size  ** 2, activation=activation))
#                 elif name == 'GWFFN':
#                     sub_layers.append(
#                         tr.GWFFN(planes[idx], group_size=group_size, activation=activation))
#                 else:
#                     raise ValueError(name)
#             self.layers.append(sub_layers)
#
#         # self.max_oa = layer.MaxPool1d(2, step_mode='m')
#         self.avgpool_oa = layer.AdaptiveAvgPool1d((8), step_mode='m')
#         # self.conv_trans = layer.Conv1d(planes[-1],planes[-1],3,2,4,step_mode='m')
#         self.conv = layer.Conv1d(planes[-1],planes[-1],1,1,0,step_mode='m')
#         # self.avgpool_fu = layer.AdaptiveAvgPool1d((48), step_mode='m')
#         # self.classifier = Linear(planes[-1], self.num_classes)
#         # self.linear_oa = Linear(18,40)
#         # self.linear_fu = Linear(18, 48)
#
#         # if self.future_enabled:
#         #     param = cfg.MODEL.LSTR.GEN_MODULE
#         #     self.gen_query = nn.Embedding(param[0], self.d_model)
#         #     gen_layer = tr.TransformerDecoderLayer(
#         #                 self.d_model, self.num_heads, self.dim_feedforward,
#         #                 self.dropout, self.activation,gru=True,atten=True)
#         #     self.gen_layer = tr.TransformerDecoder(
#         #                 gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
#         #             )
#         #     self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
#         #         #     # CCI
#         #     self.work_fusions = nn.ModuleList()
#         #     self.fut_fusions = nn.ModuleList()
#         #     for i in range(cfg.MODEL.LSTR.CCI_TIMES):
#         #         work_enc_layer = tr.TransformerDecoderLayer(
#         #                     self.d_model, self.num_heads, self.dim_feedforward,
#         #                     self.dropout, self.activation,gru=True,atten=True)
#         #         self.work_fusions.append(tr.TransformerDecoder(
#         #                     work_enc_layer, 1, tr.layer_norm(self.d_model, True)))
#         #         if i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
#         #                 fut_enc_layer = tr.TransformerDecoderLayer(
#         #                         self.d_model, self.num_heads, self.dim_feedforward,
#         #                         self.dropout, self.activation,gru=True,atten=True)
#         #                 self.fut_fusions.append(tr.TransformerDecoder(
#         #                         fut_enc_layer, 1, tr.layer_norm(self.d_model, True)))
#         # Build classifier
#         self.classifier = nn.Linear(planes[-1], self.num_classes)
#         if self.cfg.DATA.DATA_NAME == 'EK100':
#             self.classifier_verb = nn.Linear(self.d_model, 98)
#             self.classifier_noun = nn.Linear(self.d_model, 301)
#             self.dropout_cls = nn.Dropout(0.8)
#
#         self.init_weight()
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d)):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def transfer(self, state_dict: Mapping[str, Any]):
#         _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
#         return self.load_state_dict(_state_dict, strict=False)
#
#     # def cci(self, memory, output, mask, short_mem):
#     #
#     # # print(memory.shape,output.shape,'memory.shape,output.shape')
#     #     his_memory = torch.cat([memory, output])
#     #         # print(his_memory.shape,'his_memory.shape')
#     #     enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
#     #     future = self.gen_layer(enc_query, his_memory, knn=True)
#     #
#     #     dec_query = self.final_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
#     #     future_rep = [future]
#     #     short_rep = [output]
#     #     for i in range(self.cfg.MODEL.LSTR.CCI_TIMES):
#     #         mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
#     #         mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
#     #         the_mask = torch.cat((mask1, mask, mask2), dim=-1)
#     #         total_memory = torch.cat([memory, output, future])
#     #         output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask, knn=True)
#     #             # print(output.shape,'output.shape')
#     #         short_rep.append(output)
#     #             # print(memory.shape, output.shape, future.shape, 'memory.shape,output.shape,future.shape22')
#     #         total_memory = torch.cat([memory, output, future])
#     #             # print(total_memory.shape, 'total_memory.shape22')
#     #         if i == 0:
#     #             future = self.fut_fusions[i](dec_query, total_memory, knn=True)
#     #             future_rep.append(future)
#     #         elif i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
#     #                 # else:
#     #             mask1 = torch.zeros((future.shape[0], memory.shape[0] + output.shape[0])).to(output.device)
#     #             mask2 = tr.generate_square_subsequent_mask(future.shape[0]).to(output.device)
#     #             future = self.fut_fusions[i](future, total_memory, tgt_mask=mask2,
#     #                                              memory_mask=torch.cat((mask1, mask2), dim=-1), knn=True)
#     #                 # print(future.shape,'future.shape')
#     #             future_rep.append(future)
#     #     return short_rep, future_rep
#     def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=1):
#         # print(visual_inputs.shape)
#         # print(self.long_memory_num_samples)
#         if self.long_enabled:
#             # Compute long memories
#             the_long_memories = self.feature_head_long(
#                 visual_inputs[:, :self.long_memory_num_samples],
#                 motion_inputs[:, :self.long_memory_num_samples])
#             x = the_long_memories
#             the_long_memories = self.pos_encoding(the_long_memories, padding=0)
#             # short_mem = self.short_modules(short_mem)
#
#         if self.work_enabled:
#             work_memories = self.feature_head_work(
#                 visual_inputs[:, self.long_memory_num_samples:],
#                 motion_inputs[:, self.long_memory_num_samples:],
#             )
#             work_memories = self.pos_encoding(work_memories, padding=0)
#
#             # print(the_long_memories.shape,work_memories.shape,'the_long_memories.shape,work_memories.shape')
#             x = torch.cat([the_long_memories,work_memories],dim=1).permute(0,2,1)
#             # print(x.shape,'x.shape')
#
#             # Build mask
#             mask = tr.generate_square_subsequent_mask(
#                 work_memories.shape[0])
#             mask = mask.to(work_memories.device)
#             # print(work_memories.shape,memory.shape,'11')
#         if x.dim() != 4:
#             x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
#             work_memories = work_memories.unsqueeze(0).repeat(self.T, 1, 1, 1)
#             # print(work_memories.shape)
#             assert x.dim() == 4
#         else:
#             #### [B, T, C, H, W] -> [T, B, C, H, W]
#             x = x.transpose(0, 1)
#         # print(x.shape,'x')
#         x = self.prologue(x)
#         # print(x.shape,'prologue')
#         x = self.layers(x)
#
#         # print(x.shape,'layers')
#         oa_feat = self.avgpool_oa(x)
#         oa_feat_t = self.max_oa(x)
#         # print(oa_feat.shape,'oa_feat.shape')
#         # oa_feat = self.avgpool_oa(x)
#         # print(oa_feat_t.shape,'oa_feat_t')
#
#         # print(work_memories.shape,'work_memories.shape')
#         # work_memories = self.prologue_work(work_memories.permute(0,1,3,2))
#         # print(work_memories.shape, 'work_memories.shape')
#
#
#         oa_feat_f = torch.cat([oa_feat_t,oa_feat],dim=-1)
#         # print(oa_feat_f.shape,'oa_feat_f.shape')
#         work_score = self.classifier(oa_feat_f.permute(0, 1, 3, 2))
#
#         # print(oa_feat.shape,'oa_feat')
#         # print(self.future)
#         if self.future:
#             fu_feat = self.conv(oa_feat)
#             fu_feat_f = torch.cat([oa_feat_f,fu_feat],dim=-1)
#
#             print(fu_feat_f.shape, 'fu_feat_f.shape')
#             # print(fu_feat_f.shape,'fu_feat_f.shape')
#             # print(oa_feat_f.shape,'oa_feat_f.shape')
#             # print(x.shape,'flatten')
#
#             # print(work_score,'work_score')
#             future_score = self.classifier(fu_feat_f.permute(0,1,3,2))
#         # print(future_score,'future_score')
#
#             return (work_score,future_score)
#
#         return work_score

# class LSTR(nn.Module):
#
#     def __init__(self, cfg,
#                  # embed_dims=1024,
#                  embed_dim=[128, 128, 128, 1024],
#                  num_heads=4,
#                  mlp_ratios=1,
#                  depths=1,
#                  T=4,
#                  down_scale = 1,
#                  in_channels =1024,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_rate=0.0,
#                  attn_drop_rate=0.0,
#                  drop_path_rate=0.0,
#                  norm_layer=nn.LayerNorm,
#                  sr_ratios=1,
#                  prologue=None,
#                  group_size=64,
#                  activation=mem_update,
#                  **kwargs,
#                  ):
#         super(LSTR, self).__init__()
#
#         self.cfg = cfg
#         # Build long feature heads
#         self.long_enabled=True
#         self.work_enabled=True
#         self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
#         if self.long_enabled:
#             self.feature_head_long = build_feature_head(cfg)
#         if self.work_enabled:
#             self.feature_head_work = build_feature_head(cfg)
#         self.d_model = in_channels
#         self.depth = depths
#         self.down_scale = down_scale
#         self.dropout = cfg.MODEL.LSTR.DROPOUT
#         self.pos_encoding = tr.PositionalEncoding(in_channels, self.dropout)
#         self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
#         self.num_classes = cfg.DATA.NUM_CLASSES
#         self.future = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0
#         # Build position encoding
#         self.T = T
#         self.num_heads = num_heads
#         self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
#         self.activation = cfg.MODEL.LSTR.ACTIVATION
#         self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
#
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, depths)
#         ]  # stochastic depth decay rule
#
#
#         # self.down_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1,)
#         # if self.anticipation_num_samples > 0:
#         #     self.dec_query = nn.Embedding(self.anticipation_num_samples, self.d_model)
#         # self.block = nn.ModuleList(
#         #     [
#         #        tr.Block(dim=embed_dims, num_heads=num_heads,T=T)
#         #     for j in range(down_scale)
#         #     ]
#         # )
#         # self.fusion = tr.CGAFusion(dim=embed_dims)
#         self.block_w = nn.ModuleList([MS_Block(
#                     dim=embed_dim[-1],
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[j],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                 ) for j in range(depths)]
#         )
#
#
#
#
#         self.downsample1_1 = MS_DownSampling(
#             in_channels=in_channels,
#             embed_dims=embed_dim[0] // 2,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             first_layer=True,
#         )
#
#         self.ConvBlock1_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0] // 2, output_dim = embed_dim[0] // 2,mlp_ratio=mlp_ratios)]
#         )
#
#         self.MSBlock1_1 = nn.ModuleList(
#             [MS_Block(
#                 dim=embed_dim[0] //2,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratios,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[-1],
#                 norm_layer=norm_layer,
#                 sr_ratio=sr_ratios,
#             )]
#         )
#
#         self.downsample1_2 = MS_DownSampling(
#             in_channels=embed_dim[0] // 2,
#             embed_dims=embed_dim[0],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock1_2 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0], output_dim = embed_dim[0], mlp_ratio=mlp_ratios)]
#         )
#
#         self.MSBlock1_2 = nn.ModuleList(
#             [MS_Block(
#                 dim=embed_dim[0],
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratios,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[-1],
#                 norm_layer=norm_layer,
#                 sr_ratio=sr_ratios,
#             )]
#         )
#
#         self.downsample2 = MS_DownSampling(
#             in_channels=embed_dim[0],
#             embed_dims=embed_dim[1],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock2_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[1], output_dim = embed_dim[1], mlp_ratio=mlp_ratios)]
#         )
#         self.MSBlock2_1 = nn.ModuleList(
#             [MS_Block(
#                 dim=embed_dim[1],
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratios,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[-1],
#                 norm_layer=norm_layer,
#                 sr_ratio=sr_ratios,
#             )]
#         )
#
#         # self.ConvBlock2_2 = nn.ModuleList(
#         #     [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#         # )
#
#         self.downsample3 = MS_DownSampling(
#             in_channels=embed_dim[1],
#             embed_dims=embed_dim[2],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock3_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[2], output_dim = embed_dim[2], mlp_ratio=mlp_ratios)]
#         )
#
#         self.MSBlock3_1 = nn.ModuleList(
#             [MS_Block(
#                 dim=embed_dim[2],
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratios,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[-1],
#                 norm_layer=norm_layer,
#                 sr_ratio=sr_ratios,
#             )]
#         )
#
#         self.downsample4 = MS_DownSampling(
#             in_channels=embed_dim[2],
#             embed_dims=embed_dim[3],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#
#
#         self.ConvBlock4_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[3], output_dim = embed_dim[3], mlp_ratio=mlp_ratios)]
#         )
#
#         # self.WorkConvBlock = nn.ModuleList(
#         #     [MS_ConvBlock(dim=in_channels, output_dim = embed_dim[-1], mlp_ratio=mlp_ratios)]
#         # )
#         # self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
#
#         self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
#         # self.avg = layer.AdaptiveAvgPool1d((1),step_mode='m')
#         # Build classifier
#         self.future_enabled = cfg.MODEL.LSTR.FUTURE_SECONDS>0
#         if self.future_enabled:
#                 param = cfg.MODEL.LSTR.GEN_MODULE
#                 # self.gen_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], embed_dim[-1])
#
#                 self.gen_layer = MS_Block(
#                     dim=embed_dim[-1],
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[-1],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                 )
#                 # self.gen_layer = tr.Block(dim=in_channels, num_heads=num_heads,T=T)
#                 self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
#                         #     # CCI
#
#         self.classifier = nn.Linear(embed_dim[-1], self.num_classes)
#         if self.cfg.DATA.DATA_NAME == 'EK100':
#             self.classifier_verb = nn.Linear(self.d_model, 98)
#             self.classifier_noun = nn.Linear(self.d_model, 301)
#             self.dropout_cls = nn.Dropout(0.8)
#         # self.apply(self.init_weight)
#         # self.init_weight()
#     def init_weight(self,m):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d)):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)
#
#     def transfer(self, state_dict: Mapping[str, Any]):
#         _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
#         return self.load_state_dict(_state_dict, strict=False)
#
#     def cci(self, memory, work_memories,i, mask=None):
#             # print(memory.shape,work_memories.shape,'memory.shape,work_memories.shape')
#             for i in range(self.depth):
#                 output = self.block_w[i](work_memories, memory)
#                 # print(output.shape)
#                 if self.depth > 1 and i != self.depth-1:
#                     memory = torch.cat([memory,output], dim =-1)
#             # print(output.shape)
#             # print(memory.shape,output.shape)
#             his_memory = torch.cat([memory, output],dim=-1)
#             # print(his_memory.shape,'his_memory.shape')
#             dec_query = self.final_query.weight.unsqueeze(1).unsqueeze(0).repeat(self.T, 1, his_memory.shape[1], 1).permute(0,2,3,1)
#             # print(dec_query.shape,his_memory.shape)
#             future = self.gen_layer(dec_query, his_memory)
#             future_rep = [future]
#             short_rep = [output]
#             # print(output.shape,future.shape)
#             return output, future
#     # def cci(self, memory, output, mask):
#     #
#     #     # print(memory.shape,output.shape,'memory.shape,output.shape')
#     #     his_memory = torch.cat([memory, output], dim=-1)
#     #             # print(his_memory.shape,'his_memory.shape')
#     #     enc_query = self.gen_query.weight.unsqueeze(0).unsqueeze(0).repeat(self.T, his_memory.shape[1], 1, 1).permute(0,1,3,2)
#     #
#     #     if enc_query.dim() == 3:
#     #         enc_query = enc_query.unsqueeze(0).repeat(self.T,1,1,1)
#     #     # print(enc_query.shape, his_memory.shape)
#     #     future = self.gen_layer(enc_query, his_memory,his_memory)
#     #     # print(future.shape,'future.shape')
#     #     dec_query = self.final_query.weight.unsqueeze(0).unsqueeze(0).repeat(self.T, his_memory.shape[1], 1, 1).permute(0,1,3,2)
#     #     future_rep = [future]
#     #     short_rep = [output]
#     #     # for i in range(self.cfg.MODEL.LSTR.CCI_TIMES):
#     #     #
#     #     #
#     #     #     total_memory = torch.cat([memory, output, future], dim=-1)
#     #     #     output = self.work_fusions[i](output, total_memory, total_memory)
#     #     #             # print(output.shape,'output.shape')
#     #     #     short_rep.append(output)
#     #     #             # print(memory.shape, output.shape, future.shape, 'memory.shape,output.shape,future.shape22')
#     #     #     total_memory = torch.cat([memory, output, future],dim=-1)
#     #     #             # print(total_memory.shape, 'total_memory.shape22')
#     #     #     if i == 0:
#     #     #         future = self.fut_fusions[i](dec_query, total_memory, total_memory)
#     #     #         future_rep.append(future)
#     #     #     elif i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
#     #     #                 # else:
#     #     #
#     #     #         future = self.fut_fusions[i](future, total_memory, total_memory)
#     #     #                 # print(future.shape,'future.shape')
#     #     #         future_rep.append(future)
#     #     return short_rep, future_rep
#     def forward_features(self, x,output):
#         works, futs = [],[]
#
#         x = self.downsample1_1(x)
#         for blk in self.ConvBlock1_1:
#             x = blk(x)
#
#         for blk in self.MSBlock1_1:
#             x = blk(x,x)
#
#         x = self.downsample1_2(x)
#         for blk in self.ConvBlock1_2:
#             x = blk(x)
#         for blk in self.MSBlock1_2:
#             x = blk(x,x)
#
#
#
#
#         x = self.downsample2(x)
#         for blk in self.ConvBlock2_1:
#             x = blk(x)
#         for blk in self.MSBlock2_1:
#             x = blk(x,x)
#         # for blk in self.ConvBlock2_2:
#         #     x = blk(x)
#
#
#         x = self.downsample3(x)
#         for blk in self.ConvBlock3_1:
#             x = blk(x)
#
#         for blk in self.MSBlock3_1:
#             x = blk(x,x)
#         # for blk in self.block3:
#         #     x = blk(x)
#         # work, fut = self.cci(x, output,0)
#         # works.append(work)
#         # futs.append(fut)
#
#         x = self.downsample4(x)
#         for blk in self.ConvBlock4_1:
#             x = blk(x)
#
#         # for blk in self.block4:
#         #     x = blk(x)
#         work, fut = self.cci(x, output,0)
#
#         # works.append(work)
#         # futs.append(fut)
#
#         return work,fut
#         # return x  # T,B,C,N
#
#     def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=1):
#         # print(visual_inputs.shape)
#         # print(self.long_memory_num_samples)
#         if self.long_enabled:
#             # Compute long memories
#             the_long_memories = self.feature_head_long(
#                 visual_inputs[:, :self.long_memory_num_samples],
#                 motion_inputs[:, :self.long_memory_num_samples])
#             x = the_long_memories.permute(0,2,1)
#             # the_long_memories = self.pos_encoding(the_long_memories, padding=0)
#             # short_mem = self.short_modules(short_mem)
#
#         if self.work_enabled:
#             work_memories = self.feature_head_work(
#                 visual_inputs[:, self.long_memory_num_samples:],
#                 motion_inputs[:, self.long_memory_num_samples:],
#             )
#             # work_memories = self.pos_encoding(work_memories, padding=0)
#
#             # print(the_long_memories.shape,work_memories.shape,'the_long_memories.shape,work_memories.shape')
#             # x = the_long_memories.permute(0,2,1)
#             # print(x.shape,'x.shape')
#
#             # Build mask
#             work_memories = work_memories.transpose(0, 1)
#             # print(work_memories.shape,memory.shape,'11')
#
#
#         if x.dim() != 4:
#             x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
#
#             if self.anticipation_num_samples > 0:
#                 anticipation_queries = self.pos_encoding(
#                     self.final_query.weight[:self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH
#                                             :self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...].unsqueeze(1).repeat(1,
#                                                                                                                     work_memories.shape[
#                                                                                                                         1],
#                                                                                                                     1),
#                     padding=self.work_memory_num_samples)
#                 # print(work_memories.shape,anticipation_queries.shape)
#                 work_memories = torch.cat((work_memories, anticipation_queries), dim=0).permute(1,2,0)
#
#             work_memories = work_memories.unsqueeze(0).repeat(self.T, 1, 1, 1)
#             # print(work_memories.shape)
#             assert x.dim() == 4
#         else:
#             #### [B, T, C, H, W] -> [T, B, C, H, W]
#             x = x.transpose(0, 1)
#         # for blk in self.WorkConvBlock:
#         #     work_memories = blk(work_memories)
#         work, fut = self.forward_features(x, work_memories)
#
#         # mask = tr.generate_square_subsequent_mask(
#         #     work_memories.shape[-1])
#         # mask = mask.to(x.device)
#
#
#         if self.future_enabled:
#
#             # print(work.shape)
#             # work = self.avg(work.permute(0,1,3,2))
#             # print(work.shape)
#             work_score = self.classifier(work.permute(0,1,3,2)).mean(0)
#             # print(fut.shape)
#             # fut = self.avg(fut.permute(0,1,3,2))
#             # print(fut.shape)
#             fut_scores = self.classifier(fut.permute(0,1,3,2)).mean(0)
#
#             # print(work_score.shape,fut_scores.shape)
#             return ([work_score],[fut_scores])
#         else:
#             # print(x.shape)
#             work_score = self.classifier(oa_feat.permute(0, 2, 1))
#             return work_score

def fft_to_continuous(p, N):
    n = np.arange(N)  # 创建一个范围数组
    freq = np.fft.fftfreq(N) * (2 * math.pi / N)  # 计算频率
    P = torch.fft.fft(p) / N  # 执行傅里叶变换并归一化

    return n, freq, torch.abs(P)
class LSTR(nn.Module):

    def __init__(self, cfg,
                 embed_dim=[64,128, 256, 512],
                 embed_dim_w=[256, 512],
                 num_heads=4,
                 mlp_ratios=1,
                 depths=1,
                 T=4,
                 times=4,
                 in_channels =1024,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 sr_ratios=1,
                 prologue=None,
                 **kwargs,
                 ):
        super(LSTR, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_enabled=True
        self.work_enabled=True
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg,dim=in_channels)
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg,dim=embed_dim[-1])
        self.d_model = in_channels
        self.depth = depths
        self.dropout = cfg.MODEL.LSTR.DROPOUT

        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.future = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0
        # Build position encoding
        self.T = T
        self.num_heads = num_heads
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, len(embed_dim))
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, output_dim = embed_dim[0] // 2,mlp_ratio=mlp_ratios) for i in range(times)]
        )

        # self.fusion1_1 = CGAFusion_SNN(dim=embed_dim[0] // 2)


        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], output_dim = embed_dim[0], mlp_ratio=mlp_ratios) for i in range(times)]
        )

        # self.fusion1_2 = CGAFusion_SNN(dim=embed_dim[0])




        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], output_dim = embed_dim[1], mlp_ratio=mlp_ratios) for i in range(times)]
        )

        # self.fusion2_1 = CGAFusion_SNN(dim=embed_dim[1])


        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock3_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[2], output_dim = embed_dim[2], mlp_ratio=mlp_ratios) for i in range(times)]
        )

        # self.fusion3_1 = CGAFusion_SNN(dim=embed_dim[2])
        #

        #
        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )



        self.ConvBlock4_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[3], output_dim = embed_dim[3], mlp_ratio=mlp_ratios) for i in range(times)]
        )
        # self.fusion4_1 = CGAFusion_SNN(dim=embed_dim[3])





        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.block_w = nn.ModuleList([MS_Block(
            dim=embed_dim[-1],
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[j],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios,
        ) for j in range(depths)]
        )


        # Build classifier
        self.future_enabled = cfg.MODEL.LSTR.FUTURE_SECONDS>0



            # CCI
        # self.gen_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], embed_dim[-1])
        if self.future_enabled:
            param = cfg.MODEL.LSTR.GEN_MODULE
            self.gen_layer = MS_Block(
                    dim=embed_dim[-1],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[-1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
            self.pos_encoding = tr.PositionalEncoding(embed_dim[-1], self.dropout)
            self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], embed_dim[-1])
            # self.gen_query = nn.Embedding(param[0],  embed_dim[-1])

        self.cci_time = cfg.MODEL.LSTR.CCI_TIMES
        self.classifier = nn.Linear(embed_dim[-1], self.num_classes)
        if self.cfg.DATA.DATA_NAME == 'EK100':
            self.classifier_verb = nn.Linear(self.d_model, 98)
            self.classifier_noun = nn.Linear(self.d_model, 301)
            self.dropout_cls = nn.Dropout(0.8)

    def init_weight(self,m):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def cci(self, memory, work_memories, mask=None):
            # print(memory.shape,work_memories.shape,'memory.shape,work_memories.shape')
            for i in range(self.depth):
                output = self.block_w[i](work_memories, memory)
                # print(output.shape)
            his_memory = torch.cat([memory, output],dim=-1)
            # enc_query = self.gen_query.weight.unsqueeze(1).unsqueeze(0).repeat(self.T, 1, his_memory.shape[1], 1).permute(0,2,3,1)
            # fut = self.fut_layer(enc_query, his_memory)
            # print(output.shape)
            # print(memory.shape,output.shape)
            # his_memory = torch.cat([memory, output,fut],dim=-1)
            # print(his_memory.shape,'his_memory.shape')
            dec_query = self.final_query.weight.unsqueeze(1).unsqueeze(0).repeat(self.T, 1, his_memory.shape[1], 1).permute(0,2,3,1)
            # dec_query = self.gen_query.weight.unsqueeze(1).unsqueeze(0).repeat(self.T, 1, his_memory.shape[1], 1).permute(0,2,3,1)
            # print(dec_query.shape,his_memory.shape)
            future = self.gen_layer(dec_query, his_memory)
            # future += fut
            future_rep = [future]
            short_rep = [output]
            # print(output.shape,future.shape)
            return output, future

    def forward_features(self, x,x_2=None):
        works, futs = [],[]

        x = self.downsample1_1(x)

        for blk in self.ConvBlock1_1:
            x = blk(x)
        # x_2 = self.downsample1_1(x_2)
        # for blk in self.ConvBlock1_1:
        #     x_2 = blk(x_2)

        # x = self.fusion1_1(x, x_2)


        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        # x_2 = self.downsample1_2(x_2)
        # for blk in self.ConvBlock1_2:
        #     x_2 = blk(x_2)

        # x = self.fusion1_2(x, x_2)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        # x_2 = self.downsample2(x_2)
        # for blk in self.ConvBlock2_1:
        #     x_2 = blk(x_2)


        # x = self.fusion2_1(x, x_2)

        x = self.downsample3(x)
        for blk in self.ConvBlock3_1:
            x = blk(x)
        # x_2 = self.downsample3(x_2)
        # for blk in self.ConvBlock3_1:
        #     x_2 = blk(x_2)

        # x = self.fusion3_1(x, x_2)

        x = self.downsample4(x)
        for blk in self.ConvBlock4_1:
            x = blk(x)
        # x_2 = self.downsample4(x_2)
        # for blk in self.ConvBlock4_1:
        #     x_2 = blk(x_2)
        # print(x)
        # print('111111111')
        # print(x_2)

        # x = self.fusion4_1(x, x_2)

        return x,x_2
        # return x  # T,B,C,N
    def forward_features_worl(self, w,w_2):
        works, futs = [],[]

        w = self.downsample1_1_w(w)
        for blk in self.ConvBlock1_1_w:
            w = blk(w)
        w_2 = self.downsample1_1_w(w_2)
        for blk in self.ConvBlock1_1_w:
            w_2 = blk(w_2)


        w = self.fusion1_1_w(w,w_2)

        w = self.downsample1_2_w(w)
        for blk in self.ConvBlock1_2_w:
            w = blk(w)
        w_2 = self.downsample1_2_w(w_2)
        for blk in self.ConvBlock1_2_w:
            w_2 = blk(w_2)

        w = self.fusion1_2_w(w, w_2)

        w = self.downsample2_w(w)
        for blk in self.ConvBlock2_1_w:
            w = blk(w)
        w_2 = self.downsample2_w(w_2)
        for blk in self.ConvBlock2_1_w:
            w_2 = blk(w_2)


        w = self.fusion2_1_w(w, w_2)

        # x = x + x_2


        return w,w_2
        # return x  # T,B,C,N

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=1):
        # print(visual_inputs.shape,motion_inputs.shape)
        # print(self.long_memory_num_samples)
        feature_SW = list()
        feature_SF = list()
        if visual_inputs.dim() != 4:
            visual_inputs = visual_inputs.unsqueeze(0).repeat(self.T,1,1,1).permute(0,1,3,2)
        if motion_inputs.dim() != 4:
            motion_inputs = motion_inputs.unsqueeze(0).repeat(self.T,1,1,1).permute(0,1,3,2)
        # print(visual_inputs.shape,motion_inputs.shape)
        if self.long_enabled:
            # Compute long memories
            the_long_memories_visual, the_long_memories_motion = self.feature_head_long(visual_inputs[:, :, :,:self.long_memory_num_samples].permute(0,1,3,2),motion_inputs[:, :, : ,:self.long_memory_num_samples].permute(0,1,3,2))
            # print(the_long_memories_visual.shape,the_long_memories_motion.shape,'the_long_memories_visual.shape,the_long_memories_motion.shape')
            x_visual = the_long_memories_visual.permute(0,1,3,2)
            x_motion = the_long_memories_motion.permute(0,1,3,2)
            # the_long_memories = self.pos_encoding(the_long_memories, padding=0)
            # short_mem = self.short_modules(short_mem)

        if self.work_enabled:
            work_memories_visual, work_memories_motion =  self.feature_head_work(visual_inputs[:,:, :, self.long_memory_num_samples:].permute(0,1,3,2), motion_inputs[:, :, :, self.long_memory_num_samples:].permute(0,1,3,2))
            # print(work_memories_visual.shape,work_memories_motion.shape,'work_memories_visual.shape,work_memories_motion.shape')

            # work_memories = self.pos_encoding(work_memories, padding=0)

            # print(the_long_memories.shape,work_memories.shape,'the_long_memories.shape,work_memories.shape')
            # x = the_long_memories.permute(0,2,1)
            # print(x.shape,'x.shape')

            # Build mask
            work_memories_visual = work_memories_visual.permute(0,1,3,2)
            work_memories_motion = work_memories_motion.permute(0,1,3,2)
            # print(work_memories_visual.shape,'11')


        if x_visual.dim() != 4:
            x_visual = x_visual.unsqueeze(0).repeat(self.T, 1, 1, 1)
        # if x_motion.dim() != 4:
        #     x_motion = x_motion.unsqueeze(0).repeat(self.T, 1, 1, 1)

        assert x_visual.dim() == x_motion.dim() == 4
        if self.anticipation_num_samples > 0:
            anticipation_queries = self.pos_encoding(
                    self.final_query.weight[:self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH
                                            :self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...].unsqueeze(1).repeat(1,
                                                                                                                    work_memories_visual.shape[
                                                                                                                        1],
                                                                                                                    1),
                    padding=self.work_memory_num_samples)
            anticipation_queries = anticipation_queries.unsqueeze(0).repeat(self.T,1,1,1).permute(0,2,3,1)
            # print(anticipation_queries.shape)
            work_memories_visual = torch.cat((work_memories_visual, anticipation_queries), dim=-1)
            # work_memories_motion = torch.cat((work_memories_motion, anticipation_queries), dim=-1)

            if work_memories_visual.dim() != 4:
                work_memories_visual = work_memories_visual.unsqueeze(0).repeat(self.T, 1, 1, 1)
            # if work_memories_motion.dim() != 4:
            #     work_memories_motion = work_memories_motion.unsqueeze(0).repeat(self.T, 1, 1, 1)


        else:
            #### [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
        # for blk in self.WorkConvBlock:
        #     work_memories = blk(work_memories)
        # print(work_memories_visual.shape)
        x,x_2 = self.forward_features(x_visual, x_motion)
        # feature_r.append(torch.cat([x,x_2],dim=2).mean(0))
        w,w_2 = work_memories_visual,work_memories_motion
        # w, w_2 = self.forward_features_worl(work_memories_visual,work_memories_motion)
        # T,B,C = w.shape[0],w.shape[1],w.shape[2]
        # print(w.shape)
        # w =w.flatten(0,1)
        # w_2 = w_2.flatten(0,1)
        # w = F.interpolate(w, size=40,mode='linear').reshape(T,B,C,-1)
        # w_2 = F.interpolate(w_2, size=40, mode='linear').reshape(T,B,C,-1)
        # print(x.shape,w.shape)
        work_visual, fut_visual = self.cci(x, w)

        # work_motion, fut_motion = self.cci(x_2, w_2)

        # work_visual = x
        # work_motion = x_2
        # for i in range(self.cci_time):
        #
        #     work_visual = self.work_fusions[i](work_motion,work_visual)
        #     work_motion = self.work_fusions[i](work_visual,work_motion)
        #
        #     fut_visual = self.fut_fusions[i](fut_motion,fut_visual)
        #     fut_motion = self.fut_fusions[i](fut_visual,fut_motion)

        # work = self.fusion_work(work_visual,work_motion)
        # fut = self.fusion_fut(fut_visual,fut_motion)
        # work = torch.cat([work_visual,work_motion], dim=2)
        # fut = torch.cat([fut_visual,fut_motion],dim=2)
        work = work_visual
        fut = fut_visual
        # T_w,B_w,C_w,N_w = work.shape
        # work_f = work.reshape(-1,N_w)
        # n, _, work_f = fft_to_continuous(work_f, T_w*B_w*C_w)
        # work_f = work_f.reshape(T_w,B_w,C_w,N_w)
        # T_f, B_f, C_f, N_f = fut.shape
        # fut_f = fut.reshape(-1, N_f)
        # n, _, fut_f = fft_to_continuous(fut_f, T_f * B_f * C_f)
        # fut_f = fut_f.reshape( T_f, B_f, C_f, N_f )
        feature_SW.append(work.mean(0))
        feature_SF.append(fut.mean(0))
        fut_scores = []
        work_scores = []
        if self.future_enabled:
            # if self.cfg.DATA.DATA_NAME == 'EK100':
            #     noun_score = self.classifier_noun(work.permute(0,1,3,2)).mean(0)
            #     verb_score = self.classifier_verb(work.permute(0,1,3,2)).mean(0)
            #     work_score = (self.classifier(self.dropout_cls(work.permute(0, 1, 3, 2)))).mean(0)
            #
            #     fut_noun_score = self.classifier_noun(fut.permute(0,1,3,2)).mean(0)
            #     fut_verb_score = self.classifier_verb(fut.permute(0,1,3,2)).mean(0)
            #     fut_score = self.classifier(fut.permute(0, 1, 3, 2)).mean(0)
            #     return ([work_score], [fut_score], noun_score, fut_noun_score, verb_score, fut_verb_score)
            # else:

            # work_score_visual = self.classifier(work_visual.permute(0, 1, 3, 2)).mean(0)
            # work_score_motion = self.classifier(work_motion.permute(0, 1, 3, 2)).mean(0)
            # fut_score_visual = self.classifier(fut_visual.permute(0,1,3,2)).mean(0)
            # fut_score_motion = self.classifier(fut_motion.permute(0, 1, 3, 2)).mean(0)
            #
            # work_scores.append(work_score_visual)
            # work_scores.append(work_score_motion)
            # fut_scores.append(fut_score_visual)
            # fut_scores.append(fut_score_motion)
            work_score = self.classifier(work.permute(0, 1, 3, 2)).mean(0)
            # print(fut.shape)
            # T,B,C,N = fut.shape

            # fut = F.interpolate(fut.reshape(-1,C,N), size=self.future_num_samples).reshape(T,B,C,-1)
            # print(fut.shape)
            fut_score = self.classifier(fut.permute(0, 1, 3, 2)).mean(0)
            work_scores.append(work_score)
            fut_scores.append(fut_score)

            feature_SW.append(work_score)
            feature_SF.append(fut_score)
            # return ([work_score],[fut_score])
            return work_scores,fut_scores,feature_SW,feature_SF
        else:
            # print(x.shape)
            work_score = self.classifier(work.permute(0, 2, 1))
            return work_score
@registry.register('LSTR')
class LSTRStream(LSTR):

    def __init__(self, cfg):
        super(LSTRStream, self).__init__(cfg)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None

    def stream_inference(self,
                         long_visual_inputs,
                         long_motion_inputs,
                         work_visual_inputs,
                         work_motion_inputs,
                         memory_key_padding_mask=None):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        work_memories = self.pos_encoding(self.feature_head_work(
            work_visual_inputs,
            work_motion_inputs,
        ).transpose(0, 1), padding=self.long_memory_num_samples)

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)
            # work_memories = self.short_modules(work_memories)
            # print('1111')
            if self.long_memories_cache is None:
                self.long_memories_cache = long_memories
            else:
                self.long_memories_cache = torch.cat((
                    self.long_memories_cache[1:], long_memories
                ))

            long_memories = self.long_memories_cache
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # if self.cfg.MODEL.LSTR.GROUPS > 0 and (
            #         memory_key_padding_mask == float('-inf')).sum() < self.cfg.MODEL.LSTR.GROUPS:
            #
            #     T = long_memories.shape[0] // self.cfg.MODEL.LSTR.GROUPS
            #     enc_query = enc_queries[0]
            #     long_memories_list = []
            #     max_mem = []
            #     avg_mem = []
            #     for i in range(self.cfg.MODEL.LSTR.GROUPS):
            #         out = self.enc_modules[0](enc_query, long_memories[i * T:(i + 1) * T],
            #                                   memory_key_padding_mask=memory_key_padding_mask[:,
            #                                                           i * T:(i + 1) * T], knn=True,
            #                                   short_mem=work_memories)
            #         weight = self.max_polling(out.permute(1, 2, 0)).permute(2, 0, 1)
            #         out = self.average_pooling(out.permute(1, 2, 0)).permute(2, 0, 1)
            #         out = out + weight
            #         long_memories_list.append(out)
            #     long_memories = torch.cat(long_memories_list)
            # else:
            #     long_memories = self.enc_modules[0](enc_queries[0], long_memories,
            #                                         memory_key_padding_mask=memory_key_padding_mask, knn=True,
            #                                         short_mem=work_memories)

            self.compressed_long_memories_cache = long_memories
            # print('aa')
            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask,short_mem=work_memories)
            self.compressed_long_memories_cache = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]
            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        # print('11111')
        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)
            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)