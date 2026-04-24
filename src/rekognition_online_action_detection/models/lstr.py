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
from .transformer.snn_module import MS_DownSampling,MS_ConvBlock,MS_Block ,CGAFusion_SNN,MS_Block_Cross_Weight,MS_Block_Cross_No_Weight
from spikingjelly.activation_based import layer
from typing import Any, List, Mapping
from .feature_head import build_feature_head
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

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


def fft_to_continuous(p, N):
    n = np.arange(N)  # 创建一个范围数组
    freq = np.fft.fftfreq(N) * (2 * math.pi / N)  # 计算频率
    P = torch.fft.fft(p) / N  # 执行傅里叶变换并归一化

    return n, freq, torch.abs(P)
class LSTR(nn.Module):

    def __init__(self, cfg,
                 embed_dim=[64,128,256, 512],
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
        # def set_T (netT):
        #     self.T = netT
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

        # self.work_weight = nn.Parameter(torch.rand(1024, 40))
        # self.fut_weight = nn.Parameter(torch.rand(1024, 48))
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
        self.classifier = nn.Linear(embed_dim[-1]*2, self.num_classes)
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
        x_2 = self.downsample1_1(x_2)
        for blk in self.ConvBlock1_1:
            x_2 = blk(x_2)

        # x = self.fusion1_1(x, x_2)


        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x_2 = self.downsample1_2(x_2)
        for blk in self.ConvBlock1_2:
            x_2 = blk(x_2)

        # x = self.fusion1_2(x, x_2)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        x_2 = self.downsample2(x_2)
        for blk in self.ConvBlock2_1:
            x_2 = blk(x_2)


        # x = self.fusion2_1(x, x_2)

        x = self.downsample3(x)
        for blk in self.ConvBlock3_1:
            x = blk(x)
        x_2 = self.downsample3(x_2)
        for blk in self.ConvBlock3_1:
            x_2 = blk(x_2)

        # x = self.fusion3_1(x, x_2)

        x = self.downsample4(x)
        for blk in self.ConvBlock4_1:
            x = blk(x)
        x_2 = self.downsample4(x_2)
        for blk in self.ConvBlock4_1:
            x_2 = blk(x_2)
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
        # print(self.T)
        # if hasattr(self, 'T'):
        #     if epoch < 10:
        #         self.T = 4
        #     elif epoch < 20:
        #         self.T = 8
        #     else:
        #         self.T = 16

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
            work_memories_motion = torch.cat((work_memories_motion, anticipation_queries), dim=-1)

            if work_memories_visual.dim() != 4:
                work_memories_visual = work_memories_visual.unsqueeze(0).repeat(self.T, 1, 1, 1)
            if work_memories_motion.dim() != 4:
                work_memories_motion = work_memories_motion.unsqueeze(0).repeat(self.T, 1, 1, 1)


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
        # print(x.shape, w.shape,x_2.shape,w_2.shape)
        work_visual, fut_visual = self.cci(x, w)

        work_motion, fut_motion = self.cci(x_2, w_2)

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
        work = torch.cat([work_visual,work_motion], dim=2)
        fut = torch.cat([fut_visual,fut_motion],dim=2)
        # work = work_visual
        # fut = fut_visual
        # T_w,B_w,C_w,N_w = work.shape
        # work_f = work.reshape(-1,N_w)
        # n, _, work_f = fft_to_continuous(work_f, T_w*B_w*C_w)
        # work_f = work_f.reshape(T_w,B_w,C_w,N_w)
        # T_f, B_f, C_f, N_f = fut.shape
        # fut_f = fut.reshape(-1, N_f)
        # n, _, fut_f = fft_to_continuous(fut_f, T_f * B_f * C_f)
        # fut_f = fut_f.reshape( T_f, B_f, C_f, N_f )
        # feature_SW.append(work.mean(0)+self.work_weight.unsqueeze(0).repeat(work_visual.shape[1],1,1))
        # feature_SF.append(fut.mean(0)+self.fut_weight.unsqueeze(0).repeat(work_visual.shape[1],1,1))
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
