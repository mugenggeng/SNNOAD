# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
DECO ConvNet classes.
"""

import copy
from typing import Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import DropPath
from torch.cuda.amp import autocast

from .encoder_module import *
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
# from .mamba import Block
# from mamba_ssm.ops.triton.layernorm import RMSNorm,rms_norm_fn

# def create_block(
#     d_model,
#     ssm_cfg=None,
#     norm_epsilon=1e-5,
#     rms_norm=True,
#     residual_in_fp32=True,
#     fused_add_norm=True,
#     layer_idx=None,
#     device=None,
#     dtype=None,
# ):
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     mixer_cls = partial(Mamba, layer_idx=layer_idx)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
#     )
#     block = Block(
#         d_model,
#         mixer_cls,
#         norm_cls=norm_cls,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block

class DECO_ConvNet(nn.Module):
    '''DECO ConvNet class, including encoder and decoder'''

    def __init__(self, d_model=512, enc_dims=[512,1024,512], enc_depth=[2,6,2],
                 num_decoder_layers=6, normalize_before=False, return_intermediate_dec=False,qN=256):
        super().__init__()

        # object query shape
        self.qN = qN
        # self.qW = int(np.float(num_queries)/np.float(self.qH))
        # print('query shape {}x{}'.format(self.qH, self.qW))

        # encoder
        self.encoder = DecoEncoder(enc_dims=enc_dims, enc_depth=enc_depth)     

        # decoder
        decoder_layer = DecoDecoderLayer(d_model, normalize_before, self.qN)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DecoDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, qN=self.qN)
        # other initialization
        self.tgt = nn.Embedding(qN, d_model)
        self._reset_parameters()
        self.d_model = d_model
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed,mask):
        bs, c, N = src.shape
        tgt=self.tgt.weight.unsqueeze(1).repeat(1, bs, 1).permute(1,2,0)
        memory,mask = self.encoder(src,mask)
        hs,mask = self.decoder(tgt, memory, d_model=self.d_model, mask = mask,query_pos=query_embed)

        return hs.transpose(1, 2), memory,mask


class DecoEncoder(nn.Module):
    '''Define Deco Encoder'''
    def __init__(self, enc_dims=[512,512,512], enc_depth=[1,1,1]):
        super().__init__()
        self._encoder = ConvNeXt(depths=enc_depth, dims=enc_dims) 

    def forward(self, src,mask):
        output,mask = self._encoder(src,mask)
        return output,mask


class DecoDecoder(nn.Module):
    '''Define Deco Decoder'''
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, qN=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.qN = qN
        # self.qW = qW
    def stream_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None):
        output = tgt

        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        output = self.layers[0].stream_inference(output, memory, pos,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    def forward(self, tgt, memory, d_model, mask,query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        # print(output.shape)
        # print(output.shape, memory.shape, query_pos.shape, 'output.shape,memory.shape')
        for layer in self.layers:
            output,mask = layer(output, memory, mask=mask,query_pos=query_pos)
            # print(output.shape,'output11')
            # output=output.flatten(2).permute(2, 0, 1)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            # print(output.shape,'output.shape')
            output = self.norm(output.permute(0,2,1)).permute(0,2,1)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output,mask

class DecoDecoderLayer(nn.Module):
    '''Define a layer for Deco Decoder'''
    def __init__(self,d_model, normalize_before=False, qN=10,
                 drop_path=0.,layer_scale_init_value=1e-6):
        super().__init__()
        self.normalize_before = normalize_before
        self.qN = qN


        # The SIM module   
        self.dwconv1 = MaskedConv1D(d_model, d_model, kernel_size=9, padding=4,groups=d_model)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(d_model, 4 * d_model) 
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * d_model, d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.T = 4
        # The CIM module
        self.dwconv2 = MaskedConv1D(d_model, d_model, kernel_size=9, padding=4,groups=d_model)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.pwconv2_1 = nn.Linear(d_model, 4 * d_model) 
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Linear(4 * d_model, d_model)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mamaba_block = nn.ModuleList()
        for i in range(2):
            self.mamaba_block.append(Mamba(d_model//self.T).requires_grad_(False))
        # self.alph = nn.Parameter(torch.zeros(1,1),requires_grad=True)
        # self.norm = nn.LayerNorm(d_model)
        # self.num_layers = 1
        # self.gru = nn.GRU(d_model, d_model, self.num_layers, batch_first=True)
        # self.h0 = torch.zeros(self.num_layers, 1, d_model)

    def stream_inference(self, tgt, memory, pos, query_pos=None,tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # print(tgt.shape,'tgt.shape')
        b, d, N = memory.shape
        if query_pos is not None:
            tgt2 = tgt + query_pos
        else:
            tgt2 = tgt
        # if tgt2.shape[-1] != memory.shape[-1]:
        #     weight = nn.Parameter(torch.randn(tgt2.shape[-1],memory.shape[-1]))
        #     bias = nn.Parameter(torch.randn(tgt2.shape[-1]))
        #     memory = F.linear(memory)
        tgt2,tgt_mask = self.dwconv1(tgt2,tgt_mask)
        tgt2 = tgt2.permute(0, 2, 1)  # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2 # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)

        # CIM
        tgt = F.interpolate(tgt, size=[N])
        tgt2 = tgt + memory
        tgt2,tgt_mask = self.dwconv2(tgt2,tgt_mask)
        tgt2 = tgt2 + tgt
        tgt2 = tgt2.permute(0, 2, 1)  # (b,d,h,w)->(b,h,w,d)
        tgt2 = self.norm2(tgt2)
        return tgt

    def forward(self, tgt, memory, mask, query_pos: Optional[Tensor] = None):
        # SIM
        # print(tgt.shape,memory.shape,'tgt.shape')
        b, d, N = memory.shape
        B = tgt.shape[0]
        indentity = tgt

        # loca_mem = tgt.permute(0,2,1)
        loca_mem_list  = []
        num = tgt.shape[1] // self.T

        for i in range(self.T):
            loca_mem = tgt[:,i*num : (i+1)*num,:].permute(0,2,1)
            # print(type(loca_mem))
            for mod in self.mamaba_block:
                loca_mem = mod(loca_mem)
            loca_mem_list.append(loca_mem)
        loca_mem = torch.cat(loca_mem_list,dim=-1)
        # print(loca_mem)
        tgt = tgt + loca_mem.permute(0,2,1)
        # print(loca_mem)
        # h0 = self.h0.expand(-1, B, -1).to(tgt.device)
        # # print(tgt.shape)
        # tgt, _ = self.gru(tgt.permute(0, 2, 1), h0)
        # # print(tgt.shape)
        # tgt = tgt.permute(0, 2, 1) + indentity

        if query_pos is not None:
            if tgt.shape[-1] != query_pos.shape[-1]:
                weight = nn.Parameter(torch.randn(tgt.shape[-1], query_pos.shape[-1])).to(tgt.device)
                bias = nn.Parameter(torch.randn(tgt.shape[-1])).to(tgt.device)
                query_pos = F.linear(query_pos,weight,bias)
                # print(memory.shape,'memory.shape')
            # print(tgt.shape,query_pos.shape,'tgt.shape,query_pos.shape')
            tgt2 = tgt + query_pos
        else:
            tgt2 = tgt

        # tgt2 = tgt + query_pos
        tgt2,mask = self.dwconv1(tgt2,mask)
        tgt2 = tgt2.permute(0, 2, 1) # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2.permute(0,2,1) # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)
        # print(tgt.shape,'444444')
        # tgt = tgt + self.drop_path1(local_mem.permute(0, 2, 1))
        # tgt = self.norm(tgt.permute(0, 2, 1)).permute(0, 2, 1)

        # CIM
        tgt = F.interpolate(tgt, size=N)
        # print(tgt.shape,memory.shape,'tgt.shape,memory.shape')
        tgt2 = tgt + memory
        tgt2,mask = self.dwconv2(tgt2,mask)
        tgt2 = tgt2+tgt 
        tgt2 = tgt2.permute(0, 2, 1) # (b,d,h,w)->(b,h,w,d)
        tgt2=self.norm2(tgt2)
        # print(tgt2.shape, '222')

        # print(local_mem.shape,tgt2.shape,'1111')
        # local_mem = F.interpolate(local_mem, size=N)
        # tgt2 = tgt2+local_mem
        # FFN
        tgt = tgt2
        tgt2 = self.pwconv2_1(tgt2)
        tgt2 = self.act2(tgt2)
        tgt2 = self.pwconv2_2(tgt2)
        if self.gamma2 is not None:
            tgt2 = self.gamma2 * tgt2
        tgt2 = tgt2.permute(0,2,1) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt.permute(0,2,1) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt + self.drop_path1(tgt2)
        # print(tgt.shape,'111')
        # pooling


        m = nn.AdaptiveAvgPool1d(self.qN)
        tgt = m(tgt)
        # indentity = tgt

        # h0 = self.h0.expand(-1, B, -1).to(tgt.device)
        # # print(tgt.shape)
        # tgt, _ = self.gru(tgt.permute(0, 2, 1), h0)
        # # print(tgt.shape)
        # tgt = tgt.permute(0, 2, 1) + indentity



        return tgt,mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deco_convnet(args):
    return DECO_ConvNet(num_queries=args.num_queries,
                        d_model=args.hidden_dim,
                        num_decoder_layers=args.dec_layers,
                        normalize_before=args.pre_norm,
                        return_intermediate_dec=True,
                        qH=args.qH,
                        )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
