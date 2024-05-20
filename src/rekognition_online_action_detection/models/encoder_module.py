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
# Modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ============================================================================

"""
DECO Encoder class. Built with ConvNeXt blocks.
"""
from functools import partial
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .deco_convnet import *
from timm.models.layers import trunc_normal_, DropPath

class Block1(nn.Module):
    r""" ConvNeXt Block. 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = MaskedConv1D(dim, dim, kernel_size=7, padding=3,groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x,mask):
        input = x
        # print(x.shape,mask.shape,'Block')
        x ,mask= self.dwconv(x,mask)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)

        x = input + self.drop_path(x)
        return x,mask

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        depths (tuple(int)): Number of blocks at each stage. Default: [2, 6, 2]
        dims (int): Feature dimension at each stage. Default: [120, 240, 480]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, 
                 depths=[2, 6, 2], dims=[512,512,512], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, 
                 ):
        super().__init__()

        self.depths = depths

        self.downsample_layers = nn.ModuleList()
        self.layerNorm=nn.ModuleList()

        for i in range(len(depths)-1):
            downsample_layer = MaskedConv1D(dims[i], dims[i+1], kernel_size=1)
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
            self.layerNorm.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))


            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage = Block1(dim=dims[i], drop_path=dp_rates[cur],
                layer_scale_init_value=layer_scale_init_value)

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        self.apply(self._init_weights)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, mask):
        indetity =src
        src, mask=self.forward_features(src, mask)
        src = src+indetity
        return src,mask

    def forward_features(self, src, mask):
        for i in range(len(self.depths)-1):
            # print(i,'==',src.shape)
            indetity = src
            # print(src.shape,mask.shape,'1111111')
            src ,mask= self.stages[i](src,mask)
            # print(src.shape,'src.shape')
            # print(src.shape, mask.shape, '2222222222')
            src = src+indetity
            src = self.layerNorm[i](src)
            src ,mask = self.downsample_layers[i](src,mask)
            # print(src.shape, mask.shape, '33333333333333')
            # print(src,'src======')
        src,mask = self.stages[len(self.depths)-1](src,mask)
        # print(src,'src===========222222222222')
        return src,mask

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

class TemporalMaxer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride,
            padding,
            n_embd):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(
            kernel_size, stride=stride,padding=padding)

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
    def forward(self, x, mask, **kwargs):

        # out, out_mask = self.channel_att(x, mask)

        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        else:
            # masking out the features
            out_mask = mask
        # print(self.stride,self.kernel_size,self.padding,'self.stride,self.kernel_size,self.padding')
        # print(x.shape,'x4.shape========')
        x1 = self.ds_pooling(x)
        # print(x1.shape,'x5.shape============')
        # print(out_mask.shape,'out_mask')
        out = self.ds_pooling(x) * out_mask.to(x.dtype)

        return out, out_mask.bool()

class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        # print(out_conv.shape[-1],mask.shape[-1],out_conv.shape[-1]!=mask.shape[-1])
        if self.stride > 1 or out_conv.shape[-1] != mask.shape[-1]:
            # downsample the mask using nearest neighbor
            # print(mask.shape[-1])
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
            # print(out_mask.shape)
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        out_mask = out_mask.bool()
        if self.in_channels < self.out_channels:
            # print(self.out_channels,self.in_channels,'self.out_channels,self.in_channels')
            assert self.out_channels % self.in_channels == 0
            n = self.out_channels // self.in_channels
            out_mask = out_mask.repeat(1,n,1)
        elif self.in_channels > self.out_channels:
            out_mask = out_mask[:,:self.out_channels,:]
        # print(out_conv.shape,out_mask.shape)
        out_conv = out_conv * out_mask.detach()
        return out_conv, out_mask
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
    
    def forward(self, x,mask=None):
        # print(x.shape,'x')
        if self.data_format == "channels_last":
            if mask==None:
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            else:
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps),mask
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(x.shape,self.weight.shape)
            x = self.weight[:, None, None] * x.permute(1,0,2) + self.bias[:, None, None]
            if mask==None:
                return x.permute(1,0,2)
            else:
                return x.permute(1, 0, 2),mask
