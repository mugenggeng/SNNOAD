# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from .multihead_attention import MultiheadAttentionStream as MultiheadAttention
from .multihead_attention1 import MultiheadAttentionStream as MultiheadAttention1
from .layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul,mem_update
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from spikingjelly.activation_based import layer
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from einops.layers.torch import Rearrange

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from .multihead_attention import MultiheadAttentionStream as MultiheadAttention
from .multihead_attention1 import MultiheadAttentionStream as MultiheadAttention1


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask, knn=knn)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def stream_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None,short_mem=None):
        output = tgt

        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        output = self.layers[0].stream_inference(output, memory, pos,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,short_mem=short_mem)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, knn=False,short_mem=None,long_mem =None):
        output = tgt
        # print(typ)
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, knn=knn,short_mem=short_mem,long_mem=long_mem)
            # print(type(output))
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, knn=knn)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def silu(x):
    return x * torch.sigmoid(x)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', short=False,long=False,gru=True,atten=True):
        super(TransformerDecoderLayer, self).__init__()
        self.atten = atten
        if self.atten:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout,short=short)
        self.short = short
        self.long = long
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        # self.norm5 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)
        # self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.linear_soft = nn.Linear(d_model,dim_feedforward)
        self.gru = gru
        if self.gru:
            self.num_layers = 1
            # self.gru1 = nn.GRU(d_model, d_model, self.num_layers, batch_first=True)
            self.gru2 = nn.GRU(d_model, d_model, self.num_layers, batch_first=True)
            self.h0 = torch.zeros(self.num_layers, 1, d_model)
        # #



        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None,short_mem=None):

        # print(self.short)
        indetity = tgt
        B = tgt.shape[1]
        h0 = self.h0.expand(-1, B, -1).to(tgt.device)
        tgt, _ = self.gru(tgt.permute(1, 0, 2), h0) + indetity
        tgt = tgt.permute(1, 0, 2)

        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask,short_mem=short_mem)[0]



        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, knn=False, short_mem = None,long_mem=None):

        if self.gru:
            if not hasattr(self, '_flattened'):
                # self.gru1.flatten_parameters()
                self.gru2.flatten_parameters()
                setattr(self, '_flattened', True)


            indetity = tgt
            B = tgt.shape[1]
            h0 = self.h0.expand(-1, B, -1).to(tgt.device)
            tgt, _ = self.gru2(tgt.permute(1, 0, 2), h0)
            tgt = tgt.permute(1, 0, 2) + indetity

        if self.atten:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask, knn=knn)[0]

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, knn=knn,short_mem = short_mem)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class LongMemLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', short=False,long=False,gru=True,atten=False):
        super(LongMemLayer, self).__init__()

        self.atten = atten
        if self.atten:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.short = short
        self.long = long
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.linear_soft = nn.Linear(d_model,dim_feedforward)

        self.num_layers = 1
        # self.gru1 = nn.GRU(d_model, d_model, self.num_layers, batch_first=True)
        self.gru = gru
        if self.gru:
            self.gru2 = nn.GRU(d_model, d_model, self.num_layers, batch_first=True)
            self.h0 = torch.zeros(self.num_layers, 1, d_model)
        # #
        if self.short:
            self.proj = nn.Sequential(nn.Linear(d_model,dim_feedforward),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(dim_feedforward,d_model)
                                      )

            # self.norm2 = nn.LayerNorm(d_model)
            # self.dropout4 = nn.Dropout(dropout)




        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LongMemLayer, self).__setstate__(state)

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None,short_mem=None):

        # print(self.short)

        if self.tgt_cache is None:
            if self.gru:
                indetity = tgt
                B = tgt.shape[1]
                h0 = self.h0.expand(-1, B, -1).to(tgt.device)
                tgt, _ = self.gru2(tgt.permute(1, 0, 2), h0)
                tgt = tgt.permute(1, 0, 2) + indetity

            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask,short_mem=short_mem)[0]

        if self.short:
            if short_mem.shape[0] != tgt2.shape[0]:
                short_mem = F.interpolate(short_mem.permute(1, 2, 0), size=(tgt2.shape[0]), mode='nearest').permute(2,
                                                                                                                    0,
                                                                                                                    1)
            # identify = short_mem
            # short_mem =
            short_mem = short_mem + self.dropout2(self.proj(short_mem))
            # short_mem = self.norm2(short_mem)
            # short_mem = self.proj(short_mem)
            short_mem = silu(short_mem)
            # weight = torch.sum(short_mem,dim=1,keepdim=False)

            tgt2 = tgt2 * short_mem

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, knn=False, short_mem = None,long_mem=None):

        if self.gru:
            if not hasattr(self, '_flattened'):
                # self.gru1.flatten_parameters()
                self.gru2.flatten_parameters()
                setattr(self, '_flattened', True)
            indetity = tgt
            B = tgt.shape[1]
            h0 = self.h0.expand(-1, B, -1).to(tgt.device)
            tgt, _ = self.gru2(tgt.permute(1, 0, 2), h0)
            tgt = tgt.permute(1, 0, 2) + indetity
        if self.atten:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask, knn=knn)[0]

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, knn=knn)[0]

        if self.short:
            if short_mem.shape[0] != tgt2.shape[0]:
                short_mem = F.interpolate(short_mem.permute(1, 2, 0), size=(tgt2.shape[0]), mode='nearest').permute(2,
                                                                                                                      0,
                                                                                                                      1)

            short_mem = short_mem + self.dropout2(self.proj(short_mem))

            # weight = torch.sigmoid(short_mem).permute(1, 2, 0)[0]
            # print(weight.shape)
            # avg_attention_weights = torch.mean(weight, dim=1, keepdim=True).detach().cpu()
            # fig, ax = plt.subplots(figsize=(10, 10))
            #
            # # 设置色彩映射
            # norm = Normalize(vmin=0, vmax=weight.max())
            # image = ax.imshow(avg_attention_weights, cmap='viridis', norm=norm)
            #
            # # 隐藏坐标轴
            # ax.set_xticks([])
            # ax.set_yticks([])
            #
            # # 为图像添加颜色条
            # fig.colorbar(image, ax=ax)
            # plt.show()

            short_mem = silu(short_mem)


            tgt2 = tgt2 * short_mem

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class TransformerDecoder1(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder1, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

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

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer1(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer1, self).__init__()

        self.self_attn = MultiheadAttention1(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention1(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer1, self).__setstate__(state)

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))



class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(),
                    Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size),
                    BN(inner_channels),
                ))
        self.down = nn.Sequential(
            activation(),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, activation=mem_update,norm=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.norm_bn = norm
        self.momentum = 0.999
        # self.activation_in = snn.Leaky(beta=0.95)
        self.activation_in = activation()
        self.T = 4
        self.W = layer.Conv1d(dim, dim * 2,1 ,bias=False, step_mode='m')
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = activation()
        self.activation_out = activation()

        self.Wproj = Conv1x1(dim, dim)

        if self.norm_bn:
            self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [T, B, C, H, W]
        # x = x.permute(0,2,1)
        if x.dim() != 4:
            x = x.permute(0,2,1)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
            assert x.dim() == 4
        T, B, C, L = x.shape
        x_feat = x.clone()
        # if not self.training:
        #     print(x.shape,)
        x = self.activation_in(x)
        # if not self.training:
        #     print(x.shape)

        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        #
        # firing_rate_x = x.detach().mean((0, 1, 3), keepdim=True)
        # if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
        #     self.firing_rate_x = firing_rate_x
        # self.init_firing_rate_x = True
        # self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
        #         1 - self.momentum)
        # scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        # attn = attn * scale1
        attn = self.activation_attn(attn)

        # if self.training:
        #     firing_rate_attn = attn.detach().mean((0, 1, 3), keepdim=True)
        #     if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
        #         self.firing_rate_attn = firing_rate_attn
        #     self.init_firing_rate_attn = True
        #     self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (
        #         1 - self.momentum)
        # scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)
        out = self.matmul2(y2, attn)
        # out = out * scale2
        out = out.reshape(T, B, C, L)

        out = self.activation_out(out)
        # print(out)
        out = self.Wproj(out)
        if self.norm_bn:
            out = self.norm_proj(out)
        out = out + x_feat
        return out


class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=1, groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,-1).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,-1).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, T = 10, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, bias=True, kdim=None, vdim=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.T = T
        self.kdim = kdim if kdim is not None else dim
        self.vdim = vdim if vdim is not None else dim
        self._qkv_same_embed_dim = self.kdim == dim and self.vdim == dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(dim, dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)


        self.scale = 0.125
        # self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.q_lif = mem_update()

        # self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.k_lif = mem_update()
        # self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.v_lif = mem_update()
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.attn_lif = mem_update()
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.proj_lif = mem_update()
    def forward(self, q,k,v,attn_mask=None, key_padding_mask=None):
        # print(q.shape)
        if q.dim() == 3:
            q = q.unsqueeze(0).repeat(self.T, 1, 1,1)
        if k.dim() == 3:
            k = k.unsqueeze(0).repeat(self.T, 1, 1,1)
        if v.dim() == 3:
            v = v.unsqueeze(0).repeat(self.T, 1, 1,1)


        T, B, C, N = q.shape

        head_dim =  C // self.num_heads
        # print(C,self.num_heads,head_dim)
        scaling = float(head_dim) ** -0.5

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, N, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        _b = self.in_proj_bias
        _start = None
        _end = C
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        # print(q.shape,_w.shape,_b.shape)
        q_conv_out = F.linear(q.flatten(0,1).permute(2,0,1), _w, _b).permute(1,2,0)
        q_conv_out = self.q_bn(q_conv_out).unsqueeze(0).reshape(T, B, C, -1).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        _b = self.in_proj_bias
        _start = C
        _end = C * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        # print(k.shape, _w.shape, _b.shape)
        k_conv_out = F.linear(k.flatten(0,1).permute(2,0,1), _w, _b).permute(1,2,0)
        # print(k_conv_out.shape)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, -1).contiguous()
        print(k_conv_out.shape,'k_conv_out')
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        _b = self.in_proj_bias
        _start = C * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v_conv_out = F.linear(v.flatten(0,1).permute(2,0,1), _w, _b).permute(1,2,0)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,-1).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # q = q * scaling
        # x_for_qkv = q.flatten(0, 1)
        # # q = q * scaling
        # q_conv_out = self.q_conv(x_for_qkv)
        # q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        # # print(q_conv_out.shape)
        # q_conv_out = self.q_lif(q_conv_out)
        # q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        #
        # k_for_qkv = k.flatten(0, 1)
        # k_conv_out = self.k_conv(k_for_qkv)
        # k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        # k_conv_out = self.k_lif(k_conv_out)
        # v_conv_out = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        # v_conv_out = self.v_lif(v_conv_out)
        # v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
        #                                                                                                4).contiguous()

        # v_for_qkv = v.flatten(0, 1)
        # v_conv_out = self.v_conv(v_for_qkv)
        # v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        # v_conv_out = self.v_lif(v_conv_out)
        # v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        if attn_mask is not None:
            x += attn_mask
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, T=10, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, T=T, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, q,k,v,attn_mask=None, key_padding_mask=None):
        x = q
        x_attn = (self.attn(q,k,v,attn_mask=attn_mask, key_padding_mask=key_padding_mask))
        # print(x.shape,x_attn.shape,'x.shape,x_attn.shape')
        # print(x_attn)
        x = x + x_attn
        x = x + (self.mlp((x)))

        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv1d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        # T,B,C,N = x.shape
        # x = x.flaten(0,1)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        # sattn = sattn.reshape(T,B,-1,N)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.ca = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(dim//reduction),
            nn.Conv1d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        # T,C,B,N = x.shape
        # x = x.flatten(0,1)
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        # cattn = cattn.reshape(T,B,-1,N)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # T, B, C, N = x.shape
        # x = x.flatten(0,1)
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t n -> b (c t) n')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        # pattn2 = pattn2.reshape(T,B,-1,N)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv1d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        T,B,C,N = x.shape
        x = x.flatten(0,1)
        y = y.flatten(0,1)

        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        # print(cattn.shape,sattn.shape,'cattn.shape,sattn.shape')
        pattn1 = sattn + cattn
        # print(pattn1.shape,'pattn1.shape')
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)

        result = result.reshape(T,B,-1,N)
        return result


