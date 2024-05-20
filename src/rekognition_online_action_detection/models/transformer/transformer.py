# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, knn=knn,short_mem=short_mem,long_mem=long_mem)

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
