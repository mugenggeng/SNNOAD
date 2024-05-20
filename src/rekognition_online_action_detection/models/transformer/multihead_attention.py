# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0,embed_dim=None,short=False):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout
        self.short = short
        # self.linear_V = nn.Linear(embed_dim, embed_dim)
        # if self.short:
            # self.proj = nn.Sequential(
            #     nn.Linear(embed_dim,embed_dim*2),
            #     nn.Linear(embed_dim*2, embed_dim)
            #
            # )
            # self.simod = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, knn=False,short_mem = None):
        B, N1, N2 = q.shape[0], q.shape[-2], k.shape[-2]
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        # if self.short:
        #     # U = self.proj(short_mem)
        #     U = self.simod(short_mem)
        #     U = torch.log(U)+U
        #     attn_output_weights += U
        if attn_mask is not None:
            attn_output_weights += attn_mask
        if knn:
            mask=torch.zeros(B,N1,N2,device=q.device,requires_grad=False)
            index=torch.topk(attn_output_weights,k=int(N2 * 3 // 4),dim=-1,largest=True)[1]
            mask.scatter_(-1,index,1.)
            # attn_output_weights = torch.where(mask>0,attn_output_weights,torch.full_like(attn_output_weights,-1e7))
            attn_output_weights = torch.where(mask > 0, attn_output_weights, torch.full_like(attn_output_weights, float('-inf')))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        # if id_embed is not None:
        #     v = self.linear_V(v+id_embed)
        attn_output = torch.bmm(attn_output_weights, v)
        return attn_output


class DotProductAttentionStream(DotProductAttention):

    def __init__(self, dropout=0.0,embed_dim=None,short=False):
        super(DotProductAttentionStream, self).__init__(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.k_weights_cache = None
        self.k_pos_weights_cache = None





    def stream_inference(self, q, k, v, k_pos, v_pos, attn_mask=None,short_mem = None):
        if self.k_weights_cache is not None:
            k_weights_new = torch.bmm(q, k[:, [-1]].transpose(1, 2))
            k_weights = torch.cat((self.k_weights_cache[:, :, 1:], k_weights_new), dim=-1)
            self.k_weights_cache = k_weights
            k_pos_weights = self.k_pos_weights_cache
        else:
            k_weights = torch.bmm(q, k.transpose(1, 2))
            self.k_weights_cache = k_weights
            k_pos_weights = torch.bmm(q, k_pos.transpose(1, 2))
            self.k_pos_weights_cache = k_pos_weights
        attn_output_weights = k_weights + k_pos_weights

        if attn_mask is not None:
            attn_output_weights += attn_mask

        # if self.short:
        #     U = self.proj(short_mem)
        #     U = self.simod(U)
        #     U = torch.log(U) + U
        #     attn_output_weights += U

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, (v + v_pos))
        return attn_output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,short = False):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        # self.patch_wise_id_bank = nn.Conv1d(embed_dim, embed_dim, kernel_size=16, stride=16, padding=0)
        # self.id_dropout = nn.Dropout(0., True)

        self.dotproductattention = DotProductAttention(dropout=dropout,embed_dim=embed_dim,short=short)

    # def get_id_emb(self, x):
    #     print(x)
    #     id_emb = self.patch_wise_id_bank(x)
    #     id_emb = self.id_dropout(id_emb)
    #     return id_emb

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, knn=False,short_mem = None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
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

        attn_output = self.dotproductattention(q, k, v, mask, knn=knn,short_mem=short_mem)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None


class MultiheadAttentionStream(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,short=False):
        super(MultiheadAttentionStream, self).__init__(embed_dim, num_heads, dropout, bias, kdim, vdim)

        self.dotproductattention = DotProductAttentionStream(dropout,embed_dim,short=short)

        ############################
        # Cache for stream inference
        ############################
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.k_pos_cache = None
        self.v_pos_cache = None

    def stream_inference(self, q, k, v, pos, attn_mask=None, key_padding_mask=None,short_mem=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if self.q_cache is not None:
            q = self.q_cache
        else:
            _b = self.in_proj_bias
            _start = None
            _end = embed_dim
            _w = self.in_proj_weight[:_end, :]
            if _b is not None:
                _b = _b[:_end]
            q = F.linear(q, _w, _b)
            self.q_cache = q

        assert (self.k_cache is None) == (self.k_pos_cache is None)
        if self.k_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_new = F.linear(k[[-1]], _w, None)
            k = torch.cat((self.k_cache[1:], k_new))
            self.k_cache = k
            k_pos = self.k_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(k, _w, None)
            self.k_cache = k
            k_pos = F.linear(pos, _w, _b)
            self.k_pos_cache = k_pos

        assert (self.v_cache is None) == (self.v_pos_cache is None)
        if self.v_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_new = F.linear(v[[-1]], _w, None)
            v = torch.cat((self.v_cache[1:], v_new))
            self.v_cache = v
            v_pos = self.v_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, None)
            self.v_cache = v
            v_pos = F.linear(pos, _w, _b)
            self.v_pos_cache = v_pos

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k_pos = k_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v_pos = v_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
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

        attn_output = self.dotproductattention.stream_inference(q, k, v, k_pos, v_pos, mask,short_mem)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None
