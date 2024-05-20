# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import torch

# from .mamba import Mamba,Block

from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .feature_head import build_feature_head
from .deco_convnet import *
from timm.models.layers import trunc_normal_,DropPath
from .transformer.multihead_attention import MultiheadAttentionStream as MultiheadAttention
from mamba_ssm.ops.triton.layernorm import RMSNorm,rms_norm_fn


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

class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg)

        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.future_enabled = self.future_num_samples > 0
        self.d_model = self.feature_head_work.d_model
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.qN = 256

        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.activation = cfg.MODEL.LSTR.ACTIVATION

        # Build position encoding
        # self.convNet = DECO_ConvNet(d_model=self.d_model, enc_dims=[self.d_model, self.d_model , self.d_model],
        #                             enc_depth=[2, 6, 2],qN=self.qN)
        # self.query = nn.Embedding(self.qN,self.d_model)
        self.sample = nn.ModuleList()
        self.level=1
        for i in range(self.level):
            self.sample.append(TemporalMaxer(kernel_size=3,stride=2,padding=1,n_embd=self.d_model))
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
        # self.query = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        # self.query_embed = nn.Embedding(self.qN, self.d_model)
        # Build LSTR encoder
        if self.long_enabled:
            self.enc_modules = nn.ModuleList()
            self.enc_modules.append(
                DecoEncoder(enc_dims=[self.d_model,self.d_model*2,self.d_model], enc_depth=[2, 6, 2]))
            self.average_pooling = nn.AdaptiveAvgPool1d(1)
            self.long_pooling = nn.AdaptiveAvgPool1d(self.cfg.MODEL.LSTR.GROUPS)
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)
        self.normalize_before=False
        self.num_decoder_layers=2
        self.return_intermediate_dec=False
        # Build LSTR decoder

        param = cfg.MODEL.LSTR.DEC_MODULE
        decoder_layer = DecoDecoderLayer(self.d_model, self.normalize_before, qN=40)
        decoder_norm = nn.LayerNorm(self.d_model)

        if self.long_enabled:
            param = cfg.MODEL.LSTR.DEC_MODULE
            decoder_layer = DecoDecoderLayer(self.d_model, self.normalize_before, qN=40)
            decoder_norm = nn.LayerNorm(self.d_model)
            self.dec_modules = DecoDecoder(decoder_layer, param[1], decoder_norm,
                                           return_intermediate=self.return_intermediate_dec, qN=40)
            fut_layer = DecoDecoderLayer(self.d_model, self.normalize_before, qN=48)
            fut_norm = nn.LayerNorm(self.d_model)
            self.fut_modules = DecoDecoder(fut_layer, param[1], fut_norm,
                                           return_intermediate=self.return_intermediate_dec, qN=48)

        self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)




        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        if self.cfg.DATA.DATA_NAME == 'EK100':
            self.classifier_verb = nn.Linear(self.d_model, 98)
            self.classifier_noun = nn.Linear(self.d_model, 301)
            self.dropout_cls = nn.Dropout(0.8)

    def reshape_feature(self,base_feature):
        out_x = None
        for i in range(len(base_feature)):
            if out_x is None:
                out_x = base_feature[i]
            else:
                # 假设 out_x 和 base_feature[i] 的第 3 个维度长度分别为 T_out 和 T_base，则
                T_out = out_x.shape[2]
                T_base = base_feature[i].shape[2]

                # 首先计算需要在左侧和右侧各填充多少列
                left_pad = (T_out - T_base) // 2
                right_pad = T_out - T_base - left_pad

                # 使用更新后的左、右填充长度进行填充操作（注意第 1 个维度已经被忽略）
                base_feature_after = F.pad(base_feature[i], [left_pad, right_pad], value=0)

                out_x = out_x + base_feature_after

        norm = LayerNorm(out_x.shape[1])
        return out_x.cuda()

    def cci(self, memory, output, work_mask):

        his_memory = torch.cat([memory, output], dim=-1)
        enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[0], 1)
        # print(enc_query.shape,his_memory.shape,'enc_query.shape,his_memory.shap')
        enc_query = enc_query.permute(1, 2, 0).view(his_memory.shape[0], his_memory.shape[1],
                                          his_memory.shape[2])
        # print(his_memory.shape,enc_query.shape,'his_memory.shape,enc_query.shape')
        future, mask = self.gen_layer(his_memory, his_memory, enc_query, work_mask)

        # dec_query = self.final_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        # dec_query = dec_query.permute(1, 2, 0).view(his_memory.shape[0], his_memory.shape[1],
        #                                             his_memory.shape[2])
        future_rep = [future]
        short_rep = [output]
        mask1 = torch.zeros(output.shape).to(output.device)
        mask2 = torch.zeros(future.shape).to(output.device)

        # the_mask = torch.cat((mask1, work_mask, mask2), dim=-1)
        for i in range(self.cfg.MODEL.LSTR.CCI_TIMES):
            total_memory = torch.cat([memory, output, future], dim=-1)
            # print(output.shape,mask1.shape,'output.shape,mask1.shape')
            output, the_mask = self.work_fusions[i](output, total_memory, self.d_model, mask1)
            short_rep.append(output)
            total_memory = torch.cat([memory, output, future], dim=-1)
            # print(total_memory.shape)
            # print(mask1.shape,work_mask.shape,mask2.shape)
            the_mask = torch.cat((mask1, work_mask, mask2), dim=-1)
            # print(total_memory.shape)
            if i == 0:
                future, the_mask = self.fut_fusions[i](total_memory, total_memory, self.d_model, the_mask
                                                       )
                future_rep.append(future)
            elif i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
                future, the_mask = self.fut_fusions[i](future, total_memory, self.d_model, the_mask)
                future_rep.append(future)

        return short_rep, future_rep

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=0):

        the_long_memories = self.feature_head_long(
            visual_inputs[:, :self.long_memory_num_samples],
            motion_inputs[:, :self.long_memory_num_samples]).permute(0,2,1)
        memory_key_padding_mask = torch.zeros(the_long_memories.shape).to(the_long_memories.device)
        #LBN 111 1B1
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1,the_long_memories.shape[0], 1)
        # query_embed = query_embed.permute(1, 2, 0).view(the_long_memories.shape[0], the_long_memories.shape[1], the_long_memories.shape[2])
        # query,the_long_memories,memory_key_padding_mask = self.convNet(the_long_memories,query_embed,memory_key_padding_mask)
        the_long_memories_list=[]
        the_long_mask_list = []
        for j in range(len(self.sample)):
            if len(self.enc_modules) > 0:
                # Encode long memories
                if self.cfg.MODEL.LSTR.GROUPS > 0:
                    T = the_long_memories.shape[-1] // self.cfg.MODEL.LSTR.GROUPS
                    # print(T,'T')
                    avg_memories = []
                    avg_mask = []
                    for i in range(self.cfg.MODEL.LSTR.GROUPS):
                        # print(the_long_memories[:, :, i * T:(i + 1) * T].shape,'the_long_memories[i * T:(i + 1) * T].shape')
                        out, mask= self.enc_modules[0](the_long_memories[:, :, i * T:(i + 1) * T], mask = memory_key_padding_mask[:,:,
                                                                              i * T:(i + 1) * T])
                        out = self.average_pooling(out)
                        avg_memories.append(out)
                        avg_mask.append(mask)
                    avg_memories = torch.cat(avg_memories,dim=-1)
                    avg_mask = torch.cat(avg_mask,dim=-1)
                    the_long_memories_list.append(avg_memories)



        # Concatenate memories
        if self.long_enabled:
            memory = self.reshape_feature(the_long_memories_list)
            # print(memory.shape,'memory.shap')


        if self.work_enabled:
            work_memories = self.pos_encoding(self.feature_head_work(
                        visual_inputs[:, self.long_memory_num_samples:],
                        motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1), padding=0)
            if self.anticipation_num_samples > 0:
                anticipation_queries = self.pos_encoding(
                    self.final_query.weight[:self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH
                                                    :self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...].unsqueeze(1).repeat(1,
                                                                                                                            work_memories.shape[
                                                                                                                                1],
                                                                                                                            1),
                            padding=self.work_memory_num_samples)
            work_memories = torch.cat((work_memories, anticipation_queries), dim=0).permute(1, 2, 0)
            work_mask = torch.zeros(work_memories.shape).to(work_memories.device)
                    # print(work_memories.shape,'work_memories.shape')
                    # print(memory.shape,'memory.shape')
                    # Compute output

            if self.long_enabled:
                work, work_mask = self.dec_modules(
                    work_memories,
                    memory=memory,
                    d_model=self.d_model,
                    mask=work_mask
                )
                # print(anticipation_queries.shape)
                fut_mem = torch.cat([work,anticipation_queries.permute(1,2,0)],dim=-1)
                memory = torch.cat([work,memory],dim=-1)
                # print(memory.shape)
                fut_mask = torch.zeros(anticipation_queries.shape).to(memory.device)
                # print(work_mask.shape,fut_mask.shape,'work_mask.shape,fut_mask.shape')
                fut_mask = torch.cat([work_mask,fut_mask.permute(1,2,0)],dim=-1)
                # print(fut_mem.shape,'fut_mem.shape')
                fut, work_mask = self.fut_modules(
                    fut_mem,
                    memory=memory,
                    d_model=self.d_model,
                    mask = fut_mask
                )
        # print(work.shape,fut.shape,'work.shape,fut.shape')
        works, futs = [],[]
        works.append(work)
        futs.append(fut)
        if self.future_enabled:
            # works, futs = self.cci(memory, work, mask)
            work_scores = []
            fut_scores = []
            for i, work in enumerate(works):
                # print(work.shape,i,'ork.shap')
                if self.cfg.DATA.DATA_NAME == 'EK100':
                    noun_score = self.classifier_noun(work.permute(0,2,1)).transpose(0, 1)
                    verb_score = self.classifier_verb(work.permute(0,2,1)).transpose(0, 1)
                    work_scores.append(self.classifier(self.dropout_cls(work.permute(0,2,1))))
                else:
                    work_scores.append(self.classifier(work.permute(0,2,1)))
            for i, fut in enumerate(futs):
                # print(fut.shape,i,'fut.shape')
                if  self.cfg.DATA.DATA_NAME == 'EK100':
                    fut_noun_score = self.classifier_noun(fut.permute(0,2,1))
                    fut_verb_score = self.classifier_verb(fut.permute(0,2,1))
                else:
                    fut_scores.append(self.classifier(fut.permute(0, 2, 1)))

            # print(work_scores[0].shape, fut_scores[0].shape,'work_scores[0].shape, fut_scores[0].shape')
            return (work_scores, fut_scores) if self.cfg.DATA.DATA_NAME != 'EK100' else (
            work_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score)

        # Compute classification score
        score = self.classifier(work)

        return score.transpose(0, 1)


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

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)

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

            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask)
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
            for enc_module in  self.enc_modules:
                long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask

            # Compute output
            output = self.dec_modules(
                work_memories.permute(1, 2, 0),
                memory=memory.permute(1, 2, 0),
                d_model=self.d_model
            )


        # Compute classification score
        score = self.classifier(output.permute(0,2,1))
        print(score.shape,'score')
        return score.transpose(0, 1)

# class MambaBlock(nn.Module):
#     def __init__(self,d_model,expand=1):
#         super(MambaBlock, self).__init__()
#         self.d_model = d_model
#         self.long = True
#         self.short = True
#         self.expand = expand
#         if self.long:
#             self.mamba = create_block(self.d_model,expand=1)
#         if self.short:
#             self.shortlayer = nn.Sequential(
#                 nn.Conv1d(self.d_model,self.d_model,kernel_size=1),nn.GELU(),
#                 nn.Conv1d(self.d_model,self.d_model,kernel_size=3,padding=1),nn.GELU(),
#                 nn.Conv1d(self.d_model, self.d_model, kernel_size=3,padding=1),nn.GELU()
#             )
#         self.norm = nn.LayerNorm(self.d_model*2)
#     def forward(self, x):
#         # indetity = x
#         # print(x.shape)
#         long_mem, res = self.mamba(x.permute(0,2,1))
#         short_mem = self.shortlayer(x)
#         fix = torch.cat([long_mem,short_mem.permute(0,2,1)],dim=-1).permute(0,2,1)
#
#         # fix = self.norm(fix).permute(0,2,1)
#
#         return fix

class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim  # 768
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim,num_layers=1,d_state=16, d_conv=4, expand=2,drop_path=0.2):
        super().__init__()
        self.dim = dim
        # self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(dim)
        self.mamba = nn.ModuleList()
        for i in range(num_layers):
            self.mamba.append(Mamba(
                d_model=dim  # Model dimension d_model
            ))
        # self.attention = MultiheadAttention(self.dim,1)
        # self.norm1 = nn.LayerNorm(dim)
        # self.ffn = FFN(self.dim,self.dim//2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv1d):
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    def forward(self, x,mask=None):
        # print('x',x.shape)
        B, L, C = x.shape

        # print(x,'xxxxxxxxxxxxxxxxxxx')
        # x_norm = self.norm(x)
        # print(x_norm, 'x_norm0000000000000000')
        x_list = []
        for mod in self.mamba:
            # print(x.shape)

            x_norm = mod(x)

            # x_norm = mod(x)


            # print(x_norm.shape)

        # x_norm = self.norm(x_norm)
        # if mask is not None:
        #     mask  = mask.unsqueeze(-1).repeat(1,1,self.hidden_dim)
        #     print(mask.shape)
        #     x_norm = torch.bmm(mask
        # print(out,'out11111111111111111111111111')
        # out = x + self.drop_path(self.ffn(x_norm))

        # out = self.ffn(x_norm)
        # out,_ = self.attention(out,out,out)
        out = x + self.drop_path(x_norm)
        return out