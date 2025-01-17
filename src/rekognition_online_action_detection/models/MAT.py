# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .feature_head_MAT import BaseFeatureHead


class MAT(nn.Module):

    def __init__(self, cfg):
        super(MAT, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = BaseFeatureHead(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = BaseFeatureHead(cfg)

        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.future_enabled = self.future_num_samples > 0

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
        self.group = self.cfg.MODEL.LSTR.GROUPS
        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            # self.enc_mode = nn.ModuleList()
            # self.cross_attention = tr.MultiheadAttention(self.d_model,1)
            index = 0
            for param in cfg.MODEL.LSTR.ENC_MODULE:
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.LongMemLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation,short=(index==0),gru=True,atten=True)
                    self.enc_modules.append(tr.TransformerDecoder(
                            enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                    # self.enc_modules.append(self.enc_mode)
                else:
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                index = index+1
            self.average_pooling = nn.AdaptiveAvgPool1d(1)
            self.max_polling = nn.AdaptiveMaxPool1d(1)
            # self.norm = nn.LayerNorm(self.d_model)
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,gru=True,atten=True)
            self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
        else:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerEncoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = tr.TransformerEncoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
        # self.norm = nn.LayerNorm(self.d_model)
        # short_layer = tr.TransformerEncoderLayer(
        #     self.d_model, self.num_heads, self.dim_feedforward,
        #     self.dropout, self.activation)
        # self.short_modules = tr.TransformerEncoder(
        #     short_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
        # Build Anticipation Generation
        if self.future_enabled:
            param = cfg.MODEL.LSTR.GEN_MODULE
            self.gen_query = nn.Embedding(param[0], self.d_model)
            gen_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,gru=True,atten=True)
            self.gen_layer = tr.TransformerDecoder(
                gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
            )
            self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)
        #     # CCI
            self.work_fusions = nn.ModuleList()
            self.fut_fusions = nn.ModuleList()
            for i in range(cfg.MODEL.LSTR.CCI_TIMES):
                work_enc_layer = tr.TransformerDecoderLayer(
                    self.d_model, self.num_heads, self.dim_feedforward,
                    self.dropout, self.activation,gru=True,atten=True)
                self.work_fusions.append(tr.TransformerDecoder(
                    work_enc_layer, 1, tr.layer_norm(self.d_model, True)))
                if i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
                    fut_enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation,gru=True,atten=True)
                    self.fut_fusions.append(tr.TransformerDecoder(
                        fut_enc_layer, 1, tr.layer_norm(self.d_model, True)))

        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        if self.cfg.DATA.DATA_NAME == 'EK100':
            self.classifier_verb = nn.Linear(self.d_model, 98)
            self.classifier_noun = nn.Linear(self.d_model, 301)
            self.dropout_cls = nn.Dropout(0.8)

    def cci(self, memory, output, mask, short_mem):
        # print(memory.shape,output.shape,'memory.shape,output.shape')
        his_memory = torch.cat([memory, output])
        # print(his_memory.shape,'his_memory.shape')
        enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        future = self.gen_layer(enc_query, his_memory, knn=True)

        dec_query = self.final_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        future_rep = [future]
        short_rep = [output]
        for i in range(self.cfg.MODEL.LSTR.CCI_TIMES):
            mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
            mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
            the_mask = torch.cat((mask1, mask, mask2), dim=-1)
            total_memory = torch.cat([memory, output, future])
            output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask, knn=True)
            # print(output.shape,'output.shape')
            short_rep.append(output)
            # print(memory.shape, output.shape, future.shape, 'memory.shape,output.shape,future.shape22')
            total_memory = torch.cat([memory, output, future])
            # print(total_memory.shape, 'total_memory.shape22')
            if i == 0:
                future = self.fut_fusions[i](dec_query, total_memory, knn=True)
                future_rep.append(future)
            elif i != self.cfg.MODEL.LSTR.CCI_TIMES - 1:
                # else:
                mask1 = torch.zeros((future.shape[0], memory.shape[0] + output.shape[0])).to(output.device)
                mask2 = tr.generate_square_subsequent_mask(future.shape[0]).to(output.device)
                future = self.fut_fusions[i](future, total_memory, tgt_mask=mask2,
                                             memory_mask=torch.cat((mask1, mask2), dim=-1), knn=True)
                # print(future.shape,'future.shape')
                future_rep.append(future)
        return short_rep, future_rep

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None,epoch=1):
        # print(visual_inputs.shape)
        # print(self.long_memory_num_samples)
        feature_TW = list()
        feature_TF = list()
        if self.long_enabled:
            # Compute long memories
            the_long_memories = self.feature_head_long(
                visual_inputs[:, :self.long_memory_num_samples],
                motion_inputs[:, :self.long_memory_num_samples]).transpose(0, 1)

            work_memories = self.feature_head_work(
                visual_inputs[:, self.long_memory_num_samples:],
                motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1)
            work_memories = self.pos_encoding(work_memories, padding=0)
            # short_mem = self.short_modules(short_mem)
            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, the_long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    if self.cfg.MODEL.LSTR.GROUPS > 0 and (
                            memory_key_padding_mask == float('-inf')).sum() < self.cfg.MODEL.LSTR.GROUPS:

                        T = the_long_memories.shape[0] // self.cfg.MODEL.LSTR.GROUPS
                        enc_query = enc_queries[0]
                        long_memories = []
                        max_mem = []
                        avg_mem = []
                        for i in range(self.cfg.MODEL.LSTR.GROUPS):
                            out = self.enc_modules[0](enc_query,the_long_memories[i * T:(i + 1) * T],
                                                      memory_key_padding_mask=memory_key_padding_mask[:,
                                                                              i * T:(i + 1) * T], knn=True,short_mem = work_memories)
                            weight = self.max_polling(out.permute(1, 2, 0)).permute(2, 0, 1)
                            out = self.average_pooling(out.permute(1, 2, 0)).permute(2, 0, 1)
                            out = out+weight
                            # max_mem.append(weight)
                            # avg_mem.append(out)
                            long_memories.append(out)
                        long_memories = torch.cat(long_memories)

                    else:
                        # print(type(the_long_memories))
                        # print(self.enc_modules[0])
                        long_memories = self.enc_modules[0](enc_queries[0], the_long_memories,
                                                            memory_key_padding_mask=memory_key_padding_mask, knn=True,short_mem = work_memories)
                        # print(type(long_memories))
                else:
                    long_memories = self.enc_modules[0](long_memories)

                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories, knn=True)
                    else:
                        # print(long_memories.shape)
                        long_memories = enc_module(long_memories, knn=True)

        if self.long_enabled:
            memory = long_memories
        # feature_T.append(long_memories)
        if self.work_enabled:
            if self.anticipation_num_samples > 0 and self.future_enabled:
                anticipation_queries = self.pos_encoding(
                    self.final_query.weight[:self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH
                                            :self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...].unsqueeze(1).repeat(1,
                                                                                                                    work_memories.shape[
                                                                                                                        1],
                                                                                                                    1),
                    padding=self.work_memory_num_samples)
                work_memories = torch.cat((work_memories, anticipation_queries), dim=0)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)
            # print(work_memories.shape,memory.shape,'11')
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                    knn=True
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                    knn=True
                )
        work_new = list()
        fut_new = list()
        if self.future_enabled:
            works, futs = self.cci(memory, output, mask, work_memories)
            work_scores = []
            fut_scores = []
            for i, work in enumerate(works):
                # print(work.shape,'work.shape')
                if i == len(works) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
                    noun_score = self.classifier_noun(work).transpose(0, 1)
                    verb_score = self.classifier_verb(work).transpose(0, 1)
                    work_scores.append(self.classifier(self.dropout_cls(work)).transpose(0, 1))
                else:
                    work_scores.append(self.classifier(work).transpose(0, 1))
                    work_new.append(self.classifier(work).transpose(0, 1))
            for i, fut in enumerate(futs):
                # print(fut.shape,'fut.shape')
                if i == 0:
                    fut_scores.append(self.classifier(
                        F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1)).transpose(0,
                                                                                                                      1))
                    fut_new.append(F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1).permute(1, 0, 2))
                else:
                    fut_scores.append(self.classifier(fut).transpose(0, 1))
                    fut_new.append(fut.permute(1,0,2))
                    if i == len(futs) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
                        fut_noun_score = self.classifier_noun(fut).transpose(0, 1)
                        fut_verb_score = self.classifier_verb(fut).transpose(0, 1)
            # print(work_scores[0].shape, fut_scores[0].shape)
            feature_TW.append(torch.stack(works).mean(0))
            feature_TW.append(torch.stack(work_scores).mean(0))

            feature_TF.append(torch.stack(futs).mean(0))
            feature_TF.append(torch.stack(fut_scores).mean(0))
            return (work_scores, fut_scores),feature_TW, feature_TF if self.cfg.DATA.DATA_NAME != 'EK100' else (
            work_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score)

        # Compute classification score
        score = self.classifier(output)
        # print(score.shape)
        return score.transpose(0, 1)


@registry.register('MAT')
class MATStream(MAT):

    def __init__(self, cfg):
        super(MATStream, self).__init__(cfg)

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