# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

# from rekognition_online_action_detection.utils.registry import Registry

# FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'th14_mae_g_16_4': 1408,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'sensor': 8,

}


# @FEATURE_HEADS.register('THUMOS')
# @FEATURE_HEADS.register('TVSeries')
# @FEATURE_HEADS.register('EK100')
# @FEATURE_HEADS.register('HDD')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in cfg.INPUT.MODALITY
            self.with_motion = 'visual' not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        self.d_model = 512

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            if self.with_motion:
                self.motion_linear = nn.Sequential(                    # nn.BatchNorm1d(256),
                    # nn.GroupNorm(32,motion_size),
                    nn.Linear(motion_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                )
            if self.with_visual:
                self.visual_linear = nn.Sequential(
                    nn.Linear(visual_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                )
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Sequential(
                    nn.Linear(2 * self.d_model, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                )
        else:
            if self.with_motion:
                self.motion_linear = nn.Identity()
            if self.with_visual:
                self.visual_linear = nn.Identity()
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Identity()
        # self.num_layers = 1
        # self.gru = nn.GRU(self.d_model*2, self.d_model, self.num_layers, batch_first=True)
        # self.gru2 = nn.GRU(self.d_model, self.d_model, self.num_layers, batch_first=True)
        # self.h0 = torch.zeros(self.num_layers, 1, self.d_model)

    def forward(self, visual_input, motion_input):
        # if not hasattr(self, '_flattened'):
        #     self.gru.flatten_parameters()
        #     # self.gru2.flatten_parameters()
        #     setattr(self, '_flattened', True)

        # print(visual_input.shape)
        # print(motion_input.shape)
        B = visual_input.shape[0]
        if self.with_visual and self.with_motion:
            # print(visual_input.shape)
            visual_input = self.visual_linear(visual_input)
            motion_input  = self.motion_linear(motion_input)
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
            # print(visual_input.shape,'visual_input.shape')
            # print(motion_input.shape,'motion_input.shape')
            # h0 = self.h0.expand(-1, B, -1).to(visual_input.device)
            # visual_input,_ = self.gru1(visual_input,h0)
            # motion_input,_ = self.gru2(motion_input)


            # fusion_input,_ = self.gru(fusion_input,h0)
            fusion_input = self.input_linear(fusion_input)
        elif self.with_visual:
            # print(visual_input.shape)
            fusion_input = self.visual_linear(visual_input)
        elif self.with_motion:
            fusion_input = self.motion_linear(motion_input)
        # print(fusion_input.shape)
        # fusion_input = self.mamba(fusion_input)
        # print(fusion_input.shape)
        # print(fusion_input)
        return fusion_input


# def build_feature_head(cfg):
#     feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
#     return feature_head(cfg)
