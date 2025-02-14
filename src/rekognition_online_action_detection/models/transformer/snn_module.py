import math

import torch
# import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from .layers import LIF, mem_update
from functools import partial
# from .layers import mem_update
from DCLS.construct.modules import Dcls1d
from spikingjelly.activation_based import surrogate, neuron
class LIFLayer(neuron.LIFNode):

    def __init__(self, timestep=0, **cell_args):
        super(LIFLayer, self).__init__()
        assert timestep > 0, 'the number of time steps should be specified'
        self.timestep = timestep
        self.rate_flag = cell_args.get("rate_flag", False)

        tau = 1.0 / (1.0 - torch.sigmoid(cell_args['decay'])).item()

        super().__init__(tau=tau, decay_input=False, v_threshold=cell_args['thresh'], v_reset=None,
                         detach_reset=cell_args['detach_reset'], step_mode='s')

    def forward(self, x, rate=None):
        if self.rate_flag and self.training:
            assert x.shape[0] == self.timestep
            assert rate is not None
            assert x.shape[1:] == rate.shape

            self.reset()
            spikes = []
            vs = []

            self.post_elig = 0.
            self.post_elig_factor = 1.

            if isinstance(self, neuron.LIFNode):
                lam = 1.0 - 1. / self.tau
            else:
                raise NotImplementedError()

            for t in range(self.timestep):
                self.v_float_to_tensor(x[t])
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                vs.append(self.v)

                # sg = torch.autograd.grad(outputs=spike.sum(), inputs=self.v, retain_graph=True)[0]
                sigmoid_alpha = 4.0
                sgax = ((self.v - self.v_threshold) * sigmoid_alpha).sigmoid_()
                sg = (1. - sgax) * sgax * sigmoid_alpha

                spikes.append(spike)

                self.post_elig = 1. / (t + 1) * (t * self.post_elig + self.post_elig_factor * sg)

                if self.v_reset is not None:  # hard-reset
                    self.post_elig_factor = 1. + self.post_elig_factor * (lam * (1. - spike) - lam * self.cell.v * sg)
                else:  # soft-reset
                    if not self.detach_reset:  # soft-reset w/ reset_detach==False
                        self.post_elig_factor = 1. + self.post_elig_factor * (lam - lam * sg)
                    else:  # soft-reset w/ reset_detach==True
                        self.post_elig_factor = 1. + self.post_elig_factor * (lam)

                self.neuronal_reset(spike)
            out = torch.stack(spikes, dim=0)
            gu = self.post_elig.clone().detach()

            rate = out.mean(dim=0).clone().detach() + (rate * gu) - (rate * gu).detach()
            return out, rate
        else:
            self.reset()
            spikes = []
            for t in range(self.timestep):
                self.v_float_to_tensor(x[t])
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                spikes.append(spike)
                self.neuronal_reset(spike)

            out = torch.stack(spikes, dim=0)
            return out, torch.zeros_like(out[0])
class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )

            output = F.pad(output, [self.pad_pixels]*2)
            pad_values = pad_values.view(1, -1, 1)
            # print(output.shape,pad_values.shape)
            output[:, :, 0 : self.pad_pixels] = pad_values
            output[:, :, -self.pad_pixels :] = pad_values

        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()

        conv1x1 = nn.Conv1d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv1d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        # self.shortcut = nn.Conv1d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

        self.body = nn.Sequential(conv1x1, bn, conv3x3)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         if m.groups > 1:  # 深度卷积特殊初始化
        #             nn.init.normal_(m.weight, mean=0, std=0.02 / math.sqrt(m.groups))
        #         else:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0.5)  # 偏置项非零初始化

    def forward(self, x):
        return self.body(x)





class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        mlp_ratio=4.0,
        drop_path=0.0,
        bias = False,
    ):
        super().__init__()

        # self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)
        # self.temporal_proj = nn.Conv1d(
        #     dim, dim, kernel_size=1, padding=0, groups=1, bias=bias
        # ) # 新增时序投影
        # self.temporal_norm = nn.BatchNorm1d(dim)
        self.mlp_ratio=mlp_ratio
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0,
        #                                                surrogate_function=surrogate.ATan(alpha = 5.0), detach_reset=True,
        #                                                step_mode='m', decay_input=False, store_v_seq = True)
        self.conv1 = nn.Conv1d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=bias
        )
        # self.conv1_1 = nn.Conv1d(
        #     dim, dim , kernel_size=1, padding=0, groups=1, bias=bias
        # )
        # self.conv1 = Dcls1d(dim, dim * mlp_ratio, kernel_count=1, groups = 1, dilated_kernel_size = 7, bias=False, version='gauss')
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bn1 = nn.BatchNorm1d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0,
        #                                                surrogate_function=surrogate.ATan(alpha = 5.0), detach_reset=True,
        #                                                step_mode='m', decay_input=False, store_v_seq = True)
        self.conv2 = nn.Conv1d(
            dim * mlp_ratio, output_dim, kernel_size=3, padding=1, groups=1, bias=bias
        )

        # self.conv2 = Dcls1d(dim * mlp_ratio, output_dim, kernel_count=1, groups=1, dilated_kernel_size=7, bias=False, version='gauss')
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bn2 = nn.BatchNorm1d(output_dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, N = x.shape
        # if hasattr(self, 'prev_state'):
        #     x = self.temporal_norm(self.temporal_proj(self.prev_state.flatten(0,1))).reshape(T, B, C, -1)
        #     x = x + 0.5 * x
        # x = self.Conv(x) + x
        x_feat = x
        # print(x.shape)
        x = self.lif1(x).flatten(0, 1)
        # x1 = self.conv1_1(x)
        x = self.bn1(self.conv1(x)).reshape(T, B, self.mlp_ratio * C, -1)
        # print(x.shape)
        # x = F.pad(x, (3, 3), 'constant', 0)
        # print(x.shape)
        # x = x.reshape(T, B, self.mlp_ratio * C, N)
        # self.lif1(x)
        # x = x_feat + x
        x = self.lif2(x).flatten(0,1)
        # x2 = self.conv2_1(x)
        x = self.bn2(self.conv2(x)).reshape(T, B, C, -1)
        # x = x.reshape(T, B, C, N)
        # self.lif2(x)
        # x = F.pad(x, (3, 3), 'constant', 0)

        # print(x.shape)
        x = x_feat + x

        # self.prev_state = x.detach()  #

        return x

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm1d(out_planes))
            # torch.nn.init.constant_(self.bn.weight, 1)
            # torch.nn.init.constant_(self.bn.bias, 0)





class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_conv= Dcls1d(in_features, hidden_features,kernel_count=1, groups=1, padding=0, dilated_kernel_size=1, bias=False,version='gauss')
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1)
        # self.fc2_conv = Dcls1d(hidden_features, out_features, kernel_count=1, groups=1, padding=0, dilated_kernel_size=1,
        #                        bias=False, version='gauss')
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N= x.shape
        # x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()


        return x

class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


    def forward(self, q,k):

        T, B, C, N = q.shape
        x_for_q = q.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_q)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        x_for_kv = k.flatten(0, 1)
        k_conv_out = self.k_conv(x_for_kv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim = 3, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, N)
        x = self.proj_lif(x)

        return x
class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, q,k):

        T, B, C, N = q.shape
        k_ori = k
        q = self.q_lif(q)
        x_for_q = q.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_q)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,-1).contiguous()

        q = q_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()


        x_for_k = self.k_lif(k)
        x_for_k = x_for_k.flatten(0, 1)
        k_conv_out = self.k_conv(x_for_k)

        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,-1).contiguous()

        k = k_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x_for_v = self.v_lif(k_ori)
        x_for_v = x_for_v.flatten(0, 1)
        v_conv_out = self.v_conv(x_for_v)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,-1).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, -1).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,-1)

        return x
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        # self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim),RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim))
        #
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim),RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim))

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm1d(dim)
        )

        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
    def forward(self, q,k_):
        T, B, C, N = q.shape
        # N = H * W
        x = q
        # k_ = k
        # x = self.head_lif(q)
        # macac_calculator.mac_count += 3 * q.size(1) * q.size(1) * q.size(-1)  # QKV投影
        # macac_calculator.mac_count += 2 * q.size(1) * q.size(-1) * q.size(-1)  # 注意力矩阵
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, N)
        k = self.k_conv(k_.flatten(0, 1)).reshape(T, B, C, -1)
        # print(k.shape)
        v = self.v_conv(k_.flatten(0, 1)).reshape(T, B, C, -1)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        # attn = (F.normalize(k, dim=-1).transpose(-2, -1) @ F.normalize(v, dim=-1))
        # logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        # attn = attn * logit_scale
        # attn = attn.softmax(dim=-1)
        # x = q @ attn

        x = x.transpose(3, 4).reshape(T, B, C, -1).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, -1)
        x = x.reshape(T, B, C, -1)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, N)

        return x

class MS_Block_Encoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        # self.conv = nn.Conv1d(dim,dim,kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, q):
        T,B,C,N = q.shape
        # mask = self.conv(q.flatten(0,1)).reshape(T,B,C,N)
        q2 = self.attn(q,q)
        q = q+self.drop_path(q2)

        q2 = self.mlp(q2)

        x = q + self.drop_path(q2)
        return x
class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.attn1 = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        # self.conv = nn.Conv1d(dim,dim,kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, q,k):
        T,B,C,N = q.shape
        # mask = self.conv(q.flatten(0,1)).reshape(T,B,C,N)
        q2 = self.attn(q,q)
        q = q+self.drop_path(q2)

        q2 = self.attn1(q,k)
        q = q + self.drop_path(q2)

        q2 = self.mlp(q2)

        x = q + self.drop_path(q2)
        return x

class MS_Block_Cross_Weight(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.attn1 = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, q,k, atte_mask=None, knn=False):
        # res_k = k
        # res_q = q
        q2 = self.attn(q,q)
        q = q+self.drop_path(q2)
        k2 = self.attn(k, k)
        k = k + self.drop_path(k2)

        q2 = self.attn1(q,k)
        q = q + self.drop_path(q2)
        k2 = self.attn1(k,q)
        k = k+self.drop_path(k2)

        k2 = self.mlp(k2)
        q2 = self.mlp(q2)


        x = q + self.drop_path(q2)
        x1 = k+self.drop_path(k2)
        x= torch.cat([x,x1],dim=2)
        return x
        # return x,x1

class MS_Block_Cross_No_Weight(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.attn1 = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.cross_attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.cross_attn1 = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.mlp2 = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, q,k, atte_mask=None, knn=False):
        res_k = k
        res_q = q

        q2 = self.attn(q,q)
        q = q+self.drop_path(q2)

        k2 = self.attn1(k,k)
        k = k+ self.drop_path(k2)

        q2 = self.cross_attn(q,res_k)
        q = q + self.drop_path(q2)

        k2 = self.cross_attn1(k,res_q)
        k = k+self.drop_path(k2)

        # print(k2.shape,q2.s)
        k2 = self.mlp1(k2)
        q2 = self.mlp2(q2)


        x = q + self.drop_path(q2)
        x1 = k + self.drop_path(k2)
        # x = x+x1
        return x,x1
class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=1,
        stride=1,
        padding=0,
        first_layer=True,
    ):
        super().__init__()

        # self.avg = nn.AvgPool1d(kernel_size=2,stride=2)
        self.encode_conv = nn.Conv1d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # self.donw = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.dim = embed_dims
        self.encode_bn = nn.BatchNorm1d(embed_dims)
        # if not first_layer:
        #     self.encode_lif = MultiStepLIFNode(
        #         tau=2.0, detach_reset=True, backend="cupy"
        #     )

    def forward(self, x):
        T, B, _,_= x.shape

        # if hasattr(self, "encode_lif"):
        #     x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, N = x.shape

        x = self.encode_bn(x).reshape(T, B, -1, N).contiguous()
        # print(x.shape)
        return x


class SpatialAttention_SNN(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention_SNN, self).__init__()
        self.sa = nn.Conv1d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.bn = nn.BatchNorm1d(1)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x):
        T,B,C,N = x.shape
        # x = x.flaten(0,1)
        x_avg = torch.mean(x, dim=2, keepdim=True)
        x_max, _ = torch.max(x, dim=2, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=2)
        x2 = self.lif(x2)
        sattn = self.sa(x2.flatten(0,1))
        # print(sattn.shape)
        sattn = self.bn(sattn).reshape(T,B,-1,N)
        # sattn = sattn.reshape(T,B,-1,N)
        return sattn


class ChannelAttention_SNN(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_SNN, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.ca = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.BatchNorm1d(dim // reduction),
            nn.Conv1d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        T,B,C,N = x.shape
        x = self.lif(x)
        x = x.flatten(0,1)
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        cattn = cattn.reshape(T,B,C,-1)
        return cattn


class PixelAttention_SNN(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_SNN, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, pattn1):
        T, B, C, N = x.shape
        x = x.flatten(0,1)
        pattn1 = pattn1.flatten(0,1)
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t n -> b (c t) n')(x2)
        # print(x2.shape)
        x2 = self.lif(x2.reshape(T,B,-1,N))
        pattn2 = self.pa2(x2.flatten(0,1))
        pattn2 = self.bn(pattn2)
        # pattn2 = self.sigmoid(pattn2)

        pattn2 = pattn2.reshape(T,B,-1,N)

        return pattn2


class CGAFusion_SNN(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion_SNN, self).__init__()
        self.sa = SpatialAttention_SNN(dim)
        self.ca = ChannelAttention_SNN(dim, reduction)
        self.pa = PixelAttention_SNN(dim)
        self.conv = nn.Conv1d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        T,B,C,N = x.shape
        # x = x.flatten(0,1)
        # y = y.flatten(0,1)

        initial = x+y
        # print(initial.shape)
        cattn = self.ca(initial).flatten(0,1)
        sattn = self.sa(initial).flatten(0,1)
        # print(sattn.shape,cattn.shape)
        pattn1 = sattn + cattn
        pattn1 = pattn1.reshape(T,B,C,N)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result.flatten(0,1))

        result = result.reshape(T,B,-1,N)
        return result