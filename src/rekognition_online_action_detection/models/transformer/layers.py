import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron

from torch.nn.common_types import _size_2_t
decay = 0.25  # 0.25 # decay constants
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)




class LIF(neuron.LIFNode):
    def __init__(self,tau=2.0, detach_reset=True, backend="cupy",v_threshold=1.):
        super().__init__(tau=2., decay_input=True, v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)

        for t in range(x_seq.shape[0]):
            # print(v.shape, x_seq[t].shape,'v.shape, x_seq[t].shape')
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v


class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class BN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
        return functional.seq_to_ann_forward(x, self.bn)


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(layer.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, bias=bias, padding_mode='zeros',
                         step_mode='m')


class Conv1x1(layer.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode='zeros', step_mode='m')


class Linear(layer.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, step_mode='m')
