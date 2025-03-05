import torch

from torch.nn import Module, Parameter
from torch.nn import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
from torch.nn import GELU

@torch.jit.script
def gtu(inputs):
    x, y = inputs.chunk(2, dim=1)
    return torch.sigmoid(x) * torch.tanh(y)

class GTU(torch.nn.Module):
    def forward(self, inputs):
        return gtu(inputs)

class Conv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=1,
            padding=dilation * (kernel_size - 1) // 2, dilation=dilation,
            groups=groups, bias=bias,
        )
        torch.nn.utils.parametrizations.weight_norm(self)

class Upsample(torch.nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, groups=1, bias=True):
        self.scale_factor = scale_factor
        super().__init__(
            in_channels, out_channels, kernel_size=scale_factor * kernel_size, stride=scale_factor,
            padding=(kernel_size - 1) * scale_factor // 2,
            groups=groups, bias=bias,
        )
        torch.nn.utils.parametrizations.weight_norm(self)

class Downsample(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, groups=1, bias=True):
        self.scale_factor = scale_factor
        super().__init__(
            in_channels, out_channels, kernel_size=scale_factor * kernel_size, stride=scale_factor,
            padding=(kernel_size - 1) * scale_factor // 2,
            groups=groups, bias=bias,
        )
        torch.nn.utils.parametrizations.weight_norm(self)
