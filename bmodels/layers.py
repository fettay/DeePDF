import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class TransposeLayer(ModuleWrapper):

    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return torch.transpose(x, self.dim1, self.dim2)
    

class BBBConv1d(ModuleWrapper):
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape, stride=1,
                 padding=0, dilation=1, bias=True, name='BBBConv1d'):
        super(BBBConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(128))
        else:
            self.register_parameter('bias', None)

        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.reset_parameters()
        self.name = name

    def out_bias(self, input, kernel):
        return  F.conv1d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def out_nobias(self, input, kernel):
        return F.conv1d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)
            
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):

        mean = self.out_bias(x, self.weight)

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon
        
        return out

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement() * calculate_kl(self.log_alpha)


class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True, name='BBBLinear'):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_value = calculate_kl
        self.name = name

        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):

        mean = F.linear(x, self.W)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.W * self.W

        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon


        return out

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()
