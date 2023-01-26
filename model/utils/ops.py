'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad

import math
from fractions import gcd

import copy

CONV_SELECT = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d}
BN_SELECT = {'conv1d': nn.BatchNorm1d, 'conv2d': nn.BatchNorm2d, 'LN': nn.LayerNorm}

def clones(_to_clone_module, _clone_times):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(_to_clone_module) for _ in range(_clone_times)])

def clip_grads(params, clip_norm_val=35):
    params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, max_norm=clip_norm_val, norm_type=2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = None
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

"""
class CBA(nn.Module):
    # A Conv2d -> Batchnorm -> silu/leaky relu block

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", use_bn=True, pad=None, norm='BN'):
        super().__init__()
        if pad is None:
            # same padding
            pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
   
        if norm == 'GN':
            self.bn = nn.GroupNorm(32, out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = get_activation(act, inplace=True)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act((self.conv(x)))
"""

class CBA(nn.Module):
    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", use_bn=True, pad=None, norm='BN', group_num=None, conv='conv2d', res_type=False):
        super().__init__()
        self.res_type = res_type

        if pad is None:
            pad = (ksize - 1) // 2
        self.conv = CONV_SELECT[conv](
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
   
        if norm == 'GN':
            if group_num is None:
                self.bn = nn.GroupNorm(gcd(32, out_channels), out_channels)
            else:
                self.bn = nn.GroupNorm(gcd(group_num, out_channels), out_channels)
        else:
            self.bn = BN_SELECT[conv](out_channels)

        self.act = get_activation(act, inplace=True)
        if not use_bn:
            self.bn = None

        if self.res_type:
            assert norm is not None
            self.conv2 = CONV_SELECT[conv](
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
            )
            if norm == 'GN':
                 self.bn2 = nn.GroupNorm(gcd(32, out_channels), out_channels) if group_num is None else nn.GroupNorm(gcd(group_num, out_channels), out_channels)
            else:
                self.bn2 = BN_SELECT[conv](out_channels)

            if in_channels != out_channels or stride != 1:
                if norm == 'GN':
                    self.transform = nn.Sequential(
                        CONV_SELECT[conv](in_channels, out_channels, kernel_size=3 if stride!=1 else 1, stride=stride, padding=1 if stride!=1 else 0, groups=1, bias=False),
                        nn.GroupNorm(gcd(32, out_channels), out_channels) if group_num is None else nn.GroupNorm(gcd(group_num, out_channels), out_channels))
                elif norm == 'BN':
                    self.transform = nn.Sequential(
                        CONV_SELECT[conv](in_channels, out_channels, kernel_size=3 if stride!=1 else 1, stride=stride, padding=1 if stride!=1 else 0, groups=1, bias=False),
                        BN_SELECT[conv](out_channels))
                else:
                    raise NotImplementedError('Type {} not supported.'.format(norm))
            else:
                self.transform = None

    def forward(self, x):

        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.act is not None:
            out = self.act(out)

        if self.res_type:
            out = self.conv2(out)
            out = self.bn2(out)           
            if self.transform is not None:
                out += self.transform(x)
            else:
                out += x
            if self.act is not None:
                out = self.act(out)

        return out

class MLP(nn.Module):
    """A Linear -> norm -> activation block"""

    def __init__(
        self, num_in, num_out=None,  bias=True, act="relu", norm='GN', group_num=None, res_type=False):
        super().__init__()
        if num_out is None:
            num_out = num_in
            
        self.linear = nn.Linear(num_in, num_out, bias=bias)
        self.res_type = res_type

        if norm is not None:
            if norm == 'GN':
                self.norm = nn.GroupNorm(gcd(32, num_out), num_out) if group_num is None else nn.GroupNorm(gcd(group_num, num_out), num_out)
            elif norm == 'LN':
                self.norm = nn.LayerNorm(num_out)
            elif norm == 'BN':
                self.norm = nn.BatchNorm1d(num_out)
            else:
                raise NotImplementedError('Type {} not supported.'.format(norm))
        else:
            self.norm = None

        if act is not None:
            self.act = get_activation(act, inplace=True)
        else:
            self.act = None

        if self.res_type:
            assert norm is not None
            self.linear2 = nn.Linear(num_out, num_out, bias=bias)
            if norm == 'GN':
                 self.norm2 = nn.GroupNorm(gcd(32, num_out), num_out) if group_num is None else nn.GroupNorm(gcd(group_num, num_out), num_out)
            elif norm == 'LN':
                self.norm2 = nn.LayerNorm(num_out)
            elif norm == 'BN':
                self.norm2 = nn.BatchNorm1d(num_out)
            else:
                raise NotImplementedError('Type {} not supported.'.format(norm))           

            if num_in != num_out:
                if norm == 'GN':
                    self.transform = nn.Sequential(
                        nn.Linear(num_in, num_out, bias=bias),
                        nn.GroupNorm(gcd(32, num_out), num_out) if group_num is None else nn.GroupNorm(gcd(group_num, num_out), num_out))
                elif norm == 'LN':
                    self.transform = nn.Sequential(
                        nn.Linear(num_in, num_out, bias=bias),
                        nn.LayerNorm(num_out))
                elif norm == 'BN':
                    self.transform = nn.Sequential(
                        nn.Linear(num_in, num_out, bias=bias),
                        nn.BatchNorm1d(num_out))
                else:
                    raise NotImplementedError('Type {} not supported.'.format(norm))
            else:
                self.transform = None

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        if self.res_type:
            out = self.linear2(out)
            out = self.norm2(out)           
            if self.transform is not None:
                out += self.transform(x)
            else:
                out += x
            if self.act is not None:
                out = self.act(out)

        return out

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, act="silu", use_bn=True,  norm='BN'):
        super(SeparableConv, self).__init__()
        self.use_bn = use_bn
        if out_channels is None:
            out_channels = in_channels

    
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same', groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', bias=True)

        if norm == 'GN':
            self.bn = nn.GroupNorm(32, out_channels)
        else:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        
        if act is not None:
            self.act = get_activation(act, inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

class FFN(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.FC = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.FC(t_rec)
        output = output.view(T, b, -1)

        return output